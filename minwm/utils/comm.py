"""Distributed communication helpers.

Adapted from detectron2's ``detectron2/utils/comm.py``: every query degrades
gracefully so it can be called before (or without) ``torch.distributed`` being
initialized — single-process runs report rank 0, world size 1.

Two coordinate systems are exposed:

- **global** (``get_rank`` / ``get_world_size``): the flat process rank over
  ``torch.distributed``.
- **data-parallel** (``get_dp_rank`` / ``get_dp_world_size``): the rank within
  the DP dimension when sequence parallelism (SP) is active. Ranks in the same
  SP group share one DP rank, so a data sampler keyed on DP coordinates feeds
  them identical batches. Falls back to the global coordinates when SP is not
  initialized.
"""

import torch.distributed as dist

__all__ = [
    "get_world_size",
    "get_rank",
    "is_main_process",
    "synchronize",
    "all_reduce_flag",
    "get_dp_world_size",
    "get_dp_rank",
]


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def synchronize() -> None:
    """Block until every process in the global group reaches this barrier.

    A no-op when ``torch.distributed`` is unavailable / uninitialized or the
    world size is 1, so it is safe to call from single-process runs.
    """
    if not _dist_ready() or dist.get_world_size() == 1:
        return
    if dist.get_backend() == "nccl":
        import torch

        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def all_reduce_flag(flag: bool) -> bool:
    """Logical OR of ``flag`` across the global group (also a barrier).

    Lets every rank reach the same decision from a locally-observed condition —
    e.g. "did *any* rank's checkpoint write fail?" — so no rank branches away
    while its peers wait on a collective. Returns ``flag`` unchanged when
    ``torch.distributed`` is unavailable / uninitialized or the world size is 1.

    Args:
        flag (bool): this process's local vote.

    Returns:
        bool: ``True`` if ``flag`` is set on any process in the group.
    """
    if not _dist_ready() or dist.get_world_size() == 1:
        return flag
    import torch

    if dist.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    t = torch.tensor(1 if flag else 0, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def get_world_size() -> int:
    """Number of processes in the global (flat) group."""
    if not _dist_ready():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Rank of this process in the global (flat) group."""
    if not _dist_ready():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """True on global rank 0 (the process that logs / writes checkpoints)."""
    return get_rank() == 0


def _sp_active() -> bool:
    """True iff SP model-parallel groups are initialized with size > 1."""
    try:
        from minwm.distributed.sp.parallel_state import model_parallel_is_initialized
    except Exception:
        return False
    return model_parallel_is_initialized()


def get_dp_world_size() -> int:
    """Number of data-parallel replicas.

    When SP is active this is the DP group size (< global world size); otherwise
    it equals the global world size.
    """
    if _sp_active():
        from minwm.distributed.sp.parallel_state import get_dp_world_size as _dp_ws

        return _dp_ws()
    return get_world_size()


def get_dp_rank() -> int:
    """Rank of this process within the data-parallel dimension.

    Ranks sharing an SP group return the same DP rank, so a sampler keyed on
    this value hands them identical indices.
    """
    if _sp_active():
        from minwm.distributed.sp.parallel_state import get_dp_rank as _dp_rank

        return _dp_rank()
    return get_rank()
