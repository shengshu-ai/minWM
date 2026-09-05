"""Shared process/runtime initialization for training and inference engines."""

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeContext:
    """Process-local device and distributed identity.

    Args:
        device (torch.device): device used by this process.
        rank (int): global distributed rank.
        world_size (int): total number of distributed processes.
    """

    device: torch.device
    rank: int
    world_size: int


def initialize_runtime(
    *,
    tp_size: int = 1,
    sp_size: int = 1,
    hsdp_shard_size: int = 0,
    distributed: bool | None = None,
) -> RuntimeContext:
    """Initialize the shared distributed environment and resolve process identity.

    Training passes its existing torchrun detection explicitly, while inference
    may also request initialization through ``sp_size > 1``. Keeping that policy
    in each engine preserves their lifecycle semantics while sharing the actual
    bootstrap implementation.

    Args:
        tp_size (int): tensor-parallel degree.
        sp_size (int): sequence-parallel degree.
        hsdp_shard_size (int): HSDP shard degree; 0 selects the full DP dimension.
        distributed (bool, optional): whether to initialize distributed state.
            Defaults to whether torchrun environment variables are present.

    Returns:
        RuntimeContext: process-local device, rank, and world size.
    """
    if distributed is None:
        distributed = "RANK" in os.environ or int(os.environ.get("WORLD_SIZE") or 1) > 1

    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return RuntimeContext(device=device, rank=0, world_size=1)

    from minwm.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size,
        sp_size=sp_size,
        data_parallel_shard_size=hsdp_shard_size,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    import torch.distributed as dist

    return RuntimeContext(device=device, rank=dist.get_rank(), world_size=dist.get_world_size())
