"""Reproducibility helpers: global RNG seeding."""

import random

import numpy as np
import torch


def resolve_seed(seed: int | None, device: torch.device | None = None) -> int:
    """Resolve the base RNG seed, drawing + broadcasting one when unset.

    When ``seed`` is ``None`` a random base seed is drawn on global rank 0 and
    broadcast to every rank, so the whole job shares one base seed without the
    user having to pin it; an explicit ``seed`` is returned unchanged. The
    trainer then adds ``dp_rank`` to this base before calling :func:`set_seed`,
    so the per-step stream is identical within an SP group and distinct across
    DP groups.

    Args:
        seed (int | None): the configured base seed, or ``None`` to draw one.
        device (torch.device | None): device for the broadcast tensor. Defaults
            to CUDA under the nccl backend (which requires device tensors) and
            CPU otherwise (gloo), so the draw works without assuming a CUDA
            context on CPU-only runs. Ignored when ``seed`` is given or
            ``torch.distributed`` is not initialized.

    Returns:
        int: the resolved base seed, identical on every rank.
    """
    if seed is not None:
        return seed

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return random.randrange(2**31 - 1)

    # nccl can only broadcast device tensors; gloo works on CPU. Pick by backend
    # so this neither assumes CUDA on a CPU-only (gloo) run nor hands nccl a CPU
    # tensor.
    if device is None:
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    t = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64, device=device)
    dist.broadcast(t, src=0)
    return int(t.item())


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed ``random`` / ``numpy`` / ``torch`` and register the tracked RNG stream.

    The trainer calls this once after distributed init with ``base_seed +
    dp_rank`` so the stream is identical within an SP group (its ranks share one
    ``dp_rank``) and distinct across DP groups. Besides seeding the global
    generators, this registers the ``data-parallel-rng`` stream on the RNG
    tracker (see :mod:`minwm.distributed.rng`) so recipe/preprocessor draws
    wrapped in ``get_rng_states_tracker().fork()`` advance in isolation and are
    saved/restored with the checkpoint.

    Args:
        seed (int): the already ``+ dp_rank`` per-replica seed to set.
        deterministic (bool): also switch on deterministic cuDNN/torch algorithms
            (slower; off by default).
    """
    from minwm.distributed.rng import DATA_PARALLEL_RNG_TRACKER_NAME, get_rng_states_tracker

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    tracker = get_rng_states_tracker()
    tracker.reset()
    tracker.add(DATA_PARALLEL_RNG_TRACKER_NAME, seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
