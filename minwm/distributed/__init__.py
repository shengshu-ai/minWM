"""Distributed-training infrastructure for minwm.

Public surface (import directly from ``minwm.distributed``):

- **Process-group init / teardown and rank queries** — set up TP/SP/DP groups
  at training start and query rank/world-size along each axis. Backed by the
  vLLM-derived :mod:`minwm.distributed.sp.parallel_state`.
- **:class:`ParallelDims`** — flat property facade (``sp_enabled`` / ``sp_rank``
  / ``sp_group`` / …) for attention layers to consult.
- **Collective ops** — SP / TP collectives (and short ``sp_*`` / ``tp_*``
  aliases) via :mod:`minwm.distributed.collective`.

The ``sp/`` subpackage holds the implementation (GroupCoordinator, device
communicators, PyNccl bindings); treat it as internal.
"""

from .collective import (  # noqa: F401
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
    sp_all_gather,
    sp_all_to_all_4D,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tp_all_gather,
    tp_all_reduce,
)
from .parallel_dims import (  # noqa: F401
    ParallelDims,
    get_parallel_state,
    initialize_parallel_state,
)
from .rng import (  # noqa: F401
    RNGStatesTracker,
    get_rng_states_tracker,
    initialize_rng_tracker,
)
from .sp.parallel_state import (  # noqa: F401
    cleanup_dist_env_and_memory,
    get_device_mesh,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
    get_fsdp_mesh,
    get_local_torch_device,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)

__all__ = [
    # parallel state: init / teardown
    "init_distributed_environment",
    "initialize_model_parallel",
    "maybe_init_distributed_environment_and_model_parallel",
    "cleanup_dist_env_and_memory",
    "model_parallel_is_initialized",
    # parallel state: group / rank queries
    "get_world_group",
    "get_world_rank",
    "get_world_size",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "get_sp_group",
    "get_sp_parallel_rank",
    "get_sp_world_size",
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    "get_device_mesh",
    "get_fsdp_mesh",
    "get_local_torch_device",
    # parallel dims facade
    "ParallelDims",
    "get_parallel_state",
    "initialize_parallel_state",
    # rng tracker
    "RNGStatesTracker",
    "get_rng_states_tracker",
    "initialize_rng_tracker",
    # collectives
    "sequence_model_parallel_all_gather",
    "sequence_model_parallel_all_to_all_4D",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "sp_all_gather",
    "sp_all_to_all_4D",
    "tp_all_gather",
    "tp_all_reduce",
]
