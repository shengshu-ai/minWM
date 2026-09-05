# Sequence-parallel (SP) infrastructure for the minwm framework.
#
# Public API summary (import from this module):
#   - parallel_state: init_distributed_environment, get_*_group helpers,
#       initialize_model_parallel, etc.
#   - communication_ops: SP / TP collectives.
#   - parallel_dims: ParallelDims facade used by attention layers.
#   - stateless_pg.StatelessProcessGroup: side-channel metadata exchange.

from ..parallel_dims import (  # noqa: F401
    ParallelDims,
    get_parallel_state,
    initialize_parallel_state,
)
from .communication_ops import (  # noqa: F401
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from .parallel_state import (  # noqa: F401
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
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
from .stateless_pg import StatelessProcessGroup  # noqa: F401
