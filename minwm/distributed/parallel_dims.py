"""Facade exposing parallel-state queries as a flat ``ParallelDims`` object.

Attention layers want a single object with ``sp_enabled`` / ``sp_rank`` /
``sp_group`` / ``sp`` / ``dp_enabled`` properties. The underlying
``parallel_state`` module exposes the same information through ``get_*_group``
functions; this module just wraps those calls in property form.
"""

from minwm.distributed.sp.parallel_state import (
    get_dp_rank,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    model_parallel_is_initialized,
)


class ParallelDims:
    """Read-only view of the active parallel-state configuration.

    Properties degrade gracefully (return defaults) when distributed has not
    been initialized, so this can be queried at module-import time.
    """

    @property
    def sp_enabled(self) -> bool:
        if not model_parallel_is_initialized():
            return False
        return get_sp_world_size() > 1

    @property
    def sp_group(self):
        return get_sp_group().device_group

    @property
    def sp_rank(self) -> int:
        if not model_parallel_is_initialized():
            return 0
        return get_sp_parallel_rank()

    @property
    def sp(self) -> int:
        if not model_parallel_is_initialized():
            return 1
        return get_sp_world_size()

    @property
    def dp_rank(self) -> int:
        """Rank within the data-parallel group (global rank when only DP is active).

        Constant across an SP group and distinct between groups — the right axis
        to offset a per-replica seed on.
        """
        if not model_parallel_is_initialized():
            return 0
        return get_dp_rank()

    @property
    def dp_enabled(self) -> bool:
        # In FSDP setup DP is enabled iff SP is.
        return self.sp_enabled


_parallel_dims = ParallelDims()


def get_parallel_state() -> ParallelDims:
    """Return the global :class:`ParallelDims` singleton."""
    return _parallel_dims


def initialize_parallel_state(sp: int = 1) -> ParallelDims:
    """No-op kept for backwards compatibility.

    Real initialization is done by
    ``parallel_state.maybe_init_distributed_environment_and_model_parallel``.
    """
    return _parallel_dims


__all__ = ["ParallelDims", "get_parallel_state", "initialize_parallel_state"]
