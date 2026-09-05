"""Backwards-compat shim. ``ParallelDims`` now lives in
:mod:`minwm.distributed.parallel_dims` (one level up); this re-exports it so
existing ``from minwm.distributed.sp.parallel_dims import ...`` imports keep working.
"""

from ..parallel_dims import (  # noqa: F401
    ParallelDims,
    get_parallel_state,
    initialize_parallel_state,
)

__all__ = ["ParallelDims", "get_parallel_state", "initialize_parallel_state"]
