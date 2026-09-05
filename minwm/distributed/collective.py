"""SP / TP collective operations for the minwm framework.

Thin re-export of the collectives implemented in
:mod:`minwm.distributed.sp.communication_ops`, plus shorter aliases. Use these
in model code so call sites stay terse and the underlying (vLLM-derived)
implementation stays internal:

    from minwm.distributed.collective import sp_all_to_all_4D, sp_all_gather
"""

from minwm.distributed.sp.communication_ops import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

sp_all_to_all_4D = sequence_model_parallel_all_to_all_4D
sp_all_gather = sequence_model_parallel_all_gather
tp_all_reduce = tensor_model_parallel_all_reduce
tp_all_gather = tensor_model_parallel_all_gather

__all__ = [
    "sequence_model_parallel_all_gather",
    "sequence_model_parallel_all_to_all_4D",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "sp_all_to_all_4D",
    "sp_all_gather",
    "tp_all_reduce",
    "tp_all_gather",
]
