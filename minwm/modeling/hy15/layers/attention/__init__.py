"""Attention stack for the HY15 DiT: multi-mode dispatch, masks, infer-state, SSTA.

Heavy backends (flash-attn 2/3, sageattention, flex_block_attn) are imported
lazily inside the functions that use them, so importing this subpackage never
forces an optional dependency.
"""

from .attention import (
    attention,
    parallel_attention,
    sequence_parallel_attention,
    sequence_parallel_attention_txt,
    sequence_parallel_attention_vision,
)
from .flash_attn_no_pad import prepare_teacher_forcing_mask
from .infer_state import InferState, get_infer_state, initialize_infer_state, parse_range
from .ssta_attention import ssta_3d_attention

__all__ = [
    "attention",
    "sequence_parallel_attention_txt",
    "sequence_parallel_attention_vision",
    "parallel_attention",
    "sequence_parallel_attention",
    "prepare_teacher_forcing_mask",
    "InferState",
    "get_infer_state",
    "initialize_infer_state",
    "parse_range",
    "ssta_3d_attention",
]
