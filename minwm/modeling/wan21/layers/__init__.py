# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Reusable nn.Modules and functional utilities for the Wan model family.

Submodules:
    norm      — WanRMSNorm, WanLayerNorm
    rope      — rope_params, rope_apply, causal_rope_apply, sinusoidal_embedding_1d
    attention — flash_attention / attention dispatch (FA3 → FA2 → SDPA)
"""

from .attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE, attention, flash_attention
from .norm import WanLayerNorm, WanRMSNorm
from .rope import causal_rope_apply, rope_apply, rope_params, sinusoidal_embedding_1d

__all__ = [
    "WanRMSNorm",
    "WanLayerNorm",
    "sinusoidal_embedding_1d",
    "rope_params",
    "rope_apply",
    "causal_rope_apply",
    "flash_attention",
    "attention",
    "FLASH_ATTN_2_AVAILABLE",
    "FLASH_ATTN_3_AVAILABLE",
]
