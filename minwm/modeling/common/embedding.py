"""Sinusoidal (timestep) positional embedding, shared across model families.

The 1-D sin/cos frequency table is identical math across HY15 and Wan; the two
families differ only in the internal compute dtype (HY fp32, Wan fp64) and
whether odd embedding dims are zero-padded (HY pads, Wan asserts even). Both are
exposed as parameters so each family's thin wrapper preserves its own semantics.
"""

import math

import torch
from torch import Tensor


def sinusoidal_embedding(
    position: Tensor,
    dim: int,
    *,
    max_period: float = 10000,
    compute_dtype: torch.dtype = torch.float32,
    pad_odd: bool = True,
) -> Tensor:
    """Build a 1-D sinusoidal positional embedding table.

    Args:
        position (Tensor): 1-D tensor of ``N`` positions (may be fractional).
        dim (int): output embedding dimension.
        max_period (float): controls the minimum embedding frequency.
        compute_dtype (torch.dtype): dtype the frequency table is computed in
            (``float32`` for HY, ``float64`` for Wan). The result is returned in
            this dtype; the caller casts as needed.
        pad_odd (bool): if True and ``dim`` is odd, zero-pad the last column so
            the output width is exactly ``dim``. If False, ``dim`` must be even.

    Returns:
        Tensor: ``[N, dim]`` embedding, ``concat(cos, sin)`` along the last axis.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=compute_dtype, device=position.device)
        / half
    )
    args = position[:, None].to(compute_dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        if not pad_odd:
            raise ValueError(f"dim={dim} must be even when pad_odd=False")
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
