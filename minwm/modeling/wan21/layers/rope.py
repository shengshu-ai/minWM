# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Rotary positional encoding utilities."""

import torch

from minwm.modeling.common.embedding import sinusoidal_embedding

__all__ = ["sinusoidal_embedding_1d", "rope_params", "rope_apply", "causal_rope_apply"]


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Standard 1-D sinusoidal positional embedding.

    Thin wrapper over :func:`minwm.modeling.common.embedding.sinusoidal_embedding` preserving
    the Wan convention (fp64 compute, even dims only, ``(dim, position)`` order).

    Args:
        dim (int): embedding dimension (must be even).
        position (Tensor): positions to embed, shape ``[N]``.

    Returns:
        Tensor: embeddings of shape ``[N, dim]``.
    """
    return sinusoidal_embedding(position, dim, compute_dtype=torch.float64, pad_odd=False)


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Precompute complex RoPE frequency table.

    Args:
        max_seq_len (int): maximum sequence length to pre-compute.
        dim (int): per-head dimension (must be even).
        theta (float): base frequency.

    Returns:
        Tensor: complex frequencies, shape ``[max_seq_len, dim // 2]``.
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 3-D RoPE (T, H, W) to a batch of token sequences.

    Args:
        x (Tensor): token features, shape ``[B, L, N, D]``.
        grid_sizes (Tensor): per-sample ``(F, H, W)`` grids, shape ``[B, 3]``.
        freqs (Tensor): precomputed complex frequencies, shape ``[max_len, D//2]``.

    Returns:
        Tensor: RoPE-encoded features, same shape as ``x``.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor:
    """Apply 3-D RoPE with a frame offset for autoregressive / cached inference.

    Args:
        x (Tensor): token features, shape ``[B, L, N, D]``.
        grid_sizes (Tensor): per-sample ``(F, H, W)`` grids, shape ``[B, 3]``.
        freqs (Tensor): precomputed complex frequencies, shape ``[max_len, D//2]``.
        start_frame (int): frame index offset for the first frame in ``x``.

    Returns:
        Tensor: RoPE-encoded features, same shape as ``x``.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)
