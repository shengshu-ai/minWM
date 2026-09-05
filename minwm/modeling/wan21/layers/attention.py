# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Flash-attention dispatch with SDPA fallback."""

import warnings

import torch

try:
    import flash_attn_interface

    def _is_hopper_gpu() -> bool:
        if not torch.cuda.is_available():
            return False
        name = torch.cuda.get_device_name(0).lower()
        return "h100" in name or "hopper" in name

    FLASH_ATTN_3_AVAILABLE = _is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = ["flash_attention", "attention", "FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE"]


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version=None,
) -> torch.Tensor:
    """Varlen flash attention (FA3 → FA2 → error).

    Args:
        q (Tensor): queries, shape ``[B, Lq, Nq, C]``.
        k (Tensor): keys, shape ``[B, Lk, Nk, C]``.
        v (Tensor): values, shape ``[B, Lk, Nk, C]``. ``Nq`` must be divisible by ``Nk``.
        q_lens (Tensor, optional): valid query lengths per sample, shape ``[B]``.
        k_lens (Tensor, optional): valid key lengths per sample, shape ``[B]``.
        dropout_p (float): attention dropout probability.
        softmax_scale (float, optional): scale applied to ``QK^T`` before softmax.
        q_scale (float, optional): additional multiplicative scale on queries.
        causal (bool): apply causal mask.
        window_size (tuple): sliding-window local attention ``(left, right)``.
        deterministic (bool): slower but reproducible.
        dtype (torch.dtype): cast dtype when inputs are not fp16/bf16.
        version (int, optional): force FA version (2 or 3).

    Returns:
        Tensor: attention output, same shape as ``q``.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn("Flash attention 3 is not available, using flash attention 2 instead.")

    cu_q = (
        torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True)
    )
    cu_k = (
        torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(k.device, non_blocking=True)
    )

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    return x.type(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version=None,
) -> torch.Tensor:
    """Attention dispatch: flash attention when available, SDPA fallback.

    Args:
        q (Tensor): queries, shape ``[B, Lq, Nq, C]``.
        k (Tensor): keys, shape ``[B, Lk, Nk, C]``.
        v (Tensor): values, shape ``[B, Lk, Nk, C]``.
        q_lens (Tensor, optional): valid query lengths per sample.
        k_lens (Tensor, optional): valid key lengths per sample.
        dropout_p (float): attention dropout probability.
        softmax_scale (float, optional): scale for ``QK^T``.
        q_scale (float, optional): additional scale on queries.
        causal (bool): apply causal mask.
        window_size (tuple): local attention window.
        deterministic (bool): reproducible mode.
        dtype (torch.dtype): working dtype for half-precision cast.
        fa_version (int, optional): force FA version.

    Returns:
        Tensor: attention output, same shape as ``q``.
    """
    if (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE) and q.device.type == "cuda":
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )

    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention. "
            "This can have a significant impact on performance."
        )
    out_dtype = q.dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)
