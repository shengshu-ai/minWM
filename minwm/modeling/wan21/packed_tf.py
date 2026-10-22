# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Packed FlashAttention-varlen form of the teacher-forcing attention mask.

``CausalWan21Model._prepare_teacher_forcing_mask`` describes the teacher-forcing
mask over the concatenated ``[clean | noisy]`` sequence. Read as row groups it is
a short list of *contiguous* key/value ranges, with ``B`` = ``block_tokens`` =
``num_frame_per_block * frame_seqlen`` and ``N`` = ``clean_tokens // B`` blocks:

* clean block ``b`` — rows ``[b*B, (b+1)*B)`` — attends ``[0, (b+1)*B)``;
* noisy block ``b`` — rows ``[clean + b*B, clean + (b+1)*B)`` — attends
  ``[0, b*B) ∪ [clean + b*B, clean + (b+1)*B)``.

Every row of a group sees exactly what the other rows see, the ``q_idx == kv_idx``
term is already contained in those ranges, and no group is causal: each group is
one *full* attention over a contiguous KV slice, i.e. a FlashAttention-varlen
sequence with ``causal=False`` and not a single mask bit.

The segments are emitted in the *natural* row order (clean blocks ``0..N-1``,
then noisy blocks ``0..N-1``), which is what makes the lowering cheap: Q is used
exactly as stored (no gather, no padding, no reordering) and the output comes back
in that same row order (no repacking). K and V are each materialized once, by
copying the segment table's runs of rows; no per-call metadata (``cu_seqlens``,
runs, shapes) is recomputed, and the same runs make the backward a handful of
slice-adds rather than one general scatter over the duplicated rows. The price of
a flat varlen layout is the duplication — a KV slice belongs to one segment, so
the packed K/V holds ``clean_tokens * (N + 1)`` rows, 3x the sequence at the
20-frame stage-1 geometry.

The lowering needs a whole number of blocks: ``clean_tokens % block_tokens`` must
be 0 (stage 1: 20 frames / 4). A partial trailing block raises ``ValueError``
rather than guessing; such a model can keep using flex attention.

This replaces, for the teacher-forcing path only, the BlockMask execution of that
same mask: no ``create_block_mask``, no 128-aligned padding of Q/K/V, no flex
Triton kernels. Generation / i2v paths (``kv_cache`` present, blockwise-causal
masks) keep using flex attention and are untouched.

Enable with ``MINWM_TF_ATTN=packed`` (default ``flex`` = upstream behaviour).
"""

import functools
import os
from dataclasses import dataclass

import torch

from minwm.utils.logger import init_logger

__all__ = [
    "ENV_VAR",
    "TFGeometry",
    "packed_tf_attention",
    "packed_tf_enabled",
    "tf_geometry",
]

logger = init_logger(__name__)

#: ``flex`` (default, upstream) or ``packed``.
ENV_VAR = "MINWM_TF_ATTN"
_IMPLS = ("flex", "packed")
#: Geometries kept alive. Each holds a small index/seqlens table on the device,
#: so the cache is bounded: a variable-shape workload rebuilds instead of growing.
_GEOMETRY_CACHE_SIZE = 8


@dataclass(frozen=True)
class TFGeometry:
    """Segment table of the teacher-forcing mask, cached per geometry."""

    num_blocks: int
    block_tokens: int
    clean_tokens: int
    batch: int
    #: int32 ``[batch * 2 * num_blocks + 1]``, all query segments have block_tokens rows.
    cu_q: torch.Tensor
    #: int32 ``[batch * 2 * num_blocks + 1]``, kv segment lengths as lowered.
    cu_k: torch.Tensor
    #: ``(source row, length)`` runs per sample, in segment order: the rows the
    #: packed K/V holds, and the row order of the varlen call. Both directions of
    #: the packing walk this list; offsets are within one sample, whose
    #: ``sample * total_tokens`` shift the pack adds.
    kv_runs: tuple[tuple[int, int], ...]

    @property
    def total_tokens(self) -> int:
        """Sequence length of one sample, i.e. ``2 * clean_tokens``."""
        return 2 * self.clean_tokens

    @property
    def kv_tokens(self) -> int:
        """Rows in the packed K/V buffer (``3 * 62400`` at the 20-frame geometry)."""
        return self.batch * self.clean_tokens * (self.num_blocks + 1)

    @property
    def max_seqlen_q(self) -> int:
        """Longest query segment (all of them are ``block_tokens``)."""
        return self.block_tokens

    @property
    def max_seqlen_kv(self) -> int:
        """Longest kv segment: the last clean/noisy block's ``N * block_tokens``."""
        return self.num_blocks * self.block_tokens

    def describe(self) -> str:
        """One-line summary for the logs / A-B scripts."""
        return (
            f"packed TF attention: {2 * self.num_blocks} segments, "
            f"block_tokens={self.block_tokens} clean_tokens={self.clean_tokens} "
            f"total_tokens={self.total_tokens} batch={self.batch} "
            f"kv_tokens={self.kv_tokens} "
            f"({self.kv_tokens / (self.batch * self.total_tokens):.1f}x sequence)"
        )


def packed_tf_enabled() -> bool:
    """Whether the teacher-forcing path should use the packed varlen form.

    ``MINWM_TF_ATTN`` is read on every call: one ``os.environ.get`` next to an
    attention call, in exchange for no module-level state to keep in sync.

    Raises:
        ValueError: on a value that is neither ``flex`` nor ``packed`` — a typo
        must not silently train with the wrong attention path.
    """
    impl = os.environ.get(ENV_VAR, "").strip().lower()
    if impl and impl not in _IMPLS:
        raise ValueError(f"{ENV_VAR}={impl!r}: expected one of {_IMPLS}")
    return impl == "packed"


def _kv_runs(num_blocks: int, block_tokens: int, clean_tokens: int) -> list[tuple[int, int]]:
    """``(source row, length)`` runs of the packed K/V, in segment order.

    Sources are rows of one sample's ``[clean | noisy]`` sequence; segments are
    emitted clean-block-major, then noisy, so that packed Q is the original
    tensor and the output needs no repacking.
    """
    runs: list[tuple[int, int]] = []
    for b in range(num_blocks):
        runs.append((0, (b + 1) * block_tokens))  # clean b: the clean prefix
    for b in range(num_blocks):
        if b:
            runs.append((0, b * block_tokens))  # noisy b: clean prefix (empty at b=0)
        runs.append((clean_tokens + b * block_tokens, block_tokens))  # noisy b: own block
    return runs


def _build_geometry(
    *, num_blocks: int, block_tokens: int, clean_tokens: int, batch: int, device: torch.device
) -> TFGeometry:
    runs = tuple(_kv_runs(num_blocks, block_tokens, clean_tokens))

    # Every query segment has block_tokens rows. The clean and the noisy segment of
    # block b both hold (b+1) blocks of kv, spread over one run (clean) or two
    # (noisy: clean prefix + own block) that are consecutive in the packed buffer —
    # so a segment is one kv span even when it is fed by two runs. Both are
    # batch-major, matching the flattened [batch, total] row order.
    per_segment_kv = torch.tensor(
        [(b + 1) * block_tokens for b in range(num_blocks)] * 2,
        dtype=torch.int64,
        device=device,
    ).repeat(batch)
    cu_q = torch.arange(0, batch * 2 * num_blocks + 1, dtype=torch.int32, device=device).mul_(
        block_tokens
    )
    cu_k = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device=device), per_segment_kv.cumsum(0)]
    ).to(torch.int32)
    assert cu_q.numel() == cu_k.numel() == 2 * batch * num_blocks + 1
    assert int(cu_k[-1]) == batch * sum(length for _, length in runs)

    return TFGeometry(
        num_blocks=num_blocks,
        block_tokens=block_tokens,
        clean_tokens=clean_tokens,
        batch=batch,
        cu_q=cu_q,
        cu_k=cu_k,
        kv_runs=runs,
    )


@functools.lru_cache(maxsize=_GEOMETRY_CACHE_SIZE)
def _tf_geometry(
    clean_tokens: int, block_tokens: int, batch: int, device: torch.device
) -> TFGeometry:
    """Build one geometry and cache it; :func:`tf_geometry` is the validated entry point."""
    geom = _build_geometry(
        num_blocks=clean_tokens // block_tokens,
        block_tokens=block_tokens,
        clean_tokens=clean_tokens,
        batch=batch,
        device=device,
    )
    logger.info(geom.describe())
    return geom


def tf_geometry(
    *, clean_tokens: int, block_tokens: int, batch: int, device: torch.device
) -> TFGeometry:
    """Segment table for this geometry, built once and then served from cache.

    Args:
        clean_tokens (int): length of the clean half (= one sample in the batch).
        block_tokens (int): tokens per attention block
            (``num_frame_per_block * frame_seqlen``).
        batch (int): samples per forward.
        device (torch.device | str): device the cached index/seqlens live on.

    Returns:
        TFGeometry: cached segment table.

    Raises:
        ValueError: if the block size does not divide the clean half.
    """
    if clean_tokens % block_tokens:
        raise ValueError(
            f"packed TF attention needs clean_tokens ({clean_tokens}) divisible by "
            f"block_tokens ({block_tokens}) = num_frame_per_block * frame_seqlen"
        )
    return _tf_geometry(int(clean_tokens), int(block_tokens), int(batch), torch.device(device))


class _PackKV(torch.autograd.Function):
    """Pack K or V rows into the order the varlen call needs.

    The packed buffer *is* the run table, so the forward copies whole contiguous
    runs and the backward adds them back the same way. What that buys is the
    backward: a source row appears in several segments, so its gradient is the sum
    over those copies, accumulated here run by run. Spelled as a general
    ``index_select`` over the expanded rows instead, the same sum costs a kernel
    that places each gradient by searching the duplicated index
    (``indexFuncLargeIndex``: 6x these slice-adds at the stage-1 geometry) and is
    free to use atomics; the slice-adds are plain reads and writes, and
    deterministic.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        runs: tuple[tuple[int, int], ...],
        batch: int,
        total_tokens: int,
    ) -> torch.Tensor:
        ctx.runs, ctx.batch, ctx.total_tokens = runs, batch, total_tokens
        ctx.rows_per_sample = sum(length for _, length in runs)
        return torch.cat(
            [
                x[sample * total_tokens + start : sample * total_tokens + start + length]
                for sample in range(batch)
                for start, length in runs
            ]
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        out = grad.new_zeros(ctx.batch * ctx.total_tokens, *grad.shape[1:])
        for sample in range(ctx.batch):
            source = sample * ctx.total_tokens
            packed = sample * ctx.rows_per_sample
            for start, length in ctx.runs:
                out[source + start : source + start + length] += grad[packed : packed + length]
                packed += length
        return out, None, None, None


def _pack_kv(x: torch.Tensor, geom: TFGeometry) -> torch.Tensor:
    """``[B, L, H, D]`` k or v -> the rows of :attr:`TFGeometry.kv_runs`, in order."""
    return _PackKV.apply(
        x.reshape(geom.batch * geom.total_tokens, *x.shape[2:]),
        geom.kv_runs,
        geom.batch,
        geom.total_tokens,
    )


def packed_tf_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geom: TFGeometry,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Teacher-forcing attention as one FlashAttention-varlen call.

    ``q``/``k``/``v`` are the *unpadded* ``[B, L, H, D]`` tensors in the original
    ``[clean | noisy]`` order; the result has the same shape and row order, so
    callers need neither a gather on the way in nor a repack on the way out.

    Args:
        q (Tensor): queries ``[B, L, H, D]``, fp16/bf16, L == ``geom.total_tokens``.
        k (Tensor): keys, same shape as ``q``.
        v (Tensor): values, same shape as ``q``.
        geom (TFGeometry): segment table from :func:`tf_geometry`.
        softmax_scale (float, optional): defaults to ``1/sqrt(head_dim)``.

    Returns:
        Tensor: attention output ``[B, L, H, D]``.

    Raises:
        ValueError: on a shape/dtype that the lowering cannot serve.
        RuntimeError: if flash-attn is not installed.
    """
    batch, length, heads, head_dim = q.shape
    if length != geom.total_tokens or batch != geom.batch:
        raise ValueError(
            f"packed TF attention got q of shape {tuple(q.shape)} but the cached geometry "
            f"is batch={geom.batch} total_tokens={geom.total_tokens}; rebuild it with tf_geometry()"
        )
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q/k/v shapes differ: {tuple(q.shape)} vs {tuple(k.shape)}/{tuple(v.shape)}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"packed TF attention needs fp16/bf16 (got {q.dtype}); "
            f"unset {ENV_VAR} to use flex attention in full precision"
        )

    try:
        import flash_attn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{ENV_VAR}=packed requires flash-attn 2 (import flash_attn failed)"
        ) from exc

    # A view for the contiguous q the model passes — the rows are never reordered.
    q_flat = q.reshape(batch * length, heads, head_dim)
    # One pack per K/V: the segment table's runs, in order (see _PackKV).
    k_flat = _pack_kv(k, geom)
    v_flat = _pack_kv(v, geom)

    out = flash_attn.flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=geom.cu_q,
        cu_seqlens_k=geom.cu_k,
        max_seqlen_q=geom.max_seqlen_q,
        max_seqlen_k=geom.max_seqlen_kv,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        deterministic=False,
    )
    return out.reshape(batch, length, heads, head_dim)


def _reset_cache() -> None:
    """Drop the cached geometries (tests only)."""
    _tf_geometry.cache_clear()
