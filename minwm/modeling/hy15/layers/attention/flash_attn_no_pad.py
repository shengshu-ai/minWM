# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan
# works and any output and results therefrom are provided "AS IS" without any
# express or implied warranties of any kind including any warranties of title,
# merchantability, noninfringement, course of dealing, usage of trade, or
# fitness for a particular purpose. You are solely responsible for determining
# the appropriateness of using, reproducing, modifying, performing, displaying
# or distributing any of the Tencent Hunyuan works or outputs and assume any and
# all risks associated with your or a third party's use or distribution of any
# of the Tencent Hunyuan works or outputs and your exercise of rights and
# permissions under this agreement.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Variable-length attention kernels and teacher-forcing BlockMask builders.

Provides the flash-attention (v2/v3) and flex-attention varlen wrappers used by
the HY15 attention dispatch, plus the teacher-forcing ``BlockMask`` constructor
used by the AR transformers. All heavy/optional deps (``flash_attn``,
``flash_attn_interface``, ``torch.nn.attention.flex_attention`` compile) are
imported lazily so importing this module never forces them.
"""

import math
import os

import torch
from einops import rearrange

DEFAULT_FLEX_BLOCK_SIZE = 128

_flex_attention = None


def _get_flex_attention():
    """Lazily import and torch.compile ``flex_attention`` (cached)."""
    global _flex_attention
    if _flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _flex_attention = torch.compile(flex_attention, mode="default")
    return _flex_attention


def _dense_block_rows_to_ordered(
    dense_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense block mask to ordered BlockMask metadata.

    Args:
        dense_mask (torch.Tensor): boolean mask ``[num_q_blocks, num_kv_blocks]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(num_blocks, indices)`` int32 tensors.
    """
    dense_mask = dense_mask.to(dtype=torch.int32)
    num_blocks = dense_mask.sum(dim=-1).to(dtype=torch.int32, memory_format=torch.contiguous_format)
    indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
    indices = indices.to(dtype=torch.int32, memory_format=torch.contiguous_format)
    return num_blocks, indices


def _get_tf_flex_block_size() -> int:
    """Read the teacher-forcing flex block size from the environment."""
    raw = os.environ.get("HYVIDEO_TF_FLEX_BLOCK_SIZE")
    if raw is None:
        return DEFAULT_FLEX_BLOCK_SIZE
    block_size = int(raw)
    if block_size <= 0:
        raise ValueError(f"HYVIDEO_TF_FLEX_BLOCK_SIZE must be positive, got {block_size}")
    return block_size


def _iter_teacher_forcing_q_segments(
    q_start: int,
    q_end: int,
    text_seq_length: int,
    clean_start: int,
    noisy_start: int,
    total_length: int,
    attention_block_size: int,
):
    """Yield query sub-segments where the teacher-forcing attention rule is constant.

    Q layout::

        [0 ... text_seq_length) [clean_start ... noisy_start)
        [noisy_start ... total_length) [padding...)
    """
    text_end = min(q_end, clean_start)
    if q_start < clean_start and text_end > q_start:
        yield ("text", q_start, text_end, text_seq_length)

    clean_q_start = max(q_start, clean_start)
    clean_q_end = min(q_end, noisy_start)
    if clean_q_start < clean_q_end:
        block_idx = (clean_q_start - clean_start) // attention_block_size
        seg_start = clean_q_start
        while seg_start < clean_q_end:
            clean_block_end = min(
                clean_start + (block_idx + 1) * attention_block_size,
                noisy_start,
            )
            seg_end = min(clean_q_end, clean_block_end)
            yield ("clean", seg_start, seg_end, clean_block_end)
            seg_start = seg_end
            block_idx += 1

    noisy_q_start = max(q_start, noisy_start)
    noisy_q_end = min(q_end, total_length)
    if noisy_q_start < noisy_q_end:
        block_idx = (noisy_q_start - noisy_start) // attention_block_size
        seg_start = noisy_q_start
        while seg_start < noisy_q_end:
            noisy_block_start = noisy_start + block_idx * attention_block_size
            noisy_block_end = min(noisy_block_start + attention_block_size, total_length)
            seg_end = min(noisy_q_end, noisy_block_end)
            clean_context_end = clean_start + block_idx * attention_block_size
            yield (
                "noisy",
                seg_start,
                seg_end,
                clean_context_end,
                noisy_block_start,
                noisy_block_end,
            )
            seg_start = seg_end
            block_idx += 1

    pad_start = max(q_start, total_length)
    if pad_start < q_end:
        yield ("pad", pad_start, q_end, text_seq_length)


def _vec_interval_full_any(kv_starts, kv_ends, allowed_start, allowed_end):
    """Vectorized interval check across all kv blocks.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(full, any)`` boolean tensors.
    """
    if allowed_end <= allowed_start:
        z = torch.zeros(kv_starts.shape[0], dtype=torch.bool)
        return z, z.clone()
    full = (allowed_start <= kv_starts) & (kv_ends <= allowed_end)
    any_hit = (kv_starts < allowed_end) & (allowed_start < kv_ends)
    return full, any_hit


def _build_teacher_forcing_block_mask_direct(
    device,
    num_frames: int,
    frame_seqlen: int,
    text_seq_length: int,
    num_frame_per_block: int = 4,
    block_size: int = DEFAULT_FLEX_BLOCK_SIZE,
) -> "BlockMask":  # noqa: F821
    """Build BlockMask metadata directly in block space.

    Avoids ``create_block_mask``'s dense ``O(S^2)`` mask allocation.

    Args:
        device: target device for the BlockMask metadata.
        num_frames (int): number of latent frames.
        frame_seqlen (int): tokens per latent frame.
        text_seq_length (int): number of leading text tokens.
        num_frame_per_block (int): latent frames grouped into one attention block.
        block_size (int): flex-attention block size.

    Returns:
        BlockMask: teacher-forcing block mask.
    """
    from torch.nn.attention.flex_attention import BlockMask

    vision_length = num_frames * frame_seqlen
    total_length = text_seq_length + vision_length * 2
    padded_total = math.ceil(total_length / 128) * 128
    noisy_start = text_seq_length + vision_length
    attention_block_size = frame_seqlen * num_frame_per_block

    _tsl = text_seq_length
    _ns = noisy_start
    _ne = noisy_start + vision_length
    _abs = attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        text_mask = kv_idx < _tsl
        q_clean_block_end = _tsl + ((q_idx - _tsl) // _abs + 1) * _abs
        clean_mask = (
            (q_idx >= _tsl)
            & (q_idx < _ns)
            & (kv_idx >= _tsl)
            & (kv_idx < q_clean_block_end)
            & (kv_idx < _ns)
        )
        noisy_block_idx = (q_idx - _ns) // _abs
        noisy_to_clean = (
            (q_idx >= _ns)
            & (q_idx < _ne)
            & (kv_idx >= _tsl)
            & (kv_idx < _tsl + noisy_block_idx * _abs)
        )
        noisy_block_start = _ns + noisy_block_idx * _abs
        noisy_to_noisy = (
            (q_idx >= _ns)
            & (q_idx < _ne)
            & (kv_idx >= noisy_block_start)
            & (kv_idx < noisy_block_start + _abs)
            & (kv_idx < _ne)
        )
        return (q_idx == kv_idx) | text_mask | clean_mask | noisy_to_clean | noisy_to_noisy

    num_blocks = math.ceil(padded_total / block_size)

    kv_starts = torch.arange(num_blocks, dtype=torch.int64) * block_size
    kv_ends = torch.clamp(kv_starts + block_size, max=padded_total)

    full_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)
    partial_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)

    for q_bi in range(num_blocks):
        q_start = q_bi * block_size
        q_end = min(q_start + block_size, padded_total)

        segments = list(
            _iter_teacher_forcing_q_segments(
                q_start=q_start,
                q_end=q_end,
                text_seq_length=text_seq_length,
                clean_start=text_seq_length,
                noisy_start=noisy_start,
                total_length=total_length,
                attention_block_size=attention_block_size,
            )
        )
        if not segments:
            continue

        row_full = torch.ones(num_blocks, dtype=torch.bool)
        row_any = torch.zeros(num_blocks, dtype=torch.bool)

        for seg in segments:
            kind = seg[0]
            if kind == "text":
                sf, sa = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
            elif kind == "clean":
                sf, sa = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
            elif kind == "noisy":
                cf, ca = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
                nf, na = _vec_interval_full_any(kv_starts, kv_ends, seg[4], seg[5])
                sf = cf | nf
                sa = ca | na
            elif kind == "pad":
                tf, ta = _vec_interval_full_any(kv_starts, kv_ends, 0, seg[3])
                eye_any = (kv_starts < seg[2]) & (seg[1] < kv_ends)
                sf = tf
                sa = ta | eye_any
            else:
                raise ValueError(f"Unknown segment type: {kind}")

            row_full &= sf
            row_any |= sa

        full_mask[q_bi] = row_full
        partial_mask[q_bi] = row_any & ~row_full

    partial_num_blocks, partial_indices = _dense_block_rows_to_ordered(partial_mask)
    if full_mask.any():
        full_num_blocks, full_indices = _dense_block_rows_to_ordered(full_mask)
        full_num_blocks = full_num_blocks.unsqueeze(0).unsqueeze(0).to(device)
        full_indices = full_indices.unsqueeze(0).unsqueeze(0).to(device)
    else:
        full_num_blocks = None
        full_indices = None

    partial_num_blocks = partial_num_blocks.unsqueeze(0).unsqueeze(0).to(device)
    partial_indices = partial_indices.unsqueeze(0).unsqueeze(0).to(device)

    return BlockMask.from_kv_blocks(
        partial_num_blocks,
        partial_indices,
        full_num_blocks,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=attention_mask,
        seq_lengths=(padded_total, padded_total),
    )


def prepare_teacher_forcing_mask(
    device,
    num_frames: int,
    frame_seqlen: int,
    text_seq_length: int,
    num_frame_per_block: int = 4,
) -> "BlockMask":  # noqa: F821
    """Build a teacher-forcing ``BlockMask`` (memory-efficient).

    Sequence layout: ``[text_tokens | clean_vision_tokens | noisy_vision_tokens]``.
    Attention rules: every token attends to text; clean block ``i`` attends to
    clean blocks ``0..i`` (block-causal); noisy block ``i`` attends to clean
    blocks ``0..i-1`` plus its own noisy block; the diagonal is always allowed.

    Args:
        device: target device for the BlockMask metadata.
        num_frames (int): number of latent frames.
        frame_seqlen (int): tokens per latent frame.
        text_seq_length (int): number of leading text tokens.
        num_frame_per_block (int): latent frames grouped into one attention block.

    Returns:
        BlockMask: teacher-forcing block mask.
    """
    block_size = _get_tf_flex_block_size()
    return _build_teacher_forcing_block_mask_direct(
        device=device,
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        text_seq_length=text_seq_length,
        num_frame_per_block=num_frame_per_block,
        block_size=block_size,
    )


def flex_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """Chunk-wise causal flex-attention over a packed varlen qkv tensor.

    Args:
        qkv (torch.Tensor): packed tensor ``[B, S, 3, num_heads, head_dim]``.
        key_padding_mask (torch.Tensor): boolean mask ``[B, S]`` (True = keep).
        causal (bool): unused; kept for signature parity with the flash wrappers.
        dropout_p (float): unused; kept for signature parity.
        softmax_scale (float, optional): unused; kept for signature parity.
        deterministic (bool): unused; kept for signature parity.

    Returns:
        torch.Tensor: attention output ``[B, S, num_heads, head_dim]``.
    """
    from flash_attn.bert_padding import pad_input, unpad_input

    flex_attention = _get_flex_attention()

    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(2).transpose(1, 2)
    k = k.squeeze(2).transpose(1, 2)
    v = v.squeeze(2).transpose(1, 2)
    output_unpad = flex_attention(q, k, v, block_mask=None)

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """Varlen FlashAttention-2 over a packed qkv tensor with key padding.

    Args:
        qkv (torch.Tensor): packed tensor ``[B, S, 3, num_heads, head_dim]``.
        key_padding_mask (torch.Tensor): boolean mask ``[B, S]`` (True = keep).
        causal (bool): whether to apply a causal mask.
        dropout_p (float): attention dropout probability.
        softmax_scale (float, optional): softmax scale; defaults to head_dim**-0.5.
        deterministic (bool): whether to use the deterministic backward.

    Returns:
        torch.Tensor: attention output ``[B, S, num_heads, head_dim]``.
    """
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """Varlen FlashAttention-3 over a packed qkv tensor with key padding.

    Args:
        qkv (torch.Tensor): packed tensor ``[B, S, 3, num_heads, head_dim]``.
        key_padding_mask (torch.Tensor): boolean mask ``[B, S]`` (True = keep).
        causal (bool): whether to apply a causal mask.
        dropout_p (float): unused by the v3 backend; kept for signature parity.
        softmax_scale (float, optional): softmax scale; defaults to head_dim**-0.5.
        deterministic (bool): whether to use the deterministic backward.

    Returns:
        torch.Tensor: attention output ``[B, S, num_heads, head_dim]``.

    Raises:
        ImportError: if the FlashAttention-3 backend is unavailable.
    """
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output
