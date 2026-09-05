# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for
# determining the appropriateness of using, reproducing, modifying, performing, displaying or
# distributing any of the Tencent Hunyuan works or outputs and assume any and all risks associated
# with your or a third party's use or distribution of any of the Tencent Hunyuan works or outputs
# and your exercise of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.
"""Multi-mode attention dispatch for the HY15 DiT.

Provides the dense ``attention`` helper plus the sequence-parallel dispatch
(``sequence_parallel_attention`` and its ``parallel_attention`` wrapper) that
selects among eight backends: ``torch``, ``flash2``, ``flash3``, ``sageattn``,
``flex_tf`` (teacher-forcing BlockMask), ``torch_causal``, ``flex_causal`` and
``flex-block-attn`` (SSTA). KV-cache variants for the text and vision streams
support autoregressive inference. All heavy/optional backends (``flash_attn``,
``flash_attn_interface``, ``flex_block_attn``, ``sageattention``, compiled
``flex_attention``) are imported lazily so importing this module never forces
them.
"""

import math
import warnings

import einops
import numpy as np
import torch
import torch.nn.functional as F

from minwm.distributed import (
    get_parallel_state,
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)

from .flash_attn_no_pad import (
    _get_flex_attention,
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
    flex_attn_no_pad,
)
from .infer_state import get_infer_state
from .ssta_attention import ssta_3d_attention


def _is_flash2_available() -> bool:
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func  # noqa: F401

        return True
    except Exception:
        return False


def _is_flash3_available() -> bool:
    try:
        from flash_attn_interface import flash_attn_varlen_func  # noqa: F401

        return True
    except Exception:
        return False


def _is_flash_available() -> bool:
    return _is_flash2_available() or _is_flash3_available()


def _is_sparse_attn_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return "nvidia h" in torch.cuda.get_device_properties(0).name.lower()


def _is_sparse_attn_available() -> bool:
    if not _is_sparse_attn_supported():
        return False
    try:
        from flex_block_attn import flex_block_attn_func  # noqa: F401

        return True
    except Exception:
        return False


def _maybe_fallback_attn_mode(attn_mode, infer_state=None, block_idx=None):
    """Resolve the final attention mode given config and backend availability.

    Args:
        attn_mode (str): the requested attention mode.
        infer_state: optional :class:`InferState`; gates SageAttention per block.
        block_idx (int | None): current block index for the SageAttention range.

    Returns:
        str: the attention mode to actually use after fallbacks.

    Raises:
        ValueError: if ``flex-block-attn`` is requested but unavailable.
    """
    enable_sageattn = (
        infer_state is not None
        and infer_state.enable_sageattn
        and block_idx in infer_state.sage_blocks_range
    )
    assert not (
        enable_sageattn and attn_mode == "flex-block-attn"
    ), "SageAttention cannot be used with flex-block-attn mode."

    if enable_sageattn:
        return "sageattn"

    if attn_mode == "flash":
        if _is_flash3_available():
            return "flash3"
        if _is_flash2_available():
            return "flash2"
        warnings.warn("flash is not available. Falling back to torch attention.")
        return "torch"
    if attn_mode == "flash3" and not _is_flash3_available():
        warnings.warn("flash3 is not available. Falling back to torch attention.")
        return "torch"
    if attn_mode == "flash2" and not _is_flash2_available():
        warnings.warn("flash2 is not available. Falling back to torch attention.")
        return "torch"
    if attn_mode == "flex-block-attn" and not _is_sparse_attn_available():
        raise ValueError(
            f"{attn_mode} is not available for your GPU or flex-block-attn is not installed."
        )
    return attn_mode


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: torch.Tensor | None = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """Dense attention via flash-attn varlen or torch SDPA.

    Args:
        q (torch.Tensor): query tensor ``[B, L, H, D]``.
        k (torch.Tensor): key tensor ``[B, L, H, D]``.
        v (torch.Tensor): value tensor ``[B, L, H, D]``.
        drop_rate (float): attention dropout probability.
        attn_mask (torch.Tensor | None): key-padding mask ``[B, L]``.
        causal (bool): whether to apply a causal mask.
        attn_mode (str): requested attention mode (resolved via fallback).

    Returns:
        torch.Tensor: attention output ``[B, L, H*D]``.

    Raises:
        NotImplementedError: if a float ``attn_mask`` is passed to torch mode.
    """
    attn_mode = _maybe_fallback_attn_mode(attn_mode)

    if attn_mode == "torch":
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), "attention mask must be (0,1)."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    "Float attention mask is not implemented for torch attention."
                )
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=drop_rate,
            is_causal=causal,
        )

        x = x.transpose(1, 2)
        b, s, h, d = x.shape
        return x.reshape(b, s, -1)

    qkv = torch.stack([q, k, v], dim=2)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()
    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)
    b, s, a, d = x.shape
    return x.reshape(b, s, -1)


@torch.compiler.disable
def sequence_parallel_attention_txt(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
    kv_cache=None,
    cache_txt=False,
):
    """Self-attention over the text stream, optionally caching its K/V.

    Used in autoregressive inference: computes text-only attention and (when
    ``cache_txt``) returns the per-block text key/value to seed the KV cache.

    Args:
        q (torch.Tensor): text query ``[B, S_txt, H, D]``.
        k (torch.Tensor): text key ``[B, S_txt, H, D]``.
        v (torch.Tensor): text value ``[B, S_txt, H, D]``.
        img_q_len (int): unused; kept for signature parity.
        img_kv_len (int): unused; kept for signature parity.
        attn_mode (str | None): unused; kept for signature parity.
        text_mask (torch.Tensor | None): key-padding mask over text.
        attn_param (dict | None): unused; kept for signature parity.
        block_idx (int | None): block index for the SageAttention range.
        kv_cache (dict | None): unused; kept for signature parity.
        cache_txt (bool): if True, store text K/V in the returned dict.

    Returns:
        tuple[torch.Tensor, dict]: ``(encoder_hidden_states [B, S_txt, H*D], t_kv)``.
    """
    encoder_query = q
    encoder_key = k
    encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(dim, sp_rank * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    if text_mask is not None:
        text_mask = text_mask.bool().to(encoder_query.device)

    encoder_query = encoder_query.transpose(1, 2)
    encoder_key = encoder_key.transpose(1, 2)
    encoder_value = encoder_value.transpose(1, 2)
    t_kv = {}
    if cache_txt:
        t_kv["k_txt"] = encoder_key
        t_kv["v_txt"] = encoder_value
        t_kv["text_mask"] = text_mask

    infer_state = get_infer_state()
    enable_sageattn = (
        infer_state is not None
        and infer_state.enable_sageattn
        and block_idx in infer_state.sage_blocks_range
    )
    needs_text_mask = text_mask is not None and not text_mask.all().item()
    if enable_sageattn and not needs_text_mask:
        from sageattention import sageattn

        encoder_hidden_states = sageattn(
            encoder_query,
            encoder_key,
            encoder_value,
            tensor_layout="HND",
            is_causal=False,
        )
    else:
        if text_mask is not None:
            attn_mask = text_mask[:, None, None, :]
        else:
            attn_mask = None
        encoder_hidden_states = F.scaled_dot_product_attention(
            encoder_query,
            encoder_key,
            encoder_value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
    if text_mask is not None:
        encoder_hidden_states = encoder_hidden_states * text_mask[:, :, None, None].to(
            encoder_hidden_states.dtype
        )

    if enable_sp:
        encoder_hidden_states = sequence_model_parallel_all_gather(
            encoder_hidden_states, dim=2
        ).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(q.dtype)

    b, s, a, d = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.reshape(b, s, -1)

    return encoder_hidden_states, t_kv


@torch.compiler.disable
def sequence_parallel_attention_vision(
    q,
    k,
    v,
    block_idx=None,
    kv_cache=None,
    cache_vision=False,
):
    """Vision-stream attention against cached text + vision K/V.

    Used in autoregressive inference: attends vision queries over the cached
    text K/V plus accumulated vision K/V, optionally appending the current
    vision K/V to the cache.

    Args:
        q (torch.Tensor): vision query ``[B, S_vis, H, D]``.
        k (torch.Tensor): vision key ``[B, S_vis, H, D]``.
        v (torch.Tensor): vision value ``[B, S_vis, H, D]``.
        block_idx (int | None): block index into ``kv_cache``.
        kv_cache (dict): per-block cache holding ``k_txt``/``v_txt`` and
            ``k_vision``/``v_vision``; must not be None.
        cache_vision (bool): if True, return current vision K/V to cache.

    Returns:
        tuple[torch.Tensor, dict]: ``(hidden_states [B, S_vis, H*D], vision_kv)``.
    """
    assert kv_cache is not None
    query = q
    key = k
    value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    cache_vision_key = kv_cache[block_idx]["k_vision"]
    cache_vision_value = kv_cache[block_idx]["v_vision"]

    vision_kv = {}
    if cache_vision:
        vision_kv["k_vision"] = key
        vision_kv["v_vision"] = value

    if cache_vision_key is not None:
        key = torch.cat([cache_vision_key, key], dim=2)
        value = torch.cat([cache_vision_value, value], dim=2)

    encoder_key = kv_cache[block_idx]["k_txt"]
    encoder_value = kv_cache[block_idx]["v_txt"]
    text_mask = kv_cache[block_idx].get("text_mask")
    if text_mask is not None:
        text_mask = text_mask.bool().to(query.device)

    key = torch.cat([encoder_key, key], dim=2)
    value = torch.cat([encoder_value, value], dim=2)

    infer_state = get_infer_state()
    enable_sageattn = (
        infer_state is not None
        and infer_state.enable_sageattn
        and block_idx in infer_state.sage_blocks_range
    )
    needs_text_mask = text_mask is not None and not text_mask.all().item()
    if enable_sageattn and not needs_text_mask:
        from sageattention import sageattn

        hidden_states = sageattn(query, key, value, tensor_layout="HND", is_causal=False)
    else:
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (0, key.shape[2] - text_mask.shape[1]), value=True)
            attn_mask = attn_mask[:, None, None, :]
        else:
            attn_mask = None
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    hidden_states = hidden_states.transpose(1, 2)

    if enable_sp:
        hidden_states = sequence_model_parallel_all_to_all_4D(
            hidden_states, scatter_dim=1, gather_dim=2
        )
        hidden_states = hidden_states.to(query.dtype)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states, vision_kv


@torch.compiler.disable
def parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
    tf_block_mask=None,
):
    """Thin wrapper forwarding to :func:`sequence_parallel_attention`.

    Args:
        q (tuple): ``(vision_query, text_query)`` tensors ``[B, S, H, D]``.
        k (tuple): ``(vision_key, text_key)`` tensors.
        v (tuple): ``(vision_value, text_value)`` tensors.
        img_q_len (int): vision query length.
        img_kv_len (int): vision key/value length.
        attn_mode (str): attention mode to dispatch.
        text_mask (torch.Tensor | None): key-padding mask over text.
        attn_param (dict | None): backend-specific parameters.
        block_idx (int | None): current block index.
        tf_block_mask: teacher-forcing ``BlockMask`` for ``flex_tf`` mode.

    Returns:
        torch.Tensor: attention output ``[B, S, H*D]``.
    """
    return sequence_parallel_attention(
        q,
        k,
        v,
        img_q_len,
        img_kv_len,
        attn_mode,
        text_mask,
        attn_param=attn_param,
        tf_block_mask=tf_block_mask,
        block_idx=block_idx,
    )


def sequence_parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    tf_block_mask=None,
    block_idx=None,
):
    """Sequence-parallel attention dispatch across eight backends.

    Gathers heads across the SP group (all-to-all for vision, head-shrink for
    text), runs the selected backend over the concatenated vision+text stream,
    then scatters the result back. Supported ``attn_mode`` values: ``torch``,
    ``flash2``, ``flash3``, ``sageattn``, ``flex_tf``, ``torch_causal``,
    ``flex_causal`` and ``flex-block-attn``.

    Args:
        q (tuple): ``(vision_query, text_query)`` tensors ``[B, S, H, D]``.
        k (tuple): ``(vision_key, text_key)`` tensors.
        v (tuple): ``(vision_value, text_value)`` tensors.
        img_q_len (int): vision query length.
        img_kv_len (int): vision key/value length.
        attn_mode (str): attention mode to dispatch; must not be None.
        text_mask (torch.Tensor | None): key-padding mask over text.
        attn_param (dict | None): backend-specific parameters (thw, ssta knobs).
        tf_block_mask: teacher-forcing ``BlockMask`` for ``flex_tf`` mode.
        block_idx (int | None): current block index.

    Returns:
        torch.Tensor: attention output ``[B, S, H*D]``.

    Raises:
        NotImplementedError: if ``attn_mode`` is unsupported.
    """
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    sp_world_size = parallel_dims.sp if parallel_dims.sp_enabled else 1

    if sp_world_size > 1:
        sp_size = sp_world_size
        sp_rank = parallel_dims.sp_rank

        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(dim, sp_rank * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    needs_token_text_mask = text_mask is not None and not text_mask.bool().all().item()
    if needs_token_text_mask and attn_mode in {"flex-block-attn", "sageattn"}:
        # SSTA and SageAttention do not apply token-level text masks.
        attn_mode = "flash"
    infer_state = None if needs_token_text_mask else get_infer_state()
    attn_mode = _maybe_fallback_attn_mode(attn_mode, infer_state, block_idx)

    # SPA_BRANCHES
    if attn_mode == "sageattn":
        from sageattention import sageattn

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)

    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), "attention mask must be (0,1)."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    "Float attention mask is not implemented for torch attention."
                )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if attn_mask is not None:
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2)

    elif attn_mode == "torch_causal":
        vision_seq_length = query.shape[1]
        text_seq_length = encoder_query.shape[1]
        total_seq_length = vision_seq_length + text_seq_length

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        latent_seq_length = attn_param["thw"][-1] * attn_param["thw"][-2]
        chunk_seq_length = latent_seq_length * 4
        chunk_num = vision_seq_length // chunk_seq_length
        causal_mask = torch.zeros((total_seq_length, total_seq_length), device=query.device)
        causal_mask[:, :text_seq_length] = 1
        for i in range(chunk_num):
            start_i = text_seq_length + i * chunk_seq_length
            end_i = min(start_i + chunk_seq_length, total_seq_length)
            for j in range(i + 1):
                start_j = text_seq_length + j * chunk_seq_length
                end_j = min(start_j + chunk_seq_length, total_seq_length)
                causal_mask[start_i:end_i, start_j:end_j] = 1

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        causal_mask = causal_mask.expand(query.shape[0], 1, -1, -1)
        causal_mask = causal_mask.to(torch.bool)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, text_seq_length:, :, :],
            hidden_states[:, :text_seq_length, :, :],
        )
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    elif attn_mode == "flex_tf":
        block_mask = tf_block_mask
        assert block_mask is not None, "block_mask must be provided for flex_tf mode"

        text_seq_length = encoder_query.shape[1]

        if sp_world_size > 1:
            B, S, H, D = query.shape
            per_rank_half = S // (2 * sp_world_size)

            def _interleaved_to_contiguous(x):
                B, S, H, D = x.shape
                return (
                    x.reshape(B, sp_world_size, 2, per_rank_half, H, D)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(B, S, H, D)
                )

            query = _interleaved_to_contiguous(query)
            key = _interleaved_to_contiguous(key)
            value = _interleaved_to_contiguous(value)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        total_len = query.shape[1]
        padded_total = math.ceil(total_len / 128) * 128
        pad_len = padded_total - total_len
        if pad_len > 0:
            query = F.pad(query, (0, 0, 0, 0, 0, pad_len))
            key = F.pad(key, (0, 0, 0, 0, 0, pad_len))
            value = F.pad(value, (0, 0, 0, 0, 0, pad_len))

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = _get_flex_attention()(query, key, value, block_mask=block_mask)

        if pad_len > 0:
            hidden_states = hidden_states[:, :, :total_len]
        hidden_states = hidden_states.transpose(1, 2)

        vision_out = hidden_states[:, text_seq_length:, :, :]
        encoder_hidden_states = hidden_states[:, :text_seq_length, :, :]

        if sp_world_size > 1:
            B, S, H, D = vision_out.shape
            per_rank_half = S // (2 * sp_world_size)

            def _contiguous_to_interleaved(x):
                B, S, H, D = x.shape
                return (
                    x.reshape(B, 2, sp_world_size, per_rank_half, H, D)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(B, S, H, D)
                )

            vision_out = _contiguous_to_interleaved(vision_out)

        hidden_states = torch.cat([vision_out, encoder_hidden_states], dim=1)

    elif attn_mode == "flash2":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        qkv = torch.stack([query, key, value], dim=2)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            total_length = sequence_length + encoder_sequence_length
            attn_mask = torch.ones(
                query.shape[0], total_length, dtype=torch.bool, device=query.device
            )
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0)

    elif attn_mode == "flex_causal":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        qkv = torch.stack([query, key, value], dim=2)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            total_length = sequence_length + encoder_sequence_length
            attn_mask = torch.ones(
                query.shape[0], total_length, dtype=torch.bool, device=query.device
            )
        hidden_states = flex_attn_no_pad(qkv, attn_mask, causal=True, dropout_p=0.0)

    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        qkv = torch.stack([query, key, value], dim=2)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            total_length = sequence_length + encoder_sequence_length
            attn_mask = torch.ones(
                query.shape[0], total_length, dtype=torch.bool, device=query.device
            )
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0)

    elif attn_mode == "flex-block-attn":
        sparse_type = attn_param["attn_sparse_type"]
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def _get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                return (1, 16, 24)
            elif block_size == 128:
                return (1, 16, 8)
            elif block_size == 64:
                return (1, 8, 8)
            elif block_size == 16:
                return (1, 4, 4)
            else:
                raise ValueError(f"Unsupported tile_size {tile_size}; only [16, 64, 128, 384].")

        if thw[0] == 1:
            tile_size = _get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31:
            ssta_topk = ssta_topk // 2

        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states, _sparse_ratio = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f"Unsupported attention mode: {attn_mode}. "
            "Supported: torch, flash2, flash3, sageattn, flex_tf, torch_causal, "
            "flex_causal, flex-block-attn."
        )

    if sp_world_size > 1:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )
        hidden_states = sequence_model_parallel_all_to_all_4D(
            hidden_states, scatter_dim=1, gather_dim=2
        )
        encoder_hidden_states = sequence_model_parallel_all_gather(
            encoder_hidden_states, dim=2
        ).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
