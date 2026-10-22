# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan transformer blocks (bidirectional and causal)."""

import math

import torch
import torch.nn as nn

from minwm.distributed.collective import sp_all_to_all_4D
from minwm.distributed.parallel_dims import get_parallel_state
from minwm.modeling.wan21.layers.norm import WanLayerNorm

from .attention import WAN_CROSSATTENTION_CLASSES, WanSelfAttention
from .packed_tf import packed_tf_attention, packed_tf_enabled, tf_geometry

__all__ = ["WanAttentionBlock", "CausalWanSelfAttention", "CausalWanAttentionBlock"]

_COMPILED_FLEX = None


def _get_flex_attention():
    """Return a ``torch.compile``-d ``flex_attention``, compiled once and cached.

    Eager ``flex_attention`` materializes the full dense ``L×L`` score matrix —
    it does not exploit the ``block_mask`` sparsity — so the causal model's
    teacher-forcing-doubled sequence OOMs (an 87 GiB score matrix at full res).
    The compiled kernel is block-sparse / ``O(L)`` and is what real training
    needs; the math is identical. Compiled lazily (not at import) so CPU-only or
    non-causal paths pay nothing, and cached module-wide so the (slow) first
    compile happens once rather than per block/step.
    """
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX


class WanAttentionBlock(nn.Module):
    """Full DiT transformer block: self-attn + cross-attn + FFN + AdaLN modulation.

    Args:
        cross_attn_type (str): key into ``WAN_CROSSATTENTION_CLASSES``.
        dim (int): model hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        num_heads (int): number of attention heads.
        window_size (tuple): local attention window for self-attention.
        qk_norm (bool): QK normalisation.
        cross_attn_norm (bool): extra LayerNorm before cross-attention.
        eps (float): epsilon for norms.
        use_prope (bool): enable PRoPE camera conditioning on the self-attention.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        use_prope: bool = False,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, use_prope)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, viewmats=None, Ks=None
    ):
        """
        Args:
            x (Tensor): token features, shape ``[B, L, C]``.
            e (Tensor): AdaLN conditioning, shape ``[B, 6, C]``.
            seq_lens (Tensor): valid sequence lengths, shape ``[B]``.
            grid_sizes (Tensor): ``(F, H, W)`` grids, shape ``[B, 3]``.
            freqs (Tensor): RoPE frequencies.
            context (Tensor): text/image context, shape ``[B, T, C]``.
            context_lens (Tensor, optional): valid context lengths.
            viewmats (Tensor, optional): camera extrinsics for PRoPE.
            Ks (Tensor, optional): camera intrinsics for PRoPE.

        Returns:
            Tensor: output tokens, shape ``[B, L, C]``.
        """
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs, viewmats=viewmats, Ks=Ks
        )
        x = x + y * e[2]
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        x = x + self.ffn(self.norm2(x) * (1 + e[4]) + e[3]) * e[5]
        return x


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with KV cache, optional local window, sink tokens, and SP support.

    Args:
        dim (int): model hidden dimension.
        num_heads (int): number of attention heads.
        local_attn_size (int): local window in frames (``-1`` = global).
        sink_size (int): number of sink frames whose KV entries are never evicted.
        qk_norm (bool): QK normalisation.
        eps (float): epsilon for RMSNorm.
        use_prope (bool): enable PRoPE camera conditioning.
        num_frame_per_block (int): latent frames per causal block; the
            teacher-forcing mask (and the packed lowering of it) is built in
            units of these blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_prope: bool = False,
        num_frame_per_block: int = 1,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.max_attention_size = 31200 if local_attn_size == -1 else local_attn_size * 1560
        self.use_prope = use_prope

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        from minwm.modeling.wan21.layers.norm import WanRMSNorm

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        if use_prope:
            self.prope_o = nn.Linear(dim, dim)
            nn.init.zeros_(self.prope_o.weight)
            nn.init.zeros_(self.prope_o.bias)

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
    ):
        """
        Args:
            x (Tensor): shape ``[B, L, N, D]`` (pre-projected tokens).
            seq_lens (Tensor): valid lengths, shape ``[B]``.
            grid_sizes (Tensor): ``(F, H, W)`` per sample, shape ``[B, 3]``.
            freqs (Tensor): RoPE frequencies.
            block_mask (BlockMask): flex-attention block mask.
            kv_cache (dict, optional): KV cache dict (``k``, ``v``, indices).
            current_start (int): token offset for causal RoPE.
            cache_start (int, optional): cache write offset (defaults to ``current_start``).
            viewmats (Tensor, optional): camera extrinsics for PRoPE.
            Ks (Tensor, optional): camera intrinsics for PRoPE.
            prope_kv_cache (dict, optional): separate KV cache for the PRoPE path.

        Returns:
            Tensor: attention output, shape ``[B, L, C]``.
        """
        from minwm.modeling.wan21.layers.rope import causal_rope_apply, rope_apply

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        prope_enabled = (
            viewmats is not None
            and hasattr(self, "prope_o")
            and (kv_cache is None or prope_kv_cache is not None)
        )
        if prope_enabled:
            from minwm.modeling.common.prope import prope_qkv

            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            q_p, k_p, v_p = (
                q_p.permute(0, 2, 1, 3),
                k_p.permute(0, 2, 1, 3),
                v_p.permute(0, 2, 1, 3),
            )

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled

        if sp_enabled:
            q = sp_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sp_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sp_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            if prope_enabled:
                q_p = sp_all_to_all_4D(q_p, scatter_dim=2, gather_dim=1)
                k_p = sp_all_to_all_4D(k_p, scatter_dim=2, gather_dim=1)
                v_p = sp_all_to_all_4D(v_p, scatter_dim=2, gather_dim=1)

        is_tf = False
        sp_world_size = per_rank_half = 0
        if kv_cache is None:
            full_s = q.shape[1]
            is_tf = full_s > seq_lens[0].item() * 1.5
            if is_tf:
                if sp_enabled:
                    sp_world_size = parallel_dims.sp
                    chunk_size = full_s // sp_world_size
                    per_rank_half = chunk_size // 2

                    def _interleaved_to_contiguous(t):
                        B, S, H, D = t.shape
                        return (
                            t.reshape(B, sp_world_size, 2, per_rank_half, H, D)
                            .permute(0, 2, 1, 3, 4, 5)
                            .reshape(B, S, H, D)
                        )

                    q, k, v = (
                        _interleaved_to_contiguous(q),
                        _interleaved_to_contiguous(k),
                        _interleaved_to_contiguous(v),
                    )
                    if prope_enabled:
                        q_p, k_p, v_p = (
                            _interleaved_to_contiguous(q_p),
                            _interleaved_to_contiguous(k_p),
                            _interleaved_to_contiguous(v_p),
                        )

                    unpadded_half = seq_lens[0].item()
                    sp_pad_per_half = full_s // 2 - unpadded_half
                    if sp_pad_per_half > 0:

                        def _strip_sp_pad(t):
                            c = t[:, :unpadded_half]
                            nn_ = t[:, full_s // 2 : full_s // 2 + unpadded_half]
                            return torch.cat([c, nn_], dim=1)

                        q, k, v = _strip_sp_pad(q), _strip_sp_pad(k), _strip_sp_pad(v)
                        if prope_enabled:
                            q_p, k_p, v_p = (
                                _strip_sp_pad(q_p),
                                _strip_sp_pad(k_p),
                                _strip_sp_pad(v_p),
                            )

                roped_q = torch.cat(
                    [rope_apply(c, grid_sizes, freqs).type_as(v) for c in torch.chunk(q, 2, dim=1)],
                    dim=1,
                )
                roped_k = torch.cat(
                    [rope_apply(c, grid_sizes, freqs).type_as(v) for c in torch.chunk(k, 2, dim=1)],
                    dim=1,
                )

                if packed_tf_enabled():
                    # The same mask, lowered to contiguous (q, kv) segments and run
                    # as one un-masked varlen call: no padded Q/K/V, no gather of Q
                    # and no repack of the output.
                    geom = tf_geometry(
                        clean_tokens=seq_lens[0].item(),
                        block_tokens=self.num_frame_per_block * math.prod(grid_sizes[0][1:]).item(),
                        batch=q.shape[0],
                        device=q.device,
                    )
                    x = packed_tf_attention(roped_q, roped_k, v, geom)
                    if prope_enabled:
                        x_prope = packed_tf_attention(q_p, k_p, v_p, geom)
                else:
                    _flex = _get_flex_attention()
                    pad = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]

                    def _pad(t):
                        return (
                            torch.cat([t, t.new_zeros(t.shape[0], pad, *t.shape[2:])], dim=1)
                            if pad
                            else t
                        )

                    x = _flex(
                        _pad(roped_q).transpose(2, 1),
                        _pad(roped_k).transpose(2, 1),
                        _pad(v).transpose(2, 1),
                        block_mask=block_mask,
                    )
                    x = (x[:, :, : q.shape[1]] if pad else x).transpose(2, 1)

                    if prope_enabled:
                        x_prope = _flex(
                            _pad(q_p).transpose(2, 1),
                            _pad(k_p).transpose(2, 1),
                            _pad(v_p).transpose(2, 1),
                            block_mask=block_mask,
                        )
                        x_prope = (x_prope[:, :, : q_p.shape[1]] if pad else x_prope).transpose(
                            2, 1
                        )

                if sp_enabled and sp_pad_per_half > 0:
                    B_x, S_x, H_x, D_x = x.shape
                    half_v = S_x // 2
                    pad_t = x.new_zeros(B_x, sp_pad_per_half, H_x, D_x)
                    x = torch.cat([x[:, :half_v], pad_t, x[:, half_v:], pad_t], dim=1)
                    if prope_enabled:
                        x_prope = torch.cat(
                            [x_prope[:, :half_v], pad_t, x_prope[:, half_v:], pad_t], dim=1
                        )
            else:
                raise AssertionError(
                    "Diffusion Forcing is not supported. Only Teacher Forcing is supported."
                )
        else:
            from minwm.modeling.wan21.layers.attention import attention as _attn

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            roped_q = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start // frame_seqlen
            ).type_as(v)
            roped_k = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start // frame_seqlen
            ).type_as(v)

            current_end = current_start + roped_q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_size = kv_cache["k"].shape[1]
            num_new = roped_q.shape[1]

            if (
                self.local_attn_size != -1
                and current_end > kv_cache["global_end_index"].item()
                and num_new + kv_cache["local_end_index"].item() > kv_size
            ):
                n_evict = num_new + kv_cache["local_end_index"].item() - kv_size
                n_roll = kv_cache["local_end_index"].item() - n_evict - sink_tokens
                kv_cache["k"][:, sink_tokens : sink_tokens + n_roll] = kv_cache["k"][
                    :, sink_tokens + n_evict : sink_tokens + n_evict + n_roll
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + n_roll] = kv_cache["v"][
                    :, sink_tokens + n_evict : sink_tokens + n_evict + n_roll
                ].clone()
                le = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - n_evict
                )
                ls = le - num_new
            else:
                le = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                )
                ls = le - num_new

            win_start = max(0, le - self.max_attention_size)
            if torch.is_grad_enabled():
                # Self-rollout exit step (the only step carrying grad). Writing
                # the grad-carrying roped_k/v into the cache in place is rejected
                # by autograd ("view modified inplace") and would also entangle
                # the persistent cache with this step's graph. Truncated BPTT only
                # backprops this block's own forward, so: store DETACHED k/v for
                # future steps, and attend over an out-of-place concat of the
                # (detached) history window with the grad-carrying new tokens.
                kv_cache["k"][:, ls:le] = roped_k.detach()
                kv_cache["v"][:, ls:le] = v.detach()
                new_start = max(ls, win_start)
                k_win = torch.cat(
                    [kv_cache["k"][:, win_start:ls], roped_k[:, new_start - ls :]], dim=1
                )
                v_win = torch.cat([kv_cache["v"][:, win_start:ls], v[:, new_start - ls :]], dim=1)
            else:
                kv_cache["k"][:, ls:le] = roped_k
                kv_cache["v"][:, ls:le] = v
                k_win = kv_cache["k"][:, win_start:le]
                v_win = kv_cache["v"][:, win_start:le]
            x = _attn(roped_q, k_win, v_win)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(le)

            if prope_enabled and prope_kv_cache is not None:
                pc = prope_kv_cache
                pc_size = pc["k"].shape[1]
                if (
                    self.local_attn_size != -1
                    and current_end > pc["global_end_index"].item()
                    and num_new + pc["local_end_index"].item() > pc_size
                ):
                    p_evict = num_new + pc["local_end_index"].item() - pc_size
                    p_roll = pc["local_end_index"].item() - p_evict - sink_tokens
                    pc["k"][:, sink_tokens : sink_tokens + p_roll] = pc["k"][
                        :, sink_tokens + p_evict : sink_tokens + p_evict + p_roll
                    ].clone()
                    pc["v"][:, sink_tokens : sink_tokens + p_roll] = pc["v"][
                        :, sink_tokens + p_evict : sink_tokens + p_evict + p_roll
                    ].clone()
                    p_le = (
                        pc["local_end_index"].item()
                        + current_end
                        - pc["global_end_index"].item()
                        - p_evict
                    )
                    p_ls = p_le - num_new
                else:
                    p_le = (
                        pc["local_end_index"].item() + current_end - pc["global_end_index"].item()
                    )
                    p_ls = p_le - num_new
                p_win_start = max(0, p_le - self.max_attention_size)
                if torch.is_grad_enabled():
                    # See the self-attn cache above: detached write + out-of-place
                    # window so grad flows only through this block's new tokens.
                    pc["k"][:, p_ls:p_le] = k_p.detach()
                    pc["v"][:, p_ls:p_le] = v_p.detach()
                    p_new_start = max(p_ls, p_win_start)
                    pk_win = torch.cat(
                        [pc["k"][:, p_win_start:p_ls], k_p[:, p_new_start - p_ls :]], dim=1
                    )
                    pv_win = torch.cat(
                        [pc["v"][:, p_win_start:p_ls], v_p[:, p_new_start - p_ls :]], dim=1
                    )
                else:
                    pc["k"][:, p_ls:p_le] = k_p
                    pc["v"][:, p_ls:p_le] = v_p
                    pk_win = pc["k"][:, p_win_start:p_le]
                    pv_win = pc["v"][:, p_win_start:p_le]
                x_prope = _attn(q_p, pk_win, pv_win)
                pc["global_end_index"].fill_(current_end)
                pc["local_end_index"].fill_(p_le)

        if sp_enabled:
            if is_tf:
                B_x, S_x, H_x, D_x = x.shape

                def _contiguous_to_interleaved(t):
                    return (
                        t.reshape(B_x, 2, sp_world_size, per_rank_half, H_x, D_x)
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(B_x, S_x, H_x, D_x)
                    )

                x = _contiguous_to_interleaved(x)
                if prope_enabled:
                    x_prope = _contiguous_to_interleaved(x_prope)
            x = sp_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
            if prope_enabled:
                x_prope = sp_all_to_all_4D(x_prope, scatter_dim=1, gather_dim=2)

        x = self.o(x.flatten(2))

        if prope_enabled:
            x_prope = apply_fn_o(x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            x = x + self.prope_o(x_prope.flatten(2))

        return x


class CausalWanAttentionBlock(nn.Module):
    """Causal transformer block using :class:`CausalWanSelfAttention`.

    Args:
        cross_attn_type (str): key into ``WAN_CROSSATTENTION_CLASSES``.
        dim (int): model hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        num_heads (int): number of attention heads.
        local_attn_size (int): local window in frames.
        sink_size (int): number of sink frames.
        qk_norm (bool): QK normalisation.
        cross_attn_norm (bool): extra LayerNorm before cross-attention.
        eps (float): epsilon for norms.
        use_prope (bool): enable PRoPE camera conditioning on the self-attention.
        num_frame_per_block (int): latent frames per causal block, forwarded to
            the self-attention (see :class:`CausalWanSelfAttention`).
    """

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_prope=False,
        num_frame_per_block=1,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, use_prope, num_frame_per_block
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
    ):
        """
        Args:
            x (Tensor): token features, shape ``[B, L, C]``.
            e (Tensor): AdaLN cond — frame-level ``[B, F, 6, C]`` or token-level ``[B, L, 6, C]``.
            seq_lens (Tensor): valid sequence lengths, shape ``[B]``.
            grid_sizes (Tensor): ``(F, H, W)`` per sample, shape ``[B, 3]``.
            freqs (Tensor): RoPE frequencies.
            context (Tensor): text/image context.
            context_lens (Tensor, optional): valid context lengths.
            block_mask (BlockMask): flex-attention block mask.
            kv_cache (dict, optional): self-attention KV cache.
            crossattn_cache (dict, optional): cross-attention KV cache.
            current_start (int): token offset for causal RoPE.
            cache_start (int, optional): cache write offset.
            viewmats (Tensor, optional): camera extrinsics for PRoPE.
            Ks (Tensor, optional): camera intrinsics for PRoPE.
            prope_kv_cache (dict, optional): PRoPE KV cache.

        Returns:
            Tensor: output tokens, shape ``[B, L, C]``.
        """
        token_level_e = e.shape[1] == x.shape[1]
        e_num_frames = e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        if token_level_e:
            y = self.self_attn(
                self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                viewmats=viewmats,
                Ks=Ks,
                prope_kv_cache=prope_kv_cache,
            )
            x = x + y * e[2].squeeze(2)
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            x = x + self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)) * e[
                5
            ].squeeze(2)
        else:
            num_frames, frame_seqlen = e_num_frames, x.shape[1] // e_num_frames
            y = self.self_attn(
                (
                    self.norm1(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
                ).flatten(1, 2),
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                viewmats=viewmats,
                Ks=Ks,
                prope_kv_cache=prope_kv_cache,
            )
            x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[4]) + e[3]
                ).flatten(1, 2)
            )
            x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)

        return x
