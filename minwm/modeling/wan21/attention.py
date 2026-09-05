# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan-specific attention modules (self-attention + cross-attention variants)."""

import torch
import torch.nn as nn

from minwm.distributed.collective import sp_all_to_all_4D
from minwm.distributed.parallel_dims import get_parallel_state
from minwm.modeling.wan21.layers.attention import attention
from minwm.modeling.wan21.layers.norm import WanRMSNorm

__all__ = [
    "WanSelfAttention",
    "WanT2VCrossAttention",
    "WanI2VCrossAttention",
    "WanGanCrossAttention",
    "WAN_CROSSATTENTION_CLASSES",
]


class WanSelfAttention(nn.Module):
    """Self-attention with RoPE + optional PRoPE + optional sequence parallelism.

    Args:
        dim (int): model hidden dimension.
        num_heads (int): number of attention heads.
        window_size (tuple): local attention window ``(left, right)``, ``(-1, -1)`` for global.
        qk_norm (bool): apply RMSNorm to queries and keys.
        eps (float): epsilon for RMSNorm.
        use_prope (bool): build a zero-initialised PRoPE output projection
            (``prope_o``) so camera extrinsics/intrinsics modulate attention. Baked
            in at construction (not injected later) so ``from_pretrained`` round-trips
            the camera params and train/inference stay symmetric.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_prope: bool = False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_prope = use_prope

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        if use_prope:
            self.prope_o = nn.Linear(dim, dim)
            nn.init.zeros_(self.prope_o.weight)
            nn.init.zeros_(self.prope_o.bias)

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        viewmats=None,
        Ks=None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape ``[B, L, C]``.
            seq_lens (Tensor): valid sequence lengths per sample, shape ``[B]``.
            grid_sizes (Tensor): ``(F, H, W)`` grids per sample, shape ``[B, 3]``.
            freqs (Tensor): precomputed RoPE frequencies, shape ``[max_len, D//2]``.
            viewmats (Tensor, optional): camera extrinsics for PRoPE, shape ``[B, L, 4, 4]``.
            Ks (Tensor, optional): camera intrinsics for PRoPE, shape ``[B, L, 3, 3]``.

        Returns:
            Tensor: attention output, shape ``[B, L, C]``.
        """
        from minwm.modeling.wan21.layers.rope import rope_apply

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        sp_enabled = get_parallel_state().sp_enabled

        prope_enabled = self.use_prope and viewmats is not None
        if prope_enabled:
            from minwm.modeling.common.prope import prope_qkv

            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            q_p = q_p.permute(0, 2, 1, 3)
            k_p = k_p.permute(0, 2, 1, 3)
            v_p = v_p.permute(0, 2, 1, 3)

        if sp_enabled:
            q = sp_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sp_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sp_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            if prope_enabled:
                q_p = sp_all_to_all_4D(q_p, scatter_dim=2, gather_dim=1)
                k_p = sp_all_to_all_4D(k_p, scatter_dim=2, gather_dim=1)
                v_p = sp_all_to_all_4D(v_p, scatter_dim=2, gather_dim=1)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        if prope_enabled:
            x_prope = attention(q=q_p, k=k_p, v=v_p, k_lens=seq_lens, window_size=self.window_size)

        if sp_enabled:
            x = sp_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
            if prope_enabled:
                x_prope = sp_all_to_all_4D(x_prope, scatter_dim=1, gather_dim=2)

        x = self.o(x.flatten(2))

        if prope_enabled:
            x_prope = apply_fn_o(x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            x = x + self.prope_o(x_prope.flatten(2))

        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross-attention: queries from video tokens, keys/values from text context."""

    def forward(self, x, context, context_lens, crossattn_cache=None):
        """
        Args:
            x (Tensor): video tokens, shape ``[B, L, C]``.
            context (Tensor): text embeddings, shape ``[B, T, C]``.
            context_lens (Tensor, optional): valid text lengths, shape ``[B]``.
            crossattn_cache (dict, optional): cached ``k``/``v`` for text context.

        Returns:
            Tensor: cross-attention output, shape ``[B, L, C]``.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None and crossattn_cache.get("is_init"):
            k, v = crossattn_cache["k"], crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            if crossattn_cache is not None:
                crossattn_cache.update({"is_init": True, "k": k, "v": v})

        x = attention(q, k, v, k_lens=context_lens)
        return self.o(x.flatten(2))


class WanI2VCrossAttention(WanSelfAttention):
    """Image-to-video cross-attention: separate key/value projections for image tokens."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        """
        Args:
            x (Tensor): video tokens, shape ``[B, L, C]``.
            context (Tensor): image tokens (first 257) + text embeddings, ``[B, T, C]``.
            context_lens (Tensor, optional): valid text lengths, shape ``[B]``.

        Returns:
            Tensor: cross-attention output, shape ``[B, L, C]``.
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        x = attention(q, k, v, k_lens=context_lens) + attention(q, k_img, v_img)
        return self.o(x.flatten(2))


class WanGanCrossAttention(WanSelfAttention):
    """GAN-style cross-attention: context (query) attends to video tokens (key/value)."""

    def forward(self, x, context, crossattn_cache=None):
        """
        Args:
            x (Tensor): video tokens, shape ``[B, L, C]``.
            context (Tensor): GAN context (acts as query), shape ``[B, T, C]``.
            crossattn_cache: unused, kept for API compatibility.

        Returns:
            Tensor: cross-attention output from context perspective, shape ``[B, 1, C]``.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)
        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)
        x = attention(qq, kk, vv)
        return self.o(x.flatten(2))


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}
