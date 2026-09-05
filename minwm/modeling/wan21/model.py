# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan21Model: bidirectional DiT for text-to-video and image-to-video generation."""

import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from minwm.distributed.collective import sp_all_gather
from minwm.distributed.parallel_dims import get_parallel_state
from minwm.modeling.wan21.layers.norm import WanLayerNorm, WanRMSNorm
from minwm.modeling.wan21.layers.rope import rope_params, sinusoidal_embedding_1d

from .attention import WanGanCrossAttention
from .blocks import WanAttentionBlock

__all__ = ["MLPProj", "Head", "RegisterTokens", "GanAttentionBlock", "Wan21Model"]


def is_block(name: str, module: nn.Module) -> bool:
    """True for top-level transformer blocks (``blocks.<i>``), for FSDP sharding."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()


class MLPProj(nn.Module):
    """Two-layer MLP with LayerNorm for projecting image embeddings.

    Args:
        in_dim (int): input feature dimension.
        out_dim (int): output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Args:
            image_embeds (Tensor): shape ``[B, T, in_dim]``.
        Returns:
            Tensor: shape ``[B, T, out_dim]``.
        """
        return self.proj(image_embeds)


class Head(nn.Module):
    """Output head: AdaLN modulation + linear projection to patch space.

    Args:
        dim (int): hidden dimension.
        out_dim (int): output channel dimension.
        patch_size (tuple): 3-D patch size ``(T, H, W)``.
        eps (float): LayerNorm epsilon.
    """

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Args:
            x (Tensor): token features, shape ``[B, L, dim]``.
            e (Tensor): AdaLN conditioning, shape ``[B, 1, dim]``.
        Returns:
            Tensor: projected patches, shape ``[B, L, prod(patch_size)*out_dim]``.
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + e[1]) + e[0])


class RegisterTokens(nn.Module):
    """Learnable register tokens with RMSNorm.

    Args:
        num_registers (int): number of register tokens.
        dim (int): feature dimension.
    """

    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self) -> torch.Tensor:
        """Returns:
        Tensor: normalised register tokens, shape ``[num_registers, dim]``.
        """
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class GanAttentionBlock(nn.Module):
    """Cross-attention block used for classifier feature extraction.

    Context tokens (queries) attend to video tokens (keys/values).

    Args:
        dim (int): hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        num_heads (int): number of attention heads.
        window_size (tuple): local attention window.
        qk_norm (bool): QK normalisation.
        cross_attn_norm (bool): extra LayerNorm before cross-attention.
        eps (float): epsilon for norms.
    """

    def __init__(
        self,
        dim: int = 1536,
        ffn_dim: int = 8192,
        num_heads: int = 12,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanGanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Args:
            x (Tensor): video tokens, shape ``[B, L, dim]``.
            context (Tensor): register/query tokens, shape ``[B, R, dim]``.
        Returns:
            Tensor: updated context tokens, shape ``[B, R, dim]``.
        """
        token = context + self.cross_attn(self.norm3(x), context)
        return self.ffn(self.norm2(token)) + token


class Wan21Model(ModelMixin, ConfigMixin):
    """Wan bidirectional DiT backbone for text-to-video and image-to-video.

    Args:
        model_type (str): ``'t2v'`` or ``'i2v'``.
        patch_size (tuple): 3-D patch dimensions ``(T, H, W)``.
        text_len (int): fixed context sequence length.
        in_dim (int): input latent channel count.
        dim (int): transformer hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        freq_dim (int): sinusoidal time-embedding dimension.
        text_dim (int): input text-embedding dimension.
        out_dim (int): output latent channel count.
        num_heads (int): number of attention heads.
        num_layers (int): number of transformer blocks.
        window_size (tuple): local attention window for self-attention.
        qk_norm (bool): QK normalisation.
        cross_attn_norm (bool): extra LayerNorm before cross-attention.
        eps (float): epsilon for all norms.
        use_prope (bool): build per-block PRoPE camera projections (zero-init) so
            viewmats/Ks modulate self-attention. Baked in at construction so
            ``from_pretrained`` round-trips the camera params; the action-t2v
            line sets this ``True``, plain t2v leaves it ``False``.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True
    _fsdp_shard_conditions = [is_block]

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_prope: bool = False,
    ):
        super().__init__()
        assert model_type in ("t2v", "i2v")
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_prope = use_prope
        self.local_attn_size = 21

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    use_prope,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(
        self, module=None, value=False, enable=None, gradient_checkpointing_func=None
    ):
        if enable is not None:
            value = enable
        self.gradient_checkpointing = value

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Dispatch to :meth:`_forward`."""
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        classify_mode: bool = False,
        concat_time_embeddings: bool = False,
        register_tokens: nn.Module | None = None,
        cls_pred_branch: nn.Module | None = None,
        gan_ca_blocks: nn.ModuleList | None = None,
        clip_fea: torch.Tensor | None = None,
        y: list[torch.Tensor] | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the bidirectional DiT.

        Args:
            x (list[Tensor]): input video tensors, each ``[C_in, F, H, W]``.
            t (Tensor): diffusion timesteps, shape ``[B]``.
            context (list[Tensor]): text embeddings, each ``[L, C]``.
            seq_len (int): maximum padded sequence length.
            classify_mode (bool): run classifier feature extraction branch.
            concat_time_embeddings (bool): concat time embeddings for classifier.
            register_tokens (nn.Module, optional): provides register tokens.
            cls_pred_branch (nn.Module, optional): classifier head.
            gan_ca_blocks (nn.ModuleList, optional): GAN cross-attention blocks.
            clip_fea (Tensor, optional): CLIP image features for i2v, ``[B, T, C]``.
            y (list[Tensor], optional): reference frames for i2v.
            viewmats (Tensor, optional): camera extrinsics, ``[B, F, 4, 4]``.
            Ks (Tensor, optional): camera intrinsics, ``[B, F, 3, 3]``.

        Returns:
            Tensor: denoised video, shape ``[B, C_out, F, H, W]``, or
                ``(Tensor, Tensor)`` when ``classify_mode=True``.
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
        )

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        if clip_fea is not None:
            context = torch.cat([self.img_emb(clip_fea), context], dim=1)

        sp_state = get_parallel_state()
        sp_enabled = sp_state.sp_enabled
        if sp_enabled:
            sp_size = sp_state.sp
            sp_rank = sp_state.sp_rank
            chunk_size = x.shape[1] // sp_size
            chunk_start = sp_rank * chunk_size
            chunk_end = chunk_start + chunk_size
            x = x[:, chunk_start:chunk_end]

        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                ks = Ks[i, :f, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                pad_len = seq_len - f * h * w
                if pad_len > 0:
                    vm = torch.cat(
                        [vm, torch.eye(4, device=vm.device, dtype=vm.dtype).expand(pad_len, -1, -1)]
                    )
                    ks = torch.cat(
                        [ks, torch.eye(3, device=ks.device, dtype=ks.dtype).expand(pad_len, -1, -1)]
                    )
                expanded_vm.append(vm)
                expanded_ks.append(ks)
            viewmats = torch.stack(expanded_vm)
            Ks = torch.stack(expanded_ks)
            if sp_enabled:
                viewmats = viewmats[:, chunk_start:chunk_end]
                Ks = Ks[:, chunk_start:chunk_end]

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )
        if viewmats is not None:
            kwargs["viewmats"] = viewmats
            kwargs["Ks"] = Ks

        final_x = None
        if classify_mode:
            assert (
                register_tokens is not None
                and gan_ca_blocks is not None
                and cls_pred_branch is not None
            )
            final_x = []
            registers = register_tokens().unsqueeze(0).expand(x.shape[0], -1, -1)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)
            if classify_mode and ii in (13, 21, 29):
                final_x.append(gan_ca_blocks[gan_idx](x, registers[:, gan_idx : gan_idx + 1]))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            feat = (
                torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1)
                if concat_time_embeddings
                else final_x.view(final_x.shape[0], -1)
            )
            final_x = cls_pred_branch(feat)

        if sp_enabled:
            x = sp_all_gather(x, dim=1)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x
        return torch.stack(x)

    def unpatchify(
        self, x: torch.Tensor, grid_sizes: torch.Tensor, c: int | None = None
    ) -> list[torch.Tensor]:
        """Reconstruct video tensors from patch token sequence.

        Args:
            x (Tensor): patch tokens, shape ``[B, L, C_out * prod(patch_size)]``.
            grid_sizes (Tensor): ``(F, H, W)`` per sample, shape ``[B, 3]``.
            c (int, optional): override output channel count (default: ``out_dim``).

        Returns:
            list[Tensor]: reconstructed videos, each ``[C_out, F*pt, H*ph, W*pw]``.
        """
        c = self.out_dim if c is None else c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Initialise weights: Xavier uniform for Linear/Conv3d, normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
        # PRoPE output projections are a zero-init residual branch (no effect until
        # trained); re-zero after the Xavier sweep above so loading the base
        # checkpoint leaves the camera path a no-op.
        for block in self.blocks:
            if getattr(block.self_attn, "use_prope", False):
                nn.init.zeros_(block.self_attn.prope_o.weight)
                nn.init.zeros_(block.self_attn.prope_o.bias)
