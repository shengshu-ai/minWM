# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""CausalWan21Model: autoregressive DiT with KV cache, flex-attention block masks.

Supports two training regimes (teacher forcing, diffusion forcing) and a
KV-cached inference path. See Algorithm 2 of CausVid
(https://arxiv.org/abs/2412.07772) for the inference loop.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from minwm.distributed.collective import sp_all_gather
from minwm.distributed.parallel_dims import get_parallel_state
from minwm.modeling.wan21.layers.norm import WanLayerNorm
from minwm.modeling.wan21.layers.rope import rope_params, sinusoidal_embedding_1d

from .blocks import CausalWanAttentionBlock
from .model import MLPProj
from .packed_tf import packed_tf_enabled

__all__ = [
    "CausalHead",
    "CausalWan21Model",
    "is_causal_block",
]


def is_causal_block(name: str, module: nn.Module) -> bool:
    """True for top-level transformer blocks (``blocks.<i>``), for FSDP sharding."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()


def _teacher_forcing_mask_mod(
    *,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    num_frame_per_block: int = 1,
    padded_length: int = 0,
    device=None,
):
    """The teacher-forcing mask, as a flex_attention ``mask_mod``.

    ``num_frames`` frames of ``frame_seqlen`` tokens are clean, then the same
    number are noisy. A clean row sees the causal *block* prefix it belongs to
    (``num_frame_per_block`` frames per block), a noisy row sees the clean
    prefix and its own noisy block, and every row sees itself — the ``q == kv``
    term is what makes trailing alignment padding visible at all (it is
    otherwise contained in the ranges above, see ``packed_tf``).

    Args:
        num_frames (int): number of latent frames (per half).
        frame_seqlen (int): tokens per latent frame.
        num_frame_per_block (int): frames grouped into one causal chunk.
        padded_length (int): trailing padding appended to the sequence.
        device (torch.device | str): device for the index tensors.

    Returns:
        Callable: ``attention_mask(b, h, q_idx, kv_idx)`` over flat indices of
        the padded sequence. Called on dense ``q_idx``/``kv_idx`` grids it is
        the mask as a bool tensor — the reference the packed lowering
        (:mod:`minwm.modeling.wan21.packed_tf`) is checked against.
    """
    total_length = num_frames * frame_seqlen * 2
    length = total_length + padded_length
    clean_ends = num_frames * frame_seqlen

    context_ends = torch.zeros(length, device=device, dtype=torch.long)
    noise_context_starts = torch.zeros(length, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(length, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(length, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(length, device=device, dtype=torch.long)

    attention_block_size = frame_seqlen * num_frame_per_block
    frame_indices = torch.arange(
        0, num_frames * frame_seqlen, step=attention_block_size, device=device, dtype=torch.long
    )
    for start in frame_indices:
        context_ends[start : start + attention_block_size] = start + attention_block_size

    noisy_start_list = torch.arange(
        num_frames * frame_seqlen,
        total_length,
        step=attention_block_size,
        device=device,
        dtype=torch.long,
    )
    noisy_end_list = noisy_start_list + attention_block_size
    for block_index, (start, end) in enumerate(zip(noisy_start_list, noisy_end_list)):
        noise_noise_starts[start:end] = start
        noise_noise_ends[start:end] = end
        noise_context_ends[start:end] = block_index * attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
        c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        c2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
        noise_mask = (q_idx >= clean_ends) & (c1 | c2)
        return (q_idx == kv_idx) | clean_mask | noise_mask

    return attention_mask


class CausalHead(nn.Module):
    """Output head with frame-level AdaLN modulation.

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
            e (Tensor): AdaLN conditioning, shape ``[B, F, 1, dim]``.
        Returns:
            Tensor: projected patches, shape ``[B, L, prod(patch_size)*out_dim]``.
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.norm(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        return self.head(x.flatten(1, 2))


class CausalWan21Model(ModelMixin, ConfigMixin):
    """Wan autoregressive DiT backbone with KV cache and flex-attention masks.

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
        local_attn_size (int): temporal local-attention window in frames (``-1`` = global).
        sink_size (int): number of sink frames kept when rolling the KV cache.
        qk_norm (bool): QK normalisation.
        cross_attn_norm (bool): extra LayerNorm before cross-attention.
        eps (float): epsilon for all norms.
        use_prope (bool): build per-block PRoPE camera projections (zero-init) so
            viewmats/Ks modulate self-attention. Baked in at construction so
            ``from_pretrained`` round-trips the camera params.
        num_frame_per_block (int): latent frames grouped into one causal attention
            chunk (the AR block size). ``1`` = pure frame-causal; stage 1 sets ``4``.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True
    _fsdp_shard_conditions = [is_causal_block]

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
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_prope: bool = False,
        num_frame_per_block: int = 1,
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
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_prope = use_prope

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
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    use_prope,
                    # The teacher-forcing mask is built in units of these blocks
                    # here and lowered that way by packed_tf; keep the two in sync.
                    num_frame_per_block=num_frame_per_block,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)

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
        self.block_mask = None
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = False

    def _set_gradient_checkpointing(
        self, module=None, value=False, enable=None, gradient_checkpointing_func=None
    ):
        if enable is not None:
            value = enable
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        """Block-wise causal mask: each token attends to all tokens up to its chunk end.

        Args:
            device (torch.device | str): device for the mask.
            num_frames (int): number of latent frames.
            frame_seqlen (int): tokens per latent frame.
            num_frame_per_block (int): frames grouped into one causal chunk.
            local_attn_size (int): temporal window in frames (``-1`` = global).

        Returns:
            BlockMask: flex-attention block mask.
        """
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        frame_indices = torch.arange(
            0, total_length, step=frame_seqlen * num_frame_per_block, device=device
        )
        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return (
                (kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
            ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )
            print(block_mask)
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 1,
    ) -> BlockMask:
        """Teacher-forcing mask over concatenated ``[clean | noisy]`` token sequence.

        Clean tokens attend block-wise causally among themselves; each noisy block
        attends to all preceding clean blocks plus itself. The mask itself is
        :func:`_teacher_forcing_mask_mod`; this only wraps it into a ``BlockMask``.

        Args:
            device (torch.device | str): device for the mask.
            num_frames (int): number of latent frames (per half).
            frame_seqlen (int): tokens per latent frame.
            num_frame_per_block (int): frames grouped into one causal chunk.

        Returns:
            BlockMask: flex-attention block mask of length ``2 * num_frames * frame_seqlen``.
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        attention_mask = _teacher_forcing_mask_mod(
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            padded_length=padded_length,
            device=device,
        )

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            # Build the block mask under torch.compile: eager construction
            # materializes the full dense S×S index grid (int64) before
            # compressing to blocks — ~29 GiB at the doubled 20-frame
            # teacher-forcing length. In single-model stages that transient fits
            # (the mask builds on a near-empty GPU), but DMD builds it after the
            # three 1.3B nets + self-rollout already hold ~50 GiB, so the dense
            # spike OOMs. _compile=True constructs it block-sparsely, no dense
            # intermediate. Pairs with the compiled flex_attention op itself.
            _compile=True,
            device=device,
        )

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 4,
        local_attn_size: int = -1,
    ) -> BlockMask:
        """Block-wise causal mask with the first frame separated out for I2V.

        Args:
            device (torch.device | str): device for the mask.
            num_frames (int): number of latent frames.
            frame_seqlen (int): tokens per latent frame.
            num_frame_per_block (int): frames grouped into one causal chunk.
            local_attn_size (int): temporal window in frames (``-1`` = global).

        Returns:
            BlockMask: flex-attention block mask.
        """
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        ends[:frame_seqlen] = frame_seqlen

        frame_indices = torch.arange(
            frame_seqlen, total_length, step=frame_seqlen * num_frame_per_block, device=device
        )
        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return (
                (kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
            ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )
            print(block_mask)
        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
    ):
        """KV-cached autoregressive inference, one latent frame at a time.

        See Algorithm 2 of CausVid (https://arxiv.org/abs/2412.07772).

        Args:
            x (list[Tensor]): input video tensors, each ``[C_in, F, H, W]``.
            t (Tensor): diffusion timesteps, shape ``[B, F]`` or ``[B]``.
            context (list[Tensor]): text embeddings, each ``[L, C]``.
            seq_len (int): maximum sequence length.
            clip_fea (Tensor, optional): CLIP image features for i2v.
            y (list[Tensor], optional): reference frames for i2v.
            kv_cache (dict): per-block self-attention KV caches.
            crossattn_cache (dict): per-block cross-attention caches.
            current_start (int): token offset for causal RoPE.
            cache_start (int): cache write offset.
            viewmats (Tensor, optional): camera extrinsics, ``[B, F, 4, 4]``.
            Ks (Tensor, optional): camera intrinsics, ``[B, F, 3, 3]``.
            prope_kv_cache (dict, optional): per-block PRoPE KV caches.

        Returns:
            Tensor: denoised video, shape ``[B, C_out, F, H, W]``.
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
        x = torch.cat(x)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        if clip_fea is not None:
            context = torch.cat([self.img_emb(clip_fea), context], dim=1)

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            x = torch.chunk(x, sp_size, dim=1)[sp_rank]

        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            single_seq_len = seq_lens[0].item()
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                ks = Ks[i, :f, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                pad_len = single_seq_len - f * h * w
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
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )
        if viewmats is not None:
            kwargs["viewmats"] = viewmats
            kwargs["Ks"] = Ks

        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index],
                current_start=current_start,
                cache_start=cache_start,
                prope_kv_cache=prope_kv_cache[block_index] if prope_kv_cache is not None else None,
            )
            # Checkpoint the block on the grad-carrying rollout step, matching the
            # teacher-forcing path and the reference trainer. The block's in-place
            # KV-cache writes survive non-reentrant recompute: the caches are
            # updated in place *before* the checkpointed region runs (kwargs above),
            # so recompute reads the same cache tensors and produces the same
            # activations. Without this, the full-clip 30-layer activations for the
            # exit step are all retained — the dominant DMD generator-step cost that
            # pushes low-sp single-node runs OOM.
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        if sp_enabled:
            x = sp_all_gather(x, dim=1)

        x = self.head(x, e.unflatten(0, t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        viewmats=None,
        Ks=None,
    ):
        """Training forward pass (teacher forcing or diffusion forcing).

        When ``clean_x`` is given, runs teacher forcing: clean and noisy halves
        are concatenated and the teacher-forcing block mask is built — unless the
        packed attention path is on, which lowers the mask itself and needs no
        ``BlockMask``. Otherwise runs diffusion forcing with a block-wise causal
        mask.

        Args:
            x (list[Tensor]): noisy input video tensors, each ``[C_in, F, H, W]``.
            t (Tensor): diffusion timesteps, shape ``[B, F]`` or ``[B]``.
            context (list[Tensor]): text embeddings, each ``[L, C]``.
            seq_len (int): maximum padded sequence length.
            clean_x (list[Tensor], optional): clean context frames for teacher forcing.
            aug_t (Tensor, optional): timesteps for the clean half (default: zeros).
            clip_fea (Tensor, optional): CLIP image features for i2v.
            y (list[Tensor], optional): reference frames for i2v.
            viewmats (Tensor, optional): camera extrinsics, ``[B, F, 4, 4]``.
            Ks (Tensor, optional): camera intrinsics, ``[B, F, 3, 3]``.

        Returns:
            Tensor: denoised video, shape ``[B, C_out, F, H, W]``.
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.block_mask is None:
            x0 = x[0]
            num_frames = x0.shape[1]
            mask_frame_seqlen = (
                x0.shape[-2] * x0.shape[-1] // (self.patch_size[1] * self.patch_size[2])
            )
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                # The packed path lowers the mask itself and never reads a
                # BlockMask; building one here would be dead weight.
                if not packed_tf_enabled():
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=num_frames,
                        frame_seqlen=mask_frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                    )
            elif self.independent_first_frame:
                self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                    device,
                    num_frames=num_frames,
                    frame_seqlen=mask_frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device,
                    num_frames=num_frames,
                    frame_seqlen=mask_frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1) for u in x]
        )

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        if clip_fea is not None:
            context = torch.cat([self.img_emb(clip_fea), context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(
                [
                    torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1)
                    for u in clean_x
                ]
            )
            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = (
                self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(0, t.shape)
            )
            e0 = torch.cat([e0_clean, e0], dim=1)

        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            single_seq_len = seq_lens[0].item()
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                ks = Ks[i, :f, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                pad_len = single_seq_len - f * h * w
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
            if clean_x is not None:
                viewmats = torch.cat([viewmats, viewmats], dim=1)
                Ks = torch.cat([Ks, Ks], dim=1)

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        sp_seq_len_orig = x.shape[1]
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            num_frames_total = e0.shape[1]
            frame_seqlen = x.shape[1] // num_frames_total
            e0 = e0.unsqueeze(2).expand(-1, -1, frame_seqlen, -1, -1).flatten(1, 2)

            if clean_x is not None:
                half = sp_seq_len_orig // 2
                sp_pad_len = (sp_size - half % sp_size) % sp_size

                def _chunk_half(t_full, lo, hi, pad_dims):
                    seg = t_full[:, lo:hi]
                    if sp_pad_len > 0:
                        seg = F.pad(seg, pad_dims)
                    return torch.chunk(seg, sp_size, dim=1)[sp_rank]

                x = torch.cat(
                    [
                        _chunk_half(x, 0, half, (0, 0, 0, sp_pad_len)),
                        _chunk_half(x, half, sp_seq_len_orig, (0, 0, 0, sp_pad_len)),
                    ],
                    dim=1,
                )
                e0 = torch.cat(
                    [
                        _chunk_half(e0, 0, half, (0, 0, 0, 0, 0, sp_pad_len)),
                        _chunk_half(e0, half, sp_seq_len_orig, (0, 0, 0, 0, 0, sp_pad_len)),
                    ],
                    dim=1,
                )
                if viewmats is not None:
                    vm_half = viewmats.shape[1] // 2
                    ks_half = Ks.shape[1] // 2
                    viewmats = torch.cat(
                        [
                            _chunk_half(viewmats, 0, vm_half, (0, 0, 0, 0, 0, sp_pad_len)),
                            _chunk_half(
                                viewmats, vm_half, viewmats.shape[1], (0, 0, 0, 0, 0, sp_pad_len)
                            ),
                        ],
                        dim=1,
                    )
                    Ks = torch.cat(
                        [
                            _chunk_half(Ks, 0, ks_half, (0, 0, 0, 0, 0, sp_pad_len)),
                            _chunk_half(Ks, ks_half, Ks.shape[1], (0, 0, 0, 0, 0, sp_pad_len)),
                        ],
                        dim=1,
                    )
            else:
                sp_pad_len = (sp_size - x.shape[1] % sp_size) % sp_size
                if sp_pad_len > 0:
                    x = F.pad(x, (0, 0, 0, sp_pad_len))
                    e0 = F.pad(e0, (0, 0, 0, 0, 0, sp_pad_len))
                x = torch.chunk(x, sp_size, dim=1)[sp_rank]
                e0 = torch.chunk(e0, sp_size, dim=1)[sp_rank]

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )
        if viewmats is not None:
            kwargs["viewmats"] = viewmats
            kwargs["Ks"] = Ks

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2 :]

        if sp_enabled:
            x = sp_all_gather(x, dim=1)
            sp_target_len = sp_seq_len_orig // 2 if clean_x is not None else sp_seq_len_orig
            if x.shape[1] > sp_target_len:
                x = x[:, :sp_target_len]

        x = self.head(x, e.unflatten(0, t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Dispatch to inference (KV cache present) or training forward."""
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct video tensors from patch token sequence.

        Args:
            x (Tensor): patch tokens, shape ``[B, L, C_out * prod(patch_size)]``.
            grid_sizes (Tensor): ``(F, H, W)`` per sample, shape ``[B, 3]``.

        Returns:
            list[Tensor]: reconstructed videos, each ``[C_out, F*pt, H*ph, W*pw]``.
        """
        c = self.out_dim
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
