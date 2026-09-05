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
"""Autoregressive HunyuanVideo 1.5 DiT (ProPE / action) for the HY15 pipeline."""

from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange, repeat

from minwm.distributed import get_parallel_state, sequence_model_parallel_all_gather
from minwm.modeling.hy15.layers import (
    FinalLayer,
    MLPEmbedder,
    PatchEmbed,
    TextProjection,
    TimestepEmbedder,
    VisionProjection,
    get_activation_layer,
    get_nd_rotary_pos_embed,
)

from .blocks import MMDoubleStreamBlock
from .encoders import ByT5Mapper
from .text_utils import pack_text_tokens
from .token_refiner import SingleTokenRefiner


def is_double_block(n: str, m) -> bool:
    """True if ``n`` names an indexed double-stream block submodule."""
    return "double_blocks" in n and str.isdigit(n.split(".")[-1])


class ARHunyuanVideo_1_5_DiffusionTransformer(ModelMixin, ConfigMixin):
    """Autoregressive HunyuanVideo 1.5 transformer backbone.

    Mirrors the bidirectional inference base but adds the autoregressive
    training paths: teacher forcing in :meth:`forward_bi`, camera ProPE
    conditioning (``viewmats``/``Ks``), and an optional discrete-action
    embedding. ProPE is enabled at construction via ``use_prope`` and gated at
    runtime on ``viewmats`` being provided.

    Args:
        patch_size (list): patch size ``[pt, ph, pw]``.
        in_channels (int): number of input channels.
        concat_condition (bool): concatenate the condition latents in ``img_in``.
        out_channels (int, optional): number of output channels; defaults to
            ``in_channels``.
        hidden_size (int): transformer hidden size.
        heads_num (int): number of attention heads.
        mlp_width_ratio (float): MLP hidden-size expansion ratio.
        mlp_act_type (str): MLP activation type.
        mm_double_blocks_depth (int): number of double-stream blocks.
        mm_single_blocks_depth (int): number of single-stream blocks; must be 0.
        rope_dim_list (list): rotary embedding dims for ``t, h, w``.
        qkv_bias (bool): use bias in the QKV projections.
        qk_norm (bool): use QK normalization.
        qk_norm_type (str): QK normalization type.
        guidance_embed (bool): use guidance embedding for distillation.
        use_meanflow (bool): kept for signature parity; ``time_r_in`` is unused.
        text_projection (str): ``"linear"`` or ``"single_refiner"``.
        use_attention_mask (bool): apply the text attention mask.
        text_states_dim (int): text-encoder output dim.
        text_states_dim_2 (int): secondary text-encoder output dim.
        text_pool_type (str, optional): pooled-text projection toggle.
        rope_theta (int): rotary embedding theta.
        attn_mode (str): attention mode identifier.
        attn_param (dict, optional): attention parameter dict.
        glyph_byT5_v2 (bool): enable the ByT5 glyph mapper.
        vision_projection (str): vision condition embedding mode.
        vision_states_dim (int): vision-encoder states input dim.
        is_reshape_temporal_channels (bool): reshape temporal channels for video VAE.
        use_cond_type_embedding (bool): add a condition-type embedding.
        use_prope (bool): build per-block ProPE projections for camera conditioning.
        ideal_resolution (str, optional): metadata; unused at runtime.
        ideal_task (str, optional): metadata; unused at runtime.
    """

    _fsdp_shard_conditions = [is_double_block]

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int | None = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str | None = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict | None = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        use_prope: bool = False,
        ideal_resolution: str | None = None,
        ideal_task: str | None = None,
        num_frame_per_block: int = 4,
    ):
        super().__init__()
        factory_kwargs = {}

        self.patch_size = patch_size
        self.num_frame_per_block = num_frame_per_block
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=hidden_size,
                use_residual=False,
            )

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            concat_condition=concat_condition,
            **factory_kwargs,
        )

        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size
            )
        else:
            self.vision_in = None

        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )
        self.vector_in = (
            MLPEmbedder(self.config.text_states_dim_2, self.hidden_size, **factory_kwargs)
            if self.text_pool_type is not None
            else None
        )

        self.guidance_in = None
        self.time_r_in = None

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    use_prope=use_prope,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        assert mm_single_blocks_depth == 0, "No single block in HunyuanVideo 1.5 architecture"

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        if attn_param is None:
            self.attn_param = {
                "win_size": [[3, 3, 3]],
                "win_type": "fixed",
                "win_ratio": 10,
                "tile_size": [6, 8, 8],
                "ssta_topk": 64,
                "ssta_threshold": 0.0,
                "ssta_lambda": 0.7,
                "ssta_sampling_type": "importance",
                "ssta_adaptive_pool": None,
                "attn_sparse_type": "ssta",
                "attn_pad_type": "zero",
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param

        if attn_mode == "flex-block-attn":
            self.register_to_config(attn_param=self.attn_param)

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert (
                vision_projection is not None
            ), "text type embedding is only used when vision_projection is not None"
        else:
            self.cond_type_embedding = None

        self.gradient_checkpointing = False

    def enable_deterministic(self):
        """Enable deterministic attention in all double-stream blocks."""
        for block in self.double_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        """Disable deterministic attention in all double-stream blocks."""
        for block in self.double_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        """Build n-D rotary cos/sin tables for the given grid sizes.

        Args:
            rope_sizes (tuple[int, int, int]): grid sizes ``(t, h, w)`` in patches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(freqs_cos, freqs_sin)``.
        """
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(
        self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True
    ):
        """Interleave ByT5/text tokens so valid tokens precede padding.

        Args:
            byt5_txt (torch.Tensor): ByT5 token embeddings ``[B, L1, C]``.
            txt (torch.Tensor): text token embeddings ``[B, L2, C]``.
            byt5_text_mask (torch.Tensor): ByT5 token mask ``[B, L1]``.
            text_mask (torch.Tensor): text token mask ``[B, L2]``.
            zero_feat (bool): zero out padding tokens to reduce block-mask error.
            is_reorder (bool): reorder per sample; if False, just concatenate.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: reordered tokens and int64 mask.
        """
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            pad_byt5,
                            pad_text,
                        ],
                        dim=0,
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            byt5_txt_i[~byt5_text_mask_i],
                            txt_i[~text_mask_i],
                        ],
                        dim=0,
                    )
                reorder_mask_i = torch.cat(
                    [
                        byt5_text_mask_i[byt5_text_mask_i],
                        text_mask_i[text_mask_i],
                        byt5_text_mask_i[~byt5_text_mask_i],
                        text_mask_i[~text_mask_i],
                    ],
                    dim=0,
                )

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask

    def add_action_parameters(self):
        """No-op hook; discrete-action parameters are added by a separate setup."""
        pass

    def get_text_and_mask(
        self,
        encoder_attention_mask,
        text_states,
        timestep_txt,
        extra_kwargs,
        vision_states,
        mask_type,
    ):
        """Embed text (and optional ByT5/vision) tokens and build their mask.

        Args:
            encoder_attention_mask (torch.Tensor): text attention mask ``[B, L]``.
            text_states (torch.Tensor): text-encoder states ``[B, L, C]``.
            timestep_txt (torch.Tensor): per-text timestep ``[B]``.
            extra_kwargs (dict): extra inputs (ByT5 states/mask when enabled).
            vision_states (torch.Tensor): vision-encoder states or None.
            mask_type (str): task type (e.g. ``"t2v"``).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(txt, text_mask, vec_txt)``.
        """
        text_mask = encoder_attention_mask
        txt = text_states
        bs = txt.shape[0]

        vec_txt = self.time_in(timestep_txt)

        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, timestep_txt, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb

        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = self.reorder_txt_token(
                byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
            )

        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2
                    * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

            txt, text_mask = self.reorder_txt_token(
                extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
            )
        return txt, text_mask, vec_txt

    def forward_txt(
        self,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: torch.Tensor | None = None,
        mask_type="t2v",
        extra_kwargs=None,
        kv_cache: dict | None = None,
        cache_txt: bool | None = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the text stream and optionally cache per-block text K/V.

        Args:
            timestep_txt (torch.Tensor): per-text timestep ``[B]``.
            text_states (torch.Tensor): text-encoder states ``[B, L, C]``.
            encoder_attention_mask (torch.Tensor): text attention mask ``[B, L]``.
            vision_states (torch.Tensor, optional): vision-encoder states.
            mask_type (str): task type (e.g. ``"t2v"``).
            extra_kwargs (dict, optional): extra inputs (ByT5 states/mask).
            kv_cache (dict, optional): KV cache to read from.
            cache_txt (bool, optional): build and return a fresh text KV cache.

        Returns:
            list[dict] or None: per-block text KV cache when ``cache_txt`` is True.
        """
        if cache_txt:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for _ in range(transformer_num_layers):
                _kv_cache_new.append(
                    {
                        "k_vision": None,
                        "v_vision": None,
                        "k_txt": None,
                        "v_txt": None,
                        "text_mask": None,
                    }
                )

        txt, text_mask, vec_txt = self.get_text_and_mask(
            encoder_attention_mask,
            text_states,
            timestep_txt,
            extra_kwargs,
            vision_states,
            mask_type,
        )

        txt, text_mask = pack_text_tokens(txt, text_mask)

        for index, block in enumerate(self.double_blocks):
            txt, t_kv = block(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                txt=txt,
                vec_txt=vec_txt,
                text_mask=text_mask,
                attn_param=None,
                is_flash=False,
                block_idx=index,
                kv_cache=kv_cache,
                cache_txt=cache_txt,
            )

            if cache_txt:
                _kv_cache_new[index]["k_txt"] = t_kv["k_txt"]
                _kv_cache_new[index]["v_txt"] = t_kv["v_txt"]
                _kv_cache_new[index]["text_mask"] = t_kv.get("text_mask")

        if cache_txt:
            return _kv_cache_new

    def forward_vision(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_r=None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        return_dict: bool = False,
        mask_type="t2v",
        extra_kwargs=None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        cache_vision: bool = False,
        rope_temporal_size=4,
        start_rope_start_idx=0,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the vision stream against cached text K/V (autoregressive decode).

        Args:
            hidden_states (torch.Tensor): video latents ``[B, C, T, H, W]``.
            timestep (torch.LongTensor): diffusion timestep.
            timestep_r (torch.Tensor, optional): meanflow timestep.
            freqs_cos (torch.Tensor, optional): precomputed rotary cos table.
            freqs_sin (torch.Tensor, optional): precomputed rotary sin table.
            return_dict (bool): must be False.
            mask_type (str): task type (e.g. ``"t2v"``).
            extra_kwargs (dict, optional): extra inputs.
            viewmats (torch.Tensor, optional): camera view matrices ``[B, T, 4, 4]``.
            Ks (torch.Tensor, optional): camera intrinsics ``[B, T, 4, 4]``.
            kv_cache (dict, optional): text/vision KV cache.
            cache_vision (bool): build and return a fresh vision KV cache.
            rope_temporal_size (int): temporal extent for the rotary table.
            start_rope_start_idx (int): starting latent index for the rotary slice.
            action (torch.Tensor, optional): discrete-action conditioning.

        Returns:
            tuple or list[dict]: ``(img, None)`` decode output, or a KV cache when
            ``cache_vision`` is True.
        """
        if cache_vision:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for i in range(transformer_num_layers):
                _kv_cache_new.append(
                    {
                        "k_vision": None,
                        "v_vision": None,
                        "k_txt": kv_cache[i]["k_txt"],
                        "v_txt": kv_cache[i]["v_txt"],
                        "text_mask": kv_cache[i].get("text_mask"),
                    }
                )

        img = x = hidden_states
        t = timestep
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param["thw"] = [tt, th, tw]
        rope_temporal_size = rope_temporal_size // self.patch_size[0]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((rope_temporal_size, th, tw))
            per_latent_size = th * tw
            start_index = start_rope_start_idx * per_latent_size
            end_index = (start_rope_start_idx + tt) * per_latent_size
            freqs_cos = freqs_cos[start_index:end_index, ...]
            freqs_sin = freqs_sin[start_index:end_index, ...]

        img = self.img_in(img)

        t = t.reshape(-1)
        vec = self.time_in(t)
        if action is not None and hasattr(self, "action_in"):
            vec = vec + self.action_in(action.reshape(-1))
        vec = repeat(vec, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)

        if viewmats is not None:
            viewmats = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
            Ks = repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (
                    sp_size - 1
                ), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
            vec = rearrange(vec, "B S C->(B S) C")
            if viewmats is not None:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
        else:
            vec = rearrange(vec, "B S C->(B S) C")

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        for index, block in enumerate(self.double_blocks):
            self.attn_param["layer-name"] = f"double_block_{index + 1}"

            img, vision_kv = block(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                img=img,
                vec=vec,
                freqs_cis=freqs_cis,
                attn_param=self.attn_param,
                block_idx=index,
                viewmats=viewmats,
                Ks=Ks,
                kv_cache=kv_cache,
                cache_vision=cache_vision,
            )
            if cache_vision:
                _kv_cache_new[index]["k_vision"] = vision_kv["k_vision"]
                _kv_cache_new[index]["v_vision"] = vision_kv["v_vision"]

        if cache_vision:
            return _kv_cache_new

        img = self.final_layer(img, vec)
        if sp_enabled:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        features_list = None
        return (img, features_list)

    def forward_bi(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor | None = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: dict[str, Any] | None = None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        clean_x: torch.Tensor | None = None,
        aug_timesteps: torch.Tensor | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Bidirectional forward, with optional teacher forcing and ProPE.

        Args:
            hidden_states (torch.Tensor): noisy video latents ``[B, C, T, H, W]``.
            timestep (torch.LongTensor): diffusion timestep for video tokens.
            timestep_txt (torch.Tensor): timestep for text tokens.
            text_states (torch.Tensor): text-encoder states ``[B, L, C]``.
            text_states_2 (torch.Tensor): pooled text states; must be None.
            encoder_attention_mask (torch.Tensor): text attention mask ``[B, L]``.
            timestep_r (torch.Tensor, optional): meanflow timestep; must be None.
            vision_states (torch.Tensor, optional): vision-encoder states.
            output_features (bool): must be False.
            output_features_stride (int): unused; kept for signature parity.
            attention_kwargs (dict, optional): unused; kept for signature parity.
            freqs_cos (torch.Tensor, optional): precomputed rotary cos table.
            freqs_sin (torch.Tensor, optional): precomputed rotary sin table.
            return_dict (bool): must be False.
            guidance (torch.Tensor, optional): guidance strength for distillation.
            mask_type (str): task type (e.g. ``"t2v"``).
            extra_kwargs (dict, optional): extra inputs (ByT5 states/mask).
            clean_x (torch.Tensor, optional): clean latents enabling teacher forcing.
            aug_timesteps (torch.Tensor, optional): timesteps for the clean half.
            viewmats (torch.Tensor, optional): camera view matrices ``[B, T, 4, 4]``.
            Ks (torch.Tensor, optional): camera intrinsics ``[B, T, 4, 4]``.
            action (torch.Tensor, optional): discrete-action conditioning.

        Returns:
            tuple[torch.Tensor, None]: ``(img, None)`` denoised latents.
        """
        is_tf = clean_x is not None

        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)

        img = x = hidden_states
        t = timestep
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param["thw"] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        img = self.img_in(img)
        vec = self.time_in(t)
        if action is not None and hasattr(self, "action_in"):
            vec = vec + self.action_in(action.reshape(-1))
        vec = repeat(vec, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)

        if viewmats is not None:
            viewmats = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
            Ks = repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)

        clean_img = None
        vec_clean = None
        if is_tf:
            clean_img = self.img_in(clean_x)
            vec_clean = self.time_in(aug_timesteps)
            vec_clean = repeat(vec_clean, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)

        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (
                    sp_size - 1
                ), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
            vec = rearrange(vec, "B S C->(B S) C")

            if viewmats is not None:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]

            if is_tf:
                clean_img = torch.chunk(clean_img, sp_size, dim=1)[sp_rank]
                vec_clean = torch.chunk(vec_clean, sp_size, dim=1)[sp_rank]
                vec_clean = rearrange(vec_clean, "B S C->(B S) C")
        else:
            vec = rearrange(vec, "B S C->(B S) C")
            if is_tf:
                vec_clean = rearrange(vec_clean, "B S C->(B S) C")

        if is_tf:
            img = torch.cat([clean_img, img], dim=1)

        assert text_states_2 is None, "text_states_2 handling is not implemented in forward_bi"
        assert (
            self.guidance_embed is False
        ), "guidance_embed handling is not implemented in forward_bi"
        assert timestep_r is None, "timestep_r handling is not implemented in forward_bi"

        txt, text_mask, vec_txt = self.get_text_and_mask(
            encoder_attention_mask,
            text_states,
            timestep_txt,
            extra_kwargs,
            vision_states,
            mask_type,
        )

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        txt, text_mask = pack_text_tokens(txt, text_mask)
        if is_tf and text_mask is not None and not text_mask.all().item():
            raise NotImplementedError(
                "teacher-forcing forward_bi with padded batched text masks is not "
                "supported by the flex_tf BlockMask path yet."
            )

        if is_tf:
            from minwm.modeling.hy15.layers.attention import prepare_teacher_forcing_mask

            frame_seqlen = th * tw
            num_frame_per_block = self.num_frame_per_block
            cache_key = (tt, th, tw, txt.shape[1])
            if not hasattr(self, "_tf_block_mask_cache"):
                from collections import OrderedDict

                self._tf_block_mask_cache = OrderedDict()
            cache = self._tf_block_mask_cache
            if cache_key in cache:
                cache.move_to_end(cache_key)
            else:
                cache[cache_key] = prepare_teacher_forcing_mask(
                    device=img.device,
                    num_frames=tt,
                    frame_seqlen=frame_seqlen,
                    text_seq_length=txt.shape[1],
                    num_frame_per_block=num_frame_per_block,
                )
                if len(cache) > 8:
                    cache.popitem(last=False)
            tf_block_mask = cache[cache_key]
        else:
            tf_block_mask = None

        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            img, txt = block(
                bi_inference=True,
                ar_txt_inference=False,
                ar_vision_inference=False,
                img=img,
                txt=txt,
                vec_txt=vec_txt,
                vec=vec,
                freqs_cis=freqs_cis,
                text_mask=text_mask,
                attn_param=self.attn_param,
                is_flash=force_full_attn,
                block_idx=index,
                vec_clean=vec_clean,
                tf_block_mask=tf_block_mask,
                viewmats=viewmats,
                Ks=Ks,
            )

        if is_tf:
            img = img[:, img.shape[1] // 2 :]

        img = self.final_layer(img, vec)
        if sp_enabled:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        assert output_features is False, "output_features is not supported in bi-inference mode"
        features_list = None
        return (img, features_list)

    def forward(
        self,
        bi_inference=True,
        ar_txt_inference=False,
        ar_vision_inference=False,
        **kwargs,
    ):
        """Dispatch to the bidirectional, text, or vision forward path.

        Args:
            bi_inference (bool): run the bidirectional path.
            ar_txt_inference (bool): run the text-only (KV-cache build) path.
            ar_vision_inference (bool): run the vision (KV-cache decode) path.
            **kwargs: forwarded to the selected path.

        Returns:
            The output of the dispatched forward method.

        Raises:
            NotImplementedError: if no inference flag is set.
        """
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            raise NotImplementedError

    def unpatchify(self, x, t, h, w):
        """Reassemble patch tokens into a video tensor.

        Args:
            x (torch.Tensor): patch tokens ``[N, T*H*W, patch**2 * C]``.
            t (int): temporal size in patch units.
            h (int): height in patch units.
            w (int): width in patch units.

        Returns:
            torch.Tensor: video tensor ``[N, C, t*pt, h*ph, w*pw]``.
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        """Set the attention mode on the model and all double-stream blocks.

        Args:
            attn_mode (str): attention mode identifier.
        """
        self.attn_mode = attn_mode
        for block in self.double_blocks:
            block.attn_mode = attn_mode
