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
"""Unified MM double-stream block for the HY15 DiT (inference / AR / ProPE)."""

import torch
import torch.nn as nn
from einops import rearrange

from minwm.modeling.common.prope import prope_qkv
from minwm.modeling.hy15.layers import (
    MLP,
    ModulateDiT,
    apply_gate,
    apply_rotary_emb,
    get_activation_layer,
    get_norm_layer,
    modulate,
)
from minwm.modeling.hy15.layers.attention import (
    parallel_attention,
    sequence_parallel_attention_txt,
    sequence_parallel_attention_vision,
)


def is_blocks(n: str, m) -> bool:
    """True if ``n`` names an indexed double/single-stream block submodule."""
    return ("double_blocks" in n and str.isdigit(n.split(".")[-1])) or (
        "single_blocks" in n and str.isdigit(n.split(".")[-1])
    )


class MMDoubleStreamBlock(nn.Module):
    """MM double-stream block with separate image/text streams.

    Supports the inference bidirectional path, the autoregressive text/vision
    paths, the super-resolution path, and an optional ProPE (projective
    positional encoding) branch gated by ``use_prope``.

    Args:
        hidden_size (int): hidden dimension size.
        heads_num (int): number of attention heads.
        mlp_width_ratio (float): expansion ratio for the MLP hidden size.
        mlp_act_type (str): activation function type for the MLP.
        attn_mode (str, optional): attention backend mode.
        qk_norm (bool): whether to use QK normalization.
        qk_norm_type (str): type of QK normalization.
        qkv_bias (bool): whether to use bias in the Q/K/V projections.
        use_prope (bool): create a zero-initialized ProPE output projection and
            enable the ProPE attention branch when view matrices are provided.
        dtype (torch.dtype, optional): optional torch dtype.
        device (torch.device, optional): optional torch device.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str | None = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        use_prope: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.use_prope = use_prope
        self.factory_kwargs = factory_kwargs

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        if use_prope:
            self.img_attn_prope_proj = nn.Linear(
                hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
            )
            nn.init.zeros_(self.img_attn_prope_proj.weight)
            if qkv_bias:
                nn.init.zeros_(self.img_attn_prope_proj.bias)

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        """Enable deterministic attention."""
        self.deterministic = True

    def disable_deterministic(self):
        """Disable deterministic attention."""
        self.deterministic = False

    def modulate_txt(self, vec_txt: torch.Tensor, txt: torch.Tensor) -> tuple:
        """Modulate and project text tokens for attention.

        Args:
            vec_txt (torch.Tensor): text modulation conditioning ``[B, C]``.
            txt (torch.Tensor): text tokens ``[B, L, C]``.

        Returns:
            tuple: ``(txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift,
            txt_mod2_scale, txt_mod2_gate)``.
        """
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        return (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        )

    def modulate_img(self, vec: torch.Tensor, img: torch.Tensor) -> tuple:
        """Modulate and project image tokens for attention.

        Args:
            vec (torch.Tensor): image modulation conditioning ``[B, C]``.
            img (torch.Tensor): image tokens ``[B, L, C]``.

        Returns:
            tuple: ``(img_q, img_k, img_v, img_mod1_gate, img_mod2_shift,
            img_mod2_scale, img_mod2_gate)``.
        """
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        return (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        )

    def forward_txt(
        self,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        attn_param=None,
        is_flash: bool = False,
        block_idx: int | None = None,
        kv_cache: dict | None = None,
        cache_txt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive text-only stream forward.

        Args:
            txt (torch.Tensor): text tokens ``[B, L, C]``.
            vec_txt (torch.Tensor): text modulation conditioning ``[B, C]``.
            text_mask (torch.Tensor, optional): text attention mask.
            attn_param: attention backend parameters.
            is_flash (bool): unused; kept for signature parity.
            block_idx (int, optional): block index for KV caching.
            kv_cache (dict, optional): KV cache store.
            cache_txt (bool): whether to cache text keys/values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(txt, t_kv)``.
        """
        (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.modulate_txt(vec_txt, txt)

        txt_attn, t_kv = sequence_parallel_attention_txt(
            txt_q,
            txt_k,
            txt_v,
            img_q_len=txt_q.shape[1],
            img_kv_len=txt_k.shape[1],
            text_mask=text_mask,
            attn_mode="torch_causal",
            attn_param=attn_param,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_txt=cache_txt,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        return txt, t_kv

    def forward_vision(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        attn_param=None,
        block_idx: int | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        cache_vision: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive vision-only stream forward.

        Args:
            img (torch.Tensor): image tokens ``[B, L, C]``.
            vec (torch.Tensor): image modulation conditioning ``[B, C]``.
            freqs_cis (tuple, optional): rotary frequency tensors.
            attn_param: attention backend parameters.
            block_idx (int, optional): block index for KV caching.
            viewmats (torch.Tensor, optional): camera view matrices for ProPE.
            Ks (torch.Tensor, optional): camera intrinsics for ProPE.
            kv_cache (dict, optional): KV cache store.
            cache_vision (bool): whether to cache vision keys/values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(img, vision_kv)``.
        """
        (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.modulate_img(vec, img)

        use_prope = self.use_prope and viewmats is not None
        if use_prope:
            img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
                img_q.permute(0, 2, 1, 3),
                img_k.permute(0, 2, 1, 3),
                img_v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            img_q_prope = img_q_prope.permute(0, 2, 1, 3)
            img_k_prope = img_k_prope.permute(0, 2, 1, 3)
            img_v_prope = img_v_prope.permute(0, 2, 1, 3)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_qq: {img_qq.shape}, img_q: {img_q.shape}, "
                f"img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            img_q, img_k = img_qq, img_kk

        img_attn, vision_kv = sequence_parallel_attention_vision(
            img_q,
            img_k,
            img_v,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_vision=cache_vision,
        )

        img_attn_proj = self.img_attn_proj(img_attn)
        if use_prope:
            img_attn_prope, _ = sequence_parallel_attention_vision(
                img_q_prope,
                img_k_prope,
                img_v_prope,
                block_idx=block_idx,
                kv_cache=kv_cache,
                cache_vision=False,
            )
            img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
            img_attn_prope = apply_fn_o(img_attn_prope)
            img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
            img_attn_proj = img_attn_proj + self.img_attn_prope_proj(img_attn_prope)

        img = img + apply_gate(img_attn_proj, gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        return img, vision_kv

    def forward_bi(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        text_mask: torch.Tensor | None = None,
        attn_param=None,
        is_flash: bool = False,
        block_idx: int | None = None,
        vec_clean: torch.Tensor | None = None,
        tf_block_mask=None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional (joint image+text) forward — training and bi-inference.

        Args:
            img (torch.Tensor): image tokens ``[B, L_img, C]``.
            txt (torch.Tensor): text tokens ``[B, L_txt, C]``.
            vec_txt (torch.Tensor): text modulation conditioning ``[B, C]``.
            vec (torch.Tensor): image modulation conditioning ``[B, C]``.
            freqs_cis (tuple, optional): rotary frequency tensors.
            text_mask (torch.Tensor, optional): text attention mask.
            attn_param: attention backend parameters.
            is_flash (bool): force flash attention mode.
            block_idx (int, optional): block index for KV caching.
            vec_clean (torch.Tensor, optional): clean-half modulation for teacher-forcing.
            tf_block_mask: teacher-forcing block mask for flex attention.
            viewmats (torch.Tensor, optional): camera view matrices for ProPE.
            Ks (torch.Tensor, optional): camera intrinsics for ProPE.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(img, txt)``.
        """
        is_tf = vec_clean is not None
        use_prope = self.use_prope and viewmats is not None

        (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.modulate_txt(vec_txt, txt)

        if is_tf:
            half = img.shape[1] // 2
            clean_img, noisy_img = img[:, :half], img[:, half:]

            c_mod1_shift, c_mod1_scale, c_mod1_gate, c_mod2_shift, c_mod2_scale, c_mod2_gate = (
                self.img_mod(vec_clean).chunk(6, dim=-1)
            )
            n_mod1_shift, n_mod1_scale, n_mod1_gate, n_mod2_shift, n_mod2_scale, n_mod2_gate = (
                self.img_mod(vec).chunk(6, dim=-1)
            )

            clean_modulated = modulate(
                self.img_norm1(clean_img), shift=c_mod1_shift, scale=c_mod1_scale
            )
            noisy_modulated = modulate(
                self.img_norm1(noisy_img), shift=n_mod1_shift, scale=n_mod1_scale
            )
            img_modulated = torch.cat([clean_modulated, noisy_modulated], dim=1)

            img_q = self.img_attn_q(img_modulated)
            img_k = self.img_attn_k(img_modulated)
            img_v = self.img_attn_v(img_modulated)
            img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
            img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
            img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
            img_q = self.img_attn_q_norm(img_q).to(img_v)
            img_k = self.img_attn_k_norm(img_k).to(img_v)
            img_q_org = img_q.clone()
            img_k_org = img_k.clone()
            img_v_org = img_v.clone()

            if freqs_cis is not None:
                img_q_c, img_q_n = img_q.chunk(2, dim=1)
                img_k_c, img_k_n = img_k.chunk(2, dim=1)
                img_qq_c, img_kk_c = apply_rotary_emb(img_q_c, img_k_c, freqs_cis, head_first=False)
                img_qq_n, img_kk_n = apply_rotary_emb(img_q_n, img_k_n, freqs_cis, head_first=False)
                img_q = torch.cat([img_qq_c, img_qq_n], dim=1)
                img_k = torch.cat([img_kk_c, img_kk_n], dim=1)

            attn = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                text_mask=text_mask,
                attn_mode="flex_tf",
                attn_param=attn_param,
                block_idx=block_idx,
                tf_block_mask=tf_block_mask,
            )
            img_attn = attn[:, : img_q.shape[1]].contiguous()
            txt_attn = attn[:, img_q.shape[1] :].contiguous()

            if use_prope:
                clean_q, clean_k, clean_v = (
                    img_q_org[:, :half],
                    img_k_org[:, :half],
                    img_v_org[:, :half],
                )
                noisy_q, noisy_k, noisy_v = (
                    img_q_org[:, half:],
                    img_k_org[:, half:],
                    img_v_org[:, half:],
                )

                clean_q_p, clean_k_p, clean_v_p, apply_fn_clean = prope_qkv(
                    clean_q.permute(0, 2, 1, 3),
                    clean_k.permute(0, 2, 1, 3),
                    clean_v.permute(0, 2, 1, 3),
                    viewmats=viewmats,
                    Ks=Ks,
                )
                clean_q_p = clean_q_p.permute(0, 2, 1, 3)
                clean_k_p = clean_k_p.permute(0, 2, 1, 3)
                clean_v_p = clean_v_p.permute(0, 2, 1, 3)

                noisy_q_p, noisy_k_p, noisy_v_p, apply_fn_noisy = prope_qkv(
                    noisy_q.permute(0, 2, 1, 3),
                    noisy_k.permute(0, 2, 1, 3),
                    noisy_v.permute(0, 2, 1, 3),
                    viewmats=viewmats,
                    Ks=Ks,
                )
                noisy_q_p = noisy_q_p.permute(0, 2, 1, 3)
                noisy_k_p = noisy_k_p.permute(0, 2, 1, 3)
                noisy_v_p = noisy_v_p.permute(0, 2, 1, 3)

                img_q_p = torch.cat([clean_q_p, noisy_q_p], dim=1)
                img_k_p = torch.cat([clean_k_p, noisy_k_p], dim=1)
                img_v_p = torch.cat([clean_v_p, noisy_v_p], dim=1)

                attn_p = parallel_attention(
                    (img_q_p, txt_q),
                    (img_k_p, txt_k),
                    (img_v_p, txt_v),
                    img_q_len=img_q_p.shape[1],
                    img_kv_len=img_k_p.shape[1],
                    text_mask=text_mask,
                    attn_mode="flex_tf",
                    attn_param=attn_param,
                    block_idx=block_idx,
                    tf_block_mask=tf_block_mask,
                )
                img_attn_p = attn_p[:, : img_q_p.shape[1]].contiguous()
                clean_ap, noisy_ap = img_attn_p[:, :half], img_attn_p[:, half:]
                clean_ap = rearrange(clean_ap, "B L (H D) -> B H L D", H=self.heads_num)
                clean_ap = apply_fn_clean(clean_ap)
                clean_ap = rearrange(clean_ap, "B H L D -> B L (H D)")
                noisy_ap = rearrange(noisy_ap, "B L (H D) -> B H L D", H=self.heads_num)
                noisy_ap = apply_fn_noisy(noisy_ap)
                noisy_ap = rearrange(noisy_ap, "B H L D -> B L (H D)")

                clean_attn, noisy_attn = img_attn[:, :half], img_attn[:, half:]
                clean_img = clean_img + apply_gate(
                    self.img_attn_proj(clean_attn) + self.img_attn_prope_proj(clean_ap),
                    gate=c_mod1_gate,
                )
                noisy_img = noisy_img + apply_gate(
                    self.img_attn_proj(noisy_attn) + self.img_attn_prope_proj(noisy_ap),
                    gate=n_mod1_gate,
                )
            else:
                clean_attn, noisy_attn = img_attn[:, :half], img_attn[:, half:]
                clean_img = clean_img + apply_gate(self.img_attn_proj(clean_attn), gate=c_mod1_gate)
                noisy_img = noisy_img + apply_gate(self.img_attn_proj(noisy_attn), gate=n_mod1_gate)

            clean_img = clean_img + apply_gate(
                self.img_mlp(
                    modulate(self.img_norm2(clean_img), shift=c_mod2_shift, scale=c_mod2_scale)
                ),
                gate=c_mod2_gate,
            )
            noisy_img = noisy_img + apply_gate(
                self.img_mlp(
                    modulate(self.img_norm2(noisy_img), shift=n_mod2_shift, scale=n_mod2_scale)
                ),
                gate=n_mod2_gate,
            )
            img = torch.cat([clean_img, noisy_img], dim=1)

        else:
            (
                img_q,
                img_k,
                img_v,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.modulate_img(vec, img)

            if use_prope:
                img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
                    img_q.permute(0, 2, 1, 3),
                    img_k.permute(0, 2, 1, 3),
                    img_v.permute(0, 2, 1, 3),
                    viewmats=viewmats,
                    Ks=Ks,
                )
                img_q_prope = img_q_prope.permute(0, 2, 1, 3)
                img_k_prope = img_k_prope.permute(0, 2, 1, 3)
                img_v_prope = img_v_prope.permute(0, 2, 1, 3)

            if freqs_cis is not None:
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                    f"img_qq: {img_qq.shape}, img_q: {img_q.shape}, "
                    f"img_kk: {img_kk.shape}, img_k: {img_k.shape}"
                )
                img_q, img_k = img_qq, img_kk

            attn_mode = "flash" if is_flash else self.attn_mode
            attn = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                text_mask=text_mask,
                attn_mode=attn_mode,
                attn_param=attn_param,
                block_idx=block_idx,
            )
            img_attn = attn[:, : img_q.shape[1]].contiguous()
            txt_attn = attn[:, img_q.shape[1] :].contiguous()

            img_attn_proj = self.img_attn_proj(img_attn)
            if use_prope:
                attn_prope = parallel_attention(
                    (img_q_prope, txt_q),
                    (img_k_prope, txt_k),
                    (img_v_prope, txt_v),
                    img_q_len=img_q_prope.shape[1],
                    img_kv_len=img_k_prope.shape[1],
                    text_mask=text_mask,
                    attn_mode=attn_mode,
                    attn_param=attn_param,
                    block_idx=block_idx,
                )
                img_attn_prope = attn_prope[:, : img_q_prope.shape[1]].contiguous()
                img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
                img_attn_prope = apply_fn_o(img_attn_prope)
                img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
                img_attn_proj = img_attn_proj + self.img_attn_prope_proj(img_attn_prope)

            img = img + apply_gate(img_attn_proj, gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
                ),
                gate=img_mod2_gate,
            )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        return img, txt

    def forward_sr(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        text_mask: torch.Tensor | None = None,
        attn_param=None,
        is_flash: bool = False,
        block_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Super-resolution bidirectional forward (no ProPE branch).

        Args:
            img (torch.Tensor): image tokens ``[B, L_img, C]``.
            txt (torch.Tensor): text tokens ``[B, L_txt, C]``.
            vec_txt (torch.Tensor): text modulation conditioning ``[B, C]``.
            vec (torch.Tensor): image modulation conditioning ``[B, C]``.
            freqs_cis (tuple, optional): rotary frequency tensors.
            text_mask (torch.Tensor, optional): text attention mask.
            attn_param: attention backend parameters.
            is_flash (bool): force flash attention mode.
            block_idx (int, optional): block index for KV caching.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(img, txt)``.
        """
        (
            img_q,
            img_k,
            img_v,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.modulate_img(vec, img)
        (
            txt_q,
            txt_k,
            txt_v,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.modulate_txt(vec_txt, txt)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_qq: {img_qq.shape}, img_q: {img_q.shape}, "
                f"img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            img_q, img_k = img_qq, img_kk

        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )

        img_attn = attn[:, : img_q.shape[1]].contiguous()
        txt_attn = attn[:, img_q.shape[1] :].contiguous()

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt

    def forward(
        self,
        bi_inference: bool = True,
        ar_txt_inference: bool = False,
        ar_vision_inference: bool = False,
        **kwargs,
    ):
        """Dispatch to the appropriate stream forward.

        Args:
            bi_inference (bool): run the bidirectional path.
            ar_txt_inference (bool): run the autoregressive text path.
            ar_vision_inference (bool): run the autoregressive vision path.
            **kwargs: arguments forwarded to the selected stream forward.

        Returns:
            The selected stream forward's output.
        """
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            return self.forward_sr(**kwargs)
