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
"""Patch / timestep / text / vision embedding layers for the HY15 DiT."""

import collections.abc
from itertools import repeat

import torch
import torch.nn as nn

from minwm.modeling.common.embedding import sinusoidal_embedding


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """3D patch embedding via a strided Conv3d.

    Supports a concat-condition mode (multitask mask training) that widens the
    input channels and zero-initializes the extra slice.

    Args:
        patch_size (int | tuple): conv kernel/stride.
        in_chans (int): base input channels (before concat-condition widening).
        embed_dim (int): output embedding dimension.
        is_reshape_temporal_channels (bool): channel-widening variant selector.
        concat_condition (bool): if True, widen channels for concat conditioning.
        norm_layer (Callable, optional): norm applied after projection.
        flatten (bool): flatten spatial dims to a token sequence.
        bias (bool): add a conv bias.
        dtype: optional dtype.
        device: optional device.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        is_reshape_temporal_channels=False,
        concat_condition=True,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        # Only support concat mode (multitask mask training)
        orig_in_chans = in_chans
        if concat_condition:
            if is_reshape_temporal_channels:
                in_chans = in_chans + in_chans // 2 + 1
            else:
                in_chans = in_chans * 2 + 1

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )

        nn.init.xavier_uniform_(
            self.proj.weight[:, :orig_in_chans].view(
                self.proj.weight[:, :orig_in_chans].size(0), -1
            )
        )
        # Special initialization for concat mode
        nn.init.zeros_(
            self.proj.weight[:, orig_in_chans:].view(
                self.proj.weight[:, orig_in_chans:].size(0), -1
            )
        )

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """Two-layer MLP projecting text embeddings into the model space.

    Args:
        in_channels (int): input embedding dimension.
        hidden_size (int): output / hidden dimension.
        act_layer (Callable): no-arg activation constructor.
        dtype: optional dtype.
        device: optional device.
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VisionProjection(torch.nn.Module):
    """LayerNorm-GELU-LayerNorm projection for vision embeddings."""

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(),
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(self, vision_embeds):
        return self.proj(vision_embeds)


class ClipVisionProjection(nn.Module):
    """Zero-initialized up/down projection for CLIP vision features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, x):
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Thin wrapper over :func:`minwm.modeling.common.embedding.sinusoidal_embedding` preserving
    the HY convention (fp32 compute, odd dims zero-padded).

    Args:
        t (torch.Tensor): 1-D tensor of N indices, one per batch element (may be fractional).
        dim (int): output embedding dimension.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        torch.Tensor: ``[N, dim]`` positional embeddings.
    """
    return sinusoidal_embedding(
        t, dim, max_period=max_period, compute_dtype=torch.float32, pad_odd=True
    )


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Args:
        hidden_size (int): hidden dimension of the embedding MLP.
        act_layer (Callable): no-arg activation constructor.
        frequency_embedding_size (int): sinusoidal frequency dimension.
        max_period (int): controls the minimum embedding frequency.
        out_size (int, optional): output dimension; defaults to ``hidden_size``.
        dtype: optional dtype.
        device: optional device.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb
