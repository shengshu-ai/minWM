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
"""DiT modulation primitives: ModulateDiT layer plus shift/scale/gate helpers."""

from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange


class ModulateDiT(nn.Module):
    """Zero-initialized modulation projection for DiT.

    Args:
        hidden_size (int): model hidden dimension.
        factor (int): number of modulation outputs per channel.
        act_layer (Callable): no-arg callable building the activation module.
        dtype: optional dtype for the linear layer.
        device: optional device for the linear layer.
    """

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


def modulate(x, shift=None, scale=None):
    """Apply shift/scale modulation along the token axis.

    Args:
        x (torch.Tensor): input tensor, shape ``[B, L, C]``.
        shift (torch.Tensor, optional): additive shift.
        scale (torch.Tensor, optional): multiplicative scale.

    Returns:
        torch.Tensor: modulated tensor, same shape as ``x``.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        scale = scale.unsqueeze(0)
        scale = rearrange(scale, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = scale.shape[1]  # latent length
        token_length = x.shape[1] // latent_length
        # operate on the hidden_states
        scale = scale.repeat_interleave(token_length, dim=1).type_as(x)
        return x * (1 + scale)
    elif scale is None:
        shift = shift.unsqueeze(0)
        shift = rearrange(shift, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = shift.shape[1]  # latent length
        token_length = x.shape[1] // latent_length
        # operate on the hidden_states
        shift = shift.repeat_interleave(token_length, dim=1).type_as(x)
        return x + shift
    else:
        shift = shift.unsqueeze(0)
        scale = scale.unsqueeze(0)
        shift = rearrange(shift, "B (N T) C -> (B N) T C", N=x.shape[0])
        scale = rearrange(scale, "B (N T) C -> (B N) T C", N=x.shape[0])

        latent_length = shift.shape[1]  # latent length
        token_length = x.shape[1] // latent_length

        scale = scale.repeat_interleave(token_length, dim=1).type_as(x)
        shift = shift.repeat_interleave(token_length, dim=1).type_as(x)
        return x * (1 + scale) + shift


def apply_gate(x, gate=None, tanh=False):
    """Apply a per-token gate to ``x``.

    Args:
        x (torch.Tensor): input tensor, shape ``[B, L, C]``.
        gate (torch.Tensor, optional): gate tensor.
        tanh (bool): if True, squash the gate through ``tanh`` first.

    Returns:
        torch.Tensor: gated tensor, same shape as ``x``.
    """
    if gate is None:
        return x
    gate = gate.unsqueeze(0)
    gate = rearrange(gate, "B (N T) C -> (B N) T C", N=x.shape[0])
    latent_length = gate.shape[1]  # latent length
    token_length = x.shape[1] // latent_length
    gate = gate.repeat_interleave(token_length, dim=1).type_as(x)
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward
