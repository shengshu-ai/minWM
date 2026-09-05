# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""RMS and Layer normalisation modules.

``WanRMSNorm`` is a thin subclass of the shared
:class:`minwm.modeling.common.norm.RMSNorm` that keeps the Wan default
``eps=1e-5`` and the ``self.dim`` attribute; the normalisation math is identical.
"""

import torch
import torch.nn as nn

from minwm.modeling.common.norm import RMSNorm

__all__ = ["WanRMSNorm", "WanLayerNorm"]


class WanRMSNorm(RMSNorm):
    """RMS normalisation with a learnable per-channel scale.

    Args:
        dim (int): number of channels.
        eps (float): numerical stability epsilon.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__(dim, eps=eps, elementwise_affine=True)
        self.dim = dim


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm that preserves the input dtype in its output.

    Args:
        dim (int): normalised dimension.
        eps (float): numerical stability epsilon.
        elementwise_affine (bool): learnable affine parameters.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).type_as(x)
