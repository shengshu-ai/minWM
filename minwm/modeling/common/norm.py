"""Normalization modules shared across model families.

``RMSNorm`` is the root-mean-square norm used by both the HY15 and Wan DiTs; the
core ``x * rsqrt(mean(x^2) + eps)`` math is identical between them. Family-level
shims (``minwm.modeling.hy15.layers.norm``, ``minwm.modeling.wan21.layers.norm``)
re-export or subclass this to preserve their own defaults.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Args:
        dim (int): size of the input tensor's last axis.
        eps (float): value added to the denominator for numerical stability.
        elementwise_affine (bool): if True, learn a per-channel scale.
        device: optional device for the weight parameter.
        dtype: optional dtype for the weight parameter.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def reset_parameters(self) -> None:
        if hasattr(self, "weight"):
            self.weight.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output
