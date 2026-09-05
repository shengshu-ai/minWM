"""Golden equivalence tests for the shared ``RMSNorm``.

Assert the unified :class:`minwm.modeling.common.norm.RMSNorm` (and the family
shims ``hy15.layers.norm.RMSNorm`` / ``wan.layers.norm.WanRMSNorm``) reproduce the
pre-extraction per-family implementations bit-for-bit.
"""

import torch

from minwm.modeling.common.norm import RMSNorm
from minwm.modeling.hy15.layers.norm import RMSNorm as HYRMSNorm
from minwm.modeling.wan21.layers.norm import WanRMSNorm


def _ref_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """The pre-extraction ``_norm`` math (identical across both families)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _ref_forward(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    out = _ref_norm(x.float(), eps).type_as(x)
    if weight is not None:
        out = out * weight
    return out


class TestRMSNormGolden:
    def test_matches_reference_affine(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8)
        norm = RMSNorm(8, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(8))
        expected = _ref_forward(x, norm.weight, 1e-6)
        assert torch.equal(norm(x), expected)

    def test_matches_reference_no_affine(self):
        torch.manual_seed(1)
        x = torch.randn(2, 5, 8)
        norm = RMSNorm(8, eps=1e-6, elementwise_affine=False)
        assert not hasattr(norm, "weight")
        expected = _ref_forward(x, None, 1e-6)
        assert torch.equal(norm(x), expected)

    def test_hy_shim_is_shared_class(self):
        assert HYRMSNorm is RMSNorm

    def test_hy_default_eps(self):
        # HY's factory default is rms w/ eps=1e-6.
        assert RMSNorm(8).eps == 1e-6

    def test_reset_parameters(self):
        norm = RMSNorm(4)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(4))
            norm.reset_parameters()
        assert torch.equal(norm.weight, torch.ones(4))


class TestWanRMSNormGolden:
    def test_preserves_eps_and_dim(self):
        norm = WanRMSNorm(8)
        assert norm.eps == 1e-5
        assert norm.dim == 8

    def test_matches_reference(self):
        torch.manual_seed(2)
        x = torch.randn(3, 7, 8)
        norm = WanRMSNorm(8)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(8))
        expected = _ref_forward(x, norm.weight, 1e-5)
        assert torch.equal(norm(x), expected)

    def test_dtype_preserved(self):
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        norm = WanRMSNorm(8).to(torch.bfloat16)
        assert norm(x).dtype == torch.bfloat16
