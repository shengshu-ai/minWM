"""Tests for the Wan ``use_prope`` camera-conditioning construction directive."""

import torch

from minwm.modeling.wan21.causal import CausalWan21Model
from minwm.modeling.wan21.model import Wan21Model

_ARCH = dict(
    model_type="t2v",
    dim=64,
    ffn_dim=128,
    freq_dim=64,
    text_len=8,
    text_dim=64,
    num_heads=4,
    num_layers=2,
    in_dim=16,
    out_dim=16,
)


def _prope_weights(model):
    return [p for n, p in model.named_parameters() if n.endswith("prope_o.weight")]


class TestUsePropeConstruction:
    def test_disabled_by_default(self):
        model = Wan21Model(**_ARCH)
        assert _prope_weights(model) == []

    def test_enabled_builds_one_per_block(self):
        model = Wan21Model(**_ARCH, use_prope=True)
        assert len(_prope_weights(model)) == _ARCH["num_layers"]

    def test_zero_initialised_after_init_weights(self):
        # init_weights runs a Xavier sweep over every Linear; prope_o must be
        # re-zeroed so the camera branch is a no-op until trained.
        model = Wan21Model(**_ARCH, use_prope=True)
        assert all(bool((w == 0).all()) for w in _prope_weights(model))

    def test_causal_variant_enabled(self):
        model = CausalWan21Model(**_ARCH, use_prope=True)
        assert len(_prope_weights(model)) == _ARCH["num_layers"]
        assert all(bool((w == 0).all()) for w in _prope_weights(model))


def _wan_inputs(num_frames=2, h=8, w=8):
    x = [torch.randn(16, num_frames, h, w)]
    context = [torch.randn(8, 64)]
    t = torch.tensor([0.0])
    seq_len = num_frames * (h // 2) * (w // 2)
    viewmats = torch.eye(4).view(1, 1, 4, 4).repeat(1, num_frames, 1, 1)
    Ks = torch.eye(3).view(1, 1, 3, 3).repeat(1, num_frames, 1, 1)
    return x, context, t, seq_len, viewmats, Ks


def _unzero_head(model):
    """Make the zero-init output head non-zero so forward output reflects internals.

    Wan21Model zero-inits ``head.head.weight`` (standard diffusion final-layer init),
    which makes a fresh model's output identically zero — masking any internal
    change. Perturbing it lets these tests observe the PRoPE branch's effect.
    """
    with torch.no_grad():
        model.head.head.weight.add_(0.1)


class TestPropeForward:
    def test_zero_init_camera_is_noop(self):
        """A freshly built use_prope model must match the no-camera forward.

        prope_o is zero-init, so feeding viewmats/Ks adds nothing — this is the
        invariant that lets a camera model load a plain base checkpoint without
        changing behaviour. (Head un-zeroed so the output isn't trivially zero.)
        """
        torch.manual_seed(0)
        model = Wan21Model(**_ARCH, use_prope=True).eval()
        _unzero_head(model)
        x, context, t, seq_len, viewmats, Ks = _wan_inputs()
        with torch.no_grad():
            out_no_cam = model(x=x, t=t, context=context, seq_len=seq_len)
            out_cam = model(x=x, t=t, context=context, seq_len=seq_len, viewmats=viewmats, Ks=Ks)
        assert float(out_no_cam.norm()) > 0  # head un-zeroed: output is non-trivial
        torch.testing.assert_close(out_no_cam, out_cam, atol=1e-5, rtol=1e-5)

    def test_trained_camera_changes_output(self):
        """Once prope_o is non-zero, the camera path must affect the output."""
        torch.manual_seed(0)
        model = Wan21Model(**_ARCH, use_prope=True).eval()
        _unzero_head(model)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "prope_o" in n:
                    p.add_(0.1)
        x, context, t, seq_len, _, _ = _wan_inputs()
        # non-identity camera (second frame translated) so PRoPE has a geometric effect
        viewmats = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1).clone()
        viewmats[0, 1, :3, 3] = torch.tensor([0.3, 0.1, 0.2])
        Ks = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1).clone() * 2
        with torch.no_grad():
            out_no_cam = model(x=x, t=t, context=context, seq_len=seq_len)
            out_cam = model(x=x, t=t, context=context, seq_len=seq_len, viewmats=viewmats, Ks=Ks)
        assert not torch.allclose(out_no_cam, out_cam, atol=1e-5)


class TestPropeCheckpointSymmetry:
    def test_state_dict_roundtrips_with_use_prope(self):
        """A use_prope checkpoint loads strictly only into a use_prope model.

        Rebuilding without use_prope is missing prope_o keys, so a strict load
        raises instead of silently dropping the camera params.
        """
        src = Wan21Model(**_ARCH, use_prope=True)
        with torch.no_grad():
            for n, p in src.named_parameters():
                if "prope_o" in n:
                    p.add_(1.0)
        sd = src.state_dict()

        ok = Wan21Model(**_ARCH, use_prope=True)
        ok.load_state_dict(sd, strict=True)
        assert all(bool((w == 1).all()) for w in _prope_weights(ok))

        import pytest

        with pytest.raises(RuntimeError):
            Wan21Model(**_ARCH).load_state_dict(sd, strict=True)
