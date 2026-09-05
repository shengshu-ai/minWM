"""Tests for Wan21Adapter text-encoder injection (real encoder vs zero fallback)."""

import pytest
import torch

from minwm.modeling.wan21.adapter import Wan21Adapter


class _StubEncoder(torch.nn.Module):
    """Returns a per-prompt ``[len(prompt), dim]`` ones tensor, like the umt5 wrapper."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, prompts: list[str]) -> list[torch.Tensor]:
        return [torch.ones(len(p), self.dim) for p in prompts]


class TestWan21AdapterEncodeText:
    def test_zero_fallback_without_encoder(self):
        adapter = Wan21Adapter(text_len=8, text_dim=64, dtype="float32")
        out = adapter.encode_text(["hi", "yo"], 2, torch.device("cpu"))
        assert len(out) == 2
        assert all(c.shape == (8, 64) for c in out)
        assert all(bool((c == 0).all()) for c in out)

    def test_zero_fallback_when_prompts_none(self):
        adapter = Wan21Adapter(
            text_len=8, text_dim=64, dtype="float32", text_encoder=_StubEncoder(64)
        )
        out = adapter.encode_text(None, 3, torch.device("cpu"))
        assert len(out) == 3
        assert all(c.shape == (8, 64) for c in out)

    def test_uses_injected_encoder(self):
        adapter = Wan21Adapter(
            text_len=8, text_dim=64, dtype="float32", text_encoder=_StubEncoder(64)
        )
        out = adapter.encode_text(["ab", "xyz"], 2, torch.device("cpu"))
        assert [tuple(c.shape) for c in out] == [(2, 64), (3, 64)]
        assert out[0].dtype == torch.float32

    def test_prompt_count_must_match_batch_size(self):
        adapter = Wan21Adapter(
            text_len=8, text_dim=64, dtype="float32", text_encoder=_StubEncoder(64)
        )
        with pytest.raises(ValueError):
            adapter.encode_text(["only-one"], 2, torch.device("cpu"))


class TestWan21AdapterNullConditioning:
    def test_zero_text_when_no_negative_prompt(self):
        # Training default: negative_prompt is None -> base zero-text uncond, so
        # the DMD / CD unconditional stream is unchanged.
        adapter = Wan21Adapter(
            text_len=8, text_dim=64, dtype="float32", text_encoder=_StubEncoder(64)
        )
        out = adapter.null_conditioning({}, 2, torch.device("cpu"))["context"]
        assert len(out) == 2
        assert all(c.shape == (8, 64) for c in out)
        assert all(bool((c == 0).all()) for c in out)

    def test_encodes_negative_prompt_when_set(self):
        # Inference: a real negative prompt is encoded like the cond branch.
        adapter = Wan21Adapter(
            text_len=8,
            text_dim=64,
            dtype="float32",
            text_encoder=_StubEncoder(64),
            negative_prompt="bad",
        )
        out = adapter.null_conditioning({}, 2, torch.device("cpu"))["context"]
        assert [tuple(c.shape) for c in out] == [(3, 64), (3, 64)]
        assert all(bool((c == 1).all()) for c in out)
