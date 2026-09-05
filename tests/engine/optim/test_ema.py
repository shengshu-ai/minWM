"""Tests for ema.py."""

import pytest
import torch
import torch.nn as nn

from minwm.engine.optim.ema import copy_params, update_ema_params


class TestUpdateEmaParams:
    def test_basic_decay(self):
        ema = [torch.tensor([10.0])]
        model = [torch.tensor([0.0])]
        update_ema_params(ema, model, decay=0.9)
        expected = 0.9 * 10.0 + 0.1 * 0.0
        assert ema[0].item() == pytest.approx(expected)

    def test_decay_one_no_change(self):
        ema = [torch.tensor([5.0])]
        model = [torch.tensor([100.0])]
        update_ema_params(ema, model, decay=1.0)
        assert ema[0].item() == pytest.approx(5.0)

    def test_decay_zero_copies_model(self):
        ema = [torch.tensor([5.0])]
        model = [torch.tensor([100.0])]
        update_ema_params(ema, model, decay=0.0)
        assert ema[0].item() == pytest.approx(100.0)

    def test_multiple_params(self):
        ema = [torch.ones(3), torch.zeros(2)]
        model = [torch.zeros(3), torch.ones(2)]
        update_ema_params(ema, model, decay=0.5)
        torch.testing.assert_close(ema[0], torch.full((3,), 0.5))
        torch.testing.assert_close(ema[1], torch.full((2,), 0.5))

    def test_no_grad_on_model(self):
        """update_ema_params should not require grad on model params."""
        ema = [torch.tensor([1.0])]
        model = [torch.tensor([2.0], requires_grad=True)]
        update_ema_params(ema, model, decay=0.9)
        assert not ema[0].requires_grad


class TestCopyParams:
    def test_basic_copy(self):
        src = nn.Linear(4, 3, bias=False)
        dst = nn.Linear(4, 3, bias=False)
        nn.init.ones_(src.weight)
        nn.init.zeros_(dst.weight)
        copy_params(src, dst)
        torch.testing.assert_close(dst.weight.data, src.weight.data)

    def test_multi_layer(self):
        src = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        dst = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        copy_params(src, dst)
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            torch.testing.assert_close(p_dst, p_src)
