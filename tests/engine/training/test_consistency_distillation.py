"""Tests for consistency_distillation.py."""

import pytest
import torch

from minwm.engine.training.recipes.ar_cd import (
    consistency_loss,
    sample_cd_timestep_pair,
    teacher_cfg_euler_step,
)


class TestSampleCdTimestepPair:
    def test_returns_four_tensors(self):
        sigmas = torch.linspace(1.0, 0.0, 11)  # 11 entries
        timesteps = torch.linspace(1000, 0, 10)  # 10 entries
        s_t, s_tn, t, tn = sample_cd_timestep_pair(10, sigmas, timesteps, torch.device("cpu"))
        assert s_t.dim() == 0 or s_t.numel() == 1
        assert s_tn.dim() == 0 or s_tn.numel() == 1

    def test_t_greater_than_t_next(self):
        sigmas = torch.linspace(1.0, 0.0, 11)
        timesteps = torch.linspace(1000, 0, 10)
        for _ in range(50):
            _, _, t, tn = sample_cd_timestep_pair(10, sigmas, timesteps, torch.device("cpu"))
            assert t >= tn


class TestTeacherCfgEulerStep:
    def test_no_cfg(self):
        """With guidance_scale=1, v_cfg = v_cond."""
        v_cond = torch.ones(1, 2, 3, 4, 4)
        v_uncond = torch.zeros(1, 2, 3, 4, 4)
        latent = torch.zeros(1, 2, 3, 4, 4)
        t = torch.tensor([[100.0, 100.0]])
        t_next = torch.tensor([[50.0, 50.0]])
        result = teacher_cfg_euler_step(
            v_cond,
            v_uncond,
            latent,
            t,
            t_next,
            guidance_scale=1.0,
            timestep_scale=1000.0,
        )
        # dt = 50/1000 = 0.05, latent_next = 0 - 0.05 * 1 = -0.05
        expected = torch.full_like(latent, -0.05)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_cfg_scale(self):
        """CFG should amplify the difference."""
        v_cond = torch.ones(1, 1, 1, 1, 1) * 2.0
        v_uncond = torch.ones(1, 1, 1, 1, 1) * 1.0
        latent = torch.zeros(1, 1, 1, 1, 1)
        t = torch.tensor([[1000.0]])
        t_next = torch.tensor([[0.0]])
        # v_cfg = 1 + 3*(2-1) = 4
        # dt = 1000/1000 = 1
        # result = 0 - 1*4 = -4
        result = teacher_cfg_euler_step(
            v_cond,
            v_uncond,
            latent,
            t,
            t_next,
            guidance_scale=3.0,
            timestep_scale=1000.0,
        )
        assert result.item() == pytest.approx(-4.0, abs=1e-5)

    def test_timestep_scale_1(self):
        """HY15 uses timestep_scale=1."""
        v_cond = torch.ones(1, 1, 1, 1, 1)
        v_uncond = torch.zeros(1, 1, 1, 1, 1)
        latent = torch.zeros(1, 1, 1, 1, 1)
        t = torch.tensor([[0.5]])
        t_next = torch.tensor([[0.3]])
        result = teacher_cfg_euler_step(
            v_cond,
            v_uncond,
            latent,
            t,
            t_next,
            guidance_scale=1.0,
            timestep_scale=1.0,
        )
        # dt = 0.2, result = 0 - 0.2*1 = -0.2
        assert result.item() == pytest.approx(-0.2, abs=1e-5)


class TestConsistencyLoss:
    def test_zero_loss(self):
        x = torch.randn(2, 3, 4, 8, 8)
        loss = consistency_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_loss(self):
        a = torch.ones(2, 3, 4, 8, 8)
        b = torch.zeros(2, 3, 4, 8, 8)
        loss = consistency_loss(a, b)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_reduction_none(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        loss = consistency_loss(a, b, reduction="none")
        assert loss.shape == (2, 3)
