"""HY15 numerical consistency tests.

Verify that shared algorithm functions produce identical results
to the original inline HY15 code.
"""

import pytest
import torch
import torch.nn.functional as F

from minwm.engine.training.recipes.ar_cd import (
    consistency_loss,
    teacher_cfg_euler_step,
)
from minwm.engine.training.recipes.ar_dmd import (
    apply_cfg,
    compute_kl_gradient,
    dmd_critic_loss,
    dmd_generator_loss,
)
from minwm.sampling.schedulers import FlowMatchingScheduler, pred_x0_from_flow


def _rand(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------


class TestFlowMatchingHY15:
    """Verify shared flow matching ops match HY15 inline formulas."""

    def test_training_target_matches_hy15(self):
        """HY15 inline: target = noise - clean"""
        sched = FlowMatchingScheduler(shift=3.0, schedule="linear")
        clean = _rand(2, 16, 5, 8, 8)
        noise = _rand(2, 16, 5, 8, 8)
        assert torch.equal(sched.training_target(clean, noise), noise - clean)

    def test_pred_x0_from_flow_matches_hy15(self):
        """HY15 inline: x0 = sample - model_pred * sigmas"""
        sample = _rand(2, 16, 5, 8, 8)
        model_pred = _rand(2, 16, 5, 8, 8)
        sigma = torch.tensor(0.3)
        expected = sample - model_pred * sigma
        actual = pred_x0_from_flow(sample, model_pred, sigma, compute_dtype=sample.dtype)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_add_noise_matches_hy15(self):
        """HY15 inline: noisy = (1 - sigma) * clean + sigma * noise"""
        sched = FlowMatchingScheduler(shift=3.0, schedule="linear")
        clean = _rand(32, 5, 8, 8)
        noise = _rand(32, 5, 8, 8)
        ts = torch.full((32,), 500.0)
        sigma = sched.sigma_for(ts).reshape(-1, 1, 1, 1)
        expected = (1 - sigma) * clean + sigma * noise
        actual = sched.add_noise(clean, noise, ts)
        assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Consistency distillation
# ---------------------------------------------------------------------------


class TestConsistencyDistillationHY15:
    """Verify shared CD ops match HY15 inline formulas."""

    def test_teacher_cfg_euler_step_hy15_sigma_mode(self):
        """HY15 uses timestep_scale=1.0 (sigma-based).

        HY15 inline: x + v_cfg * (sigma_next - sigma)
        Shared: latent_t - dt * v_cfg where dt = (t - t_next) / scale
        """
        B, C, T, H, W = 2, 16, 5, 8, 8
        v_cond = _rand(B, C, T, H, W)
        v_uncond = _rand(B, C, T, H, W)
        latent_t = _rand(B, C, T, H, W)
        sigma = torch.tensor([[0.8], [0.8]])  # [B, 1]
        sigma_next = torch.tensor([[0.4], [0.4]])  # [B, 1]
        cfg_scale = 4.5
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        expected = latent_t + v_cfg * (sigma_next - sigma).view(B, 1, 1, 1, 1)
        actual = teacher_cfg_euler_step(
            v_cond,
            v_uncond,
            latent_t,
            sigma,
            sigma_next,
            cfg_scale,
            timestep_scale=1.0,
        )
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_consistency_loss_matches_mse(self):
        """HY15 inline: F.mse_loss(a.float(), b.float())"""
        a = _rand(2, 16, 5, 8, 8)
        b = _rand(2, 16, 5, 8, 8)
        expected = F.mse_loss(a.float(), b.float())
        actual = consistency_loss(a, b)
        assert torch.allclose(actual, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# DMD
# ---------------------------------------------------------------------------


class TestDMDHY15:
    """Verify shared DMD ops match HY15 inline formulas."""

    def test_apply_cfg_matches_hy15(self):
        """HY15 inline: uncond + scale * (cond - uncond)"""
        cond = _rand(2, 16, 5, 8, 8)
        uncond = _rand(2, 16, 5, 8, 8)
        scale = 7.5
        expected = uncond + scale * (cond - uncond)
        actual = apply_cfg(cond, uncond, scale)
        assert torch.equal(actual, expected)

    def test_compute_kl_gradient_global_matches_hy15(self):
        """HY15 inline: grad = (fake-real) / |gen-real|.mean(); nan_to_num"""
        fake = _rand(2, 16, 5, 8, 8)
        real = _rand(2, 16, 5, 8, 8)
        gen = _rand(2, 16, 5, 8, 8)
        expected = (fake - real) / torch.abs(gen - real).mean()
        expected = torch.nan_to_num(expected)
        actual = compute_kl_gradient(fake, real, gen, normalize_mode="global")
        # Allow small difference from 1e-8 clamp in shared function
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_dmd_generator_loss_matches_hy15(self):
        """HY15 inline: 0.5 * mse(x.float(), (x.float()-grad.float()).detach())"""
        x = _rand(2, 16, 5, 8, 8)
        grad = _rand(2, 16, 5, 8, 8)
        expected = 0.5 * F.mse_loss(x.float(), (x.float() - grad.float()).detach())
        actual = dmd_generator_loss(generator_output=x, kl_gradient=grad)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_dmd_critic_loss_matches_hy15(self):
        """HY15 inline: mse(pred, noise - orig)"""
        pred = _rand(4, 16, 8, 8)
        noise = _rand(4, 16, 8, 8)
        orig = _rand(4, 16, 8, 8)
        target = noise - orig
        expected = F.mse_loss(pred, target)
        actual = dmd_critic_loss(critic_flow_pred=pred, noise=noise, clean=orig)
        assert torch.allclose(actual, expected, atol=1e-6)
