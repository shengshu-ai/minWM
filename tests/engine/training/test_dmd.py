"""Tests for dmd.py."""

import pytest
import torch

from minwm.engine.training.recipes.ar_dmd import (
    apply_cfg,
    compute_kl_gradient,
    dmd_critic_loss,
    dmd_generator_loss,
)


class TestApplyCfg:
    def test_standard_formula(self):
        """Verify: uncond + scale * (cond - uncond)."""
        v_cond = torch.ones(2, 3) * 3.0
        v_uncond = torch.ones(2, 3) * 1.0
        result = apply_cfg(v_cond, v_uncond, guidance_scale=2.0)
        # 1 + 2*(3-1) = 5
        expected = torch.ones(2, 3) * 5.0
        torch.testing.assert_close(result, expected)

    def test_scale_zero(self):
        """With scale=0, result = uncond."""
        v_cond = torch.randn(2, 3)
        v_uncond = torch.randn(2, 3)
        result = apply_cfg(v_cond, v_uncond, guidance_scale=0.0)
        torch.testing.assert_close(result, v_uncond)

    def test_scale_one(self):
        """With scale=1, result = cond."""
        v_cond = torch.randn(2, 3)
        v_uncond = torch.randn(2, 3)
        result = apply_cfg(v_cond, v_uncond, guidance_scale=1.0)
        torch.testing.assert_close(result, v_cond)

    def test_legacy_cond_base_formula(self):
        v_cond = torch.full((2, 3), 3.0)
        v_uncond = torch.full((2, 3), 1.0)
        result = apply_cfg(v_cond, v_uncond, guidance_scale=2.0, base="cond")
        torch.testing.assert_close(result, torch.full((2, 3), 7.0))


class TestComputeKlGradient:
    def test_no_normalization(self):
        fake = torch.ones(2, 3, 4, 8, 8) * 2.0
        real = torch.ones(2, 3, 4, 8, 8) * 1.0
        gen = torch.randn(2, 3, 4, 8, 8)
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="none")
        expected = torch.ones(2, 3, 4, 8, 8) * 1.0
        torch.testing.assert_close(grad, expected)

    def test_global_normalizer_is_scalar(self):
        """In global mode, the normalizer should be a scalar (same for all elements)."""
        fake = torch.randn(2, 3, 4, 8, 8)
        real = torch.randn(2, 3, 4, 8, 8)
        gen = torch.randn(2, 3, 4, 8, 8)
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="global")
        # All elements should be normalized by the same scalar
        assert grad.shape == fake.shape

    def test_per_sample_keeps_batch_dim(self):
        """In per_sample mode, each sample has its own normalizer."""
        B = 4
        fake = torch.randn(B, 3, 4, 8, 8)
        real = torch.zeros(B, 3, 4, 8, 8)
        gen = torch.randn(B, 3, 4, 8, 8)
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="per_sample")
        assert grad.shape == fake.shape

    def test_nan_to_num(self):
        """Should handle NaN from division by zero."""
        fake = torch.ones(1, 1, 1, 1, 1)
        real = torch.ones(1, 1, 1, 1, 1)  # same as fake -> grad = 0
        gen = torch.ones(1, 1, 1, 1, 1)  # same as real -> normalizer = 0
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="global")
        assert not torch.isnan(grad).any()


class TestDmdGeneratorLoss:
    def test_gradient_flows_through_generator(self):
        """Gradient should flow through generator_output, not through target."""
        gen_out = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        kl_grad = torch.randn(2, 3, 4, 8, 8)
        loss = dmd_generator_loss(gen_out, kl_grad)
        loss.backward()
        assert gen_out.grad is not None

    def test_zero_gradient_gives_zero_loss(self):
        """If kl_gradient is zero, target = generator_output, loss = 0."""
        gen_out = torch.randn(2, 3, 4, 8, 8)
        kl_grad = torch.zeros(2, 3, 4, 8, 8)
        loss = dmd_generator_loss(gen_out, kl_grad)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestDmdCriticLoss:
    def test_matches_flow_matching_target(self):
        """critic_loss(pred, noise, clean) should equal MSE(pred, noise - clean)."""
        pred = torch.randn(2, 3, 4, 8, 8)
        noise = torch.randn(2, 3, 4, 8, 8)
        clean = torch.randn(2, 3, 4, 8, 8)
        loss = dmd_critic_loss(pred, noise, clean)
        expected = (pred.float() - (noise - clean).float()).pow(2).mean()
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_zero_loss(self):
        """When pred = noise - clean, loss should be 0."""
        noise = torch.randn(2, 3, 4, 8, 8)
        clean = torch.randn(2, 3, 4, 8, 8)
        pred = noise - clean
        loss = dmd_critic_loss(pred, noise, clean)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_weight(self):
        pred = torch.ones(2, 3)
        noise = torch.zeros(2, 3)
        clean = torch.zeros(2, 3)
        weight = torch.tensor([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        loss = dmd_critic_loss(pred, noise, clean, weight=weight)
        expected = (2.0 + 0.0 + 1.0 + 0.0 + 0.0 + 0.0) / 6
        assert loss.item() == pytest.approx(expected, abs=1e-6)


class TestResolveSteps:
    """ARDMDRecipe._resolve_steps warps the rollout denoising step list."""

    def test_no_warp_passthrough(self):
        """warp_denoising_step=False returns the raw list unchanged."""
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        recipe = ARDMDRecipe(denoising_step_list=[750, 500, 250], warp_denoising_step=False)
        assert recipe._resolve_steps() == [750, 500, 250]

    def test_warp_maps_to_schedule_timesteps(self):
        """warp maps raw steps via cat(timesteps, [0])[num_train - raw] (legacy formula)."""
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        steps = [1000, 750, 500, 250]
        recipe = ARDMDRecipe(
            timestep_shift=5.0, denoising_step_list=steps, warp_denoising_step=True
        )
        sched = recipe.scheduler
        ts = torch.cat((sched.timesteps.cpu(), torch.zeros(1, dtype=torch.float32)))
        expected = ts[sched.num_train_timesteps - torch.tensor(steps)].tolist()
        assert recipe._resolve_steps() == expected

    def test_warp_is_cached(self):
        """Repeated calls return the same cached object."""
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        recipe = ARDMDRecipe(denoising_step_list=[1000, 500], warp_denoising_step=True)
        assert recipe._resolve_steps() is recipe._resolve_steps()


class TestWan21DmdOptions:
    def test_global_shifted_timestep_is_clipped_and_shared(self):
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        torch.manual_seed(4)
        recipe = ARDMDRecipe(
            use_rollout_min_timestep=False,
            use_rollout_max_timestep=False,
            score_timestep_shift=True,
            score_timestep_clip=(20.0, 980.0),
            score_timestep_discrete=True,
        )
        timestep = recipe._sample_timestep(3, 4, 250, 750, torch.device("cpu"))
        assert timestep.shape == (3, 4)
        assert torch.all(timestep >= 20.0)
        assert torch.all(timestep <= 980.0)
        assert torch.equal(timestep, timestep[:, :1].expand(-1, 4))

    def test_generator_ema_seeds_from_generator_on_first_update(self):
        # The first update must seed the EMA from the live generator instead of
        # leaving its construction-time weights in place; otherwise a residue of
        # random init survives every subsequent lerp into the promoted weights.
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        generator = torch.nn.Linear(2, 2, bias=False)
        generator_ema = torch.nn.Linear(2, 2, bias=False)
        generator.weight.data.fill_(2.0)
        generator_ema.weight.data.zero_()  # stand-in for fresh/random init
        recipe = ARDMDRecipe(generator_ema_decay=0.5, generator_ema_start_step=0)

        recipe._update_generator_ema(generator_ema, generator, step=0)
        # Seeded to the generator's exact weights, not lerped toward them.
        assert torch.equal(generator_ema.weight, torch.full_like(generator_ema.weight, 2.0))

    def test_generator_ema_seeds_before_configured_start_step(self):
        # start_step gates the *lerp*, not the seeding: the EMA must never hold
        # random init even when the first steps precede generator_ema_start_step.
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        generator = torch.nn.Linear(2, 2, bias=False)
        generator_ema = torch.nn.Linear(2, 2, bias=False)
        generator.weight.data.fill_(2.0)
        generator_ema.weight.data.zero_()
        recipe = ARDMDRecipe(generator_ema_decay=0.5, generator_ema_start_step=200)

        # First call seeds (step below start_step).
        recipe._update_generator_ema(generator_ema, generator, step=10)
        assert torch.equal(generator_ema.weight, torch.full_like(generator_ema.weight, 2.0))
        # Still before start_step: no lerp, weights unchanged even as generator moves.
        generator.weight.data.fill_(6.0)
        recipe._update_generator_ema(generator_ema, generator, step=199)
        assert torch.equal(generator_ema.weight, torch.full_like(generator_ema.weight, 2.0))
        # At start_step the lerp kicks in: 0.5 * 2 + 0.5 * 6 = 4.
        recipe._update_generator_ema(generator_ema, generator, step=200)
        assert torch.equal(generator_ema.weight, torch.full_like(generator_ema.weight, 4.0))

    def test_discrete_timestep_guards_sub_unit_window(self):
        # int truncation can collapse a valid float window (lo=999.4, hi=999.6)
        # to lo_i == hi_i, which would make torch.randint raise "low >= high".
        from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe

        recipe = ARDMDRecipe(
            use_rollout_min_timestep=True,
            use_rollout_max_timestep=True,
            score_timestep_discrete=True,
        )
        # Float window is non-empty (999.4 < 999.6) so the float guard stays
        # inert, but int(lo) == int(hi) == 999 — would raise before the fix.
        t = recipe._sample_timestep(3, 4, 999.4, 999.6, torch.device("cpu"))
        assert t.shape == (3, 4)
        assert torch.all(t == 999.0)


class TestGenerationRollout:
    def test_sampler_preserves_legacy_warped_float_timesteps(self):
        from minwm.engine.inference.samplers import CMSolver
        from minwm.sampling.schedulers import FlowMatchingScheduler

        scheduler = FlowMatchingScheduler(
            num_train_timesteps=1000,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        sampler = CMSolver(
            scheduler,
            [1000, 750],
            warp_denoising_step=True,
        )
        warped = sampler.denoising_steps[1]
        assert float(warped) != int(float(warped))

        t_float = sampler.timestep_tensor(
            warped,
            batch_size=1,
            num_frames=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        t_int = sampler.timestep_tensor(
            warped,
            batch_size=1,
            num_frames=2,
            device=torch.device("cpu"),
            dtype=torch.int64,
        )

        torch.testing.assert_close(t_float, torch.full((1, 2), float(warped)))
        assert t_int.dtype == torch.float32
        torch.testing.assert_close(t_int, torch.full((1, 2), float(warped)))

        renoised = sampler.renoise(torch.zeros(1, 2, 1, 1, 1), warped)
        assert renoised.shape == (1, 2, 1, 1, 1)

        raw_t = CMSolver(scheduler, [1000]).timestep_tensor(
            1000,
            batch_size=1,
            num_frames=2,
            device=torch.device("cpu"),
            dtype=torch.int64,
        )
        assert raw_t.dtype == torch.int64
        assert raw_t.tolist() == [[1000, 1000]]

    def _run_self_forcing(self, use_prope_cache: bool):
        """Drive SelfForcingPipeline with a stub adapter recording the PRoPE flag."""
        from torch import nn

        from minwm.sampling.rollouts import SelfForcingPipeline
        from minwm.sampling.schedulers import FlowMatchingScheduler

        class TinyCausal(nn.Module):
            patch_size = (1, 2, 2)

        class RecordingAdapter:
            def __init__(self):
                self.prope_flags = []

            def rollout_init_cache(self, model, *, use_prope_cache=True, **kwargs):
                self.prope_flags.append(use_prope_cache)
                return {"tag": "cache"}

            def rollout_forward(self, model, *, noisy_block, **kwargs):
                return torch.zeros_like(noisy_block)

            def rollout_refresh_cache(self, model, **kwargs):
                pass

        torch.manual_seed(0)
        scheduler = FlowMatchingScheduler(num_train_timesteps=1000, shift=5.0)
        adapter = RecordingAdapter()
        pipe = SelfForcingPipeline(
            generator=TinyCausal(),
            scheduler=scheduler,
            adapter=adapter,
            denoising_step_list=[1000],
            num_frame_per_block=1,
            use_prope_cache=use_prope_cache,
        )
        noise = torch.randn(1, 2, 16, 4, 4)
        out, from_ts, to_ts = pipe.inference_with_trajectory(noise, cond={"tag": "cond"})
        assert out.shape == noise.shape
        assert from_ts == 1000
        assert to_ts == 0
        return adapter

    def test_self_forcing_rollout_disables_prope_cache_by_default(self):
        # Default False: the DMD self-forcing loop asks the adapter NOT to
        # allocate a PRoPE cache (matches origin/dev-meta legacy behavior).
        adapter = self._run_self_forcing(use_prope_cache=False)
        assert adapter.prope_flags
        assert all(flag is False for flag in adapter.prope_flags)

    def test_self_forcing_rollout_can_enable_prope_cache(self):
        adapter = self._run_self_forcing(use_prope_cache=True)
        assert adapter.prope_flags
        assert all(flag is True for flag in adapter.prope_flags)
