"""Tests for flow_matching.py."""

import pytest
import torch

from minwm.engine.training.losses import mse_loss
from minwm.sampling.schedulers import FlowMatchingScheduler, pred_x0_from_flow


class TestSchedulerAddNoise:
    def test_noisy_keeps_operand_dtype(self):
        """add_noise must compute in clean's dtype (bf16), not promote to fp32.

        A fp32 sigma would silently upcast the result and round differently from
        an all-bf16 compute, breaking bit-parity with bf16 training pipelines.
        """
        sched = FlowMatchingScheduler(shift=3.0, schedule="linear")
        clean = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
        noise = torch.randn_like(clean)
        ts = torch.tensor([807.0, 807.0])
        out = sched.add_noise(clean, noise, ts)
        assert out.dtype == torch.bfloat16
        sigma = sched.sigma_for(ts).reshape(-1, 1, 1, 1).to(torch.bfloat16)
        expected = (1 - sigma) * clean + sigma * noise
        assert torch.equal(out, expected)


class TestSampleTimesteps:
    """``uniform_across_frames`` must hold the same meaning in both sampling paths."""

    def _sched(self):
        return FlowMatchingScheduler(shift=3.0, schedule="linear")

    def test_uniform_shares_one_timestep_per_sample(self):
        for scheme in (None, "logit_normal"):
            g = torch.Generator(device="cpu").manual_seed(0)
            ts = self._sched().sample_timesteps(
                2,
                5,
                torch.device("cpu"),
                uniform_across_frames=True,
                weighting_scheme=scheme,
                generator=g,
            )
            assert ts.shape == (2, 5)
            for row in ts:
                assert (row == row[0]).all()

    def test_non_uniform_draws_per_frame(self):
        for scheme in (None, "logit_normal"):
            g = torch.Generator(device="cpu").manual_seed(0)
            ts = self._sched().sample_timesteps(
                2,
                5,
                torch.device("cpu"),
                uniform_across_frames=False,
                weighting_scheme=scheme,
                generator=g,
            )
            assert ts.shape == (2, 5)
        # per-frame draws should not be a single repeated value across all frames
        assert not bool((ts == ts[:, :1]).all())

    def test_uniform_weighting_draw_unchanged(self):
        """B draws for B samples when uniform — preserves the existing RNG stream."""
        g = torch.Generator(device="cpu").manual_seed(42)
        ts = self._sched().sample_timesteps(
            3,
            4,
            torch.device("cpu"),
            uniform_across_frames=True,
            weighting_scheme="logit_normal",
            generator=g,
        )
        g2 = torch.Generator(device="cpu").manual_seed(42)
        u = torch.nn.functional.sigmoid(torch.normal(0.0, 1.0, (3,), generator=g2))
        idx = (u * 1000).long()
        warped = (3.0 * (idx / 1000) / (1 + 2.0 * (idx / 1000))) * 1000
        idx = (1000 - warped).long().clamp(0, ts.shape[0] * 0 + 999)
        sched = self._sched()
        expected = sched.timesteps[idx.reshape(3, 1).expand(3, 4)]
        assert torch.equal(ts, expected)


class TestTrainingTarget:
    def test_basic(self):
        sched = FlowMatchingScheduler(shift=3.0, schedule="linear")
        clean = torch.tensor([1.0, 2.0, 3.0])
        noise = torch.tensor([4.0, 5.0, 6.0])
        target = sched.training_target(clean, noise)
        expected = torch.tensor([3.0, 3.0, 3.0])
        torch.testing.assert_close(target, expected)


class TestPredX0FromFlow:
    def test_roundtrip(self):
        """add_noise -> training_target -> pred_x0_from_flow should recover clean."""
        clean = torch.randn(2, 3, 4, dtype=torch.float32)
        noise = torch.randn_like(clean)
        sigma = torch.rand(2, 1, 1).clamp(min=0.01)
        noisy = (1 - sigma) * clean + sigma * noise
        target = noise - clean
        recovered = pred_x0_from_flow(noisy, target, sigma)
        torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)

    def test_bf16_precision(self):
        """pred_x0_from_flow should upcast to compute_dtype for stability."""
        clean = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        noise = torch.randn_like(clean)
        sigma = torch.rand(2, 1, 1, dtype=torch.bfloat16).clamp(min=0.01)
        noisy = (1 - sigma) * clean + sigma * noise
        target = noise - clean
        recovered = pred_x0_from_flow(noisy, target, sigma, compute_dtype=torch.float32)
        assert recovered.dtype == torch.bfloat16
        torch.testing.assert_close(recovered.float(), clean.float(), atol=0.05, rtol=0.05)


class TestFlowMatchingLoss:
    def test_zero_loss(self):
        x = torch.randn(2, 3)
        loss = mse_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([0.0, 0.0])
        loss = mse_loss(pred, target)
        expected = (1.0 + 4.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_with_weight(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.zeros(2)
        weight = torch.tensor([2.0, 0.5])
        loss = mse_loss(pred, target, weight=weight)
        expected = (2.0 * 1.0 + 0.5 * 4.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_with_mask(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.zeros(3)
        mask = torch.tensor([True, False, True])
        loss = mse_loss(pred, target, mask=mask)
        expected = (1.0 + 9.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_flow_loss_consumes_broadcastable_batch_mask(self):
        from minwm.engine.training.losses import FlowMatchingLoss

        pred = torch.tensor([[[[[10.0]]], [[[2.0]]]]])
        target = torch.zeros_like(pred)
        batch = {
            "target": target,
            "loss_mask": torch.tensor([False, True]).reshape(1, 2, 1, 1, 1),
        }

        loss = FlowMatchingLoss()(pred, batch, scheduler=None)

        assert loss.item() == pytest.approx(4.0)

    def test_reduction_none(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.zeros(2)
        loss = mse_loss(pred, target, reduction="none")
        assert loss.shape == (2,)


class TestFlowNoiseSeeding:
    """Seed behaviour of the flow preprocessor.

    FlowNoise draws noise/timestep off the **global** RNG (the trainer seeds it
    ``seed + dp_rank`` once at dist-init; see :func:`minwm.utils.set_seed`), so
    reproducibility and the DP-replica divergence are both governed by the global
    seed, not a per-preprocessor generator.
    """

    def _scheduler(self):
        return FlowMatchingScheduler(shift=3.0, schedule="linear")

    def test_global_seed_makes_draw_reproducible(self):
        """Same global seed before the call -> identical noise + timestep."""
        from minwm.processors.flow import FlowNoise
        from minwm.utils import set_seed

        clean = torch.randn(1, 4, 8, 4, 4)
        set_seed(123)
        out_a = FlowNoise(uniform_across_frames=True, scheduler=self._scheduler())(
            {"clean_latent": clean.clone()}, torch.device("cpu")
        )
        set_seed(123)
        out_b = FlowNoise(uniform_across_frames=True, scheduler=self._scheduler())(
            {"clean_latent": clean.clone()}, torch.device("cpu")
        )
        assert torch.equal(out_a["noise"], out_b["noise"])
        assert torch.equal(out_a["timestep"], out_b["timestep"])

    def test_distinct_global_seeds_diverge(self):
        """Different global seeds (the DP-replica axis = seed + dp_rank) diverge."""
        from minwm.processors.flow import FlowNoise
        from minwm.utils import set_seed

        clean = torch.randn(1, 4, 8, 4, 4)
        set_seed(0)
        out_a = FlowNoise(uniform_across_frames=True, scheduler=self._scheduler())(
            {"clean_latent": clean.clone()}, torch.device("cpu")
        )
        set_seed(1)
        out_b = FlowNoise(uniform_across_frames=True, scheduler=self._scheduler())(
            {"clean_latent": clean.clone()}, torch.device("cpu")
        )
        assert not torch.equal(out_a["noise"], out_b["noise"])

    def test_draws_off_shared_global_stream(self):
        """Consecutive calls (no re-seed) differ: one shared stream, no per-call reset."""
        from minwm.processors.flow import FlowNoise
        from minwm.utils import set_seed

        clean = torch.randn(2, 4, 16, 8, 8)
        set_seed(7)
        fn = FlowNoise(uniform_across_frames=False, scheduler=self._scheduler())
        out1 = fn({"clean_latent": clean.clone()}, torch.device("cpu"))
        out2 = fn({"clean_latent": clean.clone()}, torch.device("cpu"))
        assert not torch.equal(out1["noise"], out2["noise"])

    def test_blockwise_timestep_repeats_within_each_block(self):
        from minwm.processors.flow import FlowNoise
        from minwm.utils import set_seed

        clean = torch.randn(1, 8, 4, 2, 2)
        set_seed(11)
        out = FlowNoise(
            uniform_across_frames=False,
            num_frame_per_block=4,
            scheduler=self._scheduler(),
        )({"clean_latent": clean}, torch.device("cpu"))
        timestep = out["timestep"]
        assert torch.equal(timestep[:, :4], timestep[:, :1].expand(-1, 4))
        assert torch.equal(timestep[:, 4:], timestep[:, 4:5].expand(-1, 4))
