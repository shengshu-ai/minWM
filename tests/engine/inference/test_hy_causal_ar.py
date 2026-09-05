"""CPU smoke tests for the HY causal AR-rollout pipeline.

Exercise the GPU-independent seams: the shifted-σ table the pipeline shares with
Wan's :class:`~minwm.sampling.schedulers.FlowMatchingScheduler`, and the
:class:`ARGenerationLoop` block/step loop driven by a stub adapter that records
the ``rollout_*`` hook calls. Asserts shapes / finiteness / call structure, not
end-to-end numerical parity (that needs real weights + a GPU).
"""

import torch

from minwm.engine.inference.loop import ARGenerationLoop
from minwm.engine.inference.samplers import EulerSolver
from minwm.sampling.schedulers import FlowMatchingScheduler


def _euler_sampler(num_inference_steps=4, shift=5.0):
    scheduler = FlowMatchingScheduler(
        num_train_timesteps=1000,
        shift=shift,
        num_inference_steps=num_inference_steps,
        schedule="shifted",
        sigma_min=0.0,
        extra_one_step=True,
    )
    return EulerSolver(scheduler)


def test_shifted_schedule_matches_reference_deployment_table():
    """FlowMatchingScheduler(shifted, sigma_min=0, extra_one_step) is the HY table.

    The reference HY deployment sampler builds ``time_shift(linspace(1, 0, N+1))``
    and walks its first ``N`` timesteps; this config reproduces that timestep
    table exactly, and the terminal ``σ=0`` the last Euler step needs is
    synthesised inside :meth:`EulerSolver.euler_step`.
    """
    N, shift = 4, 5.0
    fm = FlowMatchingScheduler(
        num_train_timesteps=1000,
        shift=shift,
        num_inference_steps=N,
        schedule="shifted",
        sigma_min=0.0,
        extra_one_step=True,
    )

    def time_shift(t):
        return (shift * t) / (1 + (shift - 1) * t)

    ref_sigmas = time_shift(torch.linspace(1, 0, N + 1))
    assert torch.equal(fm.timesteps.float(), (ref_sigmas[:-1] * 1000).float())
    assert torch.equal(fm.sigmas.float(), ref_sigmas[:-1].float())


def test_euler_step_synthesises_terminal_sigma_zero():
    """The last step's σ_next is 0 (clean), so the final Euler step lands at x0."""
    N, shift = 4, 5.0
    fm = FlowMatchingScheduler(
        num_train_timesteps=1000,
        shift=shift,
        num_inference_steps=N,
        schedule="shifted",
        sigma_min=0.0,
        extra_one_step=True,
    )
    sampler = EulerSolver(fm)
    x = torch.randn(2, 4)  # [B*n_f, C-ish] flat
    v = torch.randn_like(x)
    last_t = fm.timesteps[-1] * torch.ones(2)
    stepped = sampler.euler_step(v, last_t, x)
    # x + v*(0 - σ_last) == x - σ_last*v
    sigma_last = fm.sigmas[-1]
    assert torch.allclose(stepped, x - sigma_last * v)
    assert torch.isfinite(stepped).all()


class _RecordingAdapter:
    """Stub adapter recording the AR-rollout hook calls; returns zero flow."""

    wants_inference_autocast = False

    def __init__(self):
        self.forward_calls = 0
        self.refresh_calls = 0
        self.init_calls = 0

    def conditioning(self, batch, batch_size, device):
        return {"tag": "cond"}

    def null_conditioning(self, batch, batch_size, device):
        return {"tag": "uncond"}

    def rollout_init_cache(self, model, **kwargs):
        self.init_calls += 1
        return {"kv": kwargs["cond"]["tag"]}

    def rollout_forward(self, model, *, noisy_block, **kwargs):
        self.forward_calls += 1
        return torch.zeros_like(noisy_block)

    def rollout_refresh_cache(self, model, **kwargs):
        self.refresh_calls += 1


class _StubGen(torch.nn.Module):
    patch_size = (1, 1, 1)


def _batch(num_frames=8):
    return {"noise": torch.randn(1, num_frames, 8, 2, 2)}


def test_ar_pipeline_no_cfg_call_structure():
    """No-CFG rollout: one init/refresh per block, steps×blocks forwards, no neg cache."""
    adapter = _RecordingAdapter()
    pipe = ARGenerationLoop(
        generator=_StubGen(),
        vae=None,
        adapter=adapter,
        sampler=_euler_sampler(num_inference_steps=4),
        guidance_scale=1.0,
        num_frame_per_block=4,
    )
    out = pipe.generate(_batch(num_frames=8))
    num_blocks = 8 // 4
    assert adapter.init_calls == 1  # single cache
    assert adapter.forward_calls == 4 * num_blocks
    assert adapter.refresh_calls == num_blocks
    assert out["latents"].shape == (1, 8, 8, 2, 2)
    assert torch.isfinite(out["latents"]).all()


def test_ar_pipeline_cfg_doubles_caches_and_forwards():
    """CFG>1 builds a second (neg) cache and runs the uncond branch each step."""
    adapter = _RecordingAdapter()
    pipe = ARGenerationLoop(
        generator=_StubGen(),
        vae=None,
        adapter=adapter,
        sampler=_euler_sampler(num_inference_steps=4),
        guidance_scale=6.0,
        num_frame_per_block=4,
    )
    pipe.generate(_batch(num_frames=8))
    num_blocks = 8 // 4
    assert adapter.init_calls == 2  # pos + neg
    assert adapter.forward_calls == 2 * 4 * num_blocks  # cond + uncond
    assert adapter.refresh_calls == 2 * num_blocks
