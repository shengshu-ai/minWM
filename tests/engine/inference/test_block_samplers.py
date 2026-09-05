"""CPU unit tests for the samplers' clip denoise loop.

Each sampler owns one clip's inner denoise loop via ``step``, given a
CFG-combined ``forward`` closure supplied by
:class:`~minwm.engine.inference.loop.ARGenerationLoop`. These tests drive each
sampler with a recording ``forward`` that returns zero flow, asserting the step
count, timestep shapes, and the shape of the returned clean latents — not
end-to-end numerical parity.
"""

import torch

from minwm.engine.inference.samplers import (
    CMSolver,
    EulerSolver,
    UniPCSolver,
)
from minwm.sampling.schedulers import FlowMatchingScheduler


def _recorder():
    """A forward closure that returns zero flow and records each call's shapes."""
    seen = []

    def forward(noisy_block, timestep):
        seen.append((tuple(noisy_block.shape), tuple(timestep.shape), timestep.dtype))
        return torch.zeros_like(noisy_block)

    return forward, seen


def _noise():
    return torch.randn(2, 3, 16, 4, 4)  # [B, n_f, C, H, W]


def test_few_step_sampler_runs_one_forward_per_denoising_step():
    scheduler = FlowMatchingScheduler(num_train_timesteps=1000, shift=5.0)
    steps = [1000, 750, 500, 250]
    sampler = CMSolver(scheduler, steps, warp_denoising_step=True)
    forward, seen = _recorder()

    noise = _noise()
    clean = sampler.step(
        forward=forward,
        noise=noise,
        batch_size=2,
        num_frames=3,
        device=torch.device("cpu"),
    )

    assert len(seen) == len(steps)  # one forward per few-step timestep
    assert all(shape == (2, 3, 16, 4, 4) for shape, _, _ in seen)
    assert all(ts_shape == (2, 3) for _, ts_shape, _ in seen)
    assert clean.shape == noise.shape
    assert torch.isfinite(clean).all()


def test_euler_sampler_runs_one_forward_per_inference_step():
    N = 4
    scheduler = FlowMatchingScheduler(
        num_train_timesteps=1000,
        shift=5.0,
        num_inference_steps=N,
        schedule="shifted",
        sigma_min=0.0,
        extra_one_step=True,
    )
    sampler = EulerSolver(scheduler)
    forward, seen = _recorder()

    noise = _noise()
    clean = sampler.step(
        forward=forward,
        noise=noise,
        batch_size=2,
        num_frames=3,
        device=torch.device("cpu"),
    )

    assert len(seen) == N
    assert all(ts_shape == (2, 3) for _, ts_shape, _ in seen)
    # Zero flow -> Euler leaves the latents untouched at every step.
    assert torch.equal(clean, noise)


def test_unipc_sampler_runs_one_forward_per_scheduler_step():
    sampler = UniPCSolver(num_inference_steps=3, shift=5.0)
    forward, seen = _recorder()

    noise = _noise()
    clean = sampler.step(
        forward=forward,
        noise=noise,
        batch_size=2,
        num_frames=3,
        device=torch.device("cpu"),
    )

    assert len(seen) == 3
    assert all(shape == (2, 3, 16, 4, 4) for shape, _, _ in seen)
    assert clean.shape == noise.shape
    assert torch.isfinite(clean).all()
