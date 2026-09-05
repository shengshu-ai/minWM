import torch

from minwm.engine.inference.loop import ARGenerationLoop, BidirectionalGenerationLoop
from minwm.engine.inference.samplers import EulerSolver, UniPCSolver
from minwm.processors import InferenceRuntime, InitialLatent
from minwm.sampling import SelfForcingPipeline
from minwm.sampling.schedulers import FlowMatchingScheduler


class _Adapter:
    wants_inference_autocast = False

    def __init__(self):
        self.forward_starts = []
        self.refresh_starts = []
        self.seen_noisy = []
        self.seen_timestep = []

    def conditioning(self, batch, batch_size, device):
        return {}

    def null_conditioning(self, batch, batch_size, device):
        return {}

    def denoise(self, model, *, noisy, **kwargs):
        if kwargs.get("timestep") is not None:
            self.seen_noisy.append(noisy.detach().clone())
            self.seen_timestep.append(kwargs["timestep"].detach().clone())
        return torch.ones_like(noisy)

    def rollout_init_cache(self, model, **kwargs):
        return {}

    def rollout_forward(self, model, *, noisy_block, meta, **kwargs):
        self.forward_starts.append(meta["f_start"])
        return torch.zeros_like(noisy_block)

    def rollout_refresh_cache(self, model, *, meta, **kwargs):
        self.refresh_starts.append(meta["f_start"])


class _Generator(torch.nn.Module):
    patch_size = (1, 1, 1)


def test_bidirectional_pipeline_restores_prefix_after_every_step():
    adapter = _Adapter()
    pipe = BidirectionalGenerationLoop(
        generator=_Generator(),
        vae=None,
        sampler=UniPCSolver(num_inference_steps=2),
        adapter=adapter,
    )
    prefix = torch.full((1, 1, 2, 2, 2), 7.0)
    noise = torch.zeros(1, 3, 2, 2, 2)

    output = pipe.generate({"noise": noise, "initial_latent": prefix})

    torch.testing.assert_close(output["latents"][:, :1], prefix)
    # The terminal restore alone would satisfy the assertion above; assert the
    # model actually *saw* a clean, timestep-0 prefix on every denoise call so a
    # dropped in-loop restore / timestep mask can't pass unnoticed.
    assert adapter.seen_noisy, "adapter.denoise was never called"
    for noisy, timestep in zip(adapter.seen_noisy, adapter.seen_timestep):
        torch.testing.assert_close(noisy[:, :1], prefix)
        assert (timestep[:, :1] == 0).all()


def test_causal_pipeline_primes_prefix_cache_and_starts_after_it():
    adapter = _Adapter()
    scheduler = FlowMatchingScheduler(num_inference_steps=1)
    pipe = ARGenerationLoop(
        generator=_Generator(),
        vae=None,
        adapter=adapter,
        sampler=EulerSolver(scheduler),
        num_frame_per_block=2,
    )
    prefix = torch.full((1, 1, 2, 2, 2), 3.0)
    noise = torch.zeros(1, 5, 2, 2, 2)

    output = pipe.generate({"noise": noise, "initial_latent": prefix})

    torch.testing.assert_close(output["latents"][:, :1], prefix)
    assert adapter.forward_starts == [1, 3]
    assert adapter.refresh_starts == [0, 1, 3]


def test_self_forcing_primes_prefix_cache_and_starts_after_it():
    adapter = _Adapter()
    pipe = SelfForcingPipeline(
        generator=_Generator(),
        scheduler=FlowMatchingScheduler(),
        adapter=adapter,
        denoising_step_list=[1000],
        num_frame_per_block=2,
    )
    prefix = torch.full((1, 1, 2, 2, 2), 5.0)
    noise = torch.zeros(1, 5, 2, 2, 2)

    output, _, _ = pipe.inference_with_trajectory(noise, cond={}, initial_latent=prefix)

    torch.testing.assert_close(output[:, :1], prefix)
    assert adapter.forward_starts == [1, 3]
    assert adapter.refresh_starts == [0, 1, 3]


class _VAE:
    def encode(self, videos):
        return [video[:, :, ::2, ::2] for video in videos]


def test_initial_latent_preprocessor_encodes_tensor_image():
    runtime = InferenceRuntime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_cfg={"latent_shape": [3, 2, 2], "num_frames": 3},
        components={"vae": _VAE()},
    )
    batch = {"prompts": ["test"], "image": torch.zeros(1, 3, 4, 4)}

    output = InitialLatent()(batch, runtime)

    assert output["initial_latent"].shape == (1, 1, 3, 2, 2)
