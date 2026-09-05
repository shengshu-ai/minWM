import pytest
import torch

from minwm.processors import FirstFrameConditioning, FlowNoise


class _TimestepScheduler:
    def sample_timesteps(self, batch_size, num_frames, device, **kwargs):
        return torch.arange(num_frames, device=device).expand(batch_size, -1)

    def add_noise(self, clean, noise, timestep):
        return clean

    def training_target(self, clean, noise):
        return noise

    def training_weight(self, timestep):
        return torch.ones_like(timestep)


def test_first_frame_conditioning_restores_prefix_and_masks_loss():
    clean = torch.arange(3.0).reshape(1, 3, 1, 1, 1)
    noisy = torch.full_like(clean, -1)
    batch = {
        "clean_latent": clean,
        "noisy": noisy,
        "timestep": torch.tensor([[800, 600, 400]]),
    }

    output = FirstFrameConditioning()(batch, torch.device("cpu"))

    torch.testing.assert_close(output["noisy"][:, :1], clean[:, :1])
    torch.testing.assert_close(output["noisy"][:, 1:], noisy[:, 1:])
    assert output["timestep"].tolist() == [[0, 600, 400]]
    assert output["loss_mask"].flatten().tolist() == [False, True, True]


def test_first_frame_conditioning_validates_explicit_prefix_shape():
    batch = {
        "clean_latent": torch.zeros(1, 3, 1, 1, 1),
        "noisy": torch.zeros(1, 3, 1, 1, 1),
        "timestep": torch.ones(1, 3),
        "initial_latent": torch.zeros(1, 2, 1, 1, 1),
    }

    with pytest.raises(ValueError, match="initial_latent must have shape"):
        FirstFrameConditioning()(batch, torch.device("cpu"))


def test_first_frame_conditioning_can_prepare_rollout_prefix_without_flow_noise():
    clean = torch.randn(1, 5, 2, 2, 2)
    output = FirstFrameConditioning()({"clean_latent": clean}, torch.device("cpu"))

    torch.testing.assert_close(output["initial_latent"], clean[:, :1])
    assert "noisy" not in output
    assert output["loss_mask"].flatten().tolist() == [False, True, True, True, True]


def test_flow_noise_groups_timesteps_after_independent_prefix():
    batch = {"clean_latent": torch.zeros(1, 9, 1, 1, 1)}
    output = FlowNoise(
        uniform_across_frames=False,
        num_frame_per_block=4,
        num_prefix_frames=1,
        scheduler=_TimestepScheduler(),
    )(batch, torch.device("cpu"))

    assert output["timestep"].tolist() == [[0, 1, 1, 1, 1, 5, 5, 5, 5]]
