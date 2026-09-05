"""Tests for the ODE-trajectory preprocessor."""

import torch

from minwm.processors.ode import ODETrajectorySample
from minwm.sampling.schedulers import FlowMatchingScheduler


class TestODETrajectorySample:
    # Real Wan ODE data: N=6 trajectory (4 denoising points + 48-step target +
    # clean), so denoising_step_list has length 4 — it bounds the sampled step
    # index and must be < N-1 so the -2 target is never fed as a noisy input.
    N = 6
    STEPS = [1000, 750, 500, 250]

    def test_output_shapes(self):
        B, F, C, H, W = 2, 3, 16, 8, 8
        traj = torch.randn(B, self.N, F, C, H, W)
        batch = ODETrajectorySample(self.STEPS)({"ode_latent": traj}, traj.device)
        assert batch["noisy"].shape == (B, F, C, H, W)
        assert batch["clean"].shape == (B, F, C, H, W)
        assert batch["target"].shape == (B, F, C, H, W)
        assert batch["timestep"].shape == (B, F)

    def test_clean_is_last(self):
        B, F, C, H, W = 1, 2, 4, 4, 4
        traj = torch.randn(B, self.N, F, C, H, W)
        batch = ODETrajectorySample(self.STEPS)({"ode_latent": traj}, traj.device)
        torch.testing.assert_close(batch["clean"], traj[:, -1])

    def test_target_is_second_to_last(self):
        B, F, C, H, W = 1, 2, 4, 4, 4
        traj = torch.randn(B, self.N, F, C, H, W)
        batch = ODETrajectorySample(self.STEPS)({"ode_latent": traj}, traj.device)
        torch.testing.assert_close(batch["target"], traj[:, -2])

    def test_timestep_in_range(self):
        B, F, C, H, W = 3, 2, 4, 4, 4
        traj = torch.randn(B, self.N, F, C, H, W)
        batch = ODETrajectorySample(self.STEPS)({"ode_latent": traj}, traj.device)
        # Step index is bounded by len(STEPS)=4, so the timestep is one of the
        # raw step values — never indexes the -2 target (would be out of bounds).
        for val in batch["timestep"].flatten().tolist():
            assert val in self.STEPS

    def test_warp_maps_to_schedule_timesteps(self):
        # warp_denoising_step remaps raw steps to the shifted-schedule timesteps
        # the ODE data was solved on: [1000,750,500,250] -> [1000,937.5,833.3,625].
        sched = FlowMatchingScheduler(
            num_train_timesteps=1000, shift=5.0, sigma_min=0.0, extra_one_step=True
        )
        pp = ODETrajectorySample(self.STEPS, warp_denoising_step=True, scheduler=sched)
        warped = pp._resolve_steps(torch.device("cpu"))
        expected = torch.tensor([1000.0, 937.5, 833.3333, 625.0])
        torch.testing.assert_close(warped, expected, atol=1e-3, rtol=0)
