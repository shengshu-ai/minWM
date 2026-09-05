"""ODETrajectorySample: sample one step from a precomputed ODE trajectory."""

import torch
from torch import Tensor

from minwm.sampling.schedulers import FlowMatchingScheduler

from .base import BatchPreprocessor


class ODETrajectorySample(BatchPreprocessor):
    """Sample one step from a precomputed ODE trajectory for regression.

    Reads ``ode_latent`` ``[B, N, F, C, H, W]`` (ordered most-noisy -> clean, so
    the last entry is the clean latent and the second-to-last is the regression
    target) and writes ``noisy``, ``clean``, ``target``, and ``timestep``
    ``[B, F]``. A single random step index is drawn per sample and shared across
    frames (teacher forcing).

    The step index is sampled from ``range(len(denoising_step_list))`` — NOT from
    ``N - 1`` — so the second-to-last entry (the near-clean regression target) is
    never fed back as a noisy input. For the legacy Wan ODE data ``N = 6`` while
    ``len(denoising_step_list) = 4``: the four noisy points sit at indices
    ``0..3`` and the 48-step target at index ``-2``, so they must not be conflated.

    Args:
        denoising_step_list (list[int]): timestep value for each ODE step,
            most-noisy -> least-noisy. Its length bounds the sampled step index.
        warp_denoising_step (bool): if True, remap each raw step to the schedule's
            actual timestep via ``cat(scheduler.timesteps, [0])[T - step]`` (the
            legacy ``warp_denoising_step``). The ODE trajectory was solved on the
            shifted flow schedule, so the label must be the warped timestep, not
            the raw value. Requires ``scheduler``.
        scheduler (FlowMatchingScheduler, optional): the recipe's shared scheduler,
            used only when ``warp_denoising_step`` is True. Left ``None`` when
            built from config; the recipe injects its runtime scheduler in
            :meth:`~minwm.engine.recipe_base.Recipe.build_preprocessors`.
    """

    def __init__(
        self,
        denoising_step_list: list[int],
        warp_denoising_step: bool = False,
        scheduler: FlowMatchingScheduler | None = None,
    ) -> None:
        self.denoising_step_list = denoising_step_list
        self.warp_denoising_step = warp_denoising_step
        self.scheduler = scheduler
        self._steps: Tensor | None = None

    def _resolve_steps(self, device: torch.device) -> Tensor:
        """Build (and cache) the per-step timestep table, warped if requested."""
        if self._steps is not None:
            return self._steps.to(device)

        raw = torch.tensor(self.denoising_step_list, dtype=torch.long)
        if not self.warp_denoising_step:
            self._steps = raw.float()
            return self._steps.to(device)

        if self.scheduler is None:
            raise ValueError("warp_denoising_step=True requires a scheduler")
        # cat a trailing 0 so step == num_train_timesteps maps to t=0, then index
        # the schedule at T - step (legacy Wan21 model/base.py warp).
        timesteps = torch.cat([self.scheduler.timesteps.cpu(), torch.zeros(1)])
        self._steps = timesteps[self.scheduler.num_train_timesteps - raw]
        return self._steps.to(device)

    def __call__(self, batch: dict, device: torch.device) -> dict:
        ode_latent: Tensor = batch["ode_latent"]
        batch_size = ode_latent.shape[0]

        clean_latent = ode_latent[:, -1]  # [B, F, C, H, W]
        target_latent = ode_latent[:, -2]  # [B, F, C, H, W]
        valid_trajectory = ode_latent[:, :-1]  # exclude clean

        num_steps = len(self.denoising_step_list)
        num_frames = valid_trajectory.shape[2]
        num_channels = valid_trajectory.shape[3]
        height = valid_trajectory.shape[4]
        width = valid_trajectory.shape[5]

        # Uniform random step index per sample (same across frames for TF)
        from minwm.distributed import get_rng_states_tracker

        with get_rng_states_tracker().fork():
            step_idx = torch.randint(0, num_steps, (batch_size, 1), device=device)
        gather_idx = step_idx.view(batch_size, 1, 1, 1, 1, 1).expand(
            batch_size, 1, num_frames, num_channels, height, width
        )
        noisy_input = torch.gather(valid_trajectory.to(device), dim=1, index=gather_idx).squeeze(
            1
        )  # [B, F, C, H, W]

        steps = self._resolve_steps(device)
        timestep = steps[step_idx.expand(-1, num_frames)]  # [B, F]

        batch["noisy"] = noisy_input
        batch["clean"] = clean_latent
        batch["target"] = target_latent
        batch["timestep"] = timestep
        return batch
