"""Flow-matching noise schedule, shared across training and inference.

The :class:`FlowMatchingScheduler` (shifted flow-matching sigma schedule + noise
and target ops) is the neutral primitive consumed by ``training`` (losses,
preprocessors, recipes), ``inference`` (samplers), and ``rollouts``. Pure torch,
no minwm dependencies. ``pred_x0_from_flow`` is the stateless flow -> x0 helper
used by the scheduler and by AR consistency distillation.
"""

import math

import torch
from torch import Tensor

__all__ = [
    "FlowMatchingScheduler",
    "pred_x0_from_flow",
]


def _compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator: torch.Generator | None = None,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 0.0,
) -> Tensor:
    """Per-item density ``u ∈ (0, 1)`` (SD3/HY): ``logit_normal``, ``mode``, or uniform."""
    if weighting_scheme == "logit_normal":
        u = torch.normal(
            mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu", generator=generator
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
    return u


class FlowMatchingScheduler:
    """Shifted flow-matching noise scheduler.

    Implements the same schedule as ``Wan21/wan_utils/scheduler.py:FlowMatchScheduler``
    but as a pure-torch object with no external dependencies. The ``sigma_min`` /
    ``extra_one_step`` knobs select which legacy *usage* to reproduce: the class
    default (``0.003/1.002`` / ``False``) matches the legacy bare-class default;
    the Wan stages' ``WanDiffusionWrapper`` overrode them to ``0.0`` / ``True``,
    so a parity run with the old framework must set those.

    Args:
        num_train_timesteps (int): total number of discrete timesteps.
        shift (float): warp strength. In ``"shifted"`` it shapes the σ table; in
            ``"linear"`` it warps the sampled timestep index.
        num_inference_steps (int): steps used to build the timestep table.
        schedule (str): ``"shifted"`` (Wan, default) or ``"linear"`` (HY). In
            ``"shifted"`` the ``sigma_min`` / ``extra_one_step`` knobs shape the
            σ table; ``"linear"`` ignores them (warp applied at sampling).
        sigma_min (float): lower end of the sigma linspace ("shifted" only).
        extra_one_step (bool): build ``num+1`` sigmas and drop the last (the
            legacy ``extra_one_step`` behaviour the Wan wrapper used; "shifted"
            only).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        num_inference_steps: int | None = None,
        schedule: str = "shifted",
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.schedule = schedule
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        # Default to a full table (one entry per train timestep) so a sampled
        # index maps directly onto a discrete timestep, matching the legacy
        # ``set_timesteps(num_train_timesteps, training=True)`` behaviour.
        self._build_schedule(num_inference_steps or num_train_timesteps)

    def _shift_transform(self, x: Tensor | float) -> Tensor | float:
        return self.shift * x / (1 + (self.shift - 1) * x)

    def _build_schedule(self, num_inference_steps: int) -> None:
        if self.schedule == "linear":
            # HY: linear σ in [1, 0), warp NOT baked in (applied at sampling).
            sigmas = torch.linspace(1, 0, num_inference_steps + 1)[:-1]
        else:
            # Wan: warp baked into the σ table over [1, sigma_min].
            sigma_max = 1.0
            if self.extra_one_step:
                sigmas = torch.linspace(sigma_max, self.sigma_min, num_inference_steps + 1)[:-1]
            else:
                sigmas = torch.linspace(sigma_max, self.sigma_min, num_inference_steps)
            sigmas = self._shift_transform(sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        # bell-shaped training weights centred at the midpoint
        x = self.timesteps
        y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
        y = y - y.min()
        self.weights = y * (num_inference_steps / y.sum())

    def sigma_for(self, timestep: Tensor) -> Tensor:
        """Look up sigma for arbitrary timestep values (shape-preserving).

        Args:
            timestep (Tensor): timestep values, any shape.

        Returns:
            Tensor: sigma at each timestep, same shape as ``timestep``.
        """
        self.sigmas = self.sigmas.to(timestep.device)
        self.timesteps = self.timesteps.to(timestep.device)
        flat_t = timestep.flatten()
        idx = (self.timesteps.unsqueeze(0) - flat_t.unsqueeze(1)).abs().argmin(dim=1)
        return self.sigmas[idx].reshape(timestep.shape)

    def predict_x0(self, noisy: Tensor, flow_pred: Tensor, timestep: Tensor) -> Tensor:
        """Convert a flow prediction to an ``x0`` estimate: ``x0 = x_t - σ·v``.

        Looks up sigma for ``timestep`` and broadcasts it over the trailing
        latent dims, so a ``[B, F]`` timestep maps onto a ``[B, F, C, H, W]``
        prediction.

        Args:
            noisy (Tensor): noisy input ``x_t`` ``[B, F, C, H, W]``.
            flow_pred (Tensor): predicted velocity ``v``, same shape as ``noisy``.
            timestep (Tensor): timestep values ``[B, F]``.

        Returns:
            Tensor: the ``x0`` estimate, same shape as ``noisy``.
        """
        sigma = self.sigma_for(timestep).reshape(*timestep.shape, 1, 1, 1)
        return pred_x0_from_flow(noisy, flow_pred, sigma)

    def sample_timesteps(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        uniform_across_frames: bool = True,
        weighting_scheme: str | None = None,
        generator: torch.Generator | None = None,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 0.0,
    ) -> Tensor:
        """Sample random training timesteps, shape ``[B, F]``.

        ``weighting_scheme=None`` is the uniform Wan path; a density scheme (HY,
        e.g. ``"logit_normal"``) draws a density, maps it to an index, then warps
        that index with :meth:`_shift_transform` — needs ``schedule="linear"``.
        ``uniform_across_frames`` applies to both paths: True draws one timestep
        per sample (shared across frames), False draws one per frame.

        Args:
            batch_size (int): batch size.
            num_frames (int): number of video frames.
            device (torch.device): target device.
            uniform_across_frames (bool): if True all frames share one timestep.
            weighting_scheme (str, optional): HY density scheme; ``None`` = uniform.
            generator (torch.Generator, optional): CPU generator; shared across an
                SP group so the draw is identical there.
            logit_mean (float): logit-normal mean (HY).
            logit_std (float): logit-normal std (HY).
            mode_scale (float): mode-weighting scale (HY).

        Returns:
            Tensor: timestep values, shape ``[B, F]``.
        """
        n = self.timesteps.shape[0]
        cols = 1 if uniform_across_frames else num_frames
        if weighting_scheme:
            u = _compute_density_for_timestep_sampling(
                weighting_scheme, batch_size * cols, generator, logit_mean, logit_std, mode_scale
            )
            idx = (u * self.num_train_timesteps).long()
            warped = (
                self._shift_transform(idx / self.num_train_timesteps) * self.num_train_timesteps
            )
            idx = (self.num_train_timesteps - warped).long()
            idx = idx.clamp(0, n - 1).to(device).reshape(batch_size, cols)
        else:
            lo, hi = int(0.02 * n), int(0.98 * n)
            if generator is not None:
                idx = torch.randint(lo, hi, (batch_size, cols), generator=generator).to(device)
            else:
                idx = torch.randint(lo, hi, (batch_size, cols), device=device)
        if uniform_across_frames:
            idx = idx.expand(batch_size, num_frames)
        self.timesteps = self.timesteps.to(device)
        return self.timesteps[idx].contiguous()

    def add_noise(self, clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        """Forward noising: ``x_t = (1-σ)*x_0 + σ*ε``.

        Args:
            clean: ``[B*F, C, H, W]`` clean latents.
            noise: same shape as clean.
            timestep: ``[B*F]`` integer timestep values.

        Returns:
            Tensor: noisy latents, same shape as clean.
        """
        sigma = self.sigma_for(timestep).reshape(-1, 1, 1, 1).to(clean.dtype)
        return (1 - sigma) * clean + sigma * noise

    def training_target(self, clean: Tensor, noise: Tensor) -> Tensor:
        """Flow target: ``v = ε - x_0``."""
        return noise - clean

    def training_weight(self, timestep: Tensor) -> Tensor:
        """Per-sample bell-shaped loss weight for ``timestep`` (shape-preserving)."""
        self.weights = self.weights.to(timestep.device)
        self.timesteps = self.timesteps.to(timestep.device)
        flat_t = timestep.flatten()
        idx = (self.timesteps.unsqueeze(1) - flat_t.unsqueeze(0)).abs().argmin(dim=0)
        return self.weights[idx].reshape(timestep.shape)


def pred_x0_from_flow(
    noisy: Tensor,
    flow_pred: Tensor,
    sigma: Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Predict clean sample from flow prediction.

    x_0 = x_t - sigma * v

    Args:
        noisy: noisy input x_t
        flow_pred: predicted velocity v
        sigma: noise level, broadcastable
        compute_dtype: dtype for the computation (default fp32 for stability)
    """
    original_dtype = noisy.dtype
    return (noisy.to(compute_dtype) - sigma.to(compute_dtype) * flow_pred.to(compute_dtype)).to(
        original_dtype
    )
