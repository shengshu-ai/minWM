"""Sampling utilities shared by generation and data-building scripts."""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from minwm.sampling.schedulers import FlowMatchingScheduler

BlockForward = Callable[[Tensor, Tensor], Tensor]


def _pin_prefix(latents: Tensor, fixed_prefix: Tensor | None) -> Tensor:
    """Restore the pinned clean prefix frames in place, returning ``latents``.

    Bidirectional generation conditions on a clean image/video prefix that must
    survive every denoise step untouched; the autoregressive loop passes
    ``fixed_prefix=None`` and this is a no-op.
    """
    if fixed_prefix is not None:
        latents[:, : fixed_prefix.shape[1]] = fixed_prefix
    return latents


def resolve_denoising_steps(
    scheduler: FlowMatchingScheduler,
    denoising_step_list: list[int] | tuple[int, ...] | Tensor,
    *,
    warp_denoising_step: bool = False,
) -> Tensor:
    """Resolve raw denoising steps into scheduler timesteps.

    Some stage configs store raw steps such as ``[1000, 750, 500, 250]`` and
    map them to the shifted schedule by indexing
    ``cat(scheduler.timesteps, [0])[T - raw]``. Keeping this in one helper avoids
    training/inference/data-prep drift.
    """

    if denoising_step_list is None:
        raise ValueError("denoising_step_list is required for few-step causal sampling")
    steps = torch.as_tensor(denoising_step_list)
    if not warp_denoising_step:
        return steps

    steps = steps.long()
    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.zeros(1, dtype=torch.float32)))
    return timesteps[scheduler.num_train_timesteps - steps]


class CMSolver:
    """Few-step flow sampler used by causal DMD/CM-style generators."""

    def __init__(
        self,
        scheduler: FlowMatchingScheduler,
        denoising_step_list: list[int] | tuple[int, ...] | Tensor,
        *,
        warp_denoising_step: bool = False,
        x0_compute_dtype: torch.dtype | None = None,
        renoise_compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.denoising_steps = resolve_denoising_steps(
            scheduler,
            denoising_step_list,
            warp_denoising_step=warp_denoising_step,
        )
        self.x0_compute_dtype = x0_compute_dtype
        self.renoise_compute_dtype = renoise_compute_dtype

    @classmethod
    def from_config(cls, cfg: dict, inference_cfg: dict, recipe_cfg: dict) -> "CMSolver":
        """Build from an ``inference.sampler`` block (Wan DMD/CD/ODE).

        Config knobs live in the ``sampler`` dict: ``denoising_step_list`` and the
        flow-match scheduler params ``num_train_timesteps`` / ``shift`` /
        ``sigma_min``. The Wan few-step deployment always warps raw steps onto the
        shifted schedule and always uses the ``extra_one_step`` σ-table, so both are
        fixed here rather than exposed as config.

        Raises:
            ValueError: if ``denoising_step_list`` is absent or unknown fields remain.
        """
        denoising_step_list = cfg.pop("denoising_step_list", recipe_cfg.get("denoising_step_list"))
        if denoising_step_list is None:
            raise ValueError("CMSolver requires sampler.denoising_step_list")
        scheduler = FlowMatchingScheduler(
            num_train_timesteps=cfg.pop("num_train_timesteps", 1000),
            shift=cfg.pop("shift", 5.0),
            sigma_min=cfg.pop("sigma_min", 0.0),
            extra_one_step=True,
        )
        if cfg:
            raise ValueError(f"unsupported CMSolver fields: {sorted(cfg)}")
        return cls(scheduler, denoising_step_list, warp_denoising_step=True)

    def timestep_tensor(
        self,
        timestep_value: Tensor | float | int,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype = torch.int64,
    ) -> Tensor:
        """Create the per-frame timestep tensor expected by causal backbones.

        If ``timestep_value`` is a warped float such as ``937.5``, PyTorch
        promotes the result to float and preserves the fractional timestep. If
        it is a raw integer, the result stays integer. Keep that promotion
        behavior because the time embedding participates in rollout parity.
        """

        value = torch.as_tensor(timestep_value, device=device)
        return torch.ones([batch_size, num_frames], device=device, dtype=dtype) * value

    def predict_x0(self, noisy: Tensor, flow_pred: Tensor, timestep: Tensor) -> Tensor:
        """Convert flow prediction to x0, optionally using a configured compute dtype."""

        if self.x0_compute_dtype is None:
            return self.scheduler.predict_x0(noisy, flow_pred, timestep)

        compute_dtype = self.x0_compute_dtype
        sigma = self.scheduler.sigma_for(timestep).reshape(*timestep.shape, 1, 1, 1)
        x0 = noisy.to(compute_dtype) - sigma.to(compute_dtype) * flow_pred.to(compute_dtype)
        return x0.to(noisy.dtype)

    def renoise(self, denoised: Tensor, next_timestep_value: Tensor | float | int) -> Tensor:
        """Re-noise a denoised block to the next few-step timestep."""

        batch_size, num_frames = denoised.shape[:2]
        next_value = torch.as_tensor(next_timestep_value, device=denoised.device)
        next_t = (
            torch.ones([batch_size * num_frames], device=denoised.device, dtype=torch.long)
            * next_value
        )
        flat = denoised.flatten(0, 1)
        noise = torch.randn_like(flat)
        if self.renoise_compute_dtype is None:
            return self.scheduler.add_noise(flat, noise, next_t).unflatten(0, denoised.shape[:2])

        compute_dtype = self.renoise_compute_dtype
        sigma = self.scheduler.sigma_for(next_t).reshape(-1, 1, 1, 1)
        renoised = (1 - sigma.to(compute_dtype)) * flat.to(compute_dtype)
        renoised = renoised + sigma.to(compute_dtype) * noise.to(compute_dtype)
        return renoised.to(noise.dtype).unflatten(0, denoised.shape[:2])

    def step(
        self,
        *,
        forward: BlockForward,
        noise: Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> Tensor:
        """Denoise a clip by predict_x0 + renoise over the few-step grid.

        Reproduces the Wan DMD/CD/ODE deployment inner loop: at each warped step
        predict ``x0`` from the CFG-combined flow, then re-noise to the next step.
        Runs unconditionally (few-step distilled generators carry no CFG). Drives
        the autoregressive loop only; the clean-prefix pin lives on
        :meth:`UniPCSolver.step`, the sole full-clip bidirectional path.

        Args:
            forward (BlockForward): ``(noisy, timestep) -> flow`` closure.
            noise (Tensor): the clip's initial noise ``[B, F, C, H, W]``.
            batch_size (int): batch size ``B``.
            num_frames (int): latent frames in this clip ``F``.
            device (torch.device): compute device.

        Returns:
            Tensor: denoised latents ``[B, F, C, H, W]``.
        """

        noisy = noise
        denoised = None
        steps = self.denoising_steps
        for index, ts_val in enumerate(steps):
            timestep = self.timestep_tensor(
                ts_val,
                batch_size=batch_size,
                num_frames=num_frames,
                device=device,
                dtype=torch.int64,
            )
            flow_pred = forward(noisy, timestep)
            denoised = self.predict_x0(noisy, flow_pred, timestep)
            if index < len(steps) - 1:
                noisy = self.renoise(denoised, steps[index + 1])
        return denoised


class EulerSolver:
    """Euler flow sampler for HunyuanVideo causal AR (few-step and teacher-forcing)."""

    def __init__(self, scheduler: FlowMatchingScheduler) -> None:
        self.scheduler = scheduler

    @classmethod
    def from_config(cls, cfg: dict, inference_cfg: dict, recipe_cfg: dict) -> "EulerSolver":
        """Build from an ``inference.sampler`` block (HY few-step + teacher-forcing).

        Drives HunyuanVideo few-step *and* teacher-forcing over the shifted
        schedule; the two modes differ only in ``num_inference_steps`` and the CFG
        scale applied by the loop. Knobs: ``num_inference_steps`` / ``shift`` /
        ``num_train_timesteps``.
        """
        scheduler = FlowMatchingScheduler(
            num_train_timesteps=cfg.pop("num_train_timesteps", 1000),
            shift=cfg.pop("shift", 5.0),
            num_inference_steps=cfg.pop("num_inference_steps", 4),
            schedule="shifted",
            sigma_min=cfg.pop("sigma_min", 0.0),
            extra_one_step=cfg.pop("extra_one_step", True),
        )
        if cfg:
            raise ValueError(f"unsupported EulerSolver fields: {sorted(cfg)}")
        return cls(scheduler)

    def euler_step(self, flow: Tensor, timestep: Tensor, latents: Tensor) -> Tensor:
        """One flow-matching Euler step ``x + v * (sigma_next - sigma)``."""

        flat_t = timestep.flatten()
        timesteps = self.scheduler.timesteps.to(flat_t.device)
        sigmas = self.scheduler.sigmas.to(flat_t.device)
        idx = (timesteps.unsqueeze(0) - flat_t.unsqueeze(1)).abs().argmin(dim=1)
        tail = (1,) * (latents.ndim - timestep.ndim)
        sigma = sigmas[idx].reshape(*timestep.shape, *tail)
        at_last = (idx + 1 >= len(timesteps)).reshape(*timestep.shape, *tail)
        sigma_next = torch.where(
            at_last,
            torch.zeros_like(sigma),
            sigmas[torch.clamp(idx + 1, max=len(sigmas) - 1)].reshape(*timestep.shape, *tail),
        )
        return latents + flow * (sigma_next - sigma)

    def step(
        self,
        *,
        forward: BlockForward,
        noise: Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> Tensor:
        """Denoise a clip with a flow-matching Euler walk.

        Reproduces the HunyuanVideo causal-AR inner loop: an Euler walk over
        *all* inference steps of the shifted schedule, with the terminal
        ``sigma_next=0`` synthesised inside :meth:`euler_step`. The two HY modes
        (teacher-forcing / few-step) differ only in step count and guidance
        scale, both captured by the ``forward`` closure and the scheduler.

        Args:
            forward (BlockForward): ``(noisy, timestep) -> flow`` closure with CFG
                already applied.
            noise (Tensor): the clip's initial noise ``[B, F, C, H, W]``.
            batch_size (int): batch size ``B``.
            num_frames (int): latent frames in this clip ``F``.
            device (torch.device): compute device.

        Returns:
            Tensor: denoised latents ``[B, F, C, H, W]``.
        """

        dtype = noise.dtype
        latents = noise
        for t in self.scheduler.timesteps:
            timestep = torch.full((batch_size, num_frames), t, device=device, dtype=dtype)
            flow = forward(latents, timestep)
            # euler_step matches this fp32 timestep against the scheduler table;
            # the last step's σ_next=0 is synthesised inside it. Flatten
            # [B,F] -> [B*F] to match its convention.
            step_t = t * torch.ones(batch_size * num_frames, device=device, dtype=torch.float32)
            latents = (
                self.euler_step(flow.flatten(0, 1), step_t, latents.flatten(0, 1))
                .unflatten(0, (batch_size, num_frames))
                .to(dtype)
            )
        return latents


class UniPCSolver:
    """Flow-matching UniPC sampler for full-diffusion CFG inference.

    Uses diffusers' flow-sigma UniPC path with
    ``prediction_type="flow_prediction"``.
    """

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        solver_order: int = 2,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.shift = float(shift)
        self.solver_order = int(solver_order)

    @classmethod
    def from_config(cls, cfg: dict, inference_cfg: dict, recipe_cfg: dict) -> "UniPCSolver":
        """Build from an ``inference.sampler`` block (Wan teacher-forcing / bi-SFT)."""
        gen, recipe = inference_cfg, recipe_cfg
        num_train_timesteps = cfg.pop(
            "num_train_timesteps",
            gen.get("num_train_timesteps", recipe.get("num_train_timesteps", 1000)),
        )
        shift = cfg.pop("shift", gen.get("timestep_shift", recipe.get("timestep_shift", 5.0)))
        num_inference_steps = cfg.pop("num_inference_steps", gen.get("num_inference_steps", 50))
        solver_order = cfg.pop("solver_order", 2)
        if cfg:
            raise ValueError(f"unsupported UniPCSolver fields: {sorted(cfg)}")
        return cls(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            shift=shift,
            solver_order=solver_order,
        )

    def new_scheduler(self, device: torch.device):
        """Create a fresh stateful UniPC scheduler for one denoising loop."""

        from diffusers import UniPCMultistepScheduler

        scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            solver_order=self.solver_order,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=self.shift,
            final_sigmas_type="zero",
        )
        self._set_dev_timesteps(scheduler, device)
        return scheduler

    def _set_dev_timesteps(self, scheduler, device: torch.device) -> None:
        """Set timesteps with the shifted flow-sigma formula."""

        sigma_max = np.float32(1.0 - 1.0 / self.num_train_timesteps).item()
        sigmas = np.linspace(sigma_max, 0.0, self.num_inference_steps + 1).copy()[:-1]
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0]]).astype(np.float32)

        scheduler.sigmas = torch.from_numpy(sigmas).to("cpu")
        scheduler.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        scheduler.num_inference_steps = len(timesteps)
        scheduler.model_outputs = [None] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0
        scheduler.last_sample = None
        scheduler._step_index = None
        scheduler._begin_index = None

    def step(
        self,
        *,
        forward: BlockForward,
        noise: Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        fixed_prefix: Tensor | None = None,
    ) -> Tensor:
        """Denoise a clip with a fresh, stateful UniPC scheduler.

        Reproduces both the Wan teacher-forcing per-block loop and the full-clip
        bidirectional-SFT loop — they differ only in ``fixed_prefix`` (the
        bidirectional clean prefix pinned across every step). A fresh UniPC
        scheduler is created per call and stepped over its timesteps; CFG is
        applied inside ``forward`` before the scheduler step.

        Args:
            forward (BlockForward): ``(noisy, timestep) -> flow`` closure with CFG
                already applied.
            noise (Tensor): the clip's initial noise ``[B, F, C, H, W]``.
            batch_size (int): batch size ``B``.
            num_frames (int): latent frames in this clip ``F``.
            device (torch.device): compute device.
            fixed_prefix (Tensor | None): clean prefix frames to pin, or None.

        Returns:
            Tensor: denoised latents ``[B, F, C, H, W]``.
        """

        latents = noise
        scheduler = self.new_scheduler(device)
        for t in scheduler.timesteps:
            _pin_prefix(latents, fixed_prefix)
            timestep = t * torch.ones([batch_size, num_frames], device=device, dtype=torch.float32)
            if fixed_prefix is not None:
                timestep[:, : fixed_prefix.shape[1]] = 0
            flow = forward(latents, timestep)
            latents = _pin_prefix(
                scheduler.step(flow, t, latents, return_dict=False)[0], fixed_prefix
            )
        return latents


def _resolve_sampler_cls(name: str) -> type:
    """Resolve ``inference.sampler.solver`` to a sampler class.

    Accepts a class name defined in this module (e.g. ``"UniPCSolver"``) or
    an explicit ``"module.path:Name"`` import path for out-of-tree samplers. This
    mirrors the ``type`` -> :func:`locate` convention used for model / vae /
    adapter / loop nodes (the sampler block just spells the key ``solver``), so
    the config names the class directly with no alias registry.

    Args:
        name (str): the configured sampler class name or import path.

    Returns:
        type: the resolved sampler class (exposing ``from_config``).

    Raises:
        NotImplementedError: if the name is neither a sampler class in this module
            nor a resolvable import path.
    """
    if ":" in name:
        from minwm.config import locate

        return locate(name)
    cls = globals().get(name)
    if not isinstance(cls, type):
        raise NotImplementedError(
            f"inference.sampler.solver={name!r} is not a sampler class in "
            "minwm.engine.inference.samplers; name a class or use 'module.path:Name'"
        )
    return cls


def build_sampler(
    sampler_cfg: dict | None = None,
    *,
    inference_cfg: dict | None = None,
    recipe_cfg: dict | None = None,
):
    """Build a full-diffusion sampler from an ``inference.sampler`` block.

    The block names its class directly via ``solver`` — e.g.
    ``sampler=dict(solver="UniPCSolver", num_inference_steps=50)`` — and the
    resolved class's :meth:`from_config` extracts the remaining fields.
    """

    gen = dict(inference_cfg or {})
    recipe = dict(recipe_cfg or {})
    cfg = dict(sampler_cfg or gen.get("sampler") or {})
    sampler_cls = _resolve_sampler_cls(cfg.pop("solver", "UniPCSolver"))
    return sampler_cls.from_config(cfg, gen, recipe)
