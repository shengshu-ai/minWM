"""ARCDRecipe: Stage 2b causal consistency distillation.

Equivalent to Wan21/wan_trainer/camera_naive_cd.py +
Wan21/model/camera_naive_consistency.py.

Three model copies are involved:
    - student  (``model``, trainer-owned, trainable) -> prediction at t
    - ema      (EMA of the student, frozen)          -> target at t_next
    - teacher  (frozen reference)                    -> CFG Euler step t -> t_next

The teacher + EMA are trainer-owned ``auxiliary_models`` (frozen, FSDP-sharded
alongside the student), passed into :meth:`train_one_step`. On the first step
they are seeded from the student's weights via :func:`copy_params` (state-dict
parameter copy, FSDP-safe — unlike ``deepcopy``, which FSDP rejects), so all
three start identical whether the student came from the base or a Stage-1/2a
checkpoint. The config must declare ``auxiliary_models=dict(teacher=..., ema=...)``
(both ``trainable=False``).

Batch contract (from CameraLatentLMDBDataset):
    prompts:      list[str], length B
    clean_latent: Tensor [B, F, C, H, W]
    viewmats:     Tensor [B, F, 4, 4]  (optional, for PRoPE)
    Ks:           Tensor [B, F, 3, 3]  (optional, for PRoPE)
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from minwm.engine.events import record_time
from minwm.engine.optim.ema import copy_params, update_ema_params
from minwm.sampling.schedulers import pred_x0_from_flow

from ._flow_base import FlowMatchingRecipeBase


class ARCDRecipe(FlowMatchingRecipeBase):
    """Causal consistency-distillation step (Causal Forcing++).

    Samples an adjacent timestep pair ``(t, t_next)`` from a discretised
    schedule. The frozen teacher does a CFG Euler step ``t -> t_next`` to make
    the target latent; the student predicts ``x0`` at ``t`` and the EMA network
    predicts ``x0`` at ``t_next``; the loss is the MSE between the two ``x0``
    predictions (consistency).

    Args:
        optimizer (dict, optional): optimizer config for :func:`build_optimizer`
            — optional ``name`` (``"module:Class"``, default ``torch.optim:AdamW``)
            plus constructor kwargs (``lr``, ``betas``, ``weight_decay``). Defaults
            to AdamW with ``lr=2e-6, betas=(0.0, 0.999), weight_decay=0.0``.
        adapter (ModelAdapter, optional): model call-convention bridge. Defaults
            to :class:`~minwm.modeling.wan21.adapter.Wan21Adapter`.
        max_grad_norm (float): gradient clipping norm.
        skip_grad_norm (float, optional): skip the step when the global grad norm
            exceeds this (all ranks in unison); ``None`` disables skipping.
        num_train_timesteps (int): scheduler timestep table size.
        timestep_shift (float): flow schedule shift parameter.
        discrete_cd_n (int): number of discrete CD steps (size of the sampled
            timestep table). ``include_terminal_pair`` controls whether index
            ``N-1`` may pair with the synthetic zero-sigma terminal.
        guidance_scale (float): teacher CFG scale.
        ema_decay (float): EMA decay for the consistency target network.
        include_terminal_pair (bool): whether the sampled adjacent pair may end
            at the synthetic zero-sigma terminal. Defaults to ``True``, which the
            CD math requires; both HY and Wan21 use the default.
        sigma_min (float): lower end of the sigma schedule; set to ``0.0`` to match
            the legacy Wan stages (see :class:`FlowMatchingScheduler`).
        extra_one_step (bool): legacy ``extra_one_step`` sigma construction; set
            ``True`` to match the legacy Wan stages.
    """

    def __init__(
        self,
        optimizer: dict | None = None,
        adapter: Any | None = None,
        max_grad_norm: float = 10.0,
        skip_grad_norm: float | None = None,
        num_train_timesteps: int = 1000,
        timestep_shift: float = 5.0,
        discrete_cd_n: int = 48,
        guidance_scale: float = 5.0,
        ema_decay: float = 0.95,
        include_terminal_pair: bool = True,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
        batch_preprocessors: list | None = None,
    ):
        super().__init__(
            optimizer=optimizer,
            adapter=adapter,
            max_grad_norm=max_grad_norm,
            skip_grad_norm=skip_grad_norm,
            num_train_timesteps=num_train_timesteps,
            timestep_shift=timestep_shift,
            num_inference_steps=discrete_cd_n,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
            batch_preprocessors=batch_preprocessors,
        )
        self.discrete_cd_n = discrete_cd_n
        self.guidance_scale = guidance_scale
        self.ema_decay = ema_decay
        self.include_terminal_pair = include_terminal_pair
        self.cd_sigmas = torch.cat([self.scheduler.sigmas, self.scheduler.sigmas.new_zeros(1)])
        self.cd_timesteps = self.scheduler.timesteps
        self.teacher: nn.Module | None = None
        self.ema: nn.Module | None = None

    def _lazy_init_aux(
        self, model: nn.Module, auxiliary_models: dict[str, nn.Module] | None
    ) -> None:
        """Bind teacher + EMA from ``auxiliary_models``, seeding from the student.

        The teacher and EMA are trainer-owned, FSDP-sharded ``auxiliary_models``
        (declared frozen in the config). On the first step they are seeded from
        the student's weights via :func:`copy_params` so all three start
        identical. ``copy_params`` is a state-dict-level parameter copy, which is
        FSDP-safe (the student and aux nets share the same sharding) — unlike
        ``deepcopy``, which FSDP rejects.

        Args:
            model (nn.Module): the student (its weights seed teacher + EMA).
            auxiliary_models (dict): must contain ``"teacher"`` and ``"ema"``.

        Raises:
            ValueError: if ``auxiliary_models`` lacks ``teacher`` / ``ema``.
        """
        if self.teacher is not None:
            return
        if (
            not auxiliary_models
            or "teacher" not in auxiliary_models
            or "ema" not in auxiliary_models
        ):
            raise ValueError(
                "ARCDRecipe needs auxiliary_models with "
                "'teacher' and 'ema' (declare them in the config, trainable=False)"
            )
        self.teacher = auxiliary_models["teacher"]
        self.ema = auxiliary_models["ema"]
        copy_params(model, self.teacher)
        copy_params(model, self.ema)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.ema.requires_grad_(False)
        self.ema.eval()

    def train_one_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
        auxiliary_models: dict[str, nn.Module] | None = None,
    ) -> dict[str, float]:
        """One consistency-distillation optimisation step.

        Args:
            model (nn.Module): student model (trainable).
            batch (dict): ``clean_latent`` ``[B,F,C,H,W]`` plus whatever
                conditioning the adapter reads (``prompts`` for Wan; the TI2V
                stream for HY) and optional PRoPE ``viewmats`` / ``Ks``.
            optimizers (dict): must contain ``"main"``.
            step (int): current global step.
            auxiliary_models (dict): must contain ``"teacher"`` and ``"ema"``.

        Returns:
            dict[str, float]: ``{"loss": <scalar>}``.
        """
        self._lazy_init_aux(model, auxiliary_models)
        device = next(model.parameters()).device
        if self.batch_preprocessors_cfg:
            batch = self.preprocess(batch, device)
        clean = batch["clean_latent"].to(device=device, dtype=self.dtype)
        B, F, C, H, W = clean.shape
        initial_latent = batch.get("initial_latent")
        prefix_frames = 0 if initial_latent is None else initial_latent.shape[1]

        viewmats: Tensor | None = None
        Ks: Tensor | None = None
        if "viewmats" in batch:
            viewmats = batch["viewmats"].to(device=device, dtype=self.dtype)
            Ks = batch["Ks"].to(device=device, dtype=self.dtype)

        self.cd_sigmas = self.cd_sigmas.to(device)
        self.cd_timesteps = self.cd_timesteps.to(device)
        # Sample from (0, self.discrete_cd_n) so the pair may end at the synthetic
        # zero-sigma terminal — an extra 0 is padded to the end of sigmas for it
        # (the terminal pair, σ_next=0). include_terminal_pair=False falls back to
        # (0, self.discrete_cd_n - 1), the legacy Wan release behavior.
        from minwm.distributed import get_rng_states_tracker

        with get_rng_states_tracker().fork():
            max_idx = self.discrete_cd_n if self.include_terminal_pair else self.discrete_cd_n - 1
            idx = int(torch.randint(0, max_idx, (1,), device=device).item())
            sigma_t = self.cd_sigmas[idx].to(self.dtype)
            sigma_t_next = self.cd_sigmas[idx + 1].to(self.dtype)
            t = self.cd_timesteps[idx].to(self.dtype)
            t_next = (
                self.cd_timesteps[idx + 1].to(self.dtype)
                if idx + 1 < self.discrete_cd_n
                else torch.zeros((), device=device, dtype=self.dtype)
            )
            timestep = t * torch.ones(B, F, device=device, dtype=self.dtype)
            timestep_next = t_next * torch.ones(B, F, device=device, dtype=self.dtype)

            noise = torch.randn_like(clean)
        latent_t = (1.0 - sigma_t) * clean + sigma_t * noise
        if initial_latent is not None:
            initial_latent = initial_latent.to(device=device, dtype=self.dtype)
            latent_t[:, :prefix_frames] = initial_latent
            timestep[:, :prefix_frames] = 0
            timestep_next[:, :prefix_frames] = 0
        cond = self.adapter.conditioning(batch, B, device)
        uncond = self.adapter.null_conditioning(batch, B, device)
        shared = dict(clean=clean, viewmats=viewmats, Ks=Ks)

        # teacher CFG Euler step t -> t_next in σ-space (no grad): the refactor's
        with torch.no_grad(), self.autocast():
            v_cond = self.adapter.denoise(
                self.teacher, noisy=latent_t, timestep=timestep, **cond, **shared
            )
            v_uncond = self.adapter.denoise(
                self.teacher, noisy=latent_t, timestep=timestep, **uncond, **shared
            )
            latent_t_next = teacher_cfg_euler_step(
                v_cond=v_cond,
                v_uncond=v_uncond,
                latent_t=latent_t,
                t=sigma_t.reshape(1, 1).expand(B, F),
                t_next=sigma_t_next.reshape(1, 1).expand(B, F),
                guidance_scale=self.guidance_scale,
                timestep_scale=1.0,
            )
            if initial_latent is not None:
                latent_t_next[:, :prefix_frames] = initial_latent

        # student x0 prediction at t
        with record_time("forward"), self.autocast():
            cm_pred_t = self._forward_x0(model, latent_t, timestep, sigma_t, cond, shared)

        with torch.no_grad(), self.autocast():
            cm_pred_t_next = self._forward_x0(
                self.ema, latent_t_next, timestep_next, sigma_t_next, cond, shared
            )

        loss = consistency_loss(
            cm_pred_t[:, prefix_frames:],
            cm_pred_t_next[:, prefix_frames:],
            reduction="mean",
        )

        self.optimizer_step(loss, optimizers["main"], model.parameters(), batch=batch)

        # Refactor updates EMA at the end of train_one_step, after optimizer.step().
        update_ema_params(self.ema.parameters(), model.parameters(), self.ema_decay)
        return {"loss": loss.item()}

    def _forward_x0(
        self,
        net: nn.Module,
        noisy: Tensor,
        timestep: Tensor,
        sigma: Tensor,
        cond: dict[str, Any],
        shared: dict[str, Any],
    ) -> Tensor:
        """Run ``net`` with teacher forcing, return x0 = noisy − σ·v [B,F,C,H,W]."""
        flow_pred = self.adapter.denoise(net, noisy=noisy, timestep=timestep, **cond, **shared)
        return pred_x0_from_flow(noisy, flow_pred, sigma)


def sample_cd_timestep_pair(
    num_steps: int,
    sigmas: Tensor,
    timesteps: Tensor,
    device: torch.device,
    include_terminal: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample adjacent ``(t, t_next)`` pair for consistency distillation.

    Args:
        num_steps (int): total number of discrete steps.
        sigmas (Tensor): [N+1] sigma schedule from scheduler.
        timesteps (Tensor): [N] timestep schedule from scheduler.
        device (torch.device): target device.
        include_terminal (bool): if True, allow sampling the terminal step.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: ``(sigma_t, sigma_t_next, t,
        t_next)`` scalars at steps ``t`` and ``t+1``.
    """
    max_idx = num_steps - 1 if not include_terminal else num_steps
    idx = torch.randint(0, max_idx, (1,), device=device).item()
    t = timesteps[idx]
    t_next = timesteps[idx + 1] if idx + 1 < len(timesteps) else torch.tensor(0.0, device=device)
    sigma_t = sigmas[idx]
    sigma_t_next = sigmas[idx + 1]
    return sigma_t, sigma_t_next, t, t_next


def teacher_cfg_euler_step(
    v_cond: Tensor,
    v_uncond: Tensor,
    latent_t: Tensor,
    t: Tensor,
    t_next: Tensor,
    guidance_scale: float,
    timestep_scale: float = 1000.0,
) -> Tensor:
    """Teacher CFG + single Euler step to produce the target latent.

    ``v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)``;
    ``dt = (t - t_next) / timestep_scale``; ``latent_t_next = latent_t - dt * v_cfg``.

    Args:
        v_cond (Tensor): teacher conditional velocity prediction.
        v_uncond (Tensor): teacher unconditional velocity prediction.
        latent_t (Tensor): noisy latent at timestep ``t``.
        t (Tensor): current timestep [B, F] or scalar.
        t_next (Tensor): next timestep [B, F] or scalar.
        guidance_scale (float): CFG scale.
        timestep_scale (float): divisor for dt (1000 for Wan21, 1 for HY15).

    Returns:
        Tensor: the target latent at ``t_next``.
    """
    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    dt = (t - t_next) / timestep_scale
    # Reshape dt for broadcasting: [B, F] -> [B, F, 1, 1, 1]
    while dt.dim() < v_cfg.dim():
        dt = dt.unsqueeze(-1)
    return latent_t - dt * v_cfg


def consistency_loss(
    cm_pred_t: Tensor,
    cm_pred_t_next: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Consistency distillation loss: MSE between student(t) and EMA(t_next).

    Args:
        cm_pred_t (Tensor): student model's consistency prediction at ``t``.
        cm_pred_t_next (Tensor): EMA model's consistency prediction at ``t_next``
            (detached).
        reduction (str): ``"mean"`` or ``"none"``.

    Returns:
        Tensor: scalar loss when ``reduction="mean"``, else the per-element loss.
    """
    loss = (cm_pred_t.float() - cm_pred_t_next.float()).pow(2)
    if reduction == "mean":
        return loss.mean()
    return loss
