"""ARDMDRecipe: Stage 3 Distribution Matching Distillation.

Equivalent to Wan21/wan_trainer/distillation.py + Wan21/model/dmd.py +
Wan21/model/camera_dmd.py, and (unified here) the HY
``ARHunyuanDMDDistillationPipeline``.

Four model copies, two optimizers, alternating per step:
    - generator  (``model``, trainable, causal)  -> backward-simulated student
    - real_score (frozen teacher, non-causal)     -> fixed score, CFG
    - fake_score (trainable critic, non-causal)   -> learns generator's dist
    - generator_ema (optional, frozen)             -> promoted inference weights

The generator step backward-simulates a fake video via the self-forcing
inference pipeline, then matches the real/fake score x0 distributions (KL
gradient). The critic step trains ``fake_score`` to denoise the generator's
(no-grad) output.

The critic/generator cadence is set by ``critic_schedule``:
    - ``"every_step"`` (Wan): critic every step, generator every
      ``dfake_gen_update_ratio`` steps (both may fire on the same step);
    - ``"alternating"`` (HY): generator on ``step % ratio == 0``, critic
      otherwise (mutually exclusive).

All model-family differences (KV-cache topology, cached AR forward, conditioning
stream, negative-prompt CFG) live in the :class:`~minwm.modeling.adapter.ModelAdapter`;
this recipe is model-agnostic — there is no per-model branch.

Batch contract:
    prompts / conditioning stream: read by the adapter (Wan text list, HY TI2V dict)
    clean_latent: Tensor [B, F, C, H, W]   (only its shape is used)
    viewmats:     Tensor [B, F, 4, 4]  (optional, for PRoPE)
    Ks:           Tensor [B, F, 3, 3]  (optional, for PRoPE)
    action:       Tensor [B, F]        (optional, discrete-action conditioning)
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from minwm.engine.events import record_time
from minwm.engine.optim.builder import build_optimizer
from minwm.engine.optim.ema import copy_params, update_ema_params
from minwm.engine.recipe_base import DEFAULT_OPTIMIZER_CFG
from minwm.sampling.rollouts import SelfForcingPipeline

from ._flow_base import FlowMatchingRecipeBase


class ARDMDRecipe(FlowMatchingRecipeBase):
    """Distribution Matching Distillation step (generator + critic).

    Args:
        optimizer (dict, optional): generator optimizer config for
            :func:`build_optimizer`. Defaults to AdamW with
            ``lr=2e-6, betas=(0.0, 0.999), weight_decay=0.0``.
        critic_optimizer (dict, optional): fake_score (critic) optimizer config,
            same shape and default as ``optimizer``.
        adapter (ModelAdapter, optional): model call-convention bridge. Defaults
            to :class:`~minwm.modeling.wan21.adapter.Wan21Adapter`.
        score_adapter (ModelAdapter, optional): adapter used for real/fake score
            networks. Defaults to ``adapter``; model families whose generator is
            causal but score networks are bidirectional can provide a second view.
        max_grad_norm (float): gradient clipping norm.
        num_train_timesteps (int): scheduler timestep table size.
        timestep_shift (float): flow schedule shift parameter.
        denoising_step_list (list[int]): generator denoising timesteps,
            most-noisy -> least-noisy.
        warp_denoising_step (bool): if True, map the raw ``denoising_step_list``
            values onto the (shifted) schedule timesteps before the self-rollout
            uses them.
        num_frame_per_block (int): frames per causal temporal block (must divide F).
        dfake_gen_update_ratio (int): generator update period.
        guidance_scale (float): real_score CFG scale.
        min_timestep (int): lower bound for score timestep sampling.
        max_timestep (int): upper bound for score timestep sampling.
        use_rollout_min_timestep (bool): use rollout's lower timestep bound when true.
        use_rollout_max_timestep (bool): use rollout's upper timestep bound when true.
        score_timestep_shift (bool): apply the scheduler shift after sampling.
        score_timestep_clip (tuple, optional): clamp shifted score timesteps.
        score_timestep_discrete (bool): draw integer rather than continuous values.
        cfg_base (str): CFG base prediction, ``"uncond"`` or legacy ``"cond"``.
        generator_ema_decay (float, optional): enable generator EMA updates.
        generator_ema_start_step (int): first step that updates generator EMA.
        critic_schedule (str): ``"every_step"`` (Wan) or ``"alternating"`` (HY).
        normalize_mode (str): KL-gradient normalization — ``"per_sample"`` (Wan),
            ``"global"`` (HY), or ``"none"``.
        context_noise (int): timestep used by the adapter's cache-refresh pass.
        sigma_min (float): lower end of the σ schedule. **Must match the score
            nets' training schedule** — real_score/fake_score seed from
            phase1/CD, which use ``0.0``; leave it at the bare-class
            ``0.003/1.002`` default and the DMD renoise noise level diverges from
            what the score nets learned. Set ``0.0`` in the stage-3 configs.
        extra_one_step (bool): legacy ``extra_one_step`` σ construction; set
            ``True`` to match the phase1/CD schedule (see ``sigma_min``).
        self_rollout_use_prope_cache (bool): enable PRoPE's separate KV cache in
            DMD self-rollout. Defaults False (matching ``dev-meta``, whose legacy
            self-rollout never allocated it); camera DMD configs opt in with
            ``True``. A no-op for non-PRoPE models.
    """

    def __init__(
        self,
        optimizer: dict | None = None,
        critic_optimizer: dict | None = None,
        adapter: Any | None = None,
        score_adapter: Any | None = None,
        max_grad_norm: float = 10.0,
        num_train_timesteps: int = 1000,
        timestep_shift: float = 5.0,
        denoising_step_list: list | None = None,
        warp_denoising_step: bool = False,
        num_frame_per_block: int = 1,
        dfake_gen_update_ratio: int = 5,
        guidance_scale: float = 5.0,
        min_timestep: int = 0,
        max_timestep: int = 1000,
        use_rollout_min_timestep: bool = True,
        use_rollout_max_timestep: bool = True,
        score_timestep_shift: bool = False,
        score_timestep_clip: tuple[float, float] | None = None,
        score_timestep_discrete: bool = False,
        cfg_base: str = "uncond",
        generator_ema_decay: float | None = None,
        generator_ema_start_step: int = 0,
        critic_schedule: str = "every_step",
        normalize_mode: str = "per_sample",
        context_noise: int = 0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
        self_rollout_use_prope_cache: bool = False,
        batch_preprocessors: list | None = None,
    ):
        super().__init__(
            optimizer=optimizer,
            adapter=adapter,
            max_grad_norm=max_grad_norm,
            num_train_timesteps=num_train_timesteps,
            timestep_shift=timestep_shift,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
            batch_preprocessors=batch_preprocessors,
        )
        self.critic_optimizer_cfg = critic_optimizer or dict(DEFAULT_OPTIMIZER_CFG)
        self.score_adapter = score_adapter or self.adapter
        self.denoising_step_list = denoising_step_list or [750, 500, 250]
        self.warp_denoising_step = warp_denoising_step
        self._warped_steps: list[float] | None = None
        self.num_frame_per_block = num_frame_per_block
        self.dfake_gen_update_ratio = dfake_gen_update_ratio
        self.guidance_scale = guidance_scale
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.use_rollout_min_timestep = use_rollout_min_timestep
        self.use_rollout_max_timestep = use_rollout_max_timestep
        self.score_timestep_shift = score_timestep_shift
        self.score_timestep_clip = score_timestep_clip
        self.score_timestep_discrete = score_timestep_discrete
        if cfg_base not in {"uncond", "cond"}:
            raise ValueError("cfg_base must be 'uncond' or 'cond'")
        self.cfg_base = cfg_base
        self.generator_ema_decay = generator_ema_decay
        self.generator_ema_start_step = generator_ema_start_step
        self._generator_ema_seeded = False
        self.critic_schedule = critic_schedule
        self.normalize_mode = normalize_mode
        self.context_noise = context_noise
        self.self_rollout_use_prope_cache = self_rollout_use_prope_cache
        self.pipeline: SelfForcingPipeline | None = None
        self._last_generator_loss = 0.0

    def build_optimizers(self, model: nn.Module) -> dict[str, torch.optim.Optimizer]:
        return {"generator": build_optimizer(model, self.optimizer_cfg)}

    def build_auxiliary_optimizers(
        self, auxiliary_models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        fake_score = auxiliary_models["fake_score"]
        return {"critic": build_optimizer(fake_score, self.critic_optimizer_cfg)}

    def _resolve_steps(self) -> list:
        """Return the rollout denoising steps, warped onto the schedule if requested.

        Warping maps each raw step value onto its (shifted) schedule timestep via
        ``cat(scheduler.timesteps, [0])[num_train_timesteps - raw]`` — the legacy
        ``warp_denoising_step``. Cached after the first call.

        Returns:
            list: the (possibly warped) denoising step values, most-noisy first.
        """
        if not self.warp_denoising_step:
            return self.denoising_step_list
        if self._warped_steps is None:
            ts = torch.cat((self.scheduler.timesteps.cpu(), torch.zeros(1, dtype=torch.float32)))
            raw = torch.tensor(self.denoising_step_list, dtype=torch.long)
            self._warped_steps = ts[self.scheduler.num_train_timesteps - raw].tolist()
        return self._warped_steps

    def _draw_noise(self, shape: tuple, device: torch.device) -> Tensor:
        """Draw rollout / renoise noise off the tracked (SP-synced, DP-diverse) stream."""
        from minwm.distributed import get_rng_states_tracker

        with get_rng_states_tracker().fork():
            return torch.randn(*shape, device=device, dtype=self.dtype)

    def train_one_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
        auxiliary_models: dict[str, nn.Module] | None = None,
    ) -> dict[str, float]:
        """One DMD step; cadence per ``critic_schedule``.

        Args:
            model (nn.Module): generator (trainable, causal).
            batch (dict): ``clean_latent`` [B,F,C,H,W] (shape only) plus whatever
                conditioning the adapter reads, optional ``viewmats``/``Ks``/``action``.
            optimizers (dict): ``"generator"`` + ``"critic"``.
            step (int): current global step.
            auxiliary_models (dict): ``"real_score"`` (frozen) + ``"fake_score"``.

        Returns:
            dict[str, float]: ``critic_loss`` and ``generator_loss``.
        """
        assert auxiliary_models is not None, "ARDMDRecipe requires auxiliary_models"
        real_score = auxiliary_models["real_score"]
        fake_score = auxiliary_models["fake_score"]
        generator_ema = auxiliary_models.get("generator_ema")
        if self.generator_ema_decay is not None and generator_ema is None:
            raise ValueError("generator_ema_decay requires auxiliary_models['generator_ema']")

        device = next(model.parameters()).device
        if self.batch_preprocessors_cfg:
            batch = self.preprocess(batch, device)
        if self.pipeline is None:
            self.pipeline = SelfForcingPipeline(
                generator=model,
                scheduler=self.scheduler,
                adapter=self.adapter,
                denoising_step_list=self._resolve_steps(),
                num_frame_per_block=self.num_frame_per_block,
                context_noise=self.context_noise,
                use_prope_cache=self.self_rollout_use_prope_cache,
            )

        clean = batch["clean_latent"].to(device=device, dtype=self.dtype)
        B, F, C, H, W = clean.shape
        shape = (B, F, C, H, W)

        viewmats: Tensor | None = None
        Ks: Tensor | None = None
        action: Tensor | None = None
        initial_latent = batch.get("initial_latent")
        if initial_latent is not None:
            initial_latent = initial_latent.to(device=device, dtype=self.dtype)
        if batch.get("viewmats") is not None:
            viewmats = batch["viewmats"].to(device=device, dtype=self.dtype)
            Ks = batch["Ks"].to(device=device, dtype=self.dtype)
        if batch.get("action") is not None:
            action = batch["action"].to(device=device)

        cond = self.adapter.conditioning(batch, B, device)
        uncond = self.adapter.null_conditioning(batch, B, device)

        args = (cond, uncond, viewmats, Ks, action, initial_latent, device, batch)
        gen_due = step % self.dfake_gen_update_ratio == 0
        metrics: dict[str, float] = {}

        # PARITY-BREAK: critic-schedule. The former HY path was generator-XOR-critic
        # ("alternating"); Wan runs critic every step plus a periodic generator
        # ("every_step"). Both preserved as a config knob, defaulting per model.
        if self.critic_schedule == "alternating":
            if gen_due:
                self._last_generator_loss = metrics["generator_loss"] = self._generator_step(
                    model,
                    real_score,
                    fake_score,
                    optimizers["generator"],
                    shape,
                    *args,
                )
                self._update_generator_ema(generator_ema, model, step)
            else:
                metrics["critic_loss"] = self._critic_step(
                    fake_score,
                    optimizers["critic"],
                    shape,
                    cond,
                    viewmats,
                    Ks,
                    action,
                    initial_latent,
                    device,
                    batch,
                )
            metrics.setdefault("critic_loss", 0.0)
            metrics.setdefault("generator_loss", self._last_generator_loss)
            return metrics

        # Generator first, then critic: the generator's KL gradient must read the
        # *pre-update* fake_score (the legacy Wan order, distillation.py:313-334).
        # Running the critic first would let the generator see a fake_score that
        # already took a step on this same batch, shifting the DMD min-max coupling.
        if gen_due:
            self._last_generator_loss = self._generator_step(
                model, real_score, fake_score, optimizers["generator"], shape, *args
            )
            self._update_generator_ema(generator_ema, model, step)
        metrics["generator_loss"] = self._last_generator_loss
        metrics["critic_loss"] = self._critic_step(
            fake_score,
            optimizers["critic"],
            shape,
            cond,
            viewmats,
            Ks,
            action,
            initial_latent,
            device,
            batch,
        )
        return metrics

    def _generator_step(
        self,
        generator: nn.Module,
        real_score: nn.Module,
        fake_score: nn.Module,
        optimizer: torch.optim.Optimizer,
        shape: tuple,
        cond: dict,
        uncond: dict,
        viewmats: Tensor | None,
        Ks: Tensor | None,
        action: Tensor | None,
        initial_latent: Tensor | None,
        device: torch.device,
        batch: Any | None = None,
    ) -> float:
        """Generator (KL) step: self-rollout -> renoise -> real/fake CFG -> DMD loss."""
        B, F, C, H, W = shape
        noise = self._draw_noise((B, F, C, H, W), device)

        with record_time("generator_forward"), self.autocast():
            fake_video, from_ts, to_ts = self.pipeline.inference_with_trajectory(
                noise, cond, viewmats, Ks, action, initial_latent
            )

        timestep = self._sample_timestep(B, F, to_ts, from_ts, device)
        prefix_frames = 0 if initial_latent is None else initial_latent.shape[1]
        if initial_latent is not None:
            timestep[:, :prefix_frames] = 0
        gen_noise = self._draw_noise(tuple(fake_video.shape), device)
        noisy = self.scheduler.add_noise(
            fake_video.flatten(0, 1), gen_noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (B, F))

        with torch.no_grad(), self.autocast():
            real_cond = self._score_x0(real_score, noisy, timestep, cond, viewmats, Ks, action)
            real_uncond = self._score_x0(real_score, noisy, timestep, uncond, viewmats, Ks, action)
            real_x0 = apply_cfg(real_cond, real_uncond, self.guidance_scale, base=self.cfg_base)
            fake_x0 = self._score_x0(fake_score, noisy, timestep, cond, viewmats, Ks, action)
            grad = compute_kl_gradient(
                fake_x0, real_x0, fake_video, normalize_mode=self.normalize_mode
            )

        loss = dmd_generator_loss(fake_video[:, prefix_frames:], grad[:, prefix_frames:])
        self.optimizer_step(
            loss,
            optimizer,
            generator.parameters(),
            backward_name="generator_backward",
            batch=batch,
        )
        return loss.item()

    def _critic_step(
        self,
        fake_score: nn.Module,
        optimizer: torch.optim.Optimizer,
        shape: tuple,
        cond: dict,
        viewmats: Tensor | None,
        Ks: Tensor | None,
        action: Tensor | None,
        initial_latent: Tensor | None,
        device: torch.device,
        batch: Any | None = None,
    ) -> float:
        """Critic step: no-grad self-rollout -> renoise -> fake_score flow-matching loss."""
        B, F, C, H, W = shape
        noise = self._draw_noise((B, F, C, H, W), device)

        with torch.no_grad(), self.autocast():
            fake_video, from_ts, to_ts = self.pipeline.inference_with_trajectory(
                noise, cond, viewmats, Ks, action, initial_latent
            )

        timestep = self._sample_timestep(B, F, to_ts, from_ts, device)
        prefix_frames = 0 if initial_latent is None else initial_latent.shape[1]
        if initial_latent is not None:
            timestep[:, :prefix_frames] = 0
        critic_noise = self._draw_noise(tuple(fake_video.shape), device)
        noisy = self.scheduler.add_noise(
            fake_video.flatten(0, 1), critic_noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (B, F))

        with record_time("critic_forward"), self.autocast():
            fake_x0 = self._score_x0(fake_score, noisy, timestep, cond, viewmats, Ks, action)
            # x0 -> flow: v = (x_t - x0) / sigma. Unified critic-flow derivation for
            # both families (the former HY path used the model's raw flow output).
            sigma = self.scheduler.sigma_for(timestep).reshape(B, F, 1, 1, 1)
            flow_pred = (noisy.float() - fake_x0.float()) / sigma.float().clamp(min=1e-8)
            loss = dmd_critic_loss(
                flow_pred[:, prefix_frames:],
                critic_noise[:, prefix_frames:],
                fake_video[:, prefix_frames:],
            )

        self.optimizer_step(
            loss,
            optimizer,
            fake_score.parameters(),
            backward_name="critic_backward",
            batch=batch,
        )
        return loss.item()

    def _score_x0(
        self,
        net: nn.Module,
        noisy: Tensor,
        timestep: Tensor,
        cond: dict,
        viewmats: Tensor | None,
        Ks: Tensor | None,
        action: Tensor | None,
    ) -> Tensor:
        """Run a score net (plain denoiser), return x0 prediction [B,F,C,H,W]."""
        shared: dict[str, Any] = {"viewmats": viewmats, "Ks": Ks}
        # ``cond`` already carries ``action`` on the real path (adapter.conditioning);
        # only thread it here when it isn't already there (the zero-text mock fallback
        # returns just ``context``), else the ``**cond``/``**shared`` splat collides.
        if action is not None and "action" not in cond:
            shared["action"] = action
        flow_pred = self.score_adapter.denoise(
            net, noisy=noisy, timestep=timestep, **cond, **shared
        )
        return self.scheduler.predict_x0(noisy, flow_pred, timestep)

    def _update_generator_ema(
        self, generator_ema: nn.Module | None, generator: nn.Module, step: int
    ) -> None:
        if generator_ema is None or self.generator_ema_decay is None:
            return
        # Seed the EMA from the live generator before the first lerp. The EMA is
        # a trainer-owned auxiliary_model constructed from config (fresh/random
        # weights), so without this seeding ``update_ema_params`` would leave a
        # ``decay^n`` residue of those construction-time weights in the promoted
        # inference checkpoint. Mirrors ARCDRecipe's copy_params seeding, but
        # deferred to the first update so start_step > 0 still seeds correctly.
        if not self._generator_ema_seeded:
            copy_params(generator, generator_ema)
            self._generator_ema_seeded = True
            return
        if step >= self.generator_ema_start_step:
            update_ema_params(
                generator_ema.parameters(),
                generator.parameters(),
                self.generator_ema_decay,
            )

    def _sample_timestep(
        self,
        batch_size: int,
        num_frames: int,
        min_ts: int,
        max_ts: int,
        device: torch.device,
    ) -> Tensor:
        """Sample one timestep value per sample (shared across frames).

        Sampled and returned in fp32: at t≈1000 the bf16 grid spacing is ~4-8, so
        drawing ``uniform_`` directly into a bf16 tensor would quantise the renoise
        timestep onto that coarse grid (and bias it via the rounding). The timestep
        only feeds ``sigma_for`` (an argmin lookup that benefits from fp32) and
        ``adapter.denoise`` (which casts to the model dtype at its own boundary),
        so keep it fp32 here.
        """
        lo = float(min_ts) if self.use_rollout_min_timestep else float(self.min_timestep)
        hi = float(max_ts) if self.use_rollout_max_timestep else float(self.max_timestep)
        if hi <= lo:
            hi = lo + 1.0
        # Score timesteps are part of the sample-level stochastic state just like
        # rollout noise: sequence-parallel ranks must draw the same value while
        # data-parallel ranks remain independent.
        from minwm.distributed import get_rng_states_tracker

        with get_rng_states_tracker().fork():
            if self.score_timestep_discrete:
                # The float guard above is not enough for randint: a sub-unit
                # window (e.g. lo=999.4, hi=999.6) truncates to lo_i == hi_i and
                # torch.randint raises "low >= high". Guard in the integer domain.
                lo_i, hi_i = int(lo), int(hi)
                if hi_i <= lo_i:
                    hi_i = lo_i + 1
                t = torch.randint(lo_i, hi_i, (batch_size, 1), device=device).float()
            else:
                t = torch.empty(batch_size, 1, device=device, dtype=torch.float32).uniform_(lo, hi)
        if self.score_timestep_shift and self.scheduler.shift != 1.0:
            ratio = t / self.scheduler.num_train_timesteps
            t = (
                self.scheduler.shift
                * ratio
                / (1.0 + (self.scheduler.shift - 1.0) * ratio)
                * self.scheduler.num_train_timesteps
            )
        if self.score_timestep_clip is not None:
            t = t.clamp(*self.score_timestep_clip)
        return t.expand(batch_size, num_frames).contiguous()


def apply_cfg(
    v_cond: Tensor,
    v_uncond: Tensor,
    guidance_scale: float,
    base: str = "uncond",
) -> Tensor:
    """Classifier-Free Guidance with an explicit base prediction.

    Args:
        v_cond (Tensor): conditional prediction.
        v_uncond (Tensor): unconditional prediction.
        guidance_scale (float): CFG scale factor.
        base (str): base prediction, ``"uncond"`` (standard) or ``"cond"``
            (legacy Wan21 DMD convention).

    Returns:
        Tensor: the guided prediction.
    """
    if base == "uncond":
        return v_uncond + guidance_scale * (v_cond - v_uncond)
    if base == "cond":
        return v_cond + guidance_scale * (v_cond - v_uncond)
    raise ValueError(f"Unknown CFG base: {base}")


def compute_kl_gradient(
    fake_x0: Tensor,
    real_x0: Tensor,
    generator_output: Tensor,
    normalize_mode: str = "global",
) -> Tensor:
    """Compute KL gradient for DMD (eq. 7 in https://arxiv.org/abs/2311.18828).

    ``grad = fake_x0 - real_x0``, optionally normalized by
    ``|generator_output - real_x0|``.

    Args:
        fake_x0 (Tensor): fake score's x0 prediction (after CFG if applicable).
        real_x0 (Tensor): real score's x0 prediction (after CFG).
        generator_output (Tensor): the generator's estimated clean output (for
            normalization).
        normalize_mode (str): ``"global"`` (mean over all dims except batch,
            HY15), ``"per_sample"`` (mean over spatial dims, keep batch, Wan21),
            or ``"none"`` (no normalization).

    Returns:
        Tensor: the KL gradient (NaNs replaced by zeros).

    Raises:
        ValueError: if ``normalize_mode`` is not a recognised value.
    """
    grad = fake_x0 - real_x0

    if normalize_mode == "none":
        pass
    elif normalize_mode == "global":
        p_real = generator_output - real_x0
        normalizer = torch.abs(p_real).mean()
        grad = grad / normalizer.clamp(min=1e-8)
    elif normalize_mode == "per_sample":
        p_real = generator_output - real_x0
        normalizer = torch.abs(p_real).mean(dim=list(range(1, p_real.dim())), keepdim=True)
        grad = grad / normalizer.clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    return torch.nan_to_num(grad)


def dmd_generator_loss(
    generator_output: Tensor,
    kl_gradient: Tensor,
    loss_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """DMD generator loss: ``0.5 * MSE(x, x - grad)``.

    The gradient only flows through ``generator_output``; the target
    (``generator_output - kl_gradient``) is detached.

    Args:
        generator_output (Tensor): generator's clean prediction [B, F, C, H, W].
        kl_gradient (Tensor): KL gradient from :func:`compute_kl_gradient`
            (detached).
        loss_dtype (torch.dtype): dtype for loss computation (default float32).

    Returns:
        Tensor: scalar generator loss.
    """
    x = generator_output.to(loss_dtype)
    target = (generator_output - kl_gradient).detach().to(loss_dtype)
    return 0.5 * torch.nn.functional.mse_loss(x, target, reduction="mean")


def dmd_critic_loss(
    critic_flow_pred: Tensor,
    noise: Tensor,
    clean: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    """Critic denoising loss: ``MSE(flow_pred, noise - clean)``.

    Equivalent to ``mse_loss(pred, noise - clean)``.

    Args:
        critic_flow_pred (Tensor): critic's flow prediction.
        noise (Tensor): the noise added to the clean sample.
        clean (Tensor): the clean sample.
        weight (Tensor, optional): per-element weight.

    Returns:
        Tensor: scalar critic loss.
    """
    target = noise - clean
    loss = (critic_flow_pred.float() - target.float()).pow(2)
    if weight is not None:
        loss = loss * weight
    return loss.mean()
