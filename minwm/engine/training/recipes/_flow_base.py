"""Shared implementation for single-forward flow-matching recipes."""

from typing import Any

import torch
import torch.nn as nn

from minwm.engine.events import record_time
from minwm.engine.recipe_base import Recipe
from minwm.sampling.schedulers import FlowMatchingScheduler

from ..losses import BasicLoss


class FlowMatchingRecipeBase(Recipe):
    """Base for flow-matching recipes; builds the shared scheduler.

    Subclasses get ``self.scheduler`` and implement their own training step.

    Args:
        optimizer (dict, optional): optimizer config; see
            :meth:`~minwm.engine.recipe_base.Recipe.__init__`.
        adapter (ModelAdapter, optional): model call-convention bridge; owns the
            model-facing text dims / working dtype.
        max_grad_norm (float): gradient clipping norm.
        skip_grad_norm (float, optional): skip the step when the global grad norm
            exceeds this (all ranks in unison); ``None`` disables skipping.
        num_train_timesteps (int): scheduler timestep table size.
        timestep_shift (float): flow schedule shift parameter.
        schedule (str): ``"shifted"`` (Wan, default) or ``"linear"`` (HY).
        num_inference_steps (int, optional): inference-step table size; only the
            discrete stages (consistency distillation) set this, the rest leave
            it ``None`` for a full per-train-timestep table.
        sigma_min (float): lower end of the sigma schedule; set to ``0.0`` to match
            the legacy Wan stages (see :class:`FlowMatchingScheduler`).
        extra_one_step (bool): legacy ``extra_one_step`` sigma construction; set
            ``True`` to match the legacy Wan stages.
        batch_preprocessors (list, optional): the recipe's preprocessor chain.
    """

    def __init__(
        self,
        optimizer: dict | None = None,
        adapter: Any | None = None,
        max_grad_norm: float = 10.0,
        skip_grad_norm: float | None = None,
        num_train_timesteps: int = 1000,
        timestep_shift: float = 5.0,
        schedule: str = "shifted",
        num_inference_steps: int | None = None,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
        batch_preprocessors: list | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            adapter=adapter,
            max_grad_norm=max_grad_norm,
            skip_grad_norm=skip_grad_norm,
            batch_preprocessors=batch_preprocessors,
        )
        self.scheduler = FlowMatchingScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=timestep_shift,
            num_inference_steps=num_inference_steps,
            schedule=schedule,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
        )


class SingleForwardFlowRecipe(FlowMatchingRecipeBase):
    """Generic single-model flow-matching step.

    One optimization step: preprocess, assemble adapter-owned conditioning, run
    ``adapter.denoise``, compute the configured loss, then backward + step.

    The shared inputs are whatever subset of :attr:`SHARED_INPUT_KEYS` the
    preprocessor chain left on the batch (e.g. SFT leaves just ``noisy`` /
    ``timestep``; AR diffusion adds ``clean`` / ``aug_t``; ODE adds ``clean``).
    The adapter pulls them by keyword, so the step never hand-picks per stage.

    Args:
        loss (BasicLoss): the loss component, built from config (e.g.
            :class:`~minwm.engine.training.losses.FlowMatchingLoss` or
            :class:`~minwm.engine.training.losses.ODERegressionLoss`).
        optimizer (dict, optional): optimizer config; see
            :meth:`~minwm.engine.recipe_base.Recipe.__init__`.
        adapter (ModelAdapter, optional): model call-convention bridge.
        max_grad_norm (float): gradient clipping norm.
        skip_grad_norm (float, optional): skip the step when the global grad norm
            exceeds this (all ranks in unison); ``None`` disables skipping.
        num_train_timesteps (int): scheduler timestep table size.
        timestep_shift (float): flow schedule shift parameter.
        schedule (str): scheduler σ-table mode — ``"shifted"`` (Wan, default) or
            ``"linear"`` (HY).
        num_inference_steps (int, optional): inference-step table size.
        sigma_min (float): lower end of the sigma schedule; set to ``0.0`` to match
            the legacy Wan stages (see :class:`FlowMatchingScheduler`).
        extra_one_step (bool): legacy ``extra_one_step`` sigma construction; set
            ``True`` to match the legacy Wan stages.
        batch_preprocessors (list, optional): the recipe's preprocessor chain.

    Raises:
        ValueError: if no ``loss`` component is supplied.
    """

    SHARED_INPUT_KEYS = ("noisy", "timestep", "clean", "aug_t", "viewmats", "Ks")

    def __init__(
        self,
        loss: BasicLoss | None = None,
        optimizer: dict | None = None,
        adapter: Any | None = None,
        max_grad_norm: float = 10.0,
        skip_grad_norm: float | None = None,
        num_train_timesteps: int = 1000,
        timestep_shift: float = 5.0,
        schedule: str = "shifted",
        num_inference_steps: int | None = None,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
        batch_preprocessors: list | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            adapter=adapter,
            max_grad_norm=max_grad_norm,
            skip_grad_norm=skip_grad_norm,
            num_train_timesteps=num_train_timesteps,
            timestep_shift=timestep_shift,
            schedule=schedule,
            num_inference_steps=num_inference_steps,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
            batch_preprocessors=batch_preprocessors,
        )
        if loss is None:
            raise ValueError(
                "SingleForwardFlowRecipe requires a 'loss' component (a BasicLoss type-dict); "
                "none was supplied."
            )
        self.loss = loss

    def train_one_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
        auxiliary_models: dict[str, nn.Module] | None = None,
    ) -> dict[str, float]:
        """Run one single-optimizer flow-matching step.

        Args:
            model (nn.Module): the trainer-owned primary model.
            batch (Any): one dataloader batch.
            optimizers (dict[str, torch.optim.Optimizer]): steps ``optimizers["main"]``.
            step (int): current global step (unused).
            auxiliary_models (dict[str, nn.Module], optional): unused.

        Returns:
            dict[str, float]: ``{"loss": <scalar>}``.
        """
        device = next(model.parameters()).device
        batch = self.preprocess(batch, device)
        batch_size = batch["noisy"].shape[0]
        cond = self.adapter.conditioning(batch, batch_size, device)
        shared = {k: batch[k] for k in self.SHARED_INPUT_KEYS if batch.get(k) is not None}

        with record_time("forward"), self.autocast():
            pred = self.adapter.denoise(model, **shared, **cond)
            loss = self.loss(pred, batch, self.scheduler)

        self.optimizer_step(loss, optimizers["main"], model.parameters(), batch=batch)
        return {"loss": loss.item()}
