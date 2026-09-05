"""Loss computation for flow-matching stages: the pure MSE op + config components.

``mse_loss`` is the stateless weighted/masked MSE shared by every
stage. A :class:`BasicLoss` wraps it into the config-buildable
``(pred, batch, scheduler) -> loss`` seam used by the flow recipes; configs pick
one via ``type``::

    recipe:
      type: BiSFTRecipe
      loss:
        type: ODERegressionLoss
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from minwm.sampling.schedulers import FlowMatchingScheduler

__all__ = ["mse_loss", "BasicLoss", "FlowMatchingLoss", "ODERegressionLoss"]


def mse_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Weighted, optionally masked MSE between a prediction and its target.

    Args:
        pred (Tensor): model prediction (velocity or ``x0``).
        target (Tensor): ground-truth target, same shape as ``pred``.
        weight (Tensor, optional): per-element or per-sample weight, broadcastable.
        mask (Tensor, optional): boolean mask, ``True`` = include in the loss.
        reduction (str): ``"mean"`` (scalar) or ``"none"`` (per-element).

    Returns:
        Tensor: scalar loss when ``reduction="mean"``, else the per-element loss.
    """
    loss = (pred.float() - target.float()).pow(2)
    if weight is not None:
        loss = loss * weight
    if mask is not None:
        mask = mask.to(device=loss.device, dtype=torch.bool)
        if mask.shape != loss.shape:
            # Left-align a leading-dims mask (e.g. per-frame ``[B, F]``) against
            # the ``[B, F, C, H, W]`` loss by padding trailing singleton dims,
            # then broadcast. ``torch.broadcast_to`` alone is right-aligned and
            # would try to match ``F`` against ``W``.
            if mask.ndim < loss.ndim:
                mask = mask.reshape(mask.shape + (1,) * (loss.ndim - mask.ndim))
            mask = torch.broadcast_to(mask, loss.shape)
        loss = loss[mask]
    if reduction == "mean":
        return loss.mean()
    return loss


class BasicLoss(ABC):
    """Map a model's flow prediction + the preprocessed batch to a scalar loss."""

    @abstractmethod
    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        """Compute the scalar training loss.

        Args:
            pred (Tensor): the model's flow (velocity) prediction ``[B, F, C, H, W]``.
            batch (dict): the preprocessed batch (``target``, ``weight``, ``noisy``,
                ``timestep``, ... — keys depend on the preprocessor chain).
            scheduler (FlowMatchingScheduler): the recipe's flow-match scheduler, for
                losses that need a flow → ``x0`` conversion.

        Returns:
            Tensor: scalar loss to backpropagate.
        """
        raise NotImplementedError


class FlowMatchingLoss(BasicLoss):
    """Weighted velocity MSE: ``mse_loss(pred, target, weight)``.

    Used by bidirectional SFT and AR diffusion teacher forcing.
    """

    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        return mse_loss(
            pred,
            batch["target"],
            weight=batch.get("weight"),
            mask=batch.get("loss_mask"),
        )


class ODERegressionLoss(BasicLoss):
    """ODE regression in ``x0`` space: convert the flow prediction to ``x0`` and
    regress against the trajectory's near-clean target, masking already-clean frames.
    """

    def __call__(self, pred: Tensor, batch: dict, scheduler: FlowMatchingScheduler) -> Tensor:
        timestep = batch["timestep"]
        x0_pred = scheduler.predict_x0(batch["noisy"], pred, timestep)
        mask = timestep != 0
        return mse_loss(x0_pred, batch["target"], mask=mask)
