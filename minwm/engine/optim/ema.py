"""EMA (Exponential Moving Average) utilities.

Pure parameter-level operations, no model/framework coupling.
"""

from typing import Iterable

import torch
from torch import Tensor


@torch.no_grad()
def update_ema_params(
    ema_params: Iterable[Tensor],
    model_params: Iterable[Tensor],
    decay: float,
) -> None:
    """Update EMA parameters in-place.

    p_ema = decay * p_ema + (1 - decay) * p_model

    Args:
        ema_params: iterable of EMA parameter tensors
        model_params: iterable of model parameter tensors (same order)
        decay: EMA decay factor, typically 0.999 or 0.9999
    """
    for p_ema, p_model in zip(ema_params, model_params):
        p_ema.mul_(decay).add_(p_model.detach(), alpha=1.0 - decay)


@torch.no_grad()
def copy_params(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> None:
    """Copy all parameters from src to dst (in-place).

    Useful for initializing EMA from model or restoring model from EMA. Operates
    on the parameters directly (not ``.data``) so it stays correct under FSDP2,
    where both sides are equally-sharded ``DTensor``\\s — touching ``.data`` would
    unwrap one side to a local ``Tensor`` and break the distributed ``copy_``.
    """
    for p_dst, p_src in zip(dst_module.parameters(), src_module.parameters()):
        p_dst.copy_(p_src)
