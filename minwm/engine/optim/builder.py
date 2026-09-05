"""Optimizer construction helpers.

Centralizes the trainable-parameter selection + optimizer wiring that every
single-optimizer recipe repeated inline. A recipe's ``build_optimizers`` becomes
a one-liner over :func:`build_optimizer`; multi-optimizer recipes (e.g. DMD's
generator + critic) call it once per group.

The optimizer class is config-selectable: the ``cfg`` dict's ``name`` key is a
short framework class name such as ``"Muon"``, or an explicit external path
such as the default ``"torch.optim:AdamW"``. The remaining keys are passed
through as constructor kwargs. The class can't live under a ``type`` key in the
recipe node because the optimizer needs the module's parameters injected at build
time, which only exist after the model is constructed — so the lazy config engine
must leave the node as a plain dict rather than instantiating it eagerly.
"""

import torch
from torch import Tensor, nn

from minwm.config.lazy import locate

_DEFAULT_OPTIMIZER = "torch.optim:AdamW"


def trainable_params(module: nn.Module) -> list[Tensor]:
    """Return the module's parameters that require gradients.

    Args:
        module (nn.Module): the module whose parameters are optimized.

    Returns:
        list[Tensor]: parameters with ``requires_grad=True``, in module order.
    """
    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(module: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build a config-selected optimizer over a module's trainable parameters.

    Args:
        module (nn.Module): the module to optimize.
        cfg (dict): optimizer config. The optional ``name`` key is a
            short framework name or ``"module.path:ClassName"`` string
            (default ``torch.optim:AdamW``);
            every other key is forwarded as a constructor kwarg (e.g. ``lr``,
            ``betas``, ``weight_decay``).

    Returns:
        torch.optim.Optimizer: optimizer over the module's trainable parameters.
    """
    cls = locate(cfg.get("name", _DEFAULT_OPTIMIZER))
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(trainable_params(module), **kwargs)
