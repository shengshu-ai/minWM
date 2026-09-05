"""Autoregressive ODE-regression recipe."""

from ._flow_base import SingleForwardFlowRecipe


class ARODERecipe(SingleForwardFlowRecipe):
    """Autoregressive ODE distillation: regress denoised trajectory targets."""
