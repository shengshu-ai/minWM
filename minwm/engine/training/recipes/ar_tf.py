"""Autoregressive teacher-forcing recipe."""

from ._flow_base import SingleForwardFlowRecipe


class ARTFRecipe(SingleForwardFlowRecipe):
    """Autoregressive teacher forcing: single causal forward with clean context."""
