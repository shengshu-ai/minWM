"""Bidirectional SFT recipe."""

from ._flow_base import SingleForwardFlowRecipe


class BiSFTRecipe(SingleForwardFlowRecipe):
    """Bidirectional SFT: single full-clip flow-matching forward."""
