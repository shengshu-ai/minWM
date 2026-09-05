"""Model components shared across model families.

Holds cross-family modeling pieces that aren't specific to one backbone, both
parameterized modules and stateless ops:

- :class:`~minwm.modeling.common.norm.RMSNorm` — the parameterized RMS norm
  reused by both the HY15 and Wan DiTs.
- ``sinusoidal_embedding`` — sinusoidal (timestep) positional embedding.
- ``prope_qkv`` / ``add_prope_parameters`` — the stateless PRoPE geometry and
  the Wan-side learnable-parameter registration helper.

Family-specific layers live under ``minwm.modeling.{family}.layers``.
"""

from .embedding import sinusoidal_embedding
from .norm import RMSNorm
from .prope import add_prope_parameters, prope_qkv

__all__ = ["RMSNorm", "sinusoidal_embedding", "add_prope_parameters", "prope_qkv"]
