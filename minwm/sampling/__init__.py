"""Model-agnostic generative-process primitives shared by training and inference.

Two neutral pieces that sit between :mod:`minwm.modeling` and the
``training`` / ``inference`` orchestration layers, depending only downward
(``schedulers`` on nothing but torch; ``rollouts`` on ``schedulers`` +
``distributed``):

- :class:`FlowMatchingScheduler` (``schedulers``): the shifted flow-matching
  sigma schedule + noise / target ops, plus the stateless ``pred_x0_from_flow``.
- :class:`SelfForcingPipeline` (``rollouts``): the model-agnostic autoregressive
  denoising loop used by DMD backward-simulation and AR inference.
"""

from .rollouts import SelfForcingPipeline, sample_exit_step
from .schedulers import FlowMatchingScheduler, pred_x0_from_flow

__all__ = [
    "FlowMatchingScheduler",
    "pred_x0_from_flow",
    "SelfForcingPipeline",
    "sample_exit_step",
]
