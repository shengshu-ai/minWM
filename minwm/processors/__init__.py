"""Composable batch/input preprocessors, model-agnostic ``_cls``-buildable stages.

Two lifecycles share one flat package:

* **training** — the per-stage data-prep chain a
  :class:`~minwm.engine.recipe_base.Recipe` runs before the forward pass
  (device placement, flow-matching noise, clean-context aug, ODE sampling).
* **inference** — the input-preparation chain the generation workflow runs
  (initial latent, camera trajectory, conditioning).

They depend only on :mod:`minwm.sampling`, so they sit above neither training
nor inference — both consume them.
"""

from .base import BatchPreprocessor, InferenceInputPreprocessor, InferenceRuntime
from .camera import CameraTrajectory, make_camera_tensors, parse_trajectory
from .flow import CleanContextNoiseAug, FirstFrameConditioning, FlowNoise
from .hy import HYConditioning
from .latent import InitialLatent, LatentNoise, LatentToDevice
from .ode import ODETrajectorySample

__all__ = [
    "BatchPreprocessor",
    "InferenceRuntime",
    "InferenceInputPreprocessor",
    "LatentToDevice",
    "FlowNoise",
    "FirstFrameConditioning",
    "CleanContextNoiseAug",
    "ODETrajectorySample",
    "LatentNoise",
    "InitialLatent",
    "CameraTrajectory",
    "make_camera_tensors",
    "parse_trajectory",
    "HYConditioning",
]
