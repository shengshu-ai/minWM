"""Input preprocessors for generation workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class InferenceRuntime:
    """Runtime values shared by inference preprocessors.

    ``components`` carries recipe-built model objects (e.g. ``vae`` /
    ``text_encoder`` / ``vision_encoder``) that a preprocessor needs at call time
    but cannot be expressed in a static config spec. Preprocessors that need a
    component look it up by name; the runtime stays family-agnostic.
    """

    device: torch.device
    dtype: torch.dtype
    inference_cfg: dict[str, Any]
    components: dict[str, Any] = field(default_factory=dict)


class InferenceInputPreprocessor(ABC):
    """One stage in the inference input-preparation chain."""

    @abstractmethod
    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        """Transform one generation batch and return it."""
        raise NotImplementedError


class BatchPreprocessor(ABC):
    """One stage in a recipe's data-prep chain: ``batch -> batch``."""

    @abstractmethod
    def __call__(self, batch: dict, device: torch.device) -> dict:
        """Transform one batch and return it.

        Args:
            batch (dict): the working batch dict.
            device (torch.device): target device for moved/created tensors.

        Returns:
            dict: the same batch dict, extended with this stage's outputs.
        """
        raise NotImplementedError
