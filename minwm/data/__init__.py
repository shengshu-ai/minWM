"""minwm data: datasets + dataloader factories.

The factory functions (:func:`build_dataset`, :func:`build_dataloader`,
:func:`cycle`) are imported eagerly — they only depend on torch + minwm config.
Dataset classes are re-exported lazily (via ``__getattr__``) so that importing
``minwm.data`` does not pull in optional deps like ``lmdb`` / ``scipy`` / ``PIL``.
"""

from typing import Any

# Eagerly import samplers subpackage so users can do:
#   from minwm.data.samplers import build_sampler, DPSPBatchSampler
from . import samplers  # noqa: F401
from .build import build_dataloader, build_dataset, cycle
from .collate import HYCollator

_DATASET_NAMES = {
    "TextDataset",
    "LatentLMDBDataset",
    "ODERegressionLMDBDataset",
    "ShardingLMDBDataset",
    "CameraLatentLMDBDataset",
    "CameraODERegressionLMDBDataset",
    "TextImagePairDataset",
    "MockCameraLatentDataset",
    "CameraPluckerDataset",
    "CausalODEDataset",
}

__all__ = ["build_dataset", "build_dataloader", "cycle", "HYCollator", *sorted(_DATASET_NAMES)]


def __getattr__(name: str) -> Any:
    if name in _DATASET_NAMES:
        from minwm.data import datasets

        return getattr(datasets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
