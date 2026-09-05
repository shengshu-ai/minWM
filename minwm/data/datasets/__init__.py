"""minwm dataset classes (lazily imported).

Heavy optional deps (``lmdb``, ``scipy``, ``PIL``) are imported only when the
corresponding dataset class is actually accessed. Importing this package — or
resolving an unrelated ``type`` path through it — never forces them, so a run
that only uses, say, ``TextDataset`` does not require ``lmdb`` to be installed.
"""

import importlib
from typing import Any

# class name -> submodule (relative to this package) that defines it
_EXPORTS = {
    "TextDataset": ".text",
    "LatentLMDBDataset": ".lmdb",
    "ODERegressionLMDBDataset": ".lmdb",
    "ShardingLMDBDataset": ".lmdb",
    "CameraLatentLMDBDataset": ".lmdb",
    "CameraODERegressionLMDBDataset": ".lmdb",
    "TextImagePairDataset": ".image",
    "MockCameraLatentDataset": ".mock",
    "CameraPluckerDataset": ".camera_pt",
    "CausalODEDataset": ".camera_pt",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(submodule, __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)
