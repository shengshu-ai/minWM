"""Public model objects used by short ``type`` config names.

Exports stay lazy so resolving one config object does not import both model
families and all of their optional dependencies.
"""

import importlib
from typing import Any

from .build import build_model

_EXPORTS = {
    "ARHunyuanVideo_1_5_DiffusionTransformer": ".hy15.causal",
    "HunyuanVideo_1_5_DiffusionTransformer": ".hy15.model",
    "HYAdapter": ".hy15.adapter",
    "TextEncoder": ".hy15.encoders.text_encoder",
    "VisionEncoder": ".hy15.encoders.vision_encoder",
    "AutoencoderKLConv3D": ".hy15.vae",
    "Wan21Model": ".wan21.model",
    "CausalWan21Model": ".wan21.causal",
    "Wan21Adapter": ".wan21.adapter",
    "Wan21TextEncoder": ".wan21.text_encoder",
    "Wan21VAE": ".wan21.vae",
}

__all__ = ["build_model", *_EXPORTS]


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(submodule, __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)
