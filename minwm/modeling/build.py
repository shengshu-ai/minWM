"""Model build entry point.

Models are constructed from a config node with a ``type`` key containing a
framework class name (or an explicit ``'module.path:ClassName'`` for external
objects). Two construction modes:

- plain init: ``cls(**kwargs)``
- transformer-style pretrained load: if the cfg has a ``_from_pretrained``
  key, call ``cls.from_pretrained(path, **kwargs)`` instead.

Example cfg::

    model = {
        "type": "Wan21Model",
        "_from_pretrained": "./ckpts/Wan21/Action2V/bidirectional",
        "is_causal": True,
    }

    model = build_model(model)
"""

from typing import Any

from omegaconf import OmegaConf

from minwm.config.lazy import build, locate
from minwm.distributed.fsdp import resolve_dtype

_FROM_PRETRAINED = "_from_pretrained"


def build_model(cfg: dict) -> Any:
    """Build a model (typically ``nn.Module``) from a ``type`` config node.

    Nested ``type`` sub-configs (e.g. a sub-module passed as a kwarg) are
    resolved recursively via :func:`minwm.config.build`.
    """
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError(f"build_model expects a dict with a 'type' key, got {cfg!r}")

    cls = locate(cfg["type"])
    pretrained = cfg.get(_FROM_PRETRAINED)
    kwargs = {k: build(v) for k, v in cfg.items() if k not in ("type", _FROM_PRETRAINED)}

    if isinstance(kwargs.get("torch_dtype"), str):
        kwargs["torch_dtype"] = resolve_dtype(kwargs["torch_dtype"])

    if pretrained is not None:
        if not hasattr(cls, "from_pretrained"):
            raise TypeError(
                f"{cls.__name__} has no 'from_pretrained'; cannot honor " f"'{_FROM_PRETRAINED}'"
            )
        return cls.from_pretrained(pretrained, **kwargs)
    return cls(**kwargs)
