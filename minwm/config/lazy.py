"""LazyConfig: ``type``-based lazy build for minwm configs.

Config dicts with a ``type`` key like::

    model:
      type: Wan21Model
      is_causal: true

are resolved by :func:`build` into ``Wan21Model(is_causal=True)``.

Use :func:`lazy` to defer instantiation (returns the class + kwargs dict
without calling ``__init__``).
"""

import importlib
from typing import Any

from omegaconf import OmegaConf

# Public facades searched for framework-provided short names. Each facade owns
# its exported names through ``__all__``; adding a new object therefore requires
# only one public export. Duplicate exports are rejected as ambiguous.
DEFAULT_MODULES = (
    "minwm.modeling",
    "minwm.engine.training.recipes",
    "minwm.engine.training.losses",
    "minwm.processors",
    "minwm.data",
    "minwm.engine.optim",
)


def locate(path: str) -> Any:
    """Resolve a config object by short name or explicit import path.

    Framework-provided objects should use a short public name such as
    ``"Wan21Model"``, resolved by scanning :data:`DEFAULT_MODULES` and matching
    against each facade's ``__all__``. Third-party or private objects keep using
    the unambiguous ``"module.path:Name"`` form, for example ``"torch.optim:AdamW"``.

    Args:
        path (str): a short public name (``"Wan21Model"``) exported by exactly one
            module in :data:`DEFAULT_MODULES`, or an explicit ``"module.path:Name"``
            import path.

    Returns:
        Any: the resolved object (class or function).

    Raises:
        ValueError: if ``path`` is not a non-empty string, is a malformed colon
            path (empty module or attribute), or is a dotted name with no colon
            (ambiguous between a short name and an import path).
        ImportError: if an explicit path's attribute is missing, if a short name
            is exported by more than one facade (ambiguous), if a facade
            advertises the name in ``__all__`` but cannot produce it, or if no
            importable facade exports the name.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"type must be a non-empty string, got {path!r}")

    if ":" in path:
        mod_path, attr = path.rsplit(":", 1)
        if not mod_path or not attr:
            raise ValueError(f"type must be 'Name' or 'module:Name', got {path!r}")
        mod = importlib.import_module(mod_path)
        try:
            return getattr(mod, attr)
        except AttributeError as exc:
            raise ImportError(f"{attr!r} not found in {mod_path!r}") from exc

    if "." in path:
        raise ValueError(f"type must be 'Name' or 'module:Name', got {path!r}")

    matches = []
    import_errors = []
    for mod_path in DEFAULT_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as exc:
            # A facade that fails to import (e.g. a missing optional dep) must
            # not poison resolution of names owned by a different facade. Defer
            # the error: surface it only if no other facade provides the name.
            import_errors.append((mod_path, exc))
            continue
        if path in getattr(mod, "__all__", ()):
            matches.append((mod_path, mod))

    if len(matches) > 1:
        modules = ", ".join(mod_path for mod_path, _ in matches)
        raise ImportError(f"ambiguous short type {path!r}; exported by: {modules}")
    if matches:
        mod_path, mod = matches[0]
        try:
            return getattr(mod, path)
        except (AttributeError, ImportError) as exc:
            # ``__all__`` advertises the name but the facade can't produce it
            # (broken lazy ``__getattr__``, stale export). Report cleanly.
            raise ImportError(
                f"{path!r} is exported by {mod_path!r} but could not be resolved"
            ) from exc

    if import_errors:
        broken = ", ".join(mod_path for mod_path, _ in import_errors)
        raise ImportError(
            f"{path!r} is not exported by any importable default module; "
            f"could not import: {broken}"
        ) from import_errors[0][1]
    raise ImportError(f"{path!r} is not exported by any default module")


def build(cfg: Any) -> Any:
    """Instantiate a config node that has a ``type`` key.

    Nested dicts with their own ``type`` are instantiated first (depth-first).
    Dicts without ``type`` are left as plain dicts.
    """
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict):
        return cfg
    if "type" not in cfg:
        return {k: build(v) for k, v in cfg.items()}

    cls = locate(cfg["type"])
    kwargs = {k: build(v) for k, v in cfg.items() if k != "type"}
    return cls(**kwargs)


def lazy(cfg: dict) -> tuple[type, dict]:
    """Return ``(cls, kwargs)`` without calling ``__init__``.

    Useful when the caller needs to wrap the class (e.g. FSDP) before init.
    """
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if "type" not in cfg:
        raise ValueError("cfg must have a 'type' key to use lazy()")
    cls = locate(cfg["type"])
    kwargs = {k: build(v) for k, v in cfg.items() if k != "type"}
    return cls, kwargs
