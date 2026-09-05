"""Config loader: load a config from a ``.py`` or ``.yaml`` file.

YAML files are loaded via OmegaConf and returned as a plain dict.
Python files are imported as a module; their module-level names become the
config namespace (detectron2 LazyConfig style). A ``.py`` config can build
arbitrary nested dicts with ``type`` keys programmatically.

A ``.py`` config may declare ``_base_`` (a path string or list of paths,
relative to the config's own directory) to inherit from shared base configs;
bases are deep-merged left-to-right and the leaf config's own keys win
(detectron2 / mmcv style). Configs without ``_base_`` are returned unchanged.
"""

import importlib.util
import os
import types
import uuid

from omegaconf import OmegaConf


def _load_yaml(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _load_py(path: str) -> dict:
    mod_name = f"minwm_cfg_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load python config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_spec = vars(module).get("_base_")
    cfg = {
        k: v
        for k, v in vars(module).items()
        if not k.startswith("_") and not isinstance(v, types.ModuleType)
    }
    if base_spec is None:
        return cfg
    base_paths = [base_spec] if isinstance(base_spec, str) else list(base_spec)
    config_dir = os.path.dirname(os.path.abspath(path))
    merged: dict = {}
    for bp in base_paths:
        bp = bp if os.path.isabs(bp) else os.path.join(config_dir, bp)
        merged = merge(merged, load(bp))
    return merge(merged, cfg)


def load(path: str) -> dict:
    """Load a config file (``.py``, ``.yaml`` or ``.yml``) into a dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return _load_py(path)
    if ext in (".yaml", ".yml"):
        return _load_yaml(path)
    raise ValueError(f"Unsupported config extension {ext!r}: {path}")


def merge(*cfgs: dict) -> dict:
    """Deep-merge config dicts left-to-right (later overrides earlier)."""
    merged = OmegaConf.merge(*[OmegaConf.create(c) for c in cfgs])
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotlist overrides (e.g. ``training.max_steps=10000``) to *cfg*.

    Args:
        cfg (dict): base config dict.
        overrides (list[str]): OmegaConf-style ``key=value`` strings.

    Returns:
        dict: merged config with overrides applied.
    """
    if not overrides:
        return cfg
    patch = OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True)
    return merge(cfg, patch)  # type: ignore[arg-type]
