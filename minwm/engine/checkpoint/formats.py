"""Checkpoint format detection and model-weight selection.

This is the Checkpointer's ``_load_file`` layer (detectron2 shape): sniff a
checkpoint's on-disk format, then read model weights out of the single-file
formats into a normalized, bare-model state dict.

It also owns the weight-selection primitives shared across training, inference,
and export — unwrapping ``model`` / ``generator`` / ``generator_ema`` payloads
and stripping FSDP / compile wrapper prefixes so weights are keyed exactly as a
bare model expects.

Sharded DCP directories are loaded by the Checkpointer itself (via a DCP
``StorageReader`` into live modules). Weight-only initialization additionally
accepts torch ``.pt`` / ``.safetensors`` files and Diffusers model directories.
"""

import json
from typing import Any, Callable

from .storage import Storage, join

DCP_DIR = "dcp"
TORCH = "torch"
SAFETENSORS = "safetensors"
DIFFUSERS = "diffusers"
UNKNOWN = "unknown"

_TORCH_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt")
_WRAPPER_PREFIXES = ("model._fsdp_wrapped_module.", "model.", "_orig_mod.")
_GENERATOR_KEYS = ("generator_ema", "generator")
_DIFFUSERS_WEIGHT_FILES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
)
_DIFFUSERS_INDEX_FILES = tuple(f"{name}.index.json" for name in _DIFFUSERS_WEIGHT_FILES)


def strip_wrapper_prefix(key: str) -> str:
    """Drop nested model, FSDP, and compile wrapper prefixes from a key."""
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def normalize_state_dict(
    weights: dict[str, Any],
    *,
    prefixes: tuple[str, ...] = _WRAPPER_PREFIXES,
    recursive: bool = True,
) -> dict[str, Any]:
    """Return model weights with configured wrapper prefixes removed.

    Args:
        weights (dict[str, Any]): state dict to normalize.
        prefixes (tuple[str, ...]): wrapper prefixes recognized by this format.
        recursive (bool): repeatedly remove nested wrappers when true.

    Returns:
        dict[str, Any]: normalized state dict.
    """
    normalized = {}
    for key, value in weights.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = recursive
                    break
        normalized[key] = value
    return normalized


def select_state_dict(state: dict, key: str = "auto", prefer_ema: bool = False) -> dict:
    """Select and normalize model weights from a raw or wrapped checkpoint.

    Args:
        state (dict): object loaded from a checkpoint.
        key (str): explicit wrapper key, or ``"auto"`` to inspect known keys.
        prefer_ema (bool): prefer ``generator_ema`` over ``generator`` in auto mode.

    Returns:
        dict: normalized bare-model state dict.

    Raises:
        KeyError: if an explicitly requested key is absent.
    """
    if key != "auto":
        if key not in state:
            raise KeyError(f"checkpoint has no key {key!r}; available keys: {list(state.keys())}")
        weights = state[key]
    else:
        candidates = (
            ("generator_ema", "generator", "model")
            if prefer_ema
            else ("generator", "generator_ema", "model")
        )
        weights = next((state[name] for name in candidates if name in state), state)
    return normalize_state_dict(weights)


def select_training_pretrained_state(state: dict[str, Any]) -> dict[str, Any]:
    """Select weights using the trainer's backwards-compatible checkpoint policy.

    Native ``{"model": ...}`` and raw state dicts retain object identity and key
    names. Legacy generator wrappers prefer EMA weights and have wrapper prefixes
    removed, matching the trainer's established loading contract.
    """
    if "model" in state:
        return state["model"]
    for key in _GENERATOR_KEYS:
        if key in state:
            return normalize_state_dict(
                state[key],
                prefixes=("model._fsdp_wrapped_module.", "model."),
                recursive=False,
            )
    return state


def detect_format(path: str, storage: Storage) -> str:
    """Classify the on-disk format of a checkpoint ``path``.

    File formats are decided by suffix; directory formats by a sentinel entry
    (DCP writes a ``.metadata`` file; diffusers writes a ``config.json``), probed
    through the storage backend so it works for local and remote paths alike.

    Args:
        path (str): file or directory path/URL to classify.
        storage (Storage): backend used to probe directory sentinels.

    Returns:
        str: one of :data:`TORCH`, :data:`SAFETENSORS`, :data:`DCP_DIR`,
        :data:`DIFFUSERS`, or :data:`UNKNOWN`.
    """
    text = str(path)
    if text.endswith(".safetensors"):
        return SAFETENSORS
    if text.endswith(_TORCH_SUFFIXES):
        return TORCH
    if storage.exists(join(text, ".metadata")):
        return DCP_DIR
    if storage.exists(join(text, "config.json")):
        return DIFFUSERS
    return UNKNOWN


def _read_weight_file(path: str, storage: Storage) -> dict[str, Any]:
    """Read one torch or safetensors weight file without selecting a wrapper."""
    fmt = detect_format(path, storage)
    local = storage.get_local_path(path)
    if fmt == SAFETENSORS:
        from safetensors.torch import load_file

        return load_file(local, device="cpu")
    if fmt in (TORCH, UNKNOWN):
        import torch

        return torch.load(local, map_location="cpu", weights_only=False)
    raise ValueError(f"expected a weight file at {path!r}, found format {fmt!r}")


def _read_diffusers_state(path: str, storage: Storage) -> dict[str, Any]:
    """Read the model state from an unsharded or indexed Diffusers directory."""
    for filename in _DIFFUSERS_WEIGHT_FILES:
        weight_path = join(path, filename)
        if storage.exists(weight_path):
            return _read_weight_file(weight_path, storage)

    for filename in _DIFFUSERS_INDEX_FILES:
        index_path = join(path, filename)
        if not storage.exists(index_path):
            continue
        index = json.loads(storage.read_text(index_path))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Diffusers index {index_path!r} has no non-empty 'weight_map'")

        weights: dict[str, Any] = {}
        shard_names = list(dict.fromkeys(weight_map.values()))
        if not all(isinstance(name, str) and name for name in shard_names):
            raise ValueError(f"Diffusers index {index_path!r} contains an invalid shard name")
        for shard_name in shard_names:
            shard_path = join(path, shard_name)
            if not storage.exists(shard_path):
                raise FileNotFoundError(
                    f"Diffusers index {index_path!r} references missing shard {shard_name!r}"
                )
            shard = _read_weight_file(shard_path, storage)
            duplicates = weights.keys() & shard.keys()
            if duplicates:
                first = next(iter(duplicates))
                raise ValueError(f"Diffusers shards contain duplicate tensor key {first!r}")
            weights.update(shard)
        return weights

    expected = ", ".join((*_DIFFUSERS_WEIGHT_FILES, *_DIFFUSERS_INDEX_FILES))
    raise FileNotFoundError(
        f"Diffusers directory {path!r} has config.json but no supported model weights; "
        f"expected one of: {expected}"
    )


def read_model_state(
    path: str,
    storage: Storage,
    *,
    select: Callable[[dict], dict] | None = None,
    key: str = "auto",
    prefer_ema: bool = False,
) -> dict[str, Any]:
    """Read model weights from a checkpoint into a bare state dict.

    Handles torch ``.pt`` (and any unrecognized single file, loaded with
    ``torch.load``), ``safetensors`` files, and Diffusers model directories with
    either one standard weight file or an indexed set of shards. Sharded DCP
    directories remain the Checkpointer's responsibility. Remote weight files
    are localized before loading. The loaded object is passed through ``select``
    (default :func:`select_state_dict`) to unwrap and normalize it.

    Args:
        path (str): checkpoint file or Diffusers model directory path/URL.
        storage (Storage): backend used to localize remote files.
        select (Callable[[dict], dict] | None): selection policy applied to the
            loaded object; ``None`` uses ``select_state_dict(obj, key, prefer_ema)``.
        key (str): explicit wrapper key or ``"auto"`` (default selector only).
        prefer_ema (bool): prefer ``generator_ema`` in auto mode (default selector).

    Returns:
        dict[str, Any]: normalized bare-model state dict.

    Raises:
        ValueError: if ``path`` is a sharded DCP directory or a malformed
            Diffusers directory.
    """
    fmt = detect_format(path, storage)
    if fmt == DCP_DIR:
        raise ValueError(
            f"cannot read model weights from {path!r}: sharded DCP directories "
            "must be loaded through Checkpointer"
        )
    obj = (
        _read_diffusers_state(path, storage)
        if fmt == DIFFUSERS
        else _read_weight_file(path, storage)
    )
    if select is not None:
        return select(obj)
    return select_state_dict(obj, key=key, prefer_ema=prefer_ema)
