"""Checkpoint I/O for minWM: storage backends and the Checkpointer.

Public surface:
    - :class:`Storage` / :func:`get_storage` / :func:`is_remote` — the storage
      backend that resolves local vs object-store from a ``save_dir`` URL scheme.
    - :class:`Checkpointer` — one owner for saving/loading (DCP or single-file).
    - :func:`load_model_weights` — weights-only model initialization.

``Checkpointer`` / ``load_model_weights`` pull in ``torch``, so they are exposed
lazily (PEP 562) to keep ``import minwm.engine.checkpoint`` free of heavy deps.
"""

from typing import Any

from .storage import Storage, get_storage, is_remote

__all__ = ["Storage", "get_storage", "is_remote", "Checkpointer", "load_model_weights"]

_LAZY = {"Checkpointer", "load_model_weights"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import checkpointer

        return getattr(checkpointer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
