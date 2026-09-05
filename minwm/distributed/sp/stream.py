"""Track the current CUDA stream cheaply.

``torch.cuda.current_stream()`` constructs a fresh ``Stream`` object on every
call, which is surprisingly expensive on the hot collective path. We patch
``torch.cuda.set_stream`` to remember the last-set stream and expose it via
:func:`current_stream`.

Hypothesis: nothing in this process calls ``torch._C._cuda_setStream`` from
C/C++ (i.e. bypassing the Python wrapper). If that ever changes, this cache
would go stale.
"""

from __future__ import annotations

import torch

_prev_set_stream = torch.cuda.set_stream
_current_stream: torch.cuda.Stream | None = None


def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
    global _current_stream
    _current_stream = stream
    if stream is not None:
        _prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream | None:
    """Return the cached current CUDA stream.

    Cheap drop-in replacement for ``torch.cuda.current_stream()``.
    """
    global _current_stream
    if _current_stream is None:
        _current_stream = torch.cuda.current_stream()
    return _current_stream


__all__ = ["current_stream"]
