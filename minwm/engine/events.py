"""A minimal global scalar bus, adapted from detectron2's ``EventStorage``.

A :class:`Recipe` runs one optimization step and returns its loss dict, but it
sometimes wants to surface *extra* scalars (per-component losses, grad norms,
schedule values) without growing ``train_one_step``'s return contract. The
:class:`EventStorage` is a process-global, context-managed sink for exactly
that: anywhere inside the active context, ``get_event_storage().put_scalar(...)``
records a value the trainer can later read back via :meth:`EventStorage.latest`
and hand to the metrics processor.

Usage::

    with EventStorage(start_iter=0) as storage:
        storage.put_scalar("loss", 0.5)
        ...
        metrics = storage.latest()

The active storage is tracked on a module-global stack so the trainer pushes
once around its loop and any nested code reaches it without plumbing.

A :func:`record_time` context manager rides on the same stack: a Recipe wraps
its forward / backward in ``with record_time("forward"):`` and the elapsed
milliseconds land on the active storage as ``time/forward(ms)``. Outside any
context (e.g. a Recipe unit test calling ``train_one_step`` directly) it is a
no-op, so instrumentation never forces the trainer's storage onto callers.
"""

import time
from contextlib import contextmanager, nullcontext

import torch

_CURRENT: list["EventStorage"] = []


def get_event_storage() -> "EventStorage":
    """Return the innermost active :class:`EventStorage`.

    Returns:
        EventStorage: the storage on top of the context stack.

    Raises:
        AssertionError: if called outside any :class:`EventStorage` context.
    """
    assert _CURRENT, "get_event_storage() called outside an EventStorage context"
    return _CURRENT[-1]


def has_event_storage() -> bool:
    """True if an :class:`EventStorage` context is active.

    Lets optional instrumentation record only when the trainer provides a storage.
    """
    return bool(_CURRENT)


@contextmanager
def record_time(name: str):
    """Time the wrapped block, recording ``time/{name}(ms)`` on the active storage.

    On CUDA the block is timed with :class:`torch.cuda.Event` plus a device
    synchronize, so the value is real kernel time rather than async launch
    overhead (the synchronize adds a short stall — acceptable for the per-step
    forward/backward of a video model, where the step already dwarfs it). On CPU
    it falls back to a wall-clock timer. With no active :class:`EventStorage`
    (e.g. a Recipe ``train_one_step`` called directly in a unit test) the block
    still runs but nothing is recorded — instrumentation never forces the
    trainer's storage onto callers.

    Args:
        name (str): metric suffix; recorded under ``time/{name}(ms)``.
    """
    if not _CURRENT:
        yield
        return
    storage = _CURRENT[-1]
    if torch.cuda.is_available():
        nvtx_range = torch.cuda.nvtx.range(name) if storage.nvtx_enabled else nullcontext()
        with nvtx_range:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            try:
                yield
            finally:
                end.record()
                torch.cuda.synchronize()
                storage.put_scalar(f"time/{name}(ms)", start.elapsed_time(end))
    else:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            storage.put_scalar(f"time/{name}(ms)", 1000.0 * (time.perf_counter() - t0))


class EventStorage:
    """Process-global, context-managed store of the latest scalar per name.

    Only the most recent value of each scalar is retained (no history / no
    smoothing in v1); the trainer flushes :meth:`latest` to the metrics
    processor at its log interval and the values roll forward until overwritten.

    Args:
        start_iter (int): the iteration the storage opens at. The trainer sets
            this from ``self.step`` after checkpoint load so resumed runs report
            correct iteration numbers.
    """

    def __init__(self, start_iter: int = 0) -> None:
        self._iter = start_iter
        self._latest: dict[str, float] = {}
        self.nvtx_enabled = False

    @property
    def iter(self) -> int:
        return self._iter

    @iter.setter
    def iter(self, value: int) -> None:
        self._iter = int(value)

    def put_scalar(self, name: str, value: float) -> None:
        """Record (overwrite) the latest value of one scalar.

        Args:
            name (str): metric key.
            value (float): metric value; cast to ``float``.
        """
        self._latest[name] = float(value)

    def put_scalars(self, **kwargs: float) -> None:
        """Record several scalars at once via :meth:`put_scalar`.

        Args:
            **kwargs (float): ``name=value`` pairs to record.
        """
        for name, value in kwargs.items():
            self.put_scalar(name, value)

    def latest(self) -> dict[str, float]:
        """Return a copy of the latest value of every recorded scalar."""
        return dict(self._latest)

    def clear(self) -> None:
        """Drop all recorded scalars (the iteration counter is unaffected)."""
        self._latest.clear()

    def __enter__(self) -> "EventStorage":
        _CURRENT.append(self)
        return self

    def __exit__(self, *exc) -> None:
        assert _CURRENT[-1] is self, "EventStorage context stack corrupted"
        _CURRENT.pop()
