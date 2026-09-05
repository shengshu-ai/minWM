"""Named CPU+CUDA RNG-state tracker for reproducible, resumable noise streams.

A single global tracker (Megatron-style) holds a named dict of RNG states. Code
that must draw *SP-synced, DP-diverse* randomness wraps the draw in::

    with get_rng_states_tracker().fork():
        noise = torch.randn(...)

``fork()`` swaps the tracked state into the default CPU **and** CUDA generators,
runs the block, then saves the advanced state back and restores the caller's
original state. The tracked stream therefore advances *in isolation* — dropout
or any other RNG-consuming kernel between two draws no longer perturbs it, and a
rank that takes a divergent code path can't silently desync the shared stream.

Because the tracker is :class:`~torch.distributed.checkpoint.stateful.Stateful`
(keyed by ``dp_rank``), the noise stream is saved/restored with the checkpoint,
so a resumed run reproduces the same trajectory as an uninterrupted one.

Adapted from Megatron-LM / the ss-falcon ``distributed.random`` module; extended
here to track the CPU generator state alongside CUDA.
"""

import contextlib
import logging
import re
from typing import Any

import torch

DATA_PARALLEL_RNG_TRACKER_NAME = "data-parallel-rng"

logger = logging.getLogger(__name__)


def _get_cuda_rng_state(device: int | str | torch.device = "cuda") -> torch.Tensor:
    """Return the CUDA RNG state of ``device`` (wraps ``torch.cuda.get_rng_state``)."""
    return torch.cuda.random.get_rng_state(device=device)


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1) -> None:
    """Set the current GPU's CUDA RNG state without cloning ``new_state``.

    Adapted from ``torch.cuda.set_rng_state`` with one change: the input state is
    not cloned. Cloning caused major perf regressions in +4-GPU runs (states are
    set every ``fork()``), so we set the generator state in place.

    Args:
        new_state (torch.Tensor): the desired CUDA RNG state (ByteTensor).
        device (int): the GPU to set; ``-1`` means the current device.
    """
    from torch import _C
    from torch.cuda import _lazy_call
    from torch.cuda import device as device_ctx_manager

    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):

        def cb() -> None:
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb() -> None:
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def _strip_dp_suffix(text: str) -> str:
    """Strip the trailing ``-{dp_rank}`` from a checkpoint key.

    Uses an anchored ``-\\d+$`` match rather than ``str.rstrip`` so a stream name
    that itself ends in a digit (e.g. ``noise-stage2``) is not mangled — only the
    ``dp_rank`` suffix appended in :meth:`RNGStatesTracker.state_dict` is removed.
    """
    return re.sub(r"-\d+$", "", text)


class RNGStatesTracker:
    """Tracker for named CPU+CUDA RNG states.

    ``add(name, seed)`` snapshots a fresh state seeded from ``seed`` under
    ``name`` without disturbing the live generators. ``fork(name)`` then swaps
    that state in, runs the caller's block, and restores the original state on
    exit — advancing the named stream in isolation.

    Satisfies the ``Stateful`` protocol by duck typing (``state_dict`` /
    ``load_state_dict``), keyed by ``dp_rank`` so each replica's stream round-trips
    through a checkpoint independently.
    """

    def __init__(self) -> None:
        from minwm.distributed.parallel_dims import get_parallel_state

        self.dp_rank = get_parallel_state().dp_rank
        self.reset()

    def is_initialized(self) -> bool:
        """True once a state has been registered via :meth:`add` or :meth:`set_states`."""
        return self._is_initialized

    def reset(self) -> None:
        """Drop all tracked states (return to the empty, uninitialized tracker)."""
        self._is_initialized = False
        self.states_: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.seeds_: set[int] = set()
        self._initial_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_states(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return a shallow copy of the ``name -> (cpu_state, cuda_state)`` map."""
        return dict(self.states_)

    def set_states(self, states: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Replace the tracked states with ``states`` (no size/compat check)."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """Register a fresh CPU+CUDA state seeded from ``seed`` under ``name``.

        Snapshots the current generator states, seeds from ``seed``, records the
        resulting states, then restores the snapshot — so registering a stream
        never perturbs the live generators.

        Args:
            name (str): the stream name to register.
            seed (int): the seed the stream is initialized from.

        Raises:
            ValueError: if ``seed`` or ``name`` is already registered.
        """
        self._is_initialized = True
        if seed in self.seeds_:
            raise ValueError(f"seed {seed} already exists")
        self.seeds_.add(seed)
        if name in self.states_:
            raise ValueError(f"rng state {name} already exists")

        orig_cpu_state = torch.get_rng_state()
        has_cuda = torch.cuda.is_available()
        orig_cuda_state = _get_cuda_rng_state() if has_cuda else None

        torch.manual_seed(seed)  # seeds CPU + (if present) all CUDA devices
        cpu_state = torch.get_rng_state()
        cuda_state = _get_cuda_rng_state() if has_cuda else torch.empty(0, dtype=torch.uint8)
        self.states_[name] = (cpu_state, cuda_state)
        # Clone the reseed baseline so a later fork() replacing states_[name] (or
        # any in-place mutation) can never advance the snapshot reseed() restores.
        self._initial_states[name] = (cpu_state.clone(), cuda_state.clone())

        torch.set_rng_state(orig_cpu_state)
        if has_cuda:
            _set_cuda_rng_state(orig_cuda_state)

    def _snapshot_current(self, name: str) -> None:
        """Register ``name`` from the current generator state (auto-fork fallback).

        Used when a ``fork(name)`` is requested for a stream that was never
        explicitly :meth:`add`-ed — e.g. a script that never called ``set_seed``.
        The stream then continues from wherever the default generator is, matching
        the pre-tracker global-RNG behavior but now isolated and resettable.
        """
        self._is_initialized = True
        has_cuda = torch.cuda.is_available()
        cpu_state = torch.get_rng_state()
        cuda_state = _get_cuda_rng_state() if has_cuda else torch.empty(0, dtype=torch.uint8)
        self.states_[name] = (cpu_state, cuda_state)
        self._initial_states[name] = (cpu_state.clone(), cuda_state.clone())

    def reseed(self, name: str | None = None) -> None:
        """Reset tracked stream(s) back to the state captured when first registered.

        Discards any advance since :meth:`add` / first :meth:`fork`, so two runs
        draw an identical sequence (e.g. a run-to-run determinism floor). Does not
        touch the live generators.

        Args:
            name (str | None): the stream to reset, or ``None`` for all streams.
        """
        names = [name] if name is not None else list(self.states_.keys())
        for nm in names:
            initial = self._initial_states.get(nm)
            if initial is not None:
                self.states_[nm] = initial

    @contextlib.contextmanager
    def fork(self, name: str = DATA_PARALLEL_RNG_TRACKER_NAME):
        """Run the block with the ``name`` stream, restoring the caller's state after.

        Swaps the tracked CPU+CUDA state into the default generators, yields, then
        saves the advanced state back under ``name`` and restores the original
        generator state. If ``name`` was never registered (e.g. a script that never
        called ``set_seed``), the stream is auto-snapshotted from the current
        generator state on first use, so the draw still advances in isolation.

        Args:
            name (str): the tracked stream to fork; defaults to the data-parallel
                stream.
        """
        if name not in self.states_:
            self._snapshot_current(name)

        orig_cpu_state = torch.get_rng_state()
        has_cuda = torch.cuda.is_available()
        orig_cuda_state = _get_cuda_rng_state() if has_cuda else None

        cpu_state, cuda_state = self.states_[name]
        torch.set_rng_state(cpu_state)
        if has_cuda:
            _set_cuda_rng_state(cuda_state)
        try:
            yield
        finally:
            new_cpu = torch.get_rng_state()
            new_cuda = _get_cuda_rng_state() if has_cuda else cuda_state
            self.states_[name] = (new_cpu, new_cuda)
            torch.set_rng_state(orig_cpu_state)
            if has_cuda:
                _set_cuda_rng_state(orig_cuda_state)

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {}
        for name, (cpu_state, cuda_state) in self.states_.items():
            sd[f"{name}-{self.dp_rank}/cpu"] = cpu_state
            sd[f"{name}-{self.dp_rank}/cuda"] = cuda_state
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict.items():
            base, _, kind = key.rpartition("/")
            name = _strip_dp_suffix(base)
            if name not in self.states_:
                logger.warning(
                    "RNG checkpoint key %r maps to unknown stream %r; skipping restore "
                    "(resume will use the freshly-seeded stream instead of the saved one)",
                    key,
                    name,
                )
                continue
            cpu_state, cuda_state = self.states_[name]
            if kind == "cpu":
                self.states_[name] = (value, cuda_state)
            elif kind == "cuda":
                self.states_[name] = (cpu_state, value)


_RNG_STATE_TRACKER: RNGStatesTracker | None = None


def initialize_rng_tracker() -> None:
    """Create the global RNG tracker on first use (idempotent)."""
    global _RNG_STATE_TRACKER
    if _RNG_STATE_TRACKER is None:
        _RNG_STATE_TRACKER = RNGStatesTracker()


def get_rng_states_tracker() -> RNGStatesTracker:
    """Return the global RNG tracker, creating it if needed."""
    initialize_rng_tracker()
    assert _RNG_STATE_TRACKER is not None
    return _RNG_STATE_TRACKER
