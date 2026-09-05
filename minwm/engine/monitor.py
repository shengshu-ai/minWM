"""Metric computation + fan-out for the trainer's observability layer.

Adapted in spirit from torchtitan's ``MetricsProcessor``: the trainer hands a
plain ``dict`` of scalars (the Recipe's losses, read off the
:class:`~minwm.engine.events.EventStorage`) to :meth:`MetricsProcessor.log`,
which augments it with throughput (steps/sec, ms/step) and — on CUDA — peak
memory, prints a STDOUT line, and fans the merged dict out to the configured
backends (TensorBoard / WandB).

The processor is **decoupled from EventStorage**: it takes a dict, so it is
unit-testable without the global bus and the trainer stays in control of what
gets merged. Real writers are built lazily (on the first :meth:`log`) and only
on the main process, so constructing a trainer never triggers ``wandb.init`` or
creates a TensorBoard run directory (important for dry runs / pytest).
"""

import os
import time

import torch

from minwm.config.schema import Monitor
from minwm.utils import comm
from minwm.utils.logger import init_logger

logger = init_logger(__name__)


class BaseWriter:
    """A metric sink. Subclasses fan one :meth:`log` call out to a backend."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write ``metrics`` at ``step`` (no-op in the base class)."""

    def close(self) -> None:
        """Flush / release backend resources (no-op in the base class)."""


class TensorBoardWriter(BaseWriter):
    """Fan metrics out to TensorBoard event files.

    Args:
        log_dir (str): directory for the event files.
    """

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir)
        logger.info("TensorBoard logging to %s", log_dir)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def close(self) -> None:
        self._writer.close()


class WandBWriter(BaseWriter):
    """Fan metrics out to Weights & Biases.

    Args:
        log_dir (str): directory used as the W&B run dir.
        project (str | None): project name; falls back to ``WANDB_PROJECT`` then
            ``"minwm"``.
        run_name (str | None): run name; falls back to ``WANDB_RUN_NAME``.
    """

    def __init__(self, log_dir: str, project: str | None, run_name: str | None) -> None:
        import wandb

        self._wandb = wandb
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(
            project=project or os.getenv("WANDB_PROJECT", "minwm"),
            name=run_name or os.getenv("WANDB_RUN_NAME"),
            dir=log_dir,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb.run is not None:
            self._wandb.finish()


class WriterList(BaseWriter):
    """Fan one :meth:`log` / :meth:`close` out to a list of writers."""

    def __init__(self, writers: list[BaseWriter]) -> None:
        self._writers = writers

    def log(self, metrics: dict[str, float], step: int) -> None:
        for writer in self._writers:
            writer.log(metrics, step)

    def close(self) -> None:
        for writer in self._writers:
            writer.close()


def _build_writers(cfg: Monitor, log_dir: str) -> WriterList:
    """Construct the configured backend writers (main process only).

    Args:
        cfg (Monitor): the monitor config naming the backends.
        log_dir (str): base directory for backend output.

    Returns:
        WriterList: the fan-out container (empty when no backends configured).
    """
    writers: list[BaseWriter] = []
    for backend in cfg.backends:
        if backend == "tensorboard":
            writers.append(TensorBoardWriter(log_dir))
        elif backend == "wandb":
            writers.append(WandBWriter(log_dir, cfg.wandb_project, cfg.wandb_run_name))
        else:
            raise ValueError(f"unknown monitor backend {backend!r}; valid: tensorboard, wandb")
    return WriterList(writers)


class DeviceMemoryMonitor:
    """Tracks peak CUDA memory between log windows.

    Only constructed on CUDA; on CPU the :class:`MetricsProcessor` skips it
    entirely so no ``memory/*`` keys are emitted.
    """

    def __init__(self) -> None:
        self._capacity = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.reset_peak_memory_stats()

    def peak_stats(self) -> dict[str, float]:
        """Return peak-reserved GiB / % and OOM count since the last reset."""
        info = torch.cuda.memory_stats()
        max_reserved = info.get("reserved_bytes.all.peak", 0)
        return {
            "memory/max_reserved(GiB)": max_reserved / 1024**3,
            "memory/max_reserved(%)": 100 * max_reserved / self._capacity,
            "memory/num_ooms": info.get("num_ooms", 0),
        }

    def reset(self) -> None:
        torch.cuda.reset_peak_memory_stats()


class MetricsProcessor:
    """Compute throughput / memory, then fan metrics out to STDOUT + backends.

    Decoupled from :class:`~minwm.engine.events.EventStorage`: :meth:`log` takes a
    plain dict so it is unit-testable without the global bus, and the trainer
    stays in control of what it merges. Writers are built lazily on the first
    :meth:`log` (so constructing a trainer never triggers ``wandb.init`` or
    TensorBoard dir creation) and only on the main process.

    Args:
        cfg (Monitor): the monitor config (backends + W&B knobs).
        log_dir (str): base directory for backend output.
    """

    def __init__(self, cfg: Monitor, log_dir: str) -> None:
        self._cfg = cfg
        self._log_dir = log_dir
        self._writers: WriterList | None = None
        self._mem = DeviceMemoryMonitor() if torch.cuda.is_available() else None
        self._t_last = time.perf_counter()
        self._step_last = 0

    def start_window(self, step: int) -> None:
        """Reset the throughput window. Called at ``train()`` start, not init.

        Args:
            step (int): the step the loop is (re)starting from, so the first
                logged ``steps/sec`` excludes model-build / checkpoint-load time.
        """
        self._t_last = time.perf_counter()
        self._step_last = step
        if self._mem is not None:
            self._mem.reset()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Merge perf + caller metrics, print to STDOUT, fan out to backends.

        Args:
            step (int): the current global step (post-increment, 1-based).
            metrics (dict[str, float]): caller-supplied scalars (losses, etc.).
        """
        if not comm.is_main_process():
            return
        if self._writers is None:
            self._writers = _build_writers(self._cfg, self._log_dir)

        now = time.perf_counter()
        dt = now - self._t_last
        nsteps = step - self._step_last
        merged: dict[str, float] = dict(metrics)
        if dt > 0 and nsteps > 0:
            merged["steps/sec"] = nsteps / dt
            merged["ms/step"] = 1000 * dt / nsteps
        if self._mem is not None:
            merged.update(self._mem.peak_stats())
            self._mem.reset()

        body = "  ".join(f"{k}: {v:.4g}" for k, v in merged.items())
        logger.info("step %d  %s", step, body)
        self._writers.log(merged, step)

        self._t_last = now
        self._step_last = step

    def close(self) -> None:
        """Flush and close all writers (idempotent)."""
        if self._writers is not None:
            self._writers.close()
            self._writers = None
