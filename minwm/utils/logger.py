"""Logging utilities for the minwm framework.

Adapted from detectron2's ``detectron2/utils/logger.py``.

Usage::

    from minwm.utils.logger import setup_logger, init_logger

    # Once, at program entry (e.g. in the training script):
    setup_logger(output="logs/run.log", distributed_rank=rank)

    # In every module:
    logger = init_logger(__name__)
    logger.info("hello")
    log_first_n(logging.WARNING, "only once", n=1)
"""

import functools
import logging
import os
import sys
import time
from collections import Counter

from termcolor import colored

__all__ = [
    "setup_logger",
    "init_logger",
    "log_first_n",
    "log_every_n",
    "log_every_n_seconds",
]

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_DEFAULT_PLAIN_FMT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
_DATE_FMT = "%m/%d %H:%M:%S"


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, root_name: str, abbrev_name: str = "", **kwargs):
        self._root_name = root_name + "."
        self._abbrev_name = (abbrev_name + ".") if abbrev_name else ""
        super().__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno in (logging.ERROR, logging.CRITICAL):
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# ---------------------------------------------------------------------------
# setup_logger — call once at program entry
# ---------------------------------------------------------------------------


@functools.lru_cache  # multiple calls with the same args are no-ops
def setup_logger(
    output: str | None = None,
    distributed_rank: int = 0,
    *,
    color: bool = True,
    name: str = "minwm",
    abbrev_name: str | None = None,
) -> logging.Logger:
    """Configure and return the root ``name`` logger.

    Args:
        output: file path (or directory) to write logs to.  All ranks write;
            rank > 0 files get a ``.rankN`` suffix.
        distributed_rank: global rank of this process.  stdout handler is
            installed only on rank 0.
        color: use ANSI colours for WARNING / ERROR on stdout.
        name: logger name (also the root namespace for child loggers).
        abbrev_name: abbreviated name shown in log lines.  Defaults to ``name``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(_DEFAULT_PLAIN_FMT, datefmt=_DATE_FMT)

    if distributed_rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = (
            _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt=_DATE_FMT,
                root_name=name,
                abbrev_name=abbrev_name,
            )
            if color
            else plain_formatter
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output is not None:
        if output.endswith((".txt", ".log")):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = f"{filename}.rank{distributed_rank}"
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# init_logger — per-module factory (drop-in for logging.getLogger)
# ---------------------------------------------------------------------------


def init_logger(name: str) -> logging.Logger:
    """Return ``logging.getLogger(name)``.

    The returned logger inherits handlers from ``setup_logger``'s root logger
    as long as propagation is not disabled at an intermediate node.  Call
    :func:`setup_logger` once at program entry to configure output / rank
    filtering; individual modules just call ``init_logger(__name__)``.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Throttled helpers — adapted from detectron2 / abseil-py
# ---------------------------------------------------------------------------


def _find_caller() -> tuple[str, tuple]:
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod = frame.f_globals.get("__name__", "minwm")
            return mod, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back  # type: ignore[assignment]
    return "minwm", ("", 0, "")


_LOG_COUNTER: Counter = Counter()
_LOG_TIMER: dict = {}


def log_first_n(
    lvl: int,
    msg: str,
    n: int = 1,
    *,
    name: str | None = None,
    key: str | tuple[str, ...] = "caller",
) -> None:
    """Log only the first *n* occurrences (keyed by caller and/or message)."""
    if isinstance(key, str):
        key = (key,)
    caller_mod, caller_key = _find_caller()
    hash_key: tuple = ()
    if "caller" in key:
        hash_key += caller_key
    if "message" in key:
        hash_key += (msg,)
    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_mod).log(lvl, msg)


def log_every_n(lvl: int, msg: str, n: int = 1, *, name: str | None = None) -> None:
    """Log once every *n* calls."""
    caller_mod, key = _find_caller()
    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        logging.getLogger(name or caller_mod).log(lvl, msg)


def log_every_n_seconds(lvl: int, msg: str, n: float = 1.0, *, name: str | None = None) -> None:
    """Log no more than once per *n* seconds."""
    caller_mod, key = _find_caller()
    last = _LOG_TIMER.get(key)
    now = time.time()
    if last is None or now - last >= n:
        logging.getLogger(name or caller_mod).log(lvl, msg)
        _LOG_TIMER[key] = now
