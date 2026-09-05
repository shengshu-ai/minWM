"""Locate the NCCL shared library at runtime."""

from __future__ import annotations

import torch

from minwm.utils.logger import init_logger

from . import envs

logger = init_logger(__name__)


def find_nccl_library() -> str:
    """Resolve the path to ``libnccl.so.2`` (or ``librccl.so.1`` on ROCm).

    Honors ``TRAINER_NCCL_SO_PATH`` if set; otherwise falls back to the
    library name shipped by the active PyTorch build, which ``ctypes`` will
    resolve via ``LD_LIBRARY_PATH``.
    """
    so_file = envs.TRAINER_NCCL_SO_PATH
    if so_file:
        logger.info(
            "Found nccl from environment variable TRAINER_NCCL_SO_PATH=%s",
            so_file,
        )
        return str(so_file)

    if torch.version.cuda is not None:
        so_file = "libnccl.so.2"
    elif torch.version.hip is not None:
        so_file = "librccl.so.1"
    else:
        raise ValueError("NCCL only supports CUDA and ROCm backends.")
    logger.info("Found nccl from library %s", so_file)
    return so_file


__all__ = ["find_nccl_library"]
