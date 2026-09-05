"""Canonical string-to-``torch.dtype`` resolution shared across the framework.

A single source of truth for the ``"bfloat16"`` -> ``torch.bfloat16`` mapping so
the config layer, the trainer, and the generation runtime don't each carry their
own copy. Lives in ``utils`` (no heavy deps) so any module can import it without
pulling in distributed / modeling machinery.
"""

import torch

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Resolve a dtype spec to a ``torch.dtype``.

    Args:
        dtype (str | torch.dtype): a ``torch.dtype`` (returned as-is) or one of
            the names ``float32`` / ``fp32`` / ``float16`` / ``fp16`` /
            ``bfloat16`` / ``bf16`` / ``float64`` / ``fp64``.

    Returns:
        torch.dtype: the resolved dtype.

    Raises:
        ValueError: if ``dtype`` is an unknown string.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {dtype!r}; valid: {sorted(_DTYPES)}") from exc


def optional_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Like :func:`resolve_dtype` but pass ``None`` through unchanged."""
    return None if dtype is None else resolve_dtype(dtype)
