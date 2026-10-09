"""Activation-checkpointing policies: full block recompute, or keep attention outputs.

Full checkpointing reruns a whole block during backward. With FlashAttention-style
kernels, self-attention is the worst op to rerun: its forward costs
``O(L_q * L_k * d)`` FLOPs while its output is only ``O(L_q * d)`` values. The
``"save_attention"`` policy keeps those outputs (with their softmax statistics) and
recomputes everything else, so backward skips the attention forward. The kept
values are the ones recomputation would reproduce, so gradients are unchanged.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
    noop_context_fn,
)

_ATTENTION_FORWARD_OPS = frozenset(
    {
        "flash_attn::_flash_attn_forward",
        "flash_attn::_flash_attn_varlen_forward",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_cudnn_attention",
    }
)


def _reaches_model_width(name: str, args: tuple, kwargs: dict) -> bool:
    # At key length >= width the forward costs >= 4 * width FLOPs per kept value, twice a
    # width x width projection; short contexts (cross-attention to text) are cheaper to rerun.
    q, k = args[0], args[1]
    if name == "flash_attn::_flash_attn_varlen_forward":
        key_len, heads = kwargs.get("max_seqlen_k", args[6]), q.shape[-2]
    elif name.startswith("flash_attn::"):
        key_len, heads = k.shape[1], q.shape[-2]
    else:
        key_len, heads = k.shape[-2], q.shape[-3]
    return key_len >= heads * q.shape[-1]


def _save_attention_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    name = getattr(getattr(op, "_schema", None), "name", None)
    if name in _ATTENTION_FORWARD_OPS and _reaches_model_width(name, args, kwargs):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _save_attention_contexts() -> tuple[AbstractContextManager, AbstractContextManager]:
    return create_selective_checkpoint_contexts(_save_attention_policy)


def checkpoint_context_fn(
    policy: str,
) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]]:
    """Return the ``context_fn`` for :func:`torch.utils.checkpoint.checkpoint`.

    Args:
        policy (str): ``"full"`` recomputes every op of a checkpointed block;
            ``"save_attention"`` keeps the outputs of long-context attention ops
            (FlashAttention 2 or PyTorch SDPA) and recomputes the rest.

    Returns:
        Callable: a factory of the two context managers ``checkpoint`` expects; only
            the non-reentrant ``checkpoint`` accepts it.

    Raises:
        ValueError: if ``policy`` is neither ``"full"`` nor ``"save_attention"``.
    """
    if policy == "full":
        return noop_context_fn
    if policy == "save_attention":
        return _save_attention_contexts
    raise ValueError(f"unknown activation checkpointing policy {policy!r}")
