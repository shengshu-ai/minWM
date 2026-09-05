"""FSDP2 (``fully_shard``) helpers for the minwm trainer.

Both DiT backbones (Wan ``CausalWan21Model`` and the HunyuanVideo transformer)
expose a ``_fsdp_shard_conditions`` list — predicates ``(name, module) -> bool``
that select the per-layer submodules to wrap. This module turns that hook plus a
data-parallel :class:`~torch.distributed.device_mesh.DeviceMesh` into a sharded
model, so the engine code stays backbone-agnostic: the only per-family
difference is the shard-condition predicate each model already owns.

Typical use from the trainer::

    from minwm.distributed import get_fsdp_mesh
    from minwm.distributed.fsdp import build_mp_policy, shard_model

    mp = build_mp_policy(torch.bfloat16, torch.float32)
    shard_model(model, mesh=get_fsdp_mesh(), mp_policy=mp)
"""

from collections.abc import Callable, Iterator

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from minwm.utils.dtype import resolve_dtype  # noqa: F401  (re-exported for existing importers)
from minwm.utils.logger import init_logger

logger = init_logger(__name__)


def build_mp_policy(
    param_dtype: str | torch.dtype = torch.bfloat16,
    reduce_dtype: str | torch.dtype = torch.float32,
    output_dtype: str | torch.dtype | None = None,
) -> MixedPrecisionPolicy:
    """Build the FSDP2 mixed-precision policy.

    ``cast_forward_inputs=True`` so the (fp32) data-pipeline tensors are cast to
    ``param_dtype`` at the sharded module boundary. The recipe's flow math runs
    in fp32 (the scheduler's sigma table is fp32, so ``add_noise`` promotes the
    ``noisy`` / ``clean`` latents back to fp32 regardless of the loader dtype),
    and the DiT params are sharded to ``param_dtype`` — without this cast the very
    first layer hits ``Input type (float) and bias type (BFloat16)``. FSDP only
    casts floating-point tensors, so masks / indices / complex RoPE tables pass
    through untouched.

    Args:
        param_dtype (str | torch.dtype): dtype params are cast to for compute
            (typically ``bfloat16``).
        reduce_dtype (str | torch.dtype): dtype for the gradient reduce-scatter
            (typically ``float32`` for stable accumulation).
        output_dtype (str | torch.dtype, optional): dtype forward outputs are
            cast to; ``None`` leaves them in ``param_dtype``.

    Returns:
        MixedPrecisionPolicy: the policy passed to ``fully_shard``.
    """
    return MixedPrecisionPolicy(
        param_dtype=resolve_dtype(param_dtype),
        reduce_dtype=resolve_dtype(reduce_dtype),
        output_dtype=resolve_dtype(output_dtype) if output_dtype is not None else None,
        cast_forward_inputs=True,
    )


def shard_model(
    model: nn.Module,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
    pin_cpu_memory: bool = True,
    activation_checkpointing: bool = False,
    activation_offload: bool = False,
) -> nn.Module:
    """Shard ``model`` in place with FSDP2 using its ``_fsdp_shard_conditions``.

    Applies ``fully_shard`` to the *top-most* submodule matched by any predicate
    in ``model._fsdp_shard_conditions`` — one wrap per transformer block, via
    :func:`_iter_fsdp_targets` — then wraps the root to catch any unsharded
    stragglers. Selecting only the top-most match means a broad predicate (e.g.
    Wan's ``parts[1].isdigit()``, which also matches a block's descendants)
    shards each block exactly once instead of also sharding every leaf inside it.
    Sharding is applied in place; the same (now FSDP-managed) module is also
    returned for convenience.

    Args:
        model (nn.Module): the model to shard; must define a non-empty
            ``_fsdp_shard_conditions`` list of ``(name, module) -> bool``
            predicates.
        mesh (DeviceMesh): the data-parallel mesh to shard over (1-D for pure
            FSDP, 2-D ``(dp_replicate, dp_shard)`` for HSDP).
        mp_policy (MixedPrecisionPolicy, optional): mixed-precision policy;
            defaults to FSDP2's all-fp32 default.
        reshard_after_forward (bool): reshard params after forward. ``True`` ==
            FULL_SHARD, ``False`` == SHARD_GRAD_OP.
        cpu_offload (bool): offload params / grads / optimizer state to CPU.
        pin_cpu_memory (bool): pin the offloaded CPU memory (only with
            ``cpu_offload``).
        activation_checkpointing (bool): wrap each top-most shard layer in
            non-reentrant activation checkpointing (recompute its forward in
            backward) before sharding. With ``activation_offload``, the offloader
            installs its compatible per-layer checkpoint wrapper instead. Mutable
            KV-cache forwards use pure offload instead of unsafe recomputation.
        activation_offload (bool): asynchronously move each FSDP layer's saved
            activations to pinned CPU memory and prefetch them during backward.
            Uses the same top-most layer boundaries as FSDP sharding.

    Returns:
        nn.Module: the sharded model (same object as ``model``).

    Raises:
        ValueError: if ``model`` has no ``_fsdp_shard_conditions`` or no module
            matched any condition.
    """
    shard_conditions = getattr(model, "_fsdp_shard_conditions", None)
    if not shard_conditions:
        raise ValueError(
            f"{type(model).__name__} defines no '_fsdp_shard_conditions'; "
            "cannot select layers to shard with FSDP."
        )

    target_names = [name for name, _ in _iter_fsdp_targets(model, shard_conditions)]
    if not target_names:
        raise ValueError(
            f"No modules in {type(model).__name__} matched any FSDP shard "
            "condition; check '_fsdp_shard_conditions'."
        )

    if activation_checkpointing and not activation_offload:
        _apply_activation_checkpointing(model, target_names)

    fsdp_kwargs: dict = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mp_policy,
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

    # Resolve each target by name *after* the AC pass so we shard the (possibly
    # checkpoint-wrapped) module — keeping FSDP outside the checkpoint wrapper.
    for name in target_names:
        fully_shard(model.get_submodule(name), **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)
    if activation_offload:
        from .activation_offload import enable_activation_offloading

        layers = [model.get_submodule(name) for name in target_names]
        enable_activation_offloading(
            model,
            layers,
            activation_checkpointing=activation_checkpointing,
        )
    logger.info("FSDP2: sharded %d layer module(s) in %s", len(target_names), type(model).__name__)
    return model


def _iter_fsdp_targets(
    model: nn.Module, shard_conditions: list[Callable[[str, nn.Module], bool]]
) -> Iterator[tuple[str, nn.Module]]:
    """Yield the top-most ``(name, module)`` pairs matching any shard condition.

    Single source of truth for *which* modules get sharded and (optionally)
    activation-checkpointed. Walks the tree top-down and, on the first module a
    predicate matches, yields it without descending further — so each transformer
    block is selected exactly once even when a predicate (e.g. Wan's
    ``parts[1].isdigit()``) would also match the block's descendants. This keeps
    FSDP from over-sharding every leaf inside a block, and keeps activation
    checkpointing from nested-wrapping a block (which would recompute an inner
    forward once per enclosing checkpoint).

    Args:
        model (nn.Module): the model to walk.
        shard_conditions (list[Callable[[str, nn.Module], bool]]):
            ``(name, module) -> bool`` predicates selecting the layers to shard /
            checkpoint.

    Yields:
        tuple[str, nn.Module]: the dotted name and module of each top-most match.
    """

    def _walk(module: nn.Module, prefix: str):
        for child_name, child in module.named_children():
            name = f"{prefix}.{child_name}" if prefix else child_name
            if any(cond(name, child) for cond in shard_conditions):
                yield name, child
            else:
                yield from _walk(child, name)

    yield from _walk(model, "")


def _apply_activation_checkpointing(model: nn.Module, target_names: list[str]) -> None:
    """Wrap each named layer in non-reentrant activation checkpointing, in place.

    Recomputes a wrapped layer's forward during backward instead of storing its
    activations, so a deep model (whose own forward does not checkpoint) keeps only
    one layer's activations live at a time. ``target_names`` is the top-most set
    :func:`shard_model` shards (from :func:`_iter_fsdp_targets`), so blocks are
    never nested-wrapped — a predicate that also matches a block's descendants
    would otherwise recompute the inner forward once per enclosing checkpoint.

    Args:
        model (nn.Module): the model whose layers are wrapped in place.
        target_names (list[str]): dotted names of the layers to checkpoint (the
            same set FSDP shards).
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    for name in target_names:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, checkpoint_wrapper(getattr(parent, child_name)))

    logger.info("FSDP2: activation-checkpointed %d layer module(s)", len(target_names))
