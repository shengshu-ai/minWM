# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grouped asynchronous CPU offload for activations saved by FSDP2 layers.

Adapted from verl PR #1220 and NVIDIA TransformerEngine's CPU offload design.
Each FSDP transformer layer is one activation group. Forward overlaps group
``i``'s D2H copy with group ``i + 1``'s compute; backward prefetches group
``i - 1`` while group ``i`` computes. A separate pass state is created for
every full layer sweep so autoregressive training may keep several forward
graphs alive before one backward.
"""

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint

from minwm.utils.logger import init_logger

logger = init_logger(__name__)


def _storage_id(tensor: Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return tensor.device.type, tensor.device.index, storage.data_ptr()


def _tensor_key(tensor: Tensor) -> tuple:
    return (
        *_storage_id(tensor),
        tensor.storage_offset(),
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _module_storage_ids(module: nn.Module) -> set[tuple]:
    storages: set[tuple] = set()
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device.type != "cuda":
            continue
        try:
            storages.add(_storage_id(tensor))
        except (NotImplementedError, RuntimeError):
            continue
    return storages


@dataclass
class _ActivationEntry:
    device: torch.device
    gpu: Tensor | None
    cpu: Tensor | None = None
    remaining_uses: int = 0


class _SavedTensorRef:
    def __init__(self, group: "_ActivationGroup", key: tuple) -> None:
        self.group = group
        self.key = key


class _ActivationGroup:
    def __init__(self, index: int, stats: dict[str, int]) -> None:
        self.index = index
        self.stats = stats
        self.parameter_storages: set[tuple] = set()
        self.entries: dict[tuple, _ActivationEntry] = {}
        self.d2h_done: torch.cuda.Event | None = None
        self.h2d_done: torch.cuda.Event | None = None
        self.prefetch_started = False

    def set_parameter_storages(self, module: nn.Module) -> None:
        self.parameter_storages = _module_storage_ids(module)

    def _eligible(self, tensor: Tensor) -> bool:
        if type(tensor) is not Tensor:
            return False
        if tensor.device.type != "cuda" or tensor.layout != torch.strided:
            return False
        if tensor.numel() == 0 or not tensor.is_contiguous():
            return False
        # FSDP2 materializes sharded parameters as plain tensors during the
        # forward. They are leaf tensors requiring grad, but they are not
        # activations and must never enter the offload queue.
        if tensor.is_leaf and tensor.requires_grad:
            return False
        return _storage_id(tensor) not in self.parameter_storages

    def pack(self, tensor: Tensor) -> Tensor | _SavedTensorRef:
        if not self._eligible(tensor):
            if type(tensor) is Tensor and tensor.device.type == "cuda":
                bytes_ = tensor.numel() * tensor.element_size()
                if tensor.is_leaf and tensor.requires_grad:
                    self.stats["leaf_requires_grad_bytes"] += bytes_
                    self.stats["leaf_requires_grad_count"] += 1
                else:
                    self.stats["ineligible_bytes"] += bytes_
                    self.stats["ineligible_count"] += 1
            return tensor
        bytes_ = tensor.numel() * tensor.element_size()
        self.stats["saved_activation_bytes"] += bytes_
        self.stats["saved_activation_count"] += 1
        key = _tensor_key(tensor)
        if key not in self.entries:
            self.entries[key] = _ActivationEntry(device=tensor.device, gpu=tensor.detach())
        self.entries[key].remaining_uses += 1
        return _SavedTensorRef(self, key)

    @staticmethod
    def unpack(packed: Tensor | _SavedTensorRef) -> Tensor:
        if isinstance(packed, Tensor):
            return packed
        entry = packed.group.entries[packed.key]
        if entry.gpu is None:
            raise RuntimeError(
                f"activation group {packed.group.index} was unpacked before H2D prefetch"
            )
        tensor = entry.gpu
        entry.remaining_uses -= 1
        if entry.remaining_uses == 0:
            packed.group.entries.pop(packed.key)
            entry.cpu = None
            entry.gpu = None
        return tensor

    def offload(self, d2h_stream: torch.cuda.Stream, compute_stream: torch.cuda.Stream) -> None:
        if not self.entries:
            return
        d2h_stream.wait_stream(compute_stream)
        with torch.cuda.stream(d2h_stream):
            for entry in self.entries.values():
                assert entry.gpu is not None
                self.stats["d2h_bytes"] += entry.gpu.numel() * entry.gpu.element_size()
                entry.cpu = torch.empty(
                    entry.gpu.size(),
                    dtype=entry.gpu.dtype,
                    layout=entry.gpu.layout,
                    device="cpu",
                    pin_memory=True,
                )
                entry.cpu.copy_(entry.gpu, non_blocking=True)
                entry.gpu.record_stream(d2h_stream)
            self.d2h_done = torch.cuda.Event()
            self.d2h_done.record(d2h_stream)

    def release_gpu_after_offload(self, compute_stream: torch.cuda.Stream) -> None:
        if self.d2h_done is None:
            return
        compute_stream.wait_event(self.d2h_done)
        for entry in self.entries.values():
            entry.gpu = None

    def prefetch(self, h2d_stream: torch.cuda.Stream) -> None:
        if self.d2h_done is None or self.prefetch_started:
            return
        self.prefetch_started = True
        with torch.cuda.stream(h2d_stream):
            h2d_stream.wait_event(self.d2h_done)
            for entry in self.entries.values():
                assert entry.cpu is not None
                self.stats["h2d_bytes"] += entry.cpu.numel() * entry.cpu.element_size()
                entry.gpu = entry.cpu.to(entry.device, non_blocking=True)
            self.h2d_done = torch.cuda.Event()
            self.h2d_done.record(h2d_stream)

    def wait_for_prefetch(
        self, h2d_stream: torch.cuda.Stream, compute_stream: torch.cuda.Stream
    ) -> None:
        if self.d2h_done is None:
            return
        if not self.prefetch_started:
            self.prefetch(h2d_stream)
        assert self.h2d_done is not None
        compute_stream.wait_event(self.h2d_done)
        for entry in self.entries.values():
            assert entry.gpu is not None
            entry.gpu.record_stream(compute_stream)

    def clear(self) -> None:
        self.entries.clear()
        self.parameter_storages.clear()
        self.d2h_done = None
        self.h2d_done = None


class _ActivationPass:
    def __init__(
        self,
        controller: "_ActivationOffloadController",
        pass_id: int,
        num_layers: int,
    ) -> None:
        self.controller = controller
        self.pass_id = pass_id
        self.groups = [_ActivationGroup(index, controller.stats) for index in range(num_layers)]
        self.next_forward_group = 0

    def on_group_forward(self, index: int) -> None:
        if index != self.next_forward_group:
            raise RuntimeError(
                f"activation offload pass {self.pass_id} expected layer "
                f"{self.next_forward_group}, got {index}"
            )
        compute_stream = torch.cuda.current_stream()
        group = self.groups[index]

        if index < len(self.groups) - 1:
            group.offload(self.controller.d2h_stream, compute_stream)
        if index > 0:
            self.groups[index - 1].release_gpu_after_offload(compute_stream)
        self.next_forward_group += 1

    def on_group_backward(self, index: int) -> None:
        compute_stream = torch.cuda.current_stream()
        group = self.groups[index]

        if index < len(self.groups) - 1:
            group.wait_for_prefetch(self.controller.h2d_stream, compute_stream)
        if index > 0:
            self.groups[index - 1].prefetch(self.controller.h2d_stream)
        if index + 1 < len(self.groups):
            self.groups[index + 1].clear()


class _GroupCommit(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, state: _ActivationPass, index: int, *tensors: Tensor) -> tuple:
        state.on_group_forward(index)
        ctx.state = state
        ctx.index = index
        return tensors

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple:
        ctx.state.on_group_backward(ctx.index)
        return None, None, *grad_outputs


def _commit_group_output(output: Any, state: _ActivationPass, index: int) -> Any:
    leaves, spec = tree_flatten(output)
    tensor_indices = [
        leaf_index
        for leaf_index, leaf in enumerate(leaves)
        if isinstance(leaf, Tensor) and leaf.requires_grad
    ]
    if not tensor_indices:
        raise RuntimeError(
            f"activation-offloaded FSDP layer {index} produced no grad-requiring tensor"
        )

    tensors = [leaves[leaf_index] for leaf_index in tensor_indices]
    committed = _GroupCommit.apply(state, index, *tensors)
    if isinstance(committed, Tensor):
        committed = (committed,)
    for leaf_index, tensor in zip(tensor_indices, committed):
        leaves[leaf_index] = tensor
    return tree_unflatten(leaves, spec)


class _ActivationOffloadController:
    def __init__(self, num_layers: int, activation_checkpointing: bool) -> None:
        self.num_layers = num_layers
        self.activation_checkpointing = activation_checkpointing
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.active_pass: _ActivationPass | None = None
        self.next_pass_id = 0
        self.stats: dict[str, int] = {
            "checkpoint_forward_count": 0,
            "plain_forward_count": 0,
            "saved_activation_bytes": 0,
            "saved_activation_count": 0,
            "leaf_requires_grad_bytes": 0,
            "leaf_requires_grad_count": 0,
            "ineligible_bytes": 0,
            "ineligible_count": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
        }
        self._warned_mutable_state = False

    def _start_pass(self) -> _ActivationPass:
        if self.active_pass is not None:
            self._abort_active_pass()
            raise RuntimeError(
                "started a new activation-offload pass before the previous one ended"
            )
        state = _ActivationPass(self, self.next_pass_id, self.num_layers)
        self.next_pass_id += 1
        self.active_pass = state
        return state

    def _abort_active_pass(self) -> None:
        state = self.active_pass
        if state is None:
            return
        self.d2h_stream.synchronize()
        self.h2d_stream.synchronize()
        for group in state.groups:
            group.clear()
        self.active_pass = None

    def _checkpoint_forward(
        self,
        forward_method: Any,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        kwarg_keys = tuple(kwargs)
        flat_args = (*args, *(kwargs[key] for key in kwarg_keys))

        def run(*inputs: Any) -> Any:
            num_kwargs = len(kwarg_keys)
            positional = inputs[:-num_kwargs] if num_kwargs else inputs
            keyword_values = inputs[-num_kwargs:] if num_kwargs else ()
            return forward_method(*positional, **dict(zip(kwarg_keys, keyword_values)))

        return checkpoint(run, *flat_args, use_reentrant=True)

    def run_layer(
        self,
        index: int,
        module: nn.Module,
        forward_method: Any,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        if not torch.is_grad_enabled():
            return forward_method(*args, **kwargs)

        if index == 0:
            state = self._start_pass()
        else:
            state = self.active_pass
            if state is None:
                raise RuntimeError(
                    f"activation-offloaded layer {index} ran without a leading layer 0"
                )

        group = state.groups[index]
        group.set_parameter_storages(module)
        hooks = torch.autograd.graph.saved_tensors_hooks(group.pack, group.unpack)
        mutable_state = any(
            kwargs.get(name) is not None
            for name in ("kv_cache", "crossattn_cache", "prope_kv_cache")
        )
        use_checkpointing = self.activation_checkpointing and not mutable_state
        if self.activation_checkpointing and mutable_state and not self._warned_mutable_state:
            logger.warning(
                "activation offload disabled checkpoint recompute for mutable KV-cache "
                "forwards; activations will be offloaded without recomputation"
            )
            self._warned_mutable_state = True
        try:
            with hooks:
                if use_checkpointing:
                    self.stats["checkpoint_forward_count"] += 1
                    output = self._checkpoint_forward(forward_method, args, kwargs)
                else:
                    self.stats["plain_forward_count"] += 1
                    output = forward_method(*args, **kwargs)
            output = _commit_group_output(output, state, index)
        except BaseException:
            self._abort_active_pass()
            raise

        if index == self.num_layers - 1:
            self.active_pass = None
        return output

    def wrap_layer(self, layer: nn.Module, index: int) -> None:
        forward_method = layer.forward
        controller = self

        @functools.wraps(forward_method)
        def wrapped(module_self: nn.Module, *args: Any, **kwargs: Any) -> Any:
            return controller.run_layer(index, module_self, forward_method, args, kwargs)

        layer.forward = wrapped.__get__(layer, type(layer))


def _disable_native_activation_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        return
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False


def enable_activation_offloading(
    model: nn.Module,
    layers: list[nn.Module],
    *,
    activation_checkpointing: bool = False,
) -> bool:
    """Enable grouped asynchronous activation offload on FSDP2 transformer layers.

    Args:
        model (nn.Module): FSDP2-sharded root model.
        layers (list[nn.Module]): ordered FSDP2 transformer layers from the
            model's ``_fsdp_shard_conditions``.
        activation_checkpointing (bool): install a compatible reentrant checkpoint
            around each offloaded layer. Calls carrying mutable KV caches use pure
            offload because deferred recomputation could observe cache entries
            added by later forwards.

    Returns:
        bool: whether offload was installed. Models with fewer than three layers
            are left unchanged because there is no useful copy/compute pipeline.
    """
    if not torch.cuda.is_available():
        logger.warning(
            "activation offload requires CUDA; leaving %s unchanged",
            type(model).__name__,
        )
        return False
    if len(layers) < 3:
        logger.warning(
            "activation offload needs at least 3 FSDP layers; found %d in %s",
            len(layers),
            type(model).__name__,
        )
        return False
    if getattr(model, "_activation_offload_controller", None) is not None:
        raise RuntimeError(f"activation offload is already enabled for {type(model).__name__}")

    if activation_checkpointing:
        _disable_native_activation_checkpointing(model)

    controller = _ActivationOffloadController(len(layers), activation_checkpointing)
    for index, layer in enumerate(layers):
        controller.wrap_layer(layer, index)
    model._activation_offload_controller = controller
    logger.info(
        "activation offload: wrapped %d FSDP2 layers in %s%s",
        len(layers),
        type(model).__name__,
        " with activation checkpointing" if activation_checkpointing else "",
    )
    return True
