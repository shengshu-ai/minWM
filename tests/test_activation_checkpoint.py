"""Tests for activation-checkpointing policies (``training.activation_checkpointing_policy``)."""

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy, noop_context_fn

from minwm.config.schema import Training
from minwm.distributed.activation_checkpoint import (
    _reaches_model_width,
    _save_attention_policy,
    checkpoint_context_fn,
)
from minwm.distributed.fsdp import _apply_activation_checkpointing
from minwm.engine import BaseTrainer
from minwm.modeling.wan21.model import Wan21Model

_SDPA = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default
_ARCH = dict(
    model_type="t2v",
    dim=64,
    ffn_dim=128,
    freq_dim=64,
    text_len=8,
    text_dim=64,
    num_heads=4,
    num_layers=2,
    in_dim=16,
    out_dim=16,
    use_prope=True,
)


class _AttentionRuns(TorchDispatchMode):
    """Record the key length of every attention forward that actually executes."""

    def __init__(self):
        super().__init__()
        self.key_lengths = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is _SDPA:
            self.key_lengths.append(args[1].shape[-2])
        return func(*args, **(kwargs or {}))


def _model():
    torch.manual_seed(0)
    model = Wan21Model(**_ARCH)
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.02)
        for name, p in model.named_parameters():
            if "prope_o" in name:
                p.normal_(std=0.02)
    return model


def _step(model):
    """One forward/backward on 2 frames of 16x16 latents (128 tokens, 8 text tokens)."""
    torch.manual_seed(1)
    x = [torch.randn(16, 2, 16, 16)]
    viewmats = torch.eye(4).repeat(1, 2, 1, 1)
    viewmats[0, 1, :3, 3] = torch.tensor([0.3, 0.1, 0.2])
    out = model(
        x=x,
        t=torch.tensor([500.0]),
        context=[torch.randn(8, 64)],
        seq_len=128,
        viewmats=viewmats,
        Ks=torch.eye(3).repeat(1, 2, 1, 1) * 2,
    )
    with _AttentionRuns() as runs:
        out.square().mean().backward()
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): p.grad
        for name, p in model.named_parameters()
    }
    return runs.key_lengths, grads


def _run(policy=None, fsdp_wrapper=False):
    model = _model()
    if policy is not None and fsdp_wrapper:
        _apply_activation_checkpointing(model, ["blocks.0", "blocks.1"], policy)
    elif policy is not None:
        BaseTrainer._enable_native_activation_checkpointing(model, policy)
    return _step(model)


def test_policy_keeps_long_context_attention_only():
    q = torch.empty(1, 4, 128, 16)
    assert _save_attention_policy(None, _SDPA, q, q, q, 0.0, False) == CheckpointPolicy.MUST_SAVE
    short = torch.empty(1, 4, 8, 16)
    assert (
        _save_attention_policy(None, _SDPA, q, short, short, 0.0, False)
        == CheckpointPolicy.PREFER_RECOMPUTE
    )
    mm = torch.ops.aten.mm.default
    assert _save_attention_policy(None, mm, q, q) == CheckpointPolicy.PREFER_RECOMPUTE


def test_flash_attn_layouts():
    q = torch.empty(512, 12, 128)
    varlen = "flash_attn::_flash_attn_varlen_forward"
    assert _reaches_model_width(varlen, (q, q, q, None, None, 512, 1536), {})
    assert not _reaches_model_width(varlen, (q, q, q, None, None, 512, 512), {})
    dense = "flash_attn::_flash_attn_forward"
    assert _reaches_model_width(dense, (q[None], torch.empty(1, 6240, 12, 128)), {})
    assert not _reaches_model_width(dense, (q[None], torch.empty(1, 512, 12, 128)), {})


@pytest.mark.parametrize("fsdp_wrapper", [False, True])
def test_save_attention_skips_self_attention_recompute(fsdp_wrapper):
    baseline_runs, baseline_grads = _run()
    full_runs, full_grads = _run("full", fsdp_wrapper)
    saved_runs, saved_grads = _run("save_attention", fsdp_wrapper)
    # backward replays each layer's main + PRoPE self-attention (128 keys) and its
    # cross-attention (8 text keys) under full recompute; only the latter remains.
    assert baseline_runs == []
    assert sorted(full_runs) == [8, 8, 128, 128, 128, 128]
    assert sorted(saved_runs) == [8, 8]
    for name, grad in baseline_grads.items():
        assert grad is not None and grad.abs().sum() > 0, name
        assert torch.equal(full_grads[name], grad), name
        assert torch.equal(saved_grads[name], grad), name


def test_native_policy_installs_context_fn():
    model = _model()
    BaseTrainer._enable_native_activation_checkpointing(model)
    assert model.gradient_checkpointing and model.checkpoint_context_fn is noop_context_fn
    BaseTrainer._enable_native_activation_checkpointing(model, "save_attention")
    assert model.checkpoint_context_fn is checkpoint_context_fn("save_attention")


def test_native_policy_requires_context_fn_support():
    with pytest.raises(ValueError, match="does not support"):
        BaseTrainer._enable_native_activation_checkpointing(nn.Linear(2, 2), "save_attention")


def test_training_config_validates_policy():
    assert Training().activation_checkpointing_policy == "full"
    with pytest.raises(ValueError, match="must be 'full' or 'save_attention'"):
        Training(activation_checkpointing_policy="selective")
    with pytest.raises(ValueError, match="activation_offload"):
        Training(activation_checkpointing_policy="save_attention", activation_offload=True)
    with pytest.raises(ValueError, match="unknown activation checkpointing policy"):
        checkpoint_context_fn("selective")
