"""Smoke tests for the HY15 Stage 2 causal-forcing recipes.

The HY teacher-forcing path (``forward_bi`` with a clean context) runs on
FlexAttention, which has no CPU backward — and on this dev env even a ``no_grad``
forward fails because ``torch.compile``'s inductor backend is broken on Python
3.13. So a full ``train_one_step`` cannot run on CPU where CI executes.

The split below follows from that constraint:

* **CPU tests** verify only the structural seams that don't touch the
  transformer forward: the tiny model builds from the config dims,
  ``set_attn_mode`` propagates to every double-stream block, the ``HYAdapter``'s
  ``encode_text`` returns the right shapes/dtype, and ``build_optimizers``
  yields a ``"main"`` optimizer.
* **CUDA-gated tests** run the full ``train_one_step`` (forward + backward) at
  ``batch_size=1`` — the HY TF path varlen-packs text into a single batch-1
  sequence, so B>=2 fails by design. They run under ``autocast(bfloat16)``: the
  text token refiner attends via flash-attn varlen, which rejects fp32, so the
  realistic mixed-precision path is the only one that exercises the forward.
  These verify finite loss + finite grads, not numerical parity with a
  reference checkpoint.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from minwm.engine.training.losses import ODERegressionLoss
from minwm.engine.training.recipes.ar_cd import ARCDRecipe
from minwm.engine.training.recipes.ar_ode import ARODERecipe
from minwm.modeling.hy15.adapter import HYAdapter
from minwm.modeling.hy15.causal import ARHunyuanVideo_1_5_DiffusionTransformer

# Latent dims (num_frames, channels, height, width); patch_size [1,2,2] ->
# 4 * (8//2) * (16//2) = 128 latent tokens per sample.
_F, _C, _H, _W = 4, 16, 8, 16
_TEXT_LEN, _TEXT_DIM = 8, 64

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HY Stage 2 teacher-forcing path requires CUDA (FlexAttention has no CPU backward)",
)


def _build_tiny_hy(device: str = "cpu") -> nn.Module:
    """Build a tiny HY transformer matching the stage2 mock config dims."""
    model = ARHunyuanVideo_1_5_DiffusionTransformer(
        patch_size=[1, 2, 2],
        in_channels=_C,
        concat_condition=False,
        hidden_size=64,
        heads_num=4,
        mm_double_blocks_depth=2,
        mm_single_blocks_depth=0,
        rope_dim_list=[4, 6, 6],
        text_states_dim=_TEXT_DIM,
        text_states_dim_2=_TEXT_DIM,
        use_prope=True,
    )
    return model.to(device).train()


def _camera(batch_size: int, frames: int, device: str = "cpu") -> tuple[Tensor, Tensor]:
    """Return identity ``(viewmats [B,F,4,4], Ks [B,F,3,3])`` for the PRoPE path."""
    viewmats = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(batch_size, frames, 1, 1)
    Ks = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(batch_size, frames, 1, 1)
    return viewmats, Ks


def _grads(model: nn.Module) -> list[Tensor]:
    """Collect populated gradients on the model's trainable parameters."""
    return [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]


# --- CPU structural tests (no transformer forward) ---


def test_hy_model_builds_and_attn_mode_propagates():
    """Tiny HY model builds from config dims; set_attn_mode reaches every block."""
    model = _build_tiny_hy()
    model.set_attn_mode("flex_tf")
    assert model.attn_mode == "flex_tf"
    assert len(model.double_blocks) == 2
    assert all(block.attn_mode == "flex_tf" for block in model.double_blocks)


def test_hy_adapter_encode_text():
    """HYAdapter.encode_text yields (text_states [B,L,C], mask [B,L] bool)."""
    adapter = HYAdapter(text_len=_TEXT_LEN, text_dim=_TEXT_DIM)
    text_states, attn_mask = adapter.encode_text(
        ["a", "b"], batch_size=2, device=torch.device("cpu")
    )
    assert text_states.shape == (2, _TEXT_LEN, _TEXT_DIM)
    assert attn_mask.shape == (2, _TEXT_LEN)
    assert attn_mask.dtype == torch.bool
    assert attn_mask.all()


def test_hy_recipe_wires_adapter_encode_text():
    """A recipe built with an HYAdapter exposes the adapter's encode_text contract."""
    recipe = ARCDRecipe(
        adapter=HYAdapter(text_len=_TEXT_LEN, text_dim=_TEXT_DIM),
        optimizer=dict(lr=1e-4),
        discrete_cd_n=8,
        guidance_scale=5.0,
    )
    text_states, attn_mask = recipe.adapter.encode_text(
        None, batch_size=1, device=torch.device("cpu")
    )
    assert text_states.shape == (1, _TEXT_LEN, _TEXT_DIM)
    assert attn_mask.shape == (1, _TEXT_LEN)
    assert attn_mask.dtype == torch.bool


def test_hy_ode_build_optimizers():
    """build_optimizers exposes a ``main`` torch optimizer over model params."""
    model = _build_tiny_hy()
    recipe = ARODERecipe(
        loss=ODERegressionLoss(),
        adapter=HYAdapter(text_len=_TEXT_LEN, text_dim=_TEXT_DIM),
        optimizer=dict(lr=1e-4),
    )
    optimizers = recipe.build_optimizers(model)
    assert "main" in optimizers
    assert isinstance(optimizers["main"], torch.optim.Optimizer)


# --- CUDA-gated full train-step tests (batch_size=1, FlexAttention TF path) ---


@cuda_only
def test_hy_ode_regression_train_step():
    """Stage 2a: one HYODERegression step yields finite loss + finite grads."""
    torch.manual_seed(0)
    model = _build_tiny_hy(device="cuda")
    recipe = ARODERecipe(
        loss=ODERegressionLoss(),
        adapter=HYAdapter(text_len=_TEXT_LEN, text_dim=_TEXT_DIM, dtype="bfloat16"),
        optimizer=dict(lr=1e-4),
        batch_preprocessors=[
            dict(type="minwm.processors.latent:LatentToDevice"),
            dict(
                type="minwm.processors.ode:ODETrajectorySample",
                denoising_step_list=[1000, 750, 500, 250],
            ),
        ],
    )
    optimizers = recipe.build_optimizers(model)

    B, N = 1, 5  # N = len(denoising_step_list) + 1 (last entry = clean)
    viewmats, Ks = _camera(B, _F, device="cuda")
    batch = {
        "ode_latent": torch.randn(B, N, _F, _C, _H, _W, device="cuda"),
        "prompts": ["mock a"],
        "viewmats": viewmats,
        "Ks": Ks,
    }

    with torch.autocast("cuda", dtype=torch.bfloat16):
        metrics = recipe.train_one_step(model, batch, optimizers, step=0)

    assert "loss" in metrics
    assert math.isfinite(metrics["loss"])
    grads = _grads(model)
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@cuda_only
def test_hy_consistency_distillation_train_step():
    """Stage 2b: one HYConsistencyDistillation step yields finite loss + grads."""
    torch.manual_seed(0)
    model = _build_tiny_hy(device="cuda")
    recipe = ARCDRecipe(
        adapter=HYAdapter(text_len=_TEXT_LEN, text_dim=_TEXT_DIM, dtype="bfloat16"),
        optimizer=dict(lr=1e-4),
        discrete_cd_n=8,
        guidance_scale=5.0,
        ema_decay=0.95,
    )
    optimizers = recipe.build_optimizers(model)

    # Teacher + EMA are trainer-owned auxiliary_models; on a single process they
    # are plain (unsharded) copies the recipe seeds from the student on step 0.
    auxiliary_models = {
        "teacher": _build_tiny_hy(device="cuda"),
        "ema": _build_tiny_hy(device="cuda"),
    }

    B = 1
    viewmats, Ks = _camera(B, _F, device="cuda")
    batch = {
        "clean_latent": torch.randn(B, _F, _C, _H, _W, device="cuda"),
        "prompts": ["mock a"],
        "viewmats": viewmats,
        "Ks": Ks,
    }

    with torch.autocast("cuda", dtype=torch.bfloat16):
        metrics = recipe.train_one_step(
            model, batch, optimizers, step=0, auxiliary_models=auxiliary_models
        )

    assert math.isfinite(metrics["loss"])
    grads = _grads(model)
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
