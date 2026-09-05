"""CPU test that ARCDRecipe excludes fixed-prefix frames from the CD loss.

The full teacher-forcing forward is CUDA-only for real backbones, but the
prefix-exclusion slicing (``cm_pred_t[:, prefix_frames:]``) is pure bookkeeping
around the forward. A stub adapter whose ``denoise`` is a scalar-scaled identity
lets the whole ``train_one_step`` run on CPU, so the slice that this PR exists to
add is actually exercised — a dropped or off-by-one slice changes the frame count
handed to ``consistency_loss`` and fails the assertion.
"""

import torch
from torch import nn

import minwm.engine.training.recipes.ar_cd as ar_cd_mod
from minwm.distributed.rng import get_rng_states_tracker
from minwm.engine.training.recipes.ar_cd import ARCDRecipe


class _ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))


class _Adapter:
    dtype = torch.float32

    def conditioning(self, batch, batch_size, device):
        return {}

    def null_conditioning(self, batch, batch_size, device):
        return {}

    def denoise(self, net, *, noisy, timestep=None, **kwargs):
        return net.w * noisy


def _run_step(initial_latent):
    torch.manual_seed(0)
    get_rng_states_tracker().reset()
    model = _ScalarModel()
    recipe = ARCDRecipe(
        adapter=_Adapter(),
        optimizer=dict(lr=1e-4),
        discrete_cd_n=4,
        guidance_scale=1.0,
    )
    aux = {"teacher": _ScalarModel(), "ema": _ScalarModel()}
    optimizers = recipe.build_optimizers(model)

    B, F, C, H, W = 1, 3, 2, 2, 2
    batch = {"clean_latent": torch.randn(B, F, C, H, W), "prompts": ["x"]}
    if initial_latent is not None:
        batch["initial_latent"] = initial_latent

    recipe.train_one_step(model, batch, optimizers, step=0, auxiliary_models=aux)


def test_cd_loss_excludes_prefix_frames(monkeypatch):
    seen = []
    real_loss = ar_cd_mod.consistency_loss

    def spy(cm_pred_t, cm_pred_t_next, reduction="mean"):
        seen.append(cm_pred_t.shape[1])
        return real_loss(cm_pred_t, cm_pred_t_next, reduction=reduction)

    monkeypatch.setattr(ar_cd_mod, "consistency_loss", spy)

    _run_step(initial_latent=torch.randn(1, 1, 2, 2, 2))
    _run_step(initial_latent=None)

    # With a 1-frame prefix the loss sees F-1=2 frames; with no prefix it sees all 3.
    assert seen == [2, 3]
