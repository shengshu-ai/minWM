"""Tests for BiSFTRecipe config wiring (scheduler knobs reach the scheduler).

These guard the build path the trainer uses (``minwm.config.lazy.build`` ->
``BiSFTRecipe(**cfg)``): the Wan phase0 config passes ``sigma_min`` /
``extra_one_step`` at the recipe level, so BiSFTRecipe must accept and forward
them to its FlowMatchingScheduler through the shared flow base.
"""

import torch

from minwm.config.lazy import build
from minwm.engine.training.recipes.bi_sft import BiSFTRecipe
from minwm.sampling.schedulers import FlowMatchingScheduler


def _recipe_cfg(**overrides) -> dict:
    cfg = dict(
        type="minwm.engine.training.recipes.bi_sft:BiSFTRecipe",
        loss=dict(type="minwm.engine.training.losses:FlowMatchingLoss"),
        timestep_shift=5.0,
    )
    cfg.update(overrides)
    return cfg


class TestFlowRecipeSchedulerWiring:
    def test_builds_with_legacy_schedule_knobs(self):
        recipe = build(_recipe_cfg(sigma_min=0.0, extra_one_step=True))
        assert isinstance(recipe, BiSFTRecipe)
        assert recipe.scheduler.sigma_min == 0.0
        assert recipe.scheduler.extra_one_step is True

    def test_defaults_match_bare_scheduler(self):
        recipe = build(_recipe_cfg())
        bare = FlowMatchingScheduler()
        assert recipe.scheduler.sigma_min == bare.sigma_min
        assert recipe.scheduler.extra_one_step == bare.extra_one_step

    def test_legacy_knobs_change_sigma_table(self):
        legacy = build(_recipe_cfg(sigma_min=0.0, extra_one_step=True)).scheduler
        default = build(_recipe_cfg()).scheduler
        assert not torch.equal(legacy.sigmas, default.sigmas)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class TestGradNormGuard:
    """Two-threshold clip/skip behaviour of ``Recipe.optimizer_step`` (single proc)."""

    def _recipe(self, **kw):
        return build(_recipe_cfg(**kw))

    def test_skip_below_scale_threshold_raises(self):
        import pytest

        with pytest.raises(ValueError):
            self._recipe(max_grad_norm=10.0, skip_grad_norm=1.0)

    def _step_with_grad(self, recipe, grad_scale: float) -> bool:
        model = _Tiny()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.lin.weight.detach().clone()
        # Forge a known-norm gradient by backward on a scaled sum.
        out = model(torch.ones(1, 4))
        loss = out.sum() * grad_scale
        stepped = recipe.optimizer_step(loss, opt, model.parameters())
        moved = not torch.equal(before, model.lin.weight.detach())
        assert stepped == moved  # stepped iff weights changed
        return stepped

    def test_huge_grad_skips_step(self):
        recipe = self._recipe(max_grad_norm=1.0, skip_grad_norm=10.0)
        # grad_scale=1e6 -> norm far above 10 -> skip, weights unchanged.
        assert self._step_with_grad(recipe, 1e6) is False

    def test_small_grad_steps(self):
        recipe = self._recipe(max_grad_norm=1.0, skip_grad_norm=10.0)
        assert self._step_with_grad(recipe, 1e-3) is True

    def test_skip_disabled_always_steps(self):
        recipe = self._recipe(max_grad_norm=1.0)  # no skip threshold
        assert recipe.skip_grad_norm is None
        assert self._step_with_grad(recipe, 1e6) is True
