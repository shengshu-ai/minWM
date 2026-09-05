"""Tests for minwm.modeling.build_model and minwm.engine (BaseTrainer/Recipe)."""

import os
import sys
import types

import pytest
import torch
from torch import nn

from minwm.config.schema import Monitor
from minwm.engine import BaseTrainer, Recipe
from minwm.engine import events as events_mod
from minwm.engine.events import EventStorage, get_event_storage, record_time
from minwm.engine.monitor import BaseWriter, MetricsProcessor
from minwm.modeling import build_model


@pytest.fixture
def fake_module():
    """Register tiny model + recipe classes in an importable module."""
    mod = types.ModuleType("minwm_test_fakes")

    class TinyModel(nn.Module):
        def __init__(self, dim=4):
            super().__init__()
            self.lin = nn.Linear(dim, dim)

        def forward(self, x):
            return self.lin(x)

    class PretrainedModel(nn.Module):
        def __init__(self):
            super().__init__()

        @classmethod
        def from_pretrained(cls, path, **kw):
            obj = cls()
            obj.path = path
            obj.kw = kw
            return obj

    class TinyRecipe(Recipe):
        def build_optimizers(self, model):
            return {"main": torch.optim.SGD(model.parameters(), lr=0.1)}

        def train_one_step(self, model, batch, optimizers, step, auxiliary_models=None):
            opt = optimizers["main"]
            device = next(model.parameters()).device
            opt.zero_grad()
            loss = model(batch.to(device)).pow(2).mean()
            loss.backward()
            opt.step()
            return {"loss": loss.item()}

    mod.TinyModel = TinyModel
    mod.PretrainedModel = PretrainedModel
    mod.TinyRecipe = TinyRecipe
    sys.modules["minwm_test_fakes"] = mod
    yield mod
    del sys.modules["minwm_test_fakes"]


def test_build_model_plain(fake_module):
    m = build_model({"type": "minwm_test_fakes:TinyModel", "dim": 8})
    assert isinstance(m, fake_module.TinyModel)
    assert m.lin.in_features == 8


def test_build_model_requires_cls():
    with pytest.raises(ValueError, match="type"):
        build_model({"dim": 4})


def test_build_model_from_pretrained(fake_module):
    m = build_model(
        {
            "type": "minwm_test_fakes:PretrainedModel",
            "_from_pretrained": "/ckpt/path",
            "scale": 2,
        }
    )
    assert m.path == "/ckpt/path"
    assert m.kw == {"scale": 2}


def test_build_model_from_pretrained_unsupported(fake_module):
    with pytest.raises(TypeError, match="from_pretrained"):
        build_model(
            {
                "type": "minwm_test_fakes:TinyModel",
                "_from_pretrained": "/ckpt",
            }
        )


def test_build_recipe_type_check(fake_module):
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyModel"},  # not a Recipe
        "training": {"max_steps": 1},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4)]

    with pytest.raises(TypeError, match="Recipe"):
        T(cfg)


def test_trainer_runs_steps(fake_module):
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 15, "log_interval": 0, "ckpt_interval": 0},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4) for _ in range(4)]

    t = T(cfg)
    assert isinstance(t.model, fake_module.TinyModel)
    assert isinstance(t.recipe, fake_module.TinyRecipe)
    assert t.model.training
    assert set(t.optimizers) == {"main"}
    t.train()
    assert t.step == 15


def test_trainer_checkpoint_hook_fires(fake_module):
    calls = {"save": 0}
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 10, "log_interval": 0, "ckpt_interval": 5},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4) for _ in range(4)]

        def save_checkpoint(self):
            calls["save"] += 1

    T(cfg).train()
    # fires at step 5, 10, plus the final save after the loop
    assert calls["save"] == 3


def test_dataloader_cycles_when_exhausted(fake_module):
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 10, "log_interval": 0, "ckpt_interval": 0},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4) for _ in range(3)]

    t = T(cfg)
    t.train()
    assert t.step == 10


# ---- EventStorage --------------------------------------------------------


def test_event_storage():
    storage = EventStorage(start_iter=5)
    assert storage.iter == 5
    with storage:
        assert get_event_storage() is storage
        storage.put_scalar("loss", 1.5)
        storage.put_scalars(a=1.0, b=2.0)
        assert storage.latest() == {"loss": 1.5, "a": 1.0, "b": 2.0}
        storage.iter = 7
        assert storage.iter == 7
        storage.clear()
        assert storage.latest() == {}
        assert storage.iter == 7  # clear leaves the iteration counter alone
    assert events_mod._CURRENT == []


def test_event_storage_pops_on_exception():
    storage = EventStorage()
    with pytest.raises(RuntimeError):
        with storage:
            assert events_mod._CURRENT == [storage]
            raise RuntimeError("boom")
    assert events_mod._CURRENT == []  # __exit__ pops even on a mid-context crash
    with pytest.raises(AssertionError):
        get_event_storage()


def test_record_time_records_inside_context():
    storage = EventStorage()
    with storage:
        with record_time("forward"):
            pass
        with record_time("backward"):
            pass
    assert "time/forward(ms)" in storage.latest()
    assert "time/backward(ms)" in storage.latest()
    assert storage.latest()["time/forward(ms)"] >= 0.0


def test_record_time_is_noop_outside_context():
    assert events_mod._CURRENT == []
    with record_time("forward"):  # no active storage -> body runs, nothing recorded
        pass
    assert events_mod._CURRENT == []


def test_record_time_records_on_exception():
    storage = EventStorage()
    with pytest.raises(RuntimeError):
        with storage:
            with record_time("forward"):
                raise RuntimeError("boom")
    # the timer is in a finally, so the partial duration is still recorded
    assert "time/forward(ms)" in storage.latest()


# ---- MetricsProcessor ----------------------------------------------------


class _CaptureWriter(BaseWriter):
    """In-memory writer recording every (step, metrics) it is handed."""

    def __init__(self):
        self.captured = []

    def log(self, metrics, step):
        self.captured.append((step, dict(metrics)))


def test_metrics_processor_throughput_no_memory_on_cpu():
    proc = MetricsProcessor(Monitor(backends=()), log_dir="")
    writer = _CaptureWriter()
    proc._writers = writer  # inject; skips lazy _build_writers
    proc.start_window(0)
    proc.log(10, {"loss": 0.5})

    assert len(writer.captured) == 1
    step, merged = writer.captured[0]
    assert step == 10
    assert merged["loss"] == 0.5
    assert "steps/sec" in merged and "ms/step" in merged
    if not torch.cuda.is_available():
        assert not any(k.startswith("memory/") for k in merged)


# ---- checkpoint roundtrip ------------------------------------------------


def test_checkpoint_roundtrip(fake_module, tmp_path):
    out = str(tmp_path)
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 6, "log_interval": 0, "ckpt_interval": 0, "output_dir": out},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4) for _ in range(4)]

    t = T(cfg)
    t.train()
    assert t.step == 6
    ckpt_dir = os.path.join(out, "ckpts")
    assert os.path.exists(os.path.join(ckpt_dir, "latest.txt"))
    assert os.path.exists(os.path.join(ckpt_dir, "checkpoint_6.pt"))
    assert not any(p.endswith(".tmp") for p in os.listdir(ckpt_dir))
    saved_weight = t.model.lin.weight.detach().clone()
    saved_opt_lr = t.optimizers["main"].state_dict()["param_groups"][0]["lr"]

    resume_cfg = dict(cfg)
    resume_cfg["checkpoint"] = {"resume": True}
    t2 = T(resume_cfg)
    assert t2.step == 0  # not yet loaded
    t2.load_checkpoint()
    assert t2.step == 6
    assert torch.equal(t2.model.lin.weight, saved_weight)
    assert t2.optimizers["main"].state_dict()["param_groups"][0]["lr"] == saved_opt_lr


def test_pretrained_seeds_model_and_auxiliary(fake_module, tmp_path):
    # A source ckpt for the primary model and one for the aux nets, distinct so we
    # can tell them apart after load.
    model_w = str(tmp_path / "model.pt")
    aux_w = str(tmp_path / "aux.pt")
    m_state = {"lin.weight": torch.ones(4, 4), "lin.bias": torch.ones(4)}
    a_state = {"lin.weight": torch.full((4, 4), 2.0), "lin.bias": torch.full((4,), 2.0)}
    torch.save({"model": m_state}, model_w)
    torch.save({"model": a_state}, aux_w)

    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "auxiliary_models": {
            "real_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4, "trainable": False},
            "fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        },
        "checkpoint": {
            "pretrained": model_w,
            "auxiliary_pretrained": {"real_score": aux_w, "fake_score": aux_w},
        },
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    t.load_checkpoint()
    assert torch.equal(t.model.lin.weight.cpu(), torch.ones(4, 4))
    assert torch.equal(t.auxiliary_models["real_score"].lin.weight.cpu(), torch.full((4, 4), 2.0))
    assert torch.equal(t.auxiliary_models["fake_score"].lin.weight.cpu(), torch.full((4, 4), 2.0))


def test_auxiliary_pretrained_only_leaves_model_untouched(fake_module, tmp_path):
    aux_w = str(tmp_path / "aux.pt")
    torch.save(
        {"model": {"lin.weight": torch.full((4, 4), 3.0), "lin.bias": torch.zeros(4)}}, aux_w
    )

    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "auxiliary_models": {"fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}},
        "checkpoint": {"auxiliary_pretrained": {"fake_score": aux_w}},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    before = t.model.lin.weight.detach().clone()
    t.load_checkpoint()
    assert torch.equal(t.model.lin.weight, before)  # no pretrained -> model untouched
    assert torch.equal(t.auxiliary_models["fake_score"].lin.weight.cpu(), torch.full((4, 4), 3.0))


def test_auxiliary_pretrained_unknown_name_raises(fake_module, tmp_path):
    aux_w = str(tmp_path / "aux.pt")
    torch.save({"model": {"lin.weight": torch.zeros(4, 4), "lin.bias": torch.zeros(4)}}, aux_w)

    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "auxiliary_models": {"fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}},
        "checkpoint": {"auxiliary_pretrained": {"nonexistent": aux_w}},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    with pytest.raises(KeyError, match="nonexistent"):
        t.load_checkpoint()


def test_empty_auxiliary_pretrained_is_noop(fake_module, tmp_path):
    """Empty auxiliary_pretrained dict should be a no-op (no load, no error)."""
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "auxiliary_models": {"fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}},
        "checkpoint": {"auxiliary_pretrained": {}},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    before = t.auxiliary_models["fake_score"].lin.weight.detach().clone()
    t.load_checkpoint()
    assert torch.equal(t.auxiliary_models["fake_score"].lin.weight, before)


def test_pretrained_without_auxiliary_pretrained_leaves_aux_untouched(fake_module, tmp_path):
    """Setting pretrained without auxiliary_pretrained should load model, leave aux untouched."""
    model_w = str(tmp_path / "model.pt")
    torch.save(
        {"model": {"lin.weight": torch.full((4, 4), 5.0), "lin.bias": torch.zeros(4)}}, model_w
    )

    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "auxiliary_models": {"fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}},
        "checkpoint": {"pretrained": model_w},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    aux_before = t.auxiliary_models["fake_score"].lin.weight.detach().clone()
    t.load_checkpoint()
    assert torch.equal(t.model.lin.weight.cpu(), torch.full((4, 4), 5.0))  # model loaded
    assert torch.equal(t.auxiliary_models["fake_score"].lin.weight, aux_before)  # aux untouched


def test_auxiliary_pretrained_state_dict_mismatch_raises(fake_module, tmp_path):
    """Mismatched state_dict keys in aux weight file should raise (strict=True)."""
    aux_w = str(tmp_path / "aux.pt")
    # Save a state_dict with wrong key names
    torch.save({"model": {"wrong_key.weight": torch.zeros(4, 4)}}, aux_w)

    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "auxiliary_models": {"fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}},
        "checkpoint": {"auxiliary_pretrained": {"fake_score": aux_w}},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    # load_state_dict with strict=True (the default in _load_pretrained_into) should raise
    with pytest.raises((RuntimeError, KeyError)):
        t.load_checkpoint()


def test_resume_without_checkpoint_raises(fake_module, tmp_path):
    """``resume=True`` with no ``latest.txt`` must raise, not silently fall back.

    A silent fresh-start (or worse, a fall-through to ``pretrained``) would train
    from the wrong weights — e.g. a CD stage seeding student/teacher/ema from the
    base model instead of the promoted TF checkpoint. The run must fail loudly.
    """
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "checkpoint": {"resume": True},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4)]

    t = T(cfg)
    with pytest.raises(FileNotFoundError):
        t.load_checkpoint()  # no latest.txt -> must raise, not start fresh


def test_resume_missing_auxiliary_raises(fake_module, tmp_path):
    """Resuming a checkpoint that lacks a config-declared aux model must raise.

    Same class of bug as the missing-pointer case: a checkpoint without the
    ``fake_score`` aux would otherwise silently leave it at freshly-built weights.
    """
    out = str(tmp_path)
    # Save a checkpoint from a run WITHOUT the auxiliary model.
    base_cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 4, "log_interval": 0, "ckpt_interval": 0, "output_dir": out},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4) for _ in range(4)]

    T(base_cfg).train()

    # Resume with a config that DOES declare an aux model -> checkpoint lacks it.
    resume_cfg = dict(base_cfg)
    resume_cfg["auxiliary_models"] = {
        "fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}
    }
    resume_cfg["checkpoint"] = {"resume": True}
    t = T(resume_cfg)
    with pytest.raises(KeyError):
        t.load_checkpoint()


def test_resume_missing_auxiliary_allowed_with_flag(fake_module, tmp_path):
    """``allow_partial_resume=True`` downgrades the missing-aux error to a warning."""
    out = str(tmp_path)
    base_cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 4, "log_interval": 0, "ckpt_interval": 0, "output_dir": out},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return [torch.randn(2, 4) for _ in range(4)]

    T(base_cfg).train()

    resume_cfg = dict(base_cfg)
    resume_cfg["auxiliary_models"] = {
        "fake_score": {"type": "minwm_test_fakes:TinyModel", "dim": 4}
    }
    resume_cfg["checkpoint"] = {"resume": True, "allow_partial_resume": True}
    t = T(resume_cfg)
    t.load_checkpoint()  # no raise
    assert t.step == 4


def test_extract_pretrained_native_and_raw_are_identity():
    """Native ``{"model": ...}`` and raw state dicts pass through unchanged.

    Guards the backward-compat contract: for the two shapes the loader handled
    before the legacy-generator branch existed, ``_extract_pretrained_state``
    must return the exact same object (no key rewriting), so earlier stages'
    checkpoints load identically.
    """
    from minwm.engine.trainer import _extract_pretrained_state

    native = {"model": {"lin.weight": torch.ones(2), "model.foo": torch.zeros(2)}}
    assert _extract_pretrained_state(native) is native["model"]

    raw = {"lin.weight": torch.ones(2), "model.legit.weight": torch.zeros(2)}
    # no wrapper key -> returned as-is, and a legit ``model.``-prefixed param is
    # NOT stripped (stripping is confined to the legacy-generator branch).
    assert _extract_pretrained_state(raw) is raw


def test_extract_pretrained_legacy_generator_unwraps_and_strips():
    """Legacy DMD ``{"generator": ...}`` ckpts: unwrap + strip ``model.`` prefix."""
    from minwm.engine.trainer import _extract_pretrained_state

    gen = {
        "generator": {
            "model.lin.weight": torch.ones(4, 4),
            "model._fsdp_wrapped_module.lin.bias": torch.zeros(4),
        }
    }
    out = _extract_pretrained_state(gen)
    assert set(out) == {"lin.weight", "lin.bias"}

    # generator_ema wins over generator (promoted EMA weights).
    both = {
        "generator": {"model.lin.weight": torch.ones(1)},
        "generator_ema": {"model.lin.weight": torch.full((1,), 9.0)},
    }
    assert torch.equal(_extract_pretrained_state(both)["lin.weight"], torch.full((1,), 9.0))


def test_pretrained_seeds_model_from_legacy_generator_ckpt(fake_module, tmp_path):
    """End-to-end: a legacy generator ckpt seeds the model via load_checkpoint."""
    gen_w = str(tmp_path / "generator.pt")
    torch.save(
        {
            "generator": {
                "model.lin.weight": torch.full((4, 4), 7.0),
                "model.lin.bias": torch.zeros(4),
            }
        },
        gen_w,
    )
    cfg = {
        "model": {"type": "minwm_test_fakes:TinyModel", "dim": 4},
        "recipe": {"type": "minwm_test_fakes:TinyRecipe"},
        "training": {"max_steps": 0, "output_dir": str(tmp_path)},
        "checkpoint": {"pretrained": gen_w},
    }

    class T(BaseTrainer):
        def build_dataloader(self, data_cfg):
            return None

    t = T(cfg)
    t.load_checkpoint()
    assert torch.equal(t.model.lin.weight.cpu(), torch.full((4, 4), 7.0))


class _EmaStubTrainer:
    """Minimal trainer surface for exercising _TrainerAppState EMA-seed handling.

    Avoids the full BaseTrainer.__init__ (distributed/checkpoint deps); provides
    just the attributes state_dict()/load_state_dict() touch.
    """

    def __init__(self, has_ema_flag=True):
        self.step = 6
        self.model = nn.Linear(4, 4)
        self.auxiliary_models = {"generator_ema": nn.Linear(4, 4)}
        self.optimizers = {}
        self.cfg = types.SimpleNamespace(
            training=types.SimpleNamespace(no_load_rng=True),
            checkpoint=types.SimpleNamespace(allow_partial_resume=True),
        )
        self.recipe = types.SimpleNamespace()
        if has_ema_flag:
            self.recipe._generator_ema_seeded = True

    def _module_for_optimizer(self, opt):
        return self.model


def test_app_state_persists_generator_ema_seeded():
    """The EMA-seeded flag survives a save/restore round-trip so a resumed run
    does not re-seed (overwrite) the restored generator EMA. Regression for the
    resume-reseed bug: the flag was in-memory only and defaulted False on resume."""
    from minwm.engine.trainer import _TrainerAppState

    saver = _EmaStubTrainer(has_ema_flag=True)
    saver.recipe._generator_ema_seeded = True
    sd = _TrainerAppState(saver).state_dict()
    assert "recipe/generator_ema_seeded" in sd

    # Fresh trainer whose recipe defaults the flag to False, as on a real resume.
    loader = _EmaStubTrainer(has_ema_flag=True)
    loader.recipe._generator_ema_seeded = False
    _TrainerAppState(loader).load_state_dict(sd)
    assert loader.recipe._generator_ema_seeded is True


def test_app_state_legacy_ckpt_infers_seeded_from_aux():
    """A checkpoint predating the flag (no ``recipe/generator_ema_seeded``) but
    carrying restored EMA aux weights must still resolve to seeded=True, so the
    first post-resume update does not clobber the restored EMA."""
    from minwm.engine.trainer import _TrainerAppState

    loader = _EmaStubTrainer(has_ema_flag=True)
    legacy_sd = {
        "step": torch.tensor(6),
        "model": loader.model.state_dict(),
        "aux/generator_ema": loader.auxiliary_models["generator_ema"].state_dict(),
    }
    loader.recipe._generator_ema_seeded = False
    _TrainerAppState(loader).load_state_dict(legacy_sd)
    assert loader.recipe._generator_ema_seeded is True


def test_app_state_from_scratch_leaves_unseeded():
    """No flag and no restored EMA aux (fresh run through app-state load) leaves
    the recipe unseeded so the first update performs the initial copy_params seed."""
    from minwm.engine.trainer import _TrainerAppState

    loader = _EmaStubTrainer(has_ema_flag=True)
    # allow_partial_resume=True lets the missing aux/generator_ema slide (warn+skip),
    # so no aux weights are restored -> the flag must stay False.
    scratch_sd = {"step": torch.tensor(0), "model": loader.model.state_dict()}
    loader.recipe._generator_ema_seeded = False
    _TrainerAppState(loader).load_state_dict(scratch_sd)
    assert loader.recipe._generator_ema_seeded is False


def test_app_state_recipe_without_flag_is_untouched():
    """Recipes that do not own the EMA flag (cd/ode/tf/sft) must not gain it."""
    from minwm.engine.trainer import _TrainerAppState

    saver = _EmaStubTrainer(has_ema_flag=False)
    sd = _TrainerAppState(saver).state_dict()
    assert "recipe/generator_ema_seeded" not in sd

    loader = _EmaStubTrainer(has_ema_flag=False)
    _TrainerAppState(loader).load_state_dict(
        {
            "step": torch.tensor(3),
            "model": loader.model.state_dict(),
            "aux/generator_ema": loader.auxiliary_models["generator_ema"].state_dict(),
        }
    )
    assert not hasattr(loader.recipe, "_generator_ema_seeded")
