"""Tests for the minwm config system: locate / build / lazy / load / merge."""

import importlib
import re
import textwrap
from pathlib import Path

import pytest

from minwm.config import MWMConfig, build, lazy, load, locate, merge


def test_locate_function():
    fn = locate("math:sqrt")
    assert fn(16) == 4.0


def test_locate_class():
    cls = locate("fractions:Fraction")
    assert cls.__name__ == "Fraction"


def test_locate_rejects_ambiguous_dotted_path():
    with pytest.raises(ValueError, match="Name.*module:Name"):
        locate("math.sqrt")


def test_locate_short_framework_name():
    cls = locate("FlowMatchingLoss")
    assert cls.__module__ == "minwm.engine.training.losses"


def test_locate_short_model_name_uses_public_facade():
    cls = locate("Wan21Adapter")
    assert cls.__module__ == "minwm.modeling.wan21.adapter"


@pytest.mark.parametrize(
    "short_name, module",
    [
        ("Wan21Model", "minwm.modeling.wan21.model"),
        ("CausalWan21Model", "minwm.modeling.wan21.causal"),
        ("Wan21Adapter", "minwm.modeling.wan21.adapter"),
        ("Wan21TextEncoder", "minwm.modeling.wan21.text_encoder"),
        ("Wan21VAE", "minwm.modeling.wan21.vae"),
    ],
)
def test_locate_resolves_all_wan21_public_names(short_name, module):
    if short_name == "Wan21TextEncoder":
        # The vendored umt5 text encoder imports optional deps (ftfy, regex,
        # transformers) at module load; these are absent from the minimal CPU
        # CI env, so skip resolving this one name when they are unavailable.
        for dep in ("ftfy", "regex", "transformers"):
            pytest.importorskip(dep)
    cls = locate(short_name)
    assert cls.__module__ == module


@pytest.mark.parametrize(
    "retired_name",
    ["WanModel", "CausalWanModel", "WanAdapter", "WanTextEncoder", "WanVAE"],
)
def test_locate_rejects_retired_wan_names(retired_name):
    # The un-versioned names were renamed to Wan21*; they must no longer resolve.
    with pytest.raises(ImportError):
        locate(retired_name)


def test_locate_missing_short_name_requires_public_export():
    with pytest.raises(ImportError, match="not exported"):
        locate("NotAFrameworkObject")


def test_locate_rejects_ambiguous_short_name(monkeypatch):
    import sys
    import types

    lazy_config = importlib.import_module("minwm.config.lazy")
    modules = []
    for module_name in ("test_facade_a", "test_facade_b"):
        module = types.ModuleType(module_name)
        module.SharedType = object
        module.__all__ = ["SharedType"]
        monkeypatch.setitem(sys.modules, module_name, module)
        modules.append(module_name)
    monkeypatch.setattr(lazy_config, "DEFAULT_MODULES", tuple(modules))

    with pytest.raises(ImportError, match="ambiguous short type"):
        locate("SharedType")


def test_locate_missing_attr():
    with pytest.raises(ImportError, match="not found"):
        locate("math:nonexistent_fn")


def test_locate_isolates_broken_default_module(monkeypatch):
    # A facade that fails to import must not block resolution of a name owned
    # by a different, healthy facade scanned in the same loop.
    import sys
    import types

    lazy_config = importlib.import_module("minwm.config.lazy")

    good = types.ModuleType("test_facade_good")
    good.WantedType = object
    good.__all__ = ["WantedType"]
    monkeypatch.setitem(sys.modules, "test_facade_good", good)

    real_import = lazy_config.importlib.import_module

    def flaky_import(name, *args, **kwargs):
        if name == "test_facade_broken":
            raise ImportError("optional dependency missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(lazy_config.importlib, "import_module", flaky_import)
    monkeypatch.setattr(lazy_config, "DEFAULT_MODULES", ("test_facade_broken", "test_facade_good"))

    assert locate("WantedType") is object


def test_locate_reports_broken_module_when_name_absent(monkeypatch):
    import sys
    import types

    lazy_config = importlib.import_module("minwm.config.lazy")

    good = types.ModuleType("test_facade_good2")
    good.__all__ = []
    monkeypatch.setitem(sys.modules, "test_facade_good2", good)

    real_import = lazy_config.importlib.import_module

    def flaky_import(name, *args, **kwargs):
        if name == "test_facade_broken2":
            raise ImportError("optional dependency missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(lazy_config.importlib, "import_module", flaky_import)
    monkeypatch.setattr(
        lazy_config, "DEFAULT_MODULES", ("test_facade_broken2", "test_facade_good2")
    )

    with pytest.raises(ImportError, match="could not import"):
        locate("NoSuchType")


def test_locate_reports_facade_export_that_cannot_resolve(monkeypatch):
    # ``__all__`` advertises the name but the object can't be produced.
    import sys
    import types

    lazy_config = importlib.import_module("minwm.config.lazy")
    module = types.ModuleType("test_facade_stale")
    module.__all__ = ["Ghost"]  # advertised, but no attribute ``Ghost``
    monkeypatch.setitem(sys.modules, "test_facade_stale", module)
    monkeypatch.setattr(lazy_config, "DEFAULT_MODULES", ("test_facade_stale",))

    with pytest.raises(ImportError, match="could not be resolved"):
        locate("Ghost")


@pytest.mark.parametrize("bad", [":", ":X", "X:"])
def test_locate_rejects_malformed_colon_paths(bad):
    with pytest.raises(ValueError, match="module:Name"):
        locate(bad)


def test_build_resolves_short_type():
    obj = build({"type": "FlowMatchingLoss"})
    assert type(obj).__name__ == "FlowMatchingLoss"


def test_build_instantiates_cls():
    obj = build({"type": "fractions:Fraction", "numerator": 3, "denominator": 4})
    assert str(obj) == "3/4"


def test_build_plain_dict_passthrough():
    out = build({"a": 1, "b": {"c": 2}})
    assert out == {"a": 1, "b": {"c": 2}}


def test_build_nested_cls():
    cfg = {
        "outer": {"type": "fractions:Fraction", "numerator": 1, "denominator": 2},
        "scalar": 5,
    }
    out = build(cfg)
    assert str(out["outer"]) == "1/2"
    assert out["scalar"] == 5


def test_lazy_returns_cls_and_kwargs():
    cls, kwargs = lazy({"type": "fractions:Fraction", "numerator": 5})
    assert cls.__name__ == "Fraction"
    assert kwargs == {"numerator": 5}


def test_lazy_requires_cls_key():
    with pytest.raises(ValueError, match="type"):
        lazy({"numerator": 5})


def test_merge_override():
    assert merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_merge_deep():
    out = merge({"x": {"a": 1, "b": 2}}, {"x": {"b": 9}})
    assert out == {"x": {"a": 1, "b": 9}}


def test_load_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  type: fractions:Fraction\n  numerator: 7\n")
    cfg = load(str(p))
    assert cfg["model"]["type"] == "fractions:Fraction"
    assert cfg["model"]["numerator"] == 7


def test_load_py(tmp_path):
    p = tmp_path / "cfg.py"
    p.write_text(textwrap.dedent("""
            model = {"type": "fractions:Fraction", "numerator": 9}
            max_steps = 100
            _private = "hidden"
            """))
    cfg = load(str(p))
    assert cfg["model"]["numerator"] == 9
    assert cfg["max_steps"] == 100
    assert "_private" not in cfg


def test_load_py_base_merge(tmp_path):
    base = tmp_path / "base.py"
    base.write_text(textwrap.dedent("""
            training = {"sp_size": 1, "tp_size": 1, "eval_interval": 0}
            checkpoint = {"resume": False}
            """))
    leaf = tmp_path / "leaf.py"
    leaf.write_text(textwrap.dedent("""
            _base_ = "base.py"
            training = {"sp_size": 4, "max_steps": 3}
            model = {"dim": 128}
            """))
    cfg = load(str(leaf))
    assert cfg["training"]["sp_size"] == 4  # leaf overrides base
    assert cfg["training"]["tp_size"] == 1  # inherited from base
    assert cfg["training"]["max_steps"] == 3  # leaf adds a key
    assert cfg["checkpoint"]["resume"] is False  # whole base block inherited
    assert cfg["model"]["dim"] == 128  # leaf-only block
    assert "_base_" not in cfg  # directive is stripped from the namespace


def test_load_unknown_ext(tmp_path):
    p = tmp_path / "cfg.txt"
    p.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        load(str(p))


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load("/no/such/config.yaml")


def test_action_configs_gate_prope_and_camera_inputs():
    """Action configs carry PRoPE/camera streams; T2V/TI2V configs do not."""

    config_paths = sorted(Path("configs/wan21").glob("**/*.py"))
    config_paths += sorted(Path("configs/hy").glob("**/*.py"))
    for path in config_paths:
        cfg = load(str(path))
        path_s = path.as_posix()
        is_action = "/action2v/" in path_s
        is_non_action = "/t2v/" in path_s or "/ti2v/" in path_s
        if not (is_action or is_non_action):
            continue

        expected = is_action
        model = cfg.get("model") or {}
        if "use_prope" in model:
            assert model["use_prope"] is expected, path_s

        for name, aux_model in (cfg.get("auxiliary_models") or {}).items():
            if isinstance(aux_model, dict) and "use_prope" in aux_model:
                assert aux_model["use_prope"] is expected, f"{path_s}:{name}"

        dataset = (cfg.get("data") or {}).get("dataset") or {}
        if "with_camera" in dataset:
            assert dataset["with_camera"] is expected, path_s


def test_configs_do_not_reference_retired_package_entries():
    """Config strings should use the new layout, not deleted compatibility paths."""

    retired = (
        "minwm.recipes",
        "minwm.modules",
        "minwm.ops",
        "minwm.solver",
        "minwm.generation",
        "minwm.diffusion",
        "minwm.evaluation",
        "minwm.optim",
        "minwm.schedulers",
        "minwm.rollouts",
        "minwm.training.fm_utils",
        "minwm.engine.recipe",
        "minwm.checkpoint",
        "minwm.training",
        "minwm.inference",
        "minwm.engine.training.preprocessors",
        "minwm.engine.inference.preprocessors",
        "minwm.processors.training",
        "minwm.processors.inference",
    )

    def iter_strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from iter_strings(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                yield from iter_strings(value)

    offenders = []
    for path in sorted(Path("configs").glob("**/*.py")):
        cfg = load(str(path))
        for value in iter_strings(cfg):
            for token in retired:
                if token in value:
                    offenders.append((path.as_posix(), token, value))

    assert offenders == []


def test_short_config_targets_are_declared_by_public_facades():
    """Every migrated config target is public without importing optional deps."""

    targets = set()

    def collect(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    key in {"type", "name"}
                    and isinstance(value, str)
                    and ":" not in value
                    and value[:1].isupper()
                ):
                    targets.add(value)
                collect(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                collect(value)

    for path in sorted(Path("configs").glob("**/*.py")):
        collect(load(str(path)))

    from minwm.config.lazy import DEFAULT_MODULES

    for target in targets:
        exporters = [
            module_name
            for module_name in DEFAULT_MODULES
            if target in getattr(importlib.import_module(module_name), "__all__", ())
        ]
        assert exporters, f"{target} is not exported by a default module"
        assert len(exporters) == 1, f"{target} is exported by multiple modules: {exporters}"


def test_config_docs_reference_existing_config_files():
    """README / CLI examples should not point at renamed or deleted config files."""

    docs = sorted(Path("configs").glob("**/*.md"))
    docs += [
        Path("tools/train_mwm.py"),
        Path("tools/infer_mwm.py"),
        Path("tools/export_checkpoint.py"),
    ]

    missing = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"configs/[A-Za-z0-9_./-]+\.py", text):
            ref = match.group(0)
            if not Path(ref).exists():
                missing.append((path.as_posix(), ref))

    assert missing == []


def test_config_readmes_use_current_stage_filenames():
    """Current config READMEs should use unified stage filenames after the rename."""

    retired_filenames = {
        "sft.py",
        "tf.py",
        "ode_distill.py",
        "cm.py",
        "dmd.py",
        "bidirectional.py",
        "ar_diffusion_tf.py",
        "causal_ode.py",
        "causal_cd.py",
        "stage1_mock.py",
        "stage2a_mock.py",
        "stage2b_mock.py",
        "stage3_mock.py",
    }
    offenders = []
    for path in sorted(Path("configs").glob("*/*/README.md")):
        text = path.read_text(encoding="utf-8")
        for name in retired_filenames:
            if f"`{name}`" in text:
                offenders.append((path.as_posix(), name))

    assert offenders == []


def test_diffusers_export_examples_use_train_configs():
    """Diffusers export must build architecture from train configs, not infer targets."""

    docs = sorted(Path("configs").glob("**/*.md"))
    docs.append(Path("tools/export_checkpoint.py"))
    pattern = re.compile(r"--format diffusers(?:\s|\\)+--config\s+(configs/[A-Za-z0-9_./-]+\.py)")

    offenders = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            ref = match.group(1)
            if "/infer/" in ref:
                offenders.append((path.as_posix(), ref))

    assert offenders == []


# ---- MWMConfig -----------------------------------------------------------


def test_mwmconfig_defaults():
    cfg = MWMConfig.from_dict({})
    assert cfg.training.max_steps == 0
    assert cfg.training.log_interval == 100
    assert cfg.checkpoint.resume is False
    assert cfg.profile.enabled is False
    assert cfg.profile.start_iter == 0
    assert cfg.profile.end_iter == 1
    assert cfg.profile.rank == 0
    assert cfg.data.batch_size == 1
    assert cfg.data.shuffle is True


def test_mwmconfig_happy_path():
    cfg = MWMConfig.from_dict(
        {
            "training": {"max_steps": 5000, "sp_size": 4},
            "model": {"type": "fractions:Fraction", "dim": 128},
        }
    )
    assert cfg.training.max_steps == 5000
    assert cfg.training.tp_size == 1  # untouched field keeps its default
    assert cfg.model.dim == 128  # open section: attribute access


def test_profile_config():
    cfg = MWMConfig.from_dict(
        {"profile": {"enabled": True, "start_iter": 2, "end_iter": 4, "rank": -1}}
    )
    assert cfg.profile.enabled is True
    assert cfg.profile.start_iter == 2
    assert cfg.profile.end_iter == 4
    assert cfg.profile.rank == -1


def test_profile_window_validation():
    with pytest.raises(ValueError, match="end_iter"):
        MWMConfig.from_dict({"profile": {"start_iter": 2, "end_iter": 2}})
    with pytest.raises(ValueError, match="rank"):
        MWMConfig.from_dict({"profile": {"rank": -2}})


def test_mwmconfig_strict_rejects_typo():
    with pytest.raises(ValueError, match="max_step"):
        MWMConfig.from_dict({"training": {"max_step": 1}})


def test_mwmconfig_ignores_extra_top_level_keys():
    cfg = MWMConfig.from_dict(
        {"trainer": {"type": "pkg:Cls"}, "_helper": 7, "training": {"max_steps": 3}}
    )
    assert cfg.training.max_steps == 3


def test_mwmconfig_open_sections_are_dictconfig():
    from omegaconf import DictConfig

    cfg = MWMConfig.from_dict({"model": {"dim": 8}, "recipe": {"lr": 0.1}})
    assert isinstance(cfg.model, DictConfig)
    assert isinstance(cfg.recipe, DictConfig)
    assert cfg.recipe.lr == 0.1


def test_mwmconfig_data_dataset_is_open_node():
    from omegaconf import DictConfig

    cfg = MWMConfig.from_dict({"data": {"dataset": {"type": "pkg:DS", "n": 4}, "batch_size": 2}})
    assert isinstance(cfg.data.dataset, DictConfig)
    assert cfg.data.dataset.n == 4
    assert cfg.data.batch_size == 2


def test_mwmconfig_data_strict_rejects_typo():
    with pytest.raises(ValueError, match="batch_size"):
        MWMConfig.from_dict({"data": {"batch_sizee": 2}})


def test_mwmconfig_from_dictconfig():
    from omegaconf import OmegaConf

    cfg = MWMConfig.from_dict(OmegaConf.create({"training": {"max_steps": 9}}))
    assert cfg.training.max_steps == 9


def test_monitor_defaults():
    cfg = MWMConfig.from_dict({})
    assert cfg.monitor.backends == ()
    assert cfg.monitor.wandb_project is None


def test_monitor_backends_list_coerces_to_tuple():
    cfg = MWMConfig.from_dict({"monitor": {"backends": ["tensorboard"]}})
    assert cfg.monitor.backends == ("tensorboard",)


def test_monitor_strict_rejects_typo():
    with pytest.raises(ValueError, match="backend"):
        MWMConfig.from_dict({"monitor": {"backend": ["wandb"]}})
