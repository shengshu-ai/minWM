"""Tests for tools/auto_sample.py.

Covers safetensors completeness detection (the readiness gate that keeps
inference off a half-flushed weight), cluster job-name sanitization, and the
infer_mwm.py argument assembly — which must emit ``inference.*`` dotlist
overrides, since the entry point accepts no value-carrying flags.
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools._cluster import build_safe_job_name
from tools.auto_sample import (
    build_infer_args,
    is_safetensors_fully_written,
)


def _defaults(**over) -> argparse.Namespace:
    base = dict(
        config_file="cfg.py",
        benchmark=None,
        seed=None,
        ema=False,
        opts=[],
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_safetensors_complete_file(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    p = str(tmp_path / "m.safetensors")
    save_file({"w": torch.randn(4, 4)}, p)
    assert is_safetensors_fully_written(p)


def test_safetensors_truncated_file(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    p = str(tmp_path / "m.safetensors")
    save_file({"w": torch.randn(8, 8)}, p)
    # Truncate the tensor bytes: header math must now fail.
    full = os.path.getsize(p)
    with open(p, "r+b") as f:
        f.truncate(full - 16)
    assert not is_safetensors_fully_written(p)


def test_safetensors_missing_file(tmp_path):
    assert not is_safetensors_fully_written(str(tmp_path / "nope.safetensors"))


def test_safe_job_name_basic():
    assert build_safe_job_name("wan_sft", 1000, "sample") == "wan-sft-sample-1000"


def test_safe_job_name_long_is_bounded_and_keeps_step():
    name = build_safe_job_name("a" * 80, 12345, "sample")
    assert len(name) <= 30
    assert name.endswith("sample-12345")


def test_build_infer_args_forwards_step_values_as_dotlist():
    args = _defaults()
    out = build_infer_args(args, "model_1000.pt", "samples/1000")
    assert out[:2] == ["--config-file", "cfg.py"]
    assert "inference.checkpoint=model_1000.pt" in out
    assert "inference.output_dir=samples/1000" in out
    # No value-carrying flags: infer_mwm.py's argparse would reject them.
    assert not any(o.startswith("--") for o in out[2:])


def test_build_infer_args_omits_benchmark_when_config_supplies_it():
    out = build_infer_args(_defaults(), "m.pt", "s/1")
    assert not any(o.startswith("inference.benchmark=") for o in out)


def test_build_infer_args_benchmark_ema_and_seed():
    args = _defaults(benchmark="in.json", ema=True, seed=42)
    out = build_infer_args(args, "m.safetensors", "s/1")
    assert "inference.benchmark=in.json" in out
    assert "inference.prefer_ema=True" in out
    assert "inference.seed=42" in out


def test_build_infer_args_drops_remainder_dashdash():
    # argparse.REMAINDER keeps a leading "--"; it must not be forwarded.
    args = _defaults(opts=["--", "inference.num_frames=20"])
    out = build_infer_args(args, "m.pt", "s/1")
    assert "--" not in out
    assert "inference.num_frames=20" in out


def test_main_rejects_remote_output_dir(monkeypatch, capsys):
    # auto_sample watches weights via the local filesystem, so a remote
    # --output-dir would block forever; it must fail fast at parse time instead.
    import tools.auto_sample as m

    argv = [
        "auto_sample.py",
        "--output-dir",
        "s3://bucket/run",
        "--ckpt-steps",
        "1000",
        "--config-file",
        "cfg.py",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        m.main()
    assert exc.value.code == 1
    assert "must be a local/mounted path" in capsys.readouterr().err
