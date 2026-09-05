"""Tests for tools/auto_dump.py.

Covers path derivation, the "already dumped" short-circuit, cluster job-name
sanitization, the DCP ``.metadata`` readiness gate, and a real single-process
DCP round-trip through ``dump_local`` (save a DCP dir -> dump_local -> load the
safetensors back), exercising the same consolidation path a live run uses
without a distributed launch.
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.checkpoint.storage import Storage
from tools._cluster import build_safe_job_name
from tools.auto_dump import (
    dcp_dir_for,
    dump_local,
    is_exported,
    output_path_for,
    wait_for_dcp,
)


def test_dcp_dir_for():
    assert dcp_dir_for("run/ckpts", 1000).endswith("run/ckpts/checkpoint_1000")


def test_output_path_single_file():
    assert output_path_for("run/exp", 1000, "safetensors").endswith("model_1000.safetensors")
    assert output_path_for("run/exp", 1000, "pt").endswith("model_1000.pt")


def test_output_path_diffusers_is_dir():
    # diffusers writes a directory, not a suffixed file
    assert output_path_for("run/exp", 1000, "diffusers").endswith("model_1000")


def test_is_exported_single_file(tmp_path):
    storage = Storage()
    out = str(tmp_path / "model_5.safetensors")
    assert not is_exported(storage, out, "safetensors")
    open(out, "wb").close()
    assert is_exported(storage, out, "safetensors")


def test_is_exported_diffusers_needs_config_json(tmp_path):
    storage = Storage()
    out = str(tmp_path / "model_5")
    os.makedirs(out)
    assert not is_exported(storage, out, "diffusers")
    open(os.path.join(out, "config.json"), "wb").close()
    assert is_exported(storage, out, "diffusers")


def test_wait_for_dcp_returns_when_metadata_present(tmp_path):
    # .metadata already present -> wait_for_dcp returns immediately (no sleep).
    storage = Storage()
    dcp = tmp_path / "checkpoint_1000"
    dcp.mkdir()
    (dcp / ".metadata").write_bytes(b"x")
    wait_for_dcp(storage, str(dcp), 1000, wait_interval=0)


def test_safe_job_name_basic():
    assert build_safe_job_name("wan_sft", 10000, "dump") == "wan-sft-dump-10000"


def test_safe_job_name_long_is_bounded_and_keeps_step():
    name = build_safe_job_name("a" * 80, 12345, "dump")
    assert len(name) <= 30
    assert name.endswith("dump-12345")


def test_dump_local_roundtrip(tmp_path):
    """Save a single-process DCP dir, dump_local it, load the safetensors back."""
    pytest.importorskip("safetensors")
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import load_file

    ckpt = tmp_path / "checkpoint_1000"
    model = {
        "model._fsdp_wrapped_module.blocks.0.weight": torch.randn(4, 4),
        "model.head.weight": torch.randn(2, 4),
    }
    dcp.save({"app": {"step": torch.tensor(1000), "model": model}}, checkpoint_id=str(ckpt))

    out = str(tmp_path / "model_1000.safetensors")
    args = argparse.Namespace(format="safetensors", model_key="model", config=None)
    dump_local(1000, str(ckpt), out, args, Storage())

    loaded = load_file(out, device="cpu")
    assert set(loaded) == {"blocks.0.weight", "head.weight"}
    assert torch.equal(
        loaded["blocks.0.weight"], model["model._fsdp_wrapped_module.blocks.0.weight"]
    )


def test_dump_local_selects_aux_key(tmp_path):
    pytest.importorskip("safetensors")
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import load_file

    ckpt = tmp_path / "checkpoint_aux"
    expected = torch.full((2, 2), 7.0)
    state = {"app": {"model": {"w": torch.zeros(2, 2)}, "aux/generator_ema": {"w": expected}}}
    dcp.save(state, checkpoint_id=str(ckpt))

    out = str(tmp_path / "model_aux.safetensors")
    args = argparse.Namespace(format="safetensors", model_key="aux/generator_ema", config=None)
    dump_local(0, str(ckpt), out, args, Storage())

    assert torch.equal(load_file(out, device="cpu")["w"], expected)
