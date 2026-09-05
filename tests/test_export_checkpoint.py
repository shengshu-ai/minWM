"""Tests for tools/export_checkpoint.py.

Covers wrapper-prefix stripping, the .pt / .safetensors export helpers, the
default-output-name resolution, and a single-process DCP round-trip that
exercises the real consolidation path (``dcp.save`` -> ``_load_dcp``) without
needing a distributed launch (DCP falls back to single-process saving when no
process group is initialized).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.checkpoint.formats import select_state_dict, strip_wrapper_prefix
from tools.export_checkpoint import (
    _load_dcp,
    _resolve_output,
    export_pt,
    export_safetensors,
)


def test_strip_prefix_model():
    assert strip_wrapper_prefix("model.blocks.0.weight") == "blocks.0.weight"


def test_strip_prefix_fsdp():
    assert strip_wrapper_prefix("model._fsdp_wrapped_module.blocks.0.weight") == "blocks.0.weight"


def test_strip_prefix_orig_mod():
    assert strip_wrapper_prefix("_orig_mod.blocks.0.weight") == "blocks.0.weight"


def test_strip_prefix_layered():
    assert strip_wrapper_prefix("model._orig_mod.blocks.0.weight") == "blocks.0.weight"


def test_strip_prefix_clean():
    assert strip_wrapper_prefix("blocks.0.weight") == "blocks.0.weight"


def _sample_weights() -> dict[str, torch.Tensor]:
    return {
        "blocks.0.weight": torch.randn(4, 4),
        "blocks.0.bias": torch.zeros(4),
        "head.weight": torch.randn(2, 4),
    }


def test_export_pt_roundtrip(tmp_path):
    weights = _sample_weights()
    out = str(tmp_path / "model.pt")
    export_pt(weights, out)

    loaded = torch.load(out, map_location="cpu", weights_only=False)
    assert "model" in loaded
    for k, v in weights.items():
        assert k in loaded["model"]
        assert torch.equal(loaded["model"][k], v)


def test_export_safetensors_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    weights = _sample_weights()
    out = str(tmp_path / "model.safetensors")
    export_safetensors(weights, out)

    loaded = load_file(out, device="cpu")
    for k, v in weights.items():
        assert k in loaded
        assert torch.equal(loaded[k], v)


def test_export_creates_missing_parent_dir(tmp_path):
    # Regression: exporting to a not-yet-existing dir must create it, not crash.
    # (auto_dump submits export_checkpoint straight to the cluster with a fresh
    # <ckpts>/exported/ path and no prior mkdir.)
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    weights = _sample_weights()
    nested = tmp_path / "exported" / "deeper"
    st_out = str(nested / "model.safetensors")
    pt_out = str(nested / "model.pt")
    assert not nested.exists()

    export_safetensors(weights, st_out)
    export_pt(weights, pt_out)

    assert set(load_file(st_out, device="cpu")) == set(weights)
    assert "model" in torch.load(pt_out, map_location="cpu", weights_only=False)


def test_export_pt_file_exists(tmp_path):
    out = str(tmp_path / "model.pt")
    export_pt(_sample_weights(), out)
    assert os.path.getsize(out) > 0


def test_resolve_output_defaults_to_checkpoint_name():
    assert _resolve_output("logs/run/checkpoint_1000", None, "pt") == "checkpoint_1000.pt"
    assert (
        _resolve_output("logs/run/checkpoint_1000/", None, "safetensors")
        == "checkpoint_1000.safetensors"
    )


def test_resolve_output_respects_explicit_path():
    assert _resolve_output("logs/run/checkpoint_1000", "model.pt", "pt") == "model.pt"


def test_load_dcp_roundtrip(tmp_path):
    """Real DCP consolidation: save a single-process DCP dir, load it back.

    Mirrors the trainer's ``{"app": {"step", "model", "opt/<name>", ...}}``
    schema with wrapper-prefixed model keys, then verifies ``_load_dcp``
    consolidates the shards, unwraps ``app.model``, and strips prefixes.
    """
    import torch.distributed.checkpoint as dcp

    ckpt = tmp_path / "checkpoint_1000"
    model = {
        "model._fsdp_wrapped_module.blocks.0.weight": torch.randn(4, 4),
        "model.head.weight": torch.randn(2, 4),
    }
    state = {
        "app": {
            "step": torch.tensor(1000),
            "model": model,
            "opt/main": {"dummy": torch.zeros(1)},
        }
    }
    dcp.save(state, checkpoint_id=str(ckpt))

    weights = _load_dcp(str(ckpt))

    assert set(weights) == {"blocks.0.weight", "head.weight"}
    assert torch.equal(
        weights["blocks.0.weight"], model["model._fsdp_wrapped_module.blocks.0.weight"]
    )
    assert torch.equal(weights["head.weight"], model["model.head.weight"])


def test_load_dcp_selects_auxiliary_model(tmp_path):
    import torch.distributed.checkpoint as dcp

    ckpt = tmp_path / "checkpoint_aux"
    expected = torch.full((2, 2), 7.0)
    state = {
        "app": {
            "model": {"weight": torch.zeros(2, 2)},
            "aux/generator_ema": {"weight": expected},
        }
    }
    dcp.save(state, checkpoint_id=str(ckpt))

    weights = _load_dcp(str(ckpt), model_key="aux/generator_ema")
    assert torch.equal(weights["weight"], expected)


def test_load_dcp_missing_model_raises(tmp_path):
    import torch.distributed.checkpoint as dcp

    ckpt = tmp_path / "checkpoint_bad"
    dcp.save({"app": {"step": torch.tensor(1)}}, checkpoint_id=str(ckpt))

    with pytest.raises(RuntimeError, match="no 'model' entry"):
        _load_dcp(str(ckpt))


def test_load_dcp_missing_key_lists_available_entries(tmp_path):
    # The keyed load returns an empty dict for a missing key, so the error must
    # re-read metadata and name the real entries (else it says "keys: []").
    import torch.distributed.checkpoint as dcp

    ckpt = tmp_path / "checkpoint_entries"
    dcp.save(
        {"app": {"model": {"w": torch.zeros(2)}, "opt/main": {"x": torch.zeros(1)}}},
        checkpoint_id=str(ckpt),
    )

    with pytest.raises(RuntimeError, match=r"available entries:.*model") as exc:
        _load_dcp(str(ckpt), model_key="aux/nope")
    assert "opt/main" in str(exc.value)


def test_load_dcp_non_state_dict_error_names_selected_key(tmp_path):
    # The "not a state dict" error must reference the requested model_key, not a
    # hard-coded 'model', so a bad --model-key points at the right entry.
    import torch.distributed.checkpoint as dcp

    ckpt = tmp_path / "checkpoint_scalar_aux"
    state = {
        "app": {
            "model": {"weight": torch.zeros(2, 2)},
            "aux/step": torch.tensor(3),
        }
    }
    dcp.save(state, checkpoint_id=str(ckpt))

    with pytest.raises(RuntimeError, match="'aux/step' entry is not a state dict"):
        _load_dcp(str(ckpt), model_key="aux/step")


def test_select_state_dict_raw_dict_passes_through():
    # A bare param->tensor dict with no known weight key and no training-wrapper
    # bookkeeping keys must pass through untouched (raw state dict contract).
    raw = {"lin.weight": torch.ones(2), "lin.bias": torch.zeros(2)}
    out = select_state_dict(raw)
    assert set(out) == {"lin.weight", "lin.bias"}


def test_select_state_dict_explicit_missing_key_raises():
    with pytest.raises(KeyError, match="no key 'generator_ema'"):
        select_state_dict({"model": {"w": torch.zeros(2)}}, key="generator_ema")
