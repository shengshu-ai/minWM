"""Tests for minwm.engine.checkpoint.checkpointer and .formats.

Everything runs single-process and off-GPU. The save/load paths are exercised
backend-agnostically over two roots — a local temp dir and an fsspec
``memory://`` filesystem (the same code path a real ``s3://`` / ``oss://`` store
takes) — for both the single-file and the sharded DCP formats.
"""

import json
import os
import sys
import uuid

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.checkpoint import formats
from minwm.engine.checkpoint.checkpointer import Checkpointer, load_model_weights
from minwm.engine.checkpoint.storage import Storage, is_remote, join


class _Bundle:
    """Minimal AppState-shaped Stateful: a model plus a training step."""

    def __init__(self, model: nn.Module, step: int = 0) -> None:
        self.model = model
        self.step = step

    def state_dict(self) -> dict:
        return {"model": get_model_state_dict(self.model), "step": torch.tensor(self.step)}

    def load_state_dict(self, state_dict: dict) -> None:
        set_model_state_dict(self.model, state_dict["model"])
        self.step = int(state_dict["step"].item())


def _make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def _params_equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _write_safetensors(st, path, weights, staged):
    save_file = pytest.importorskip("safetensors.torch").save_file
    save_file(weights, str(staged))
    if is_remote(path):
        with open(staged, "rb") as fsrc, st.open(path, "wb") as fdst:
            fdst.write(fsrc.read())
    else:
        os.replace(staged, path)


@pytest.fixture(params=["local", "memory"])
def root(request, tmp_path):
    """Yield a fresh (Storage, save_dir) pair for each backend under test."""
    if request.param == "local":
        yield Storage(), str(tmp_path / "ckpts")
    else:
        base = f"memory://mwm-ckpt-{uuid.uuid4().hex[:8]}/ckpts"
        st = Storage()
        yield st, base
        try:
            st.remove(base.rsplit("/", 1)[0], recursive=True)
        except FileNotFoundError:
            pass


# --- format detection ------------------------------------------------------


def test_detect_format_by_suffix(tmp_path):
    st = Storage()
    assert formats.detect_format("run/model.safetensors", st) == formats.SAFETENSORS
    for suffix in (".pt", ".pth", ".bin", ".ckpt"):
        assert formats.detect_format(f"run/model{suffix}", st) == formats.TORCH
    assert formats.detect_format(str(tmp_path / "does-not-exist"), st) == formats.UNKNOWN


def test_detect_format_directories(tmp_path):
    st = Storage()
    diffusers = tmp_path / "transformer"
    diffusers.mkdir()
    (diffusers / "config.json").write_text("{}")
    (diffusers / "model.safetensors").write_bytes(b"")
    assert formats.detect_format(str(diffusers), st) == formats.DIFFUSERS

    dcp_dir = tmp_path / "checkpoint_1"
    dcp_dir.mkdir()
    (dcp_dir / ".metadata").write_bytes(b"")
    assert formats.detect_format(str(dcp_dir), st) == formats.DCP_DIR


# --- single-file save/load -------------------------------------------------


def test_single_file_roundtrip(root):
    st, save_dir = root
    src = _make_model(0)
    ckpt = Checkpointer(save_dir, checkpointables={"app": _Bundle(src, step=1000)}, storage=st)

    assert not ckpt.has_checkpoint()
    target = ckpt.save("checkpoint_1000")
    assert target == "checkpoint_1000.pt"
    assert ckpt.has_checkpoint()
    assert ckpt.get_checkpoint_file() == join(save_dir, "checkpoint_1000.pt")

    dst_bundle = _Bundle(_make_model(1), step=0)
    dst_ckpt = Checkpointer(save_dir, checkpointables={"app": dst_bundle}, storage=st)
    assert not _params_equal(src, dst_bundle.model)
    dst_ckpt.load(dst_ckpt.get_checkpoint_file())
    assert _params_equal(src, dst_bundle.model)
    assert dst_bundle.step == 1000


def test_resume_or_load(root):
    st, save_dir = root
    src = _make_model(2)
    Checkpointer(save_dir, checkpointables={"app": _Bundle(src, step=500)}, storage=st).save(
        "checkpoint_500"
    )

    # resume=True picks up latest.txt.
    dst = _Bundle(_make_model(3), step=0)
    ckpt = Checkpointer(save_dir, checkpointables={"app": dst}, storage=st)
    restored = ckpt.resume_or_load(resume=True)
    assert restored == join(save_dir, "checkpoint_500.pt")
    assert dst.step == 500 and _params_equal(src, dst.model)

    # resume=False with no path is a fresh start.
    fresh = _Bundle(_make_model(4), step=0)
    assert (
        Checkpointer(save_dir, checkpointables={"app": fresh}, storage=st).resume_or_load(
            resume=False
        )
        is None
    )
    assert fresh.step == 0


def test_save_skipped_without_save_dir():
    ckpt = Checkpointer("", checkpointables={"app": _Bundle(_make_model(0))})
    assert ckpt.save("checkpoint_1") is None
    assert not ckpt.has_checkpoint()


# --- sharded DCP save/load (single process) --------------------------------


@pytest.mark.parametrize("backend", ["local", "memory"])
def test_dcp_roundtrip(backend, tmp_path):
    if backend == "local":
        st, save_dir = Storage(), str(tmp_path / "dcp")
    else:
        st, save_dir = Storage(), f"memory://mwm-dcp-{uuid.uuid4().hex[:8]}/ckpts"

    src = _make_model(5)
    ckpt = Checkpointer(
        save_dir, checkpointables={"app": _Bundle(src, step=2000)}, sharded=True, storage=st
    )
    target = ckpt.save("checkpoint_2000")
    assert target == "checkpoint_2000"
    assert formats.detect_format(join(save_dir, target), st) == formats.DCP_DIR

    dst = _Bundle(_make_model(6), step=0)
    dst_ckpt = Checkpointer(save_dir, checkpointables={"app": dst}, sharded=True, storage=st)
    assert dst_ckpt.resume_or_load(resume=True) == join(save_dir, "checkpoint_2000")
    assert dst.step == 2000 and _params_equal(src, dst.model)

    if is_remote(save_dir):
        st.remove(save_dir.rsplit("/", 1)[0], recursive=True)


# --- weights-only initialization -------------------------------------------


def test_load_model_weights_native(root):
    st, save_dir = root
    st.makedirs(save_dir)
    src = _make_model(7)
    path = join(save_dir, "weights.pt")
    with st.open(path, "wb") as f:
        torch.save({"model": src.state_dict()}, f)

    dst = _make_model(8)
    assert not _params_equal(src, dst)
    load_model_weights(dst, path, storage=st)
    assert _params_equal(src, dst)


def test_load_model_weights_legacy_ema(root):
    st, save_dir = root
    st.makedirs(save_dir)
    src = _make_model(9)
    # Legacy generator wrapper: EMA weights under a "model." prefix.
    ema = {f"model.{k}": v for k, v in src.state_dict().items()}
    path = join(save_dir, "legacy.pt")
    with st.open(path, "wb") as f:
        torch.save({"generator_ema": ema, "generator": {}}, f)

    dst = _make_model(10)
    load_model_weights(dst, path, storage=st, prefer_ema=True)
    assert _params_equal(src, dst)


def test_load_model_weights_safetensors(root, tmp_path):
    st, save_dir = root
    st.makedirs(save_dir)
    src = _make_model(11)
    path = join(save_dir, "model.safetensors")

    # safetensors writes to a local path; upload the bytes for remote backends.
    staged = str(tmp_path / "staged.safetensors")
    _write_safetensors(st, path, src.state_dict(), staged)

    dst = _make_model(12)
    load_model_weights(dst, path, storage=st)
    assert _params_equal(src, dst)


def test_load_model_weights_diffusers_directory(root, tmp_path):
    st, save_dir = root
    model_dir = join(save_dir, "transformer")
    st.makedirs(model_dir)
    st.write_text(join(model_dir, "config.json"), "{}")

    src = _make_model(13)
    weight_path = join(model_dir, "diffusion_pytorch_model.safetensors")
    _write_safetensors(st, weight_path, src.state_dict(), tmp_path / "diffusers.safetensors")

    dst = _make_model(14)
    load_model_weights(dst, model_dir, storage=st)
    assert _params_equal(src, dst)


def test_load_model_weights_sharded_diffusers_directory(root, tmp_path):
    st, save_dir = root
    model_dir = join(save_dir, "transformer-sharded")
    st.makedirs(model_dir)
    st.write_text(join(model_dir, "config.json"), "{}")

    src = _make_model(15)
    items = list(src.state_dict().items())
    shard_names = (
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    )
    shards = (dict(items[:2]), dict(items[2:]))
    for index, (shard_name, shard) in enumerate(zip(shard_names, shards, strict=True)):
        _write_safetensors(
            st,
            join(model_dir, shard_name),
            shard,
            tmp_path / f"diffusers-shard-{index}.safetensors",
        )
    weight_map = {key: shard_names[0 if index < 2 else 1] for index, (key, _) in enumerate(items)}
    st.write_text(
        join(model_dir, "diffusion_pytorch_model.safetensors.index.json"),
        json.dumps({"metadata": {}, "weight_map": weight_map}),
    )

    dst = _make_model(16)
    load_model_weights(dst, model_dir, storage=st)
    assert _params_equal(src, dst)


@pytest.mark.parametrize("backend", ["local", "memory"])
def test_dcp_async_save_defers_pointer(backend, tmp_path):
    if backend == "local":
        st, save_dir = Storage(), str(tmp_path / "adcp")
    else:
        st, save_dir = Storage(), f"memory://mwm-adcp-{uuid.uuid4().hex[:8]}/ckpts"

    src = _make_model(13)
    ckpt = Checkpointer(
        save_dir,
        checkpointables={"app": _Bundle(src, step=3000)},
        sharded=True,
        storage=st,
        async_save=True,
    )
    assert ckpt.save("checkpoint_3000") == "checkpoint_3000"
    # Bytes may be mid-write; the resume pointer must not move until finalized.
    ckpt.wait_for_saves()
    assert ckpt.has_checkpoint()
    assert ckpt.get_checkpoint_file() == join(save_dir, "checkpoint_3000")

    dst = _Bundle(_make_model(14), step=0)
    dst_ckpt = Checkpointer(save_dir, checkpointables={"app": dst}, sharded=True, storage=st)
    assert dst_ckpt.resume_or_load(resume=True) == join(save_dir, "checkpoint_3000")
    assert dst.step == 3000 and _params_equal(src, dst.model)

    if is_remote(save_dir):
        st.remove(save_dir.rsplit("/", 1)[0], recursive=True)


def test_dcp_async_next_save_finalizes_previous(tmp_path):
    save_dir = str(tmp_path / "adcp2")
    bundle = _Bundle(_make_model(15), step=100)
    ckpt = Checkpointer(
        save_dir, checkpointables={"app": bundle}, sharded=True, storage=Storage(), async_save=True
    )

    ckpt.save("checkpoint_100")
    bundle.step = 200
    # A second save serializes on (and thereby finalizes) the first.
    ckpt.save("checkpoint_200")
    assert ckpt.get_checkpoint_file() == join(save_dir, "checkpoint_100")
    ckpt.wait_for_saves()
    assert ckpt.get_checkpoint_file() == join(save_dir, "checkpoint_200")


def test_read_model_state_rejects_dcp_directory(tmp_path):
    st = Storage()
    dcp = tmp_path / "checkpoint_1"
    dcp.mkdir()
    (dcp / ".metadata").write_bytes(b"")
    with pytest.raises(ValueError):
        formats.read_model_state(str(dcp), st)


# --- fail-loud partial-load enforcement ------------------------------------
#
# These tests pin the two-paths-share-behavior contract:
#   - single-file: Checkpointer.load raises KeyError on a missing top-level key
#     when allow_partial_load=False; silently skips when True.
#   - DCP: DefaultLoadPlanner raises CheckpointException on a missing tensor key
#     when allow_partial_load=False; loads ok when True.


class _BundleWithExtra:
    """A _Bundle that also persists an ``extra`` model, simulating a config that
    added an auxiliary model after the checkpoint was saved."""

    def __init__(self, model: nn.Module, extra: nn.Module, step: int = 0) -> None:
        self.model = model
        self.extra = extra
        self.step = step

    def state_dict(self) -> dict:
        return {
            "model": get_model_state_dict(self.model),
            "extra": get_model_state_dict(self.extra),
            "step": torch.tensor(self.step),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        set_model_state_dict(self.model, state_dict["model"])
        if "extra" in state_dict:
            set_model_state_dict(self.extra, state_dict["extra"])
        self.step = int(state_dict["step"].item())


def test_single_file_fails_loud_on_missing_key(root):
    """Resuming a single-file checkpoint that lacks a registered checkpointable
    raises KeyError instead of silently leaving the object at its built state."""
    st, save_dir = root
    # Save a checkpoint that has no "app" key (simulates an old flat format or a
    # completely wrong file being pointed at by latest.txt).
    st.makedirs(save_dir)
    wrong_path = join(save_dir, "checkpoint_bad.pt")
    with st.open(wrong_path, "wb") as f:
        torch.save({"not_app": 42}, f)
    st.write_text(join(save_dir, "latest.txt"), "checkpoint_bad.pt")

    bundle = _Bundle(_make_model(20), step=0)
    ckpt = Checkpointer(save_dir, checkpointables={"app": bundle}, storage=st)
    with pytest.raises(KeyError, match="checkpoint_bad.pt"):
        ckpt.load(ckpt.get_checkpoint_file())
    # Step must not have changed — nothing was loaded.
    assert bundle.step == 0


def test_single_file_allow_partial_load_skips_missing_key(root):
    """With allow_partial_load=True a missing key is skipped rather than raising."""
    st, save_dir = root
    st.makedirs(save_dir)
    wrong_path = join(save_dir, "checkpoint_partial.pt")
    with st.open(wrong_path, "wb") as f:
        torch.save({"not_app": 42}, f)
    st.write_text(join(save_dir, "latest.txt"), "checkpoint_partial.pt")

    bundle = _Bundle(_make_model(21), step=99)
    ckpt = Checkpointer(
        save_dir, checkpointables={"app": bundle}, storage=st, allow_partial_load=True
    )
    # Should not raise; bundle is left as built.
    ckpt.load(ckpt.get_checkpoint_file())
    assert bundle.step == 99  # unchanged — nothing was loaded


@pytest.mark.parametrize("backend", ["local", "memory"])
def test_dcp_fails_loud_on_missing_key(backend, tmp_path):
    """DCP resume raises when the checkpoint is missing a key that the current
    config declares, and allow_partial_load is False (the default)."""
    from torch.distributed.checkpoint.api import CheckpointException

    if backend == "local":
        st, save_dir = Storage(), str(tmp_path / "dcp_miss")
    else:
        st, save_dir = Storage(), f"memory://mwm-dcp-miss-{uuid.uuid4().hex[:8]}/ckpts"

    # Save a checkpoint WITHOUT the "extra" model.
    src = _make_model(22)
    Checkpointer(
        save_dir, checkpointables={"app": _Bundle(src, step=10)}, sharded=True, storage=st
    ).save("checkpoint_10")

    # Now load with a bundle that EXPECTS "extra" in the checkpoint.
    dst = _BundleWithExtra(_make_model(23), _make_model(24))
    ckpt = Checkpointer(
        save_dir,
        checkpointables={"app": dst},
        sharded=True,
        storage=st,
        allow_partial_load=False,
    )
    with pytest.raises(CheckpointException):
        ckpt.load(ckpt.get_checkpoint_file())

    if is_remote(save_dir):
        st.remove(save_dir.rsplit("/", 1)[0], recursive=True)


@pytest.mark.parametrize("backend", ["local", "memory"])
def test_dcp_allow_partial_load_succeeds(backend, tmp_path):
    """With allow_partial_load=True a DCP checkpoint that is missing keys loads
    without error; the missing tensors are left at their freshly-built state."""
    if backend == "local":
        st, save_dir = Storage(), str(tmp_path / "dcp_partial")
    else:
        st, save_dir = Storage(), f"memory://mwm-dcp-part-{uuid.uuid4().hex[:8]}/ckpts"

    # Save WITHOUT "extra".
    src = _make_model(25)
    Checkpointer(
        save_dir, checkpointables={"app": _Bundle(src, step=20)}, sharded=True, storage=st
    ).save("checkpoint_20")

    # Load WITH "extra" and allow_partial_load=True — must not raise.
    fresh_extra = _make_model(26)
    dst = _BundleWithExtra(_make_model(27), fresh_extra)
    ckpt = Checkpointer(
        save_dir,
        checkpointables={"app": dst},
        sharded=True,
        storage=st,
        allow_partial_load=True,
    )
    ckpt.load(ckpt.get_checkpoint_file())
    # The main model was restored; "extra" was not touched.
    assert dst.step == 20
    assert _params_equal(src, dst.model)
    assert _params_equal(fresh_extra, dst.extra)  # extra is left as built

    if is_remote(save_dir):
        st.remove(save_dir.rsplit("/", 1)[0], recursive=True)
