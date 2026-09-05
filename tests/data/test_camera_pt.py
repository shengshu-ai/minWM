"""Alignment tests for the migrated HY camera datasets.

Three layers of "alignment", in decreasing strength:

1. Pure-math differential — :func:`build_viewmats_and_Ks` and
   :func:`discretize_poses_to_actions` are run against the *actual* legacy
   ``HY15/`` implementations on synthetic inputs and must agree bit-for-bit.
   These skip automatically once ``HY15/`` is removed from the tree.
2. Golden / invariant — the same functions are checked against hand-derived
   invariants and frozen golden values, so the guard survives legacy removal.
3. Dataset contract — tiny synthetic ``.pt`` fixtures are loaded through
   :class:`CameraPluckerDataset` / :class:`CausalODEDataset` to assert the
   emitted keys, shapes, and dtypes. No real pre-encoded data required.

End-to-end training parity (real latents through ``tools/train_mwm.py``) is a
separate gate and is intentionally out of scope here.
"""

import ast
import importlib.util
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from minwm.data.datasets.action import (
    discretize_poses_to_actions,
    trajectory_str_to_action_labels,
)
from minwm.data.datasets.geometry import build_viewmats_and_Ks

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LEGACY_AU = os.path.join(_REPO_ROOT, "HY15", "trainer", "dataset_camera", "action_utils.py")
_LEGACY_DS = os.path.join(
    _REPO_ROOT, "HY15", "trainer", "dataset_camera", "ar_camera_plucker_dataset.py"
)
_LEGACY_PRESENT = os.path.isfile(_LEGACY_AU) and os.path.isfile(_LEGACY_DS)
_skip_no_legacy = pytest.mark.skipif(
    not _LEGACY_PRESENT, reason="legacy HY15/ tree removed — differential check N/A"
)


def _load_legacy_action_utils():
    """Import the standalone legacy ``action_utils`` module by file path."""
    spec = importlib.util.spec_from_file_location("legacy_action_utils", _LEGACY_AU)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_legacy_build_viewmats():
    """Extract legacy ``build_viewmats_and_Ks`` without importing its module.

    The function lives in ``ar_camera_plucker_dataset.py`` whose top-level
    imports drag in ``torchdata`` and ``trainer.*``; we compile just the one
    function in an isolated namespace instead.
    """
    src = open(_LEGACY_DS).read()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_viewmats_and_Ks"
    )
    ns = {"np": np, "Rotation": Rotation}
    exec(compile(ast.Module([fn], []), _LEGACY_DS, "exec"), ns)  # noqa: S102
    return ns["build_viewmats_and_Ks"]


def _random_poses(n: int, seed: int) -> np.ndarray:
    """``(n, 7)`` w2c poses with varied rotation + translation."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        q = Rotation.from_euler("xyz", rng.normal(scale=10, size=3), degrees=True).as_quat()
        poses[i, :3] = rng.normal(scale=0.5, size=3)
        poses[i, 3:] = q
    return poses


def _viewmats_from_poses(poses: np.ndarray) -> np.ndarray:
    """Raw (un-normalized) ``(T, 4, 4)`` w2c matrices, for action tests."""
    T = len(poses)
    vm = np.zeros((T, 4, 4), dtype=np.float32)
    for i, (tx, ty, tz, qx, qy, qz, qw) in enumerate(poses):
        vm[i, :3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        vm[i, :3, 3] = [tx, ty, tz]
        vm[i, 3, 3] = 1.0
    return vm


# ---- layer 1: differential vs legacy HY15/ -------------------------------


@_skip_no_legacy
def test_geometry_matches_legacy_bit_for_bit():
    legacy_bvk = _load_legacy_build_viewmats()
    intrinsics = np.array([0.8, 1.1, 0.5, 0.5], dtype=np.float32)
    poses = _random_poses(10, seed=1)

    vm_new, K_new = build_viewmats_and_Ks(intrinsics, poses)
    vm_old, K_old = legacy_bvk(intrinsics, poses)

    assert np.array_equal(vm_new, vm_old)
    assert np.array_equal(K_new, K_old)


@_skip_no_legacy
def test_discretize_actions_matches_legacy_bit_for_bit():
    legacy = _load_legacy_action_utils()
    viewmats = _viewmats_from_poses(_random_poses(12, seed=0))

    assert np.array_equal(
        discretize_poses_to_actions(viewmats),
        legacy.discretize_poses_to_actions(viewmats),
    )


@_skip_no_legacy
def test_trajectory_str_matches_legacy_bit_for_bit():
    legacy = _load_legacy_action_utils()
    for traj in ("w*4,a*3,l*2", "s*2,d*5,i*1,k*1", "", "j*8"):
        new = trajectory_str_to_action_labels(traj, 12).numpy()
        old = legacy.trajectory_str_to_action_labels(traj, 12).numpy()
        assert np.array_equal(new, old), traj


# ---- layer 2: golden / invariant (survives legacy removal) ---------------


def test_geometry_first_frame_is_identity():
    # Normalization anchors the trajectory at frame 0 -> viewmats[0] == I.
    poses = _random_poses(6, seed=2)
    vm, _ = build_viewmats_and_Ks(np.array([1.0, 1.0, 0.5, 0.5], np.float32), poses)
    assert np.allclose(vm[0], np.eye(4), atol=1e-5)


def test_geometry_K_is_tiled_intrinsics():
    fx, fy, cx, cy = 0.7, 0.9, 0.5, 0.5
    poses = _random_poses(4, seed=3)
    _, Ks = build_viewmats_and_Ks(np.array([fx, fy, cx, cy], np.float32), poses)
    expected = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    assert Ks.shape == (4, 3, 3)
    for k in Ks:
        assert np.array_equal(k, expected)


def test_action_frame_zero_is_no_action():
    viewmats = _viewmats_from_poses(_random_poses(8, seed=4))
    assert discretize_poses_to_actions(viewmats)[0] == 0


def test_action_static_camera_is_all_zero():
    # Identical poses -> zero relative motion -> all no-action labels.
    pose = _random_poses(1, seed=5)[0]
    viewmats = _viewmats_from_poses(np.tile(pose, (5, 1)))
    assert np.array_equal(discretize_poses_to_actions(viewmats), np.zeros(5, dtype=np.int64))


def test_trajectory_str_label_encoding():
    # forward 'w' -> trans=1, rotate=0 -> label 9; yaw-right 'l' -> 1.
    labels = trajectory_str_to_action_labels("w*2,l*1", 6)
    assert labels.dtype == torch.int64
    assert labels[0].item() == 0  # frame 0 always no-action
    assert labels[1].item() == 9
    assert labels[2].item() == 9
    assert labels[3].item() == 1
    assert labels[4:].tolist() == [0, 0]


# ---- layer 3: dataset contract via synthetic .pt fixtures ----------------

_C, _T, _H, _W = 4, 8, 5, 6  # latent (C, T, H, W); T divisible by 4
_PROMPT_LEN, _PROMPT_DIM = 3, 16
_BYT5_LEN, _BYT5_DIM = 2, 8


def _write_sample_pt(path: str, *, with_ode: bool = False) -> None:
    """Write a tiny synthetic pre-encoded ``.pt`` matching the dataset keys."""
    sample = {
        "latent": torch.randn(1, _C, _T, _H, _W),
        "intrinsics": torch.tensor([0.8, 1.1, 0.5, 0.5]),
        "poses": torch.from_numpy(_random_poses(_T, seed=7)),
        "prompt_embeds": torch.randn(1, _PROMPT_LEN, _PROMPT_DIM),
        "prompt_mask": torch.ones(1, _PROMPT_LEN, dtype=torch.long),
        "byt5_text_states": torch.randn(1, _BYT5_LEN, _BYT5_DIM),
        "byt5_text_mask": torch.ones(1, _BYT5_LEN, dtype=torch.long),
    }
    if with_ode:
        sample["ode_trajectory"] = torch.randn(1, 5, _C, _T, _H, _W)
    torch.save(sample, path)


def _write_neg_prompts(neg_path: str, byt5_path: str) -> None:
    torch.save(
        {
            "negative_prompt_embeds": torch.randn(1, _PROMPT_LEN, _PROMPT_DIM),
            "negative_prompt_mask": torch.ones(1, _PROMPT_LEN, dtype=torch.long),
        },
        neg_path,
    )
    torch.save(
        {
            "byt5_text_states": torch.randn(1, _BYT5_LEN, _BYT5_DIM),
            "byt5_text_mask": torch.ones(1, _BYT5_LEN, dtype=torch.long),
        },
        byt5_path,
    )


@pytest.fixture
def camera_fixture(tmp_path):
    """Build an index JSON + one sample ``.pt`` + neg-prompt ``.pt``s."""
    import json

    def _make(with_ode: bool = False):
        sample_pt = tmp_path / "sample.pt"
        _write_sample_pt(str(sample_pt), with_ode=with_ode)
        neg_pt = tmp_path / "neg.pt"
        byt5_pt = tmp_path / "byt5.pt"
        _write_neg_prompts(str(neg_pt), str(byt5_pt))
        index = tmp_path / "index.json"
        with open(index, "w") as f:
            json.dump([{"latent_path": str(sample_pt)}], f)
        return str(index), str(neg_pt), str(byt5_pt)

    return _make


def test_camera_plucker_contract(camera_fixture):
    from minwm.data.datasets import CameraPluckerDataset

    index, neg, byt5 = camera_fixture()
    ds = CameraPluckerDataset(index, window_frames=_T, neg_prompt_path=neg, neg_byt5_path=byt5)
    assert len(ds) == 1
    s = ds[0]

    assert s["latent"].shape == (_C, _T, _H, _W)
    assert s["viewmats"].shape == (_T, 4, 4)
    assert s["Ks"].shape == (_T, 3, 3)
    assert s["action"].shape == (_T,)
    assert s["action"].dtype == torch.int64
    assert s["prompt_embed"].shape == (_PROMPT_LEN, _PROMPT_DIM)
    assert s["byt5_text_states"].shape == (_BYT5_LEN, _BYT5_DIM)
    # t2v path emits zero image conditioning.
    assert s["image_cond"].shape == (32, 1, _H, _W)
    assert torch.count_nonzero(s["image_cond"]) == 0
    assert s["i2v_mask"].shape == s["latent"].shape
    assert s["select_window_out_flag"] == 0


def test_camera_plucker_cfg_swaps_to_negative(camera_fixture):
    from minwm.data.datasets import CameraPluckerDataset

    index, neg, byt5 = camera_fixture()
    neg_pt = torch.load(neg, weights_only=True)
    ds = CameraPluckerDataset(
        index, window_frames=_T, cfg_rate=1.0, neg_prompt_path=neg, neg_byt5_path=byt5
    )
    assert torch.equal(ds[0]["prompt_embed"], neg_pt["negative_prompt_embeds"][0])


def test_causal_ode_contract(camera_fixture):
    from minwm.data.datasets import CausalODEDataset

    index, neg, byt5 = camera_fixture(with_ode=True)
    ds = CausalODEDataset(index, window_frames=_T, neg_prompt_path=neg, neg_byt5_path=byt5)
    s = ds[0]
    assert s["latent"].shape == (_C, _T, _H, _W)
    assert s["ode_trajectory"].shape == (5, _C, _T, _H, _W)
    assert s["viewmats"].shape == (_T, 4, 4)
    assert s["action"].shape == (_T,)


def test_causal_ode_preserves_prebuilt_camera_dtype(camera_fixture):
    import json

    from minwm.data.datasets import CausalODEDataset

    index, neg, byt5 = camera_fixture(with_ode=True)
    with open(index, "r") as f:
        sample_path = json.load(f)[0]["latent_path"]
    sample = torch.load(sample_path, map_location="cpu", weights_only=True)
    sample["viewmats"] = torch.eye(4, dtype=torch.bfloat16).reshape(1, 1, 4, 4).repeat(1, _T, 1, 1)
    sample["Ks"] = torch.eye(3, dtype=torch.bfloat16).reshape(1, 1, 3, 3).repeat(1, _T, 1, 1)
    torch.save(sample, sample_path)

    ds = CausalODEDataset(index, window_frames=_T, neg_prompt_path=neg, neg_byt5_path=byt5)
    s = ds[0]

    assert s["viewmats"].dtype == torch.bfloat16
    assert s["Ks"].dtype == torch.bfloat16
