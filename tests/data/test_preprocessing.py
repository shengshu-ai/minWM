"""Alignment tests for the migrated HY data-preprocessing helpers.

Mirrors the layering in ``test_camera_pt.py``:

1. Pure-math differential — :func:`generate_camera_trajectory_local` is run
   against the legacy ``HY15/hyvideo/generate_custom_trajectory.py`` on synthetic
   motions and must agree bit-for-bit. Skips automatically once ``HY15/`` is gone.
2. Golden / invariant — frame-index sampling and trajectory shape are checked
   against hand-derived invariants that survive legacy removal.
"""

import importlib.util
import os

import numpy as np
import pytest

from minwm.data.preprocessing.hy import sample_frame_indices
from minwm.data.preprocessing.trajectory import generate_camera_trajectory_local

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LEGACY_TRAJ = os.path.join(_REPO_ROOT, "HY15", "hyvideo", "generate_custom_trajectory.py")
_LEGACY_PRESENT = os.path.isfile(_LEGACY_TRAJ)
_skip_no_legacy = pytest.mark.skipif(
    not _LEGACY_PRESENT, reason="legacy HY15/ tree removed — differential check N/A"
)


def _load_legacy_trajectory():
    """Import the standalone legacy trajectory module by file path."""
    spec = importlib.util.spec_from_file_location("legacy_traj", _LEGACY_TRAJ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_motions():
    """A motion list exercising every recognized key."""
    return (
        [{"forward": 0.08}] * 6
        + [{"yaw": np.deg2rad(3)}] * 4
        + [{"pitch": -np.deg2rad(3)}] * 3
        + [{"right": -0.08}] * 2
        + [{"third_yaw": np.deg2rad(5)}] * 2
    )


# ---- layer 1: differential vs legacy HY15/ -------------------------------


@_skip_no_legacy
def test_trajectory_matches_legacy_bit_for_bit():
    legacy = _load_legacy_trajectory()
    motions = _sample_motions()

    new = generate_camera_trajectory_local(motions)
    old = legacy.generate_camera_trajectory_local(motions)

    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)


# ---- layer 2: golden / invariant -----------------------------------------


def test_trajectory_starts_at_identity_and_length():
    motions = _sample_motions()
    poses = generate_camera_trajectory_local(motions)

    assert len(poses) == len(motions) + 1
    assert np.array_equal(poses[0], np.eye(4))
    for p in poses:
        assert p.shape == (4, 4)


def test_sample_frame_indices_length_and_alignment():
    # 20 camera frames -> n_select == 20, so the middle slice is the whole set.
    cam = list(range(0, 80, 4))  # [0, 4, ..., 76]
    idx = sample_frame_indices(cam, max_frames=77)

    assert idx is not None
    assert len(idx) == 77
    # every 4th index (the group boundary) lands on a camera frame
    assert idx[0] == cam[0]
    for k in range(1, 20):
        assert idx[4 * k] == cam[k]


def test_sample_frame_indices_too_few_returns_none():
    assert sample_frame_indices([0, 4, 8], max_frames=77) is None
