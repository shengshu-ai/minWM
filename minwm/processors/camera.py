"""Camera-condition input preparation for inference.

Also hosts the camera trajectory generators used by the ``CameraTrajectory``
preprocessor. All poses are w2c (world-to-camera) OpenCV convention, matching
the training pipeline (``build_worldplaygen_lmdb.py``: c2w -> w2c via
``np.linalg.inv``).

Trajectory string format (each step = 0.08 unit translation or 3 deg rotation)::

    w*N  -- move forward  (+Z in camera local frame)
    s*N  -- move backward (-Z)
    a*N  -- move left     (-X)
    d*N  -- move right    (+X)
    u*N  -- move up       (-Y, OpenCV Y-down)
    dn*N -- move down     (+Y)
    j*N  -- yaw left      (rotate around Y, positive)
    l*N  -- yaw right     (rotate around Y, negative)
    i*N  -- pitch up      (rotate around X, negative)
    k*N  -- pitch down    (rotate around X, positive)

Example: ``"w*19"`` -> 20-frame forward dolly (1 identity + 19 steps).
Chain segments with commas: ``"w*10,d*9"``, ``"w*9,j*10"``.
"""

import re

import numpy as np
import torch
from torch import Tensor

from .base import InferenceInputPreprocessor, InferenceRuntime


class CameraTrajectory(InferenceInputPreprocessor):
    """Attach PRoPE camera tensors when a trajectory string is supplied."""

    def __init__(self, trajectory_key: str = "trajectory") -> None:
        self.trajectory_key = trajectory_key

    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        trajectory = batch.get(self.trajectory_key)
        if not trajectory:
            return batch

        if isinstance(trajectory, (list, tuple)):
            pairs = [
                make_camera_tensors(item, device=runtime.device, dtype=runtime.dtype)
                for item in trajectory
            ]
            viewmats = torch.cat([p[0] for p in pairs], dim=0)
            ks = torch.cat([p[1] for p in pairs], dim=0)
        else:
            viewmats, ks = make_camera_tensors(
                trajectory,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            batch_size = len(batch["prompts"])
            if batch_size != viewmats.shape[0]:
                viewmats = viewmats.expand(batch_size, -1, -1, -1)
                ks = ks.expand(batch_size, -1, -1, -1)
        batch["viewmats"] = viewmats
        batch["Ks"] = ks
        return batch


_STEP = 0.08
_ROT_STEP = np.radians(3.0)  # 3.0 degrees per latent frame

# Direction -> per-step motion dict (same convention as HY-WorldPlay).
_MOTIONS: dict[str, dict[str, float]] = {
    "w": {"forward": _STEP},
    "s": {"forward": -_STEP},
    "d": {"right": _STEP},
    "a": {"right": -_STEP},
    "u": {"up": _STEP},
    "dn": {"up": -_STEP},
    "j": {"yaw": -_ROT_STEP},  # yaw left
    "l": {"yaw": _ROT_STEP},  # yaw right
    "i": {"pitch": _ROT_STEP},  # pitch up
    "k": {"pitch": -_ROT_STEP},  # pitch down
}


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _generate_c2w_trajectory(motions: list[dict[str, float]]) -> list[np.ndarray]:
    """Build c2w 4x4 matrices from per-step motion dicts.

    Exact equivalent of ``HY-WorldPlay/hyvideo/generate_custom_trajectory.py`` so
    the generated poses match the training pipeline.
    """
    T = np.eye(4)
    poses = [T.copy()]
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])
        up = move.get("up", 0.0)
        if up:
            # up in camera frame = -Y (OpenCV Y-down)
            T[:3, 3] += T[:3, :3] @ np.array([0, -up, 0])
        poses.append(T.copy())
    return poses


def parse_trajectory(traj_str: str) -> np.ndarray:
    """Parse a trajectory string into ``(T, 4, 4)`` w2c view matrices.

    Builds c2w via :func:`_generate_c2w_trajectory` (matching the training
    pipeline), then inverts to w2c via ``np.linalg.inv``. The first frame is
    always identity.

    Args:
        traj_str (str): trajectory string such as ``"w*19"`` or ``"w*10,d*9"``.

    Returns:
        np.ndarray: ``(T, 4, 4)`` float32 w2c matrices, where ``T`` is one more
        than the total step count (the leading identity frame).

    Raises:
        ValueError: if a segment is malformed or names an unknown direction.
    """
    segments = traj_str.strip().split(",")
    motions: list[dict[str, float]] = []
    for seg in segments:
        seg = seg.strip()
        m = re.fullmatch(r"([a-z]+)\*(\d+)", seg)
        if m is None:
            raise ValueError(f"Cannot parse trajectory segment: '{seg}'. Expected 'w*19'.")
        key, n = m.group(1), int(m.group(2))
        if key not in _MOTIONS:
            raise ValueError(f"Unknown direction '{key}'. Valid: {list(_MOTIONS.keys())}")
        motions.extend([_MOTIONS[key]] * n)

    c2w_list = _generate_c2w_trajectory(motions)
    T = len(c2w_list)
    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i, c2w in enumerate(c2w_list):
        viewmats[i] = np.linalg.inv(c2w)
    return viewmats


def make_camera_tensors(
    traj_str: str,
    fx: float = 0.5,
    fy: float = 0.5,
    cx: float = 0.5,
    cy: float = 0.5,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build ``(1, T, 4, 4)`` view matrices and ``(1, T, 3, 3)`` intrinsics.

    The intrinsics default to the normalized ``0.5`` placeholders used by the
    legacy ``wan_inference.py`` PRoPE path, so a config can pass its own values
    without changing call sites.

    Args:
        traj_str (str): trajectory string such as ``"w*19"``.
        fx (float): normalized focal length x. Defaults to 0.5.
        fy (float): normalized focal length y. Defaults to 0.5.
        cx (float): normalized principal point x. Defaults to 0.5.
        cy (float): normalized principal point y. Defaults to 0.5.
        device (str | torch.device): output device. Defaults to ``"cpu"``.
        dtype (torch.dtype): output dtype. Defaults to ``torch.float32``.

    Returns:
        tuple[Tensor, Tensor]:
            - viewmats (Tensor): ``(1, T, 4, 4)`` w2c matrices.
            - Ks (Tensor): ``(1, T, 3, 3)`` intrinsics.
    """
    viewmats_np = parse_trajectory(traj_str)
    T = len(viewmats_np)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    Ks_np = np.tile(K, (T, 1, 1))

    viewmats = torch.tensor(viewmats_np, dtype=dtype, device=device).unsqueeze(0)
    Ks = torch.tensor(Ks_np, dtype=dtype, device=device).unsqueeze(0)
    return viewmats, Ks
