"""Discrete action utilities for WorldPlay-style action conditioning.

Action representation: ``action_label = trans_label * 9 + rotate_label`` (81
classes total).

- Translation (9 classes): no-action(0), forward(1), backward(2), left(3),
  right(4), forward+left(5), forward+right(6), backward+left(7),
  backward+right(8).
- Rotation (9 classes): same pattern over yaw_right(1), yaw_left(2),
  pitch_up(3), pitch_down(4), ...
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

__all__ = [
    "discretize_poses_to_actions",
    "trajectory_str_to_action_labels",
    "one_hot_to_label",
]

# (forward, backward, left, right) one-hot -> translation label (0-8).
_TRANS_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

# Trajectory key -> (trans_label, rotate_label).
_TRAJ_KEY_TO_LABELS = {
    "w": (1, 0),  # forward
    "s": (2, 0),  # backward
    "a": (3, 0),  # left
    "d": (4, 0),  # right
    "j": (0, 2),  # yaw left
    "l": (0, 1),  # yaw right
    "i": (0, 3),  # pitch up
    "k": (0, 4),  # pitch down
}

_MOVE_NORM_THRESHOLD = 0.01
_ROT_THRESHOLD_DEG = 5e-2


def one_hot_to_label(one_hot: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` one-hot rows to ``(N,)`` integer labels (0-8).

    Args:
        one_hot (np.ndarray): ``(N, 4)`` one-hot direction flags.

    Returns:
        np.ndarray: ``(N,)`` int64 labels; unrecognised rows map to 0.
    """
    labels = np.zeros(len(one_hot), dtype=np.int64)
    for i, row in enumerate(one_hot):
        key = tuple(int(x) for x in row)
        labels[i] = _TRANS_MAPPING.get(key, 0)
    return labels


def discretize_poses_to_actions(viewmats) -> np.ndarray:
    """Derive discrete action labels from consecutive w2c view matrices.

    Each frame's action is classified from its relative camera motion against
    the previous frame (frame 0 is always no-action).

    Args:
        viewmats: ``(T, 4, 4)`` w2c matrices (center-normalized to the first
            frame). Accepts a numpy array or a torch tensor of any dtype.

    Returns:
        np.ndarray: ``(T,)`` int64 labels, ``trans_label * 9 + rotate_label``.
    """
    if not isinstance(viewmats, np.ndarray):
        viewmats = viewmats.float().numpy()
    T = len(viewmats)
    c2ws = np.linalg.inv(viewmats)

    trans_one_hot = np.zeros((T, 4), dtype=np.int32)  # forward, backward, left, right
    rotate_one_hot = np.zeros((T, 4), dtype=np.int32)  # yaw_right, yaw_left, pitch_up, pitch_down

    for i in range(1, T):
        rel_c2w = np.linalg.inv(c2ws[i - 1]) @ c2ws[i]
        move_dirs = rel_c2w[:3, 3]
        move_norm = np.linalg.norm(move_dirs)

        if move_norm > _MOVE_NORM_THRESHOLD:
            move_norm_dirs = move_dirs / move_norm
            angles_deg = np.degrees(np.arccos(np.clip(move_norm_dirs, -1.0, 1.0)))

            # Z-axis: forward (< 60°) / backward (> 120°)
            if angles_deg[2] < 60:
                trans_one_hot[i, 0] = 1
            elif angles_deg[2] > 120:
                trans_one_hot[i, 1] = 1

            # X-axis: left (< 60°) / right (> 120°)
            if angles_deg[0] < 60:
                trans_one_hot[i, 2] = 1
            elif angles_deg[0] > 120:
                trans_one_hot[i, 3] = 1

        rot_angles_deg = Rotation.from_matrix(rel_c2w[:3, :3]).as_euler("xyz", degrees=True)

        # Yaw (Y-axis rotation)
        if rot_angles_deg[1] > _ROT_THRESHOLD_DEG:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -_ROT_THRESHOLD_DEG:
            rotate_one_hot[i, 1] = 1

        # Pitch (X-axis rotation)
        if rot_angles_deg[0] > _ROT_THRESHOLD_DEG:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -_ROT_THRESHOLD_DEG:
            rotate_one_hot[i, 3] = 1

    trans_labels = one_hot_to_label(trans_one_hot)
    rotate_labels = one_hot_to_label(rotate_one_hot)
    return trans_labels * 9 + rotate_labels


def trajectory_str_to_action_labels(traj_str: str, num_frames: int) -> torch.Tensor:
    """Convert a trajectory string like ``"w*4,a*8,d*7"`` to action labels.

    Args:
        traj_str (str): segments ``key*count`` joined by commas; keys are
            ``w/s/a/d/j/l/i/k``. Empty / blank yields all no-action.
        num_frames (int): number of latent frames to emit.

    Returns:
        torch.Tensor: ``(num_frames,)`` int64 labels; frame 0 is always 0
        (no-action), actions fill from frame 1 and are truncated to fit.
    """
    if not traj_str or traj_str.strip() == "":
        return torch.zeros(num_frames, dtype=torch.int64)

    actions: list[int] = []
    for seg in traj_str.strip().split(","):
        seg = seg.strip()
        if "*" in seg:
            key, count_str = seg.split("*", 1)
            count = int(count_str)
        else:
            key, count = seg, 1
        trans_label, rotate_label = _TRAJ_KEY_TO_LABELS.get(key.lower().strip(), (0, 0))
        actions.extend([trans_label * 9 + rotate_label] * count)

    result = np.zeros(num_frames, dtype=np.int64)
    fill_len = min(len(actions), num_frames - 1)
    result[1 : 1 + fill_len] = actions[:fill_len]
    return torch.from_numpy(result)
