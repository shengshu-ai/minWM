"""Camera geometry helpers shared by the PRoPE-conditioned datasets.

Converts raw ``(intrinsics, poses)`` into the ``(viewmats, Ks)`` pair the
HunyuanVideo / Wan transformers consume for PRoPE attention. Poses follow the
OpenCV w2c convention: ``[tx, ty, tz, qx, qy, qz, qw]`` where the quaternion
encodes ``R_w2c`` and the translation is ``t_w2c``.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = ["build_viewmats_and_Ks", "interpolate_poses"]


def build_viewmats_and_Ks(
    intrinsics: np.ndarray, poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build 4×4 w2c view matrices and 3×3 intrinsics matrices from poses.

    All poses are normalized relative to the first frame so the trajectory is
    expressed in a canonical coordinate system anchored at the initial camera.

    Args:
        intrinsics (np.ndarray): ``(4,)`` normalized ``[fx, fy, cx, cy]``.
        poses (np.ndarray): ``(T, 7)`` ``[tx, ty, tz, qx, qy, qz, qw]`` w2c
            (OpenCV convention).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``viewmats`` ``(T, 4, 4)`` w2c SE3
        matrices (normalized to the first frame) and ``Ks`` ``(T, 3, 3)``
        intrinsics matrices (the same K tiled over all frames).
    """
    T = len(poses)
    fx, fy, cx, cy = intrinsics

    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i, (tx, ty, tz, qx, qy, qz, qw) in enumerate(poses):
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = [tx, ty, tz]
        viewmats[i, 3, 3] = 1.0

    c2w = np.linalg.inv(viewmats)
    c2w_aligned = np.linalg.inv(c2w[0]) @ c2w
    viewmats = np.linalg.inv(c2w_aligned).astype(np.float32)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return viewmats, np.tile(K, (T, 1, 1))


def interpolate_poses(
    poses: np.ndarray, camera_indices: np.ndarray, target_indices: np.ndarray
) -> np.ndarray:
    """Interpolate sparse camera poses onto dense target frame indices.

    Translation is interpolated linearly; rotation uses quaternion slerp.

    Args:
        poses (np.ndarray): ``(N_cam, 7)`` ``[tx, ty, tz, qx, qy, qz, qw]`` w2c.
        camera_indices (np.ndarray): ``(N_cam,)`` frame index of each pose.
        target_indices (np.ndarray): ``(T,)`` frame indices to interpolate to.

    Returns:
        np.ndarray: ``(T, 7)`` interpolated poses (float32).
    """
    cam_idx = camera_indices.astype(np.float64)
    target = target_indices.astype(np.float64)

    trans = np.stack([np.interp(target, cam_idx, poses[:, c]) for c in range(3)], axis=1)

    rots = Rotation.from_quat(poses[:, 3:])
    slerp = Slerp(cam_idx, rots)
    clamped = np.clip(target, cam_idx[0], cam_idx[-1])
    quats = slerp(clamped).as_quat()

    return np.concatenate([trans, quats], axis=1).astype(np.float32)
