"""Camera-trajectory synthesis for HY data preprocessing.

Builds camera-to-world (c2w) 4x4 pose matrices from a list of incremental motion
dicts (``forward``/``right`` translations, ``yaw``/``pitch``/``third_yaw``
rotations). Used by the WorldPlay-style pose-string preprocessing pipelines.
"""

import numpy as np

__all__ = ["generate_camera_trajectory_local"]


def _rot_x(theta: float) -> np.ndarray:
    """Rotation matrix about the x-axis by ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta: float) -> np.ndarray:
    """Rotation matrix about the y-axis by ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def generate_camera_trajectory_local(motions: list[dict]) -> list[np.ndarray]:
    """Accumulate incremental motions into a list of c2w pose matrices.

    The trajectory starts at identity; each motion is applied in the camera's
    local frame and a snapshot of the running pose is appended, so the result
    has ``len(motions) + 1`` poses.

    Args:
        motions (list[dict]): per-step motions. Recognized keys: ``yaw`` /
            ``pitch`` / ``third_yaw`` (radians) and ``forward`` / ``right``
            (local translations). ``yaw`` rotates left/right, ``pitch`` up/down,
            ``forward`` translates along local z, ``right`` along local x, and
            ``third_yaw`` orbits about a point one unit ahead.

    Returns:
        list[np.ndarray]: ``len(motions) + 1`` camera-to-world ``(4, 4)`` matrices.
    """
    poses = []
    T = np.eye(4)
    poses.append(T.copy())

    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])

        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])

        forward = move.get("forward", 0.0)
        if forward != 0:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])

        right = move.get("right", 0.0)
        if right != 0:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])

        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            R_y = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            T = T @ (np.linalg.inv(c_origin) @ C)

        poses.append(T.copy())

    return poses
