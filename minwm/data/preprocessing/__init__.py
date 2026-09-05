"""Data-preprocessing helpers: HY15 latent extraction + camera-trajectory synthesis.

Importing this subpackage stays light — the heavy HY15 encoders are only pulled in
when :mod:`minwm.data.preprocessing.hy` is imported.
"""

from .trajectory import generate_camera_trajectory_local

__all__ = ["generate_camera_trajectory_local"]
