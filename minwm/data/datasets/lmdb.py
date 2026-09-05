"""LMDB-backed video latent datasets.

All datasets share the same LMDB key convention::

    <array>_shape          — space-separated int tuple stored as UTF-8 string
    <array>_<i>_data       — row i as raw bytes (float16/float32/str)

Sharding is transparent: pass either a single LMDB directory (contains
``data.mdb``) or a parent directory of sharded LMDB subdirectories.
"""

import os

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import build_viewmats_and_Ks as _build_viewmats_and_Ks

# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------


def _open(path: str) -> lmdb.Environment:
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)


def _get_shape(env: lmdb.Environment, key: str) -> tuple[int, ...]:
    with env.begin() as txn:
        raw = txn.get(f"{key}_shape".encode()).decode()
    return tuple(map(int, raw.split()))


def _get_row(env: lmdb.Environment, key: str, dtype, idx: int, shape=None):
    data_key = f"{key}_{idx}_data".encode()
    with env.begin() as txn:
        raw = txn.get(data_key)
    if dtype is str:
        return raw.decode()
    arr = np.frombuffer(raw, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------


def _is_single_lmdb(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "data.mdb"))


def _open_shards(path: str) -> list[lmdb.Environment]:
    """Open all LMDB shards under *path* in sorted order."""
    envs = []
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and _is_single_lmdb(sub):
            envs.append(_open(sub))
    if not envs:
        raise FileNotFoundError(f"No LMDB shards found under {path}")
    return envs


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class _SingleLMDBDataset(Dataset):
    """Single LMDB file containing ``latents`` + ``prompts``."""

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = _open(data_path)
        self.latents_shape = _get_shape(self.env, "latents")
        self.max_pair = max_pair

    def __len__(self) -> int:
        return min(self.latents_shape[0], self.max_pair)

    def _get_latents(self, idx: int) -> np.ndarray:
        arr = _get_row(self.env, "latents", np.float16, idx, shape=self.latents_shape[1:])
        if arr.ndim == 4:
            arr = arr[None]
        return arr

    def _get_prompt(self, idx: int) -> str:
        return _get_row(self.env, "prompts", str, idx)


class _ShardedLMDBDataset(Dataset):
    """Sharded LMDB: parent directory of per-shard sub-directories."""

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = _open_shards(data_path)
        self._latents_shapes: list[tuple] = []
        self._index: list[tuple[int, int]] = []
        for sid, env in enumerate(self.envs):
            shape = _get_shape(env, "latents")
            self._latents_shapes.append(shape)
            for j in range(shape[0]):
                self._index.append((sid, j))
        self.max_pair = max_pair

    def __len__(self) -> int:
        return min(len(self._index), self.max_pair)

    def _get_latents(self, idx: int) -> tuple[np.ndarray, int, int]:
        sid, local = self._index[idx]
        arr = _get_row(
            self.envs[sid], "latents", np.float16, local, shape=self._latents_shapes[sid][1:]
        )
        if arr.ndim == 4:
            arr = arr[None]
        return arr, sid, local

    def _get_prompt(self, idx: int) -> str:
        sid, local = self._index[idx]
        return _get_row(self.envs[sid], "prompts", str, local)


# ---------------------------------------------------------------------------
# Public dataset classes
# ---------------------------------------------------------------------------


class LatentLMDBDataset(_SingleLMDBDataset):
    """Clean latent (last trajectory step) + prompt."""

    def __getitem__(self, idx: int) -> dict:
        return {
            "prompts": self._get_prompt(idx),
            "clean_latent": torch.tensor(self._get_latents(idx), dtype=torch.float32)[-1],
        }


class ODERegressionLMDBDataset(_SingleLMDBDataset):
    """Full ODE trajectory (noise→clean) + prompt."""

    def __getitem__(self, idx: int) -> dict:
        return {
            "prompts": self._get_prompt(idx),
            "ode_latent": torch.tensor(self._get_latents(idx), dtype=torch.float32),
        }


class ShardingLMDBDataset(_ShardedLMDBDataset):
    """Sharded ODE trajectory dataset."""

    def __getitem__(self, idx: int) -> dict:
        latents, _, _ = self._get_latents(idx)
        return {
            "prompts": self._get_prompt(idx),
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
        }


class CameraODERegressionLMDBDataset(ODERegressionLMDBDataset):
    """ODE trajectory + per-frame camera (viewmats, Ks) for PRoPE.

    Extra LMDB keys: ``viewmats`` (N,F,4,4) float32, ``Ks`` (N,F,3,3) float32.
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        super().__init__(data_path, max_pair)
        self._viewmats_shape = _get_shape(self.env, "viewmats")
        self._Ks_shape = _get_shape(self.env, "Ks")

    def __getitem__(self, idx: int) -> dict:
        batch = super().__getitem__(idx)
        batch["viewmats"] = torch.tensor(
            _get_row(self.env, "viewmats", np.float32, idx, shape=self._viewmats_shape[1:]),
            dtype=torch.float32,
        )
        batch["Ks"] = torch.tensor(
            _get_row(self.env, "Ks", np.float32, idx, shape=self._Ks_shape[1:]),
            dtype=torch.float32,
        )
        return batch


class CameraLatentLMDBDataset(Dataset):
    """Clean latent + camera for PRoPE.

    LMDB keys: ``intrinsics`` (N,4) [fx,fy,cx,cy], ``poses`` (N,F,7)
    [tx,ty,tz,qx,qy,qz,qw] w2c OpenCV.  viewmats/Ks are computed on-the-fly.

    ``data_path`` may be a single LMDB directory or a sharded parent directory.
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.max_pair = max_pair
        if _is_single_lmdb(data_path):
            self._sharded = False
            self._ds = LatentLMDBDataset(data_path, max_pair)
            self._intrinsics_shape = _get_shape(self._ds.env, "intrinsics")
            self._poses_shape = _get_shape(self._ds.env, "poses")
        else:
            self._sharded = True
            self._sds = _open_shards(data_path)
            self._latents_shapes: list[tuple] = []
            self._intrinsics_shapes: list[tuple] = []
            self._poses_shapes: list[tuple] = []
            self._index: list[tuple[int, int]] = []
            for sid, env in enumerate(self._sds):
                ls = _get_shape(env, "latents")
                self._latents_shapes.append(ls)
                self._intrinsics_shapes.append(_get_shape(env, "intrinsics"))
                self._poses_shapes.append(_get_shape(env, "poses"))
                for j in range(ls[0]):
                    self._index.append((sid, j))

    def __len__(self) -> int:
        if self._sharded:
            return min(len(self._index), self.max_pair)
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        if self._sharded:
            sid, local = self._index[idx]
            env = self._sds[sid]
            latents = _get_row(
                env, "latents", np.float16, local, shape=self._latents_shapes[sid][1:]
            )
            if latents.ndim == 4:
                latents = latents[None]
            prompts = _get_row(env, "prompts", str, local)
            intrinsics = _get_row(
                env, "intrinsics", np.float32, local, shape=self._intrinsics_shapes[sid][1:]
            )
            poses = _get_row(env, "poses", np.float32, local, shape=self._poses_shapes[sid][1:])
        else:
            env = self._ds.env
            latents = self._ds._get_latents(idx)
            prompts = self._ds._get_prompt(idx)
            intrinsics = _get_row(
                env, "intrinsics", np.float32, idx, shape=self._intrinsics_shape[1:]
            )
            poses = _get_row(env, "poses", np.float32, idx, shape=self._poses_shape[1:])

        viewmats, Ks = _build_viewmats_and_Ks(intrinsics, poses)
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1],
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }
