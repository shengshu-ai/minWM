"""``.pt``-file-backed camera datasets for HunyuanVideo AR training.

These load the pre-encoded ``.pt`` latents produced by
``tools/data/hy/preencode_camera_video.py`` and build
the ``viewmats``/``Ks`` PRoPE inputs (plus discrete action labels) on the fly.

Two datasets, matching the documented HY Action2V stages:

* :class:`CameraPluckerDataset` — single clean latent per sample. Used by
  Phase 1 Bidirectional SFT, Phase 2 Stage 1 (Teacher-Forcing AR), and Stage 2b
  (Consistency Distillation).
* :class:`CausalODEDataset` — full ODE trajectory per sample. Used by Phase 2
  Stage 2a (Causal ODE Distillation Initialization).

Unlike the legacy trainer these are plain map-style ``Dataset``s: the
SP-aware sampler and infinite iteration are owned by
:func:`minwm.data.build_dataloader`, so the dataset only produces samples.

``.pt`` keys consumed:
    latent, prompt_embeds, prompt_mask, byt5_text_states, byt5_text_mask,
    intrinsics (4,), poses (N_cam, 7) w2c OpenCV; plus image_cond /
    vision_states for ``task_type="i2v"``; plus ode_trajectory and optional
    pre-built viewmats / Ks for :class:`CausalODEDataset`.
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset

from .action import discretize_poses_to_actions
from .geometry import build_viewmats_and_Ks

__all__ = ["CameraPluckerDataset", "CausalODEDataset"]

# HY1.5 i2v conditioning shapes for the text-to-video (no-image) path.
_T2V_VISION_STATES_SHAPE = (1, 4096, 3584)
_T2V_IMAGE_COND_CHANNELS = 32


def _load_neg_prompts(
    neg_prompt_path: str | None, neg_byt5_path: str | None, json_path: str
) -> tuple[dict, dict]:
    """Load the negative-prompt embeddings used for classifier-free guidance."""
    if neg_prompt_path is None:
        neg_prompt_path = os.environ.get(
            "NEG_PROMPT_PT", os.path.join(os.path.dirname(json_path), "hunyuan_neg_prompt.pt")
        )
    if neg_byt5_path is None:
        neg_byt5_path = os.environ.get(
            "NEG_BYT5_PT", os.path.join(os.path.dirname(json_path), "hunyuan_neg_byt5_prompt.pt")
        )
    neg_prompt = torch.load(neg_prompt_path, map_location="cpu", weights_only=True)
    neg_byt5 = torch.load(neg_byt5_path, map_location="cpu", weights_only=True)
    return neg_prompt, neg_byt5


class CameraPluckerDataset(Dataset):
    """Pre-encoded clean-latent dataset with PRoPE camera conditioning.

    Loads one ``.pt`` per sample, truncates the latent to ``window_frames``,
    builds ``viewmats``/``Ks`` from the stored intrinsics/poses, and derives
    discrete action labels. With probability ``cfg_rate`` the text embeddings
    are swapped for the negative prompt (classifier-free guidance).

    Args:
        json_path (str): path to an index JSON; a list of ``{"latent_path": ...}``.
        window_frames (int): max latent frames ``F`` to keep per sample.
        cfg_rate (float): probability of substituting the negative prompt.
        task_type (str): ``"i2v"`` keeps stored image conditioning; anything
            else emits zero image-cond / vision-states (t2v).
        seed (int): RNG seed for CFG dropout and resampling on load failure.
        neg_prompt_path (str, optional): negative prompt ``.pt``. Defaults to
            ``$NEG_PROMPT_PT`` or ``<json dir>/hunyuan_neg_prompt.pt``.
        neg_byt5_path (str, optional): negative byT5 ``.pt``. Defaults to
            ``$NEG_BYT5_PT`` or ``<json dir>/hunyuan_neg_byt5_prompt.pt``.

    Each sample is a dict with keys: ``latent`` ``[C,F,H,W]``, ``viewmats``
    ``[F,4,4]``, ``Ks`` ``[F,3,3]``, ``action`` ``[F]``, ``prompt_embed``,
    ``prompt_mask``, ``byt5_text_states``, ``byt5_text_mask``, ``image_cond``,
    ``vision_states``, ``i2v_mask``, ``video_path``, ``select_window_out_flag``.
    """

    def __init__(
        self,
        json_path: str,
        window_frames: int,
        cfg_rate: float = 0.0,
        task_type: str = "t2v",
        seed: int = 0,
        neg_prompt_path: str | None = None,
        neg_byt5_path: str | None = None,
    ):
        with open(json_path, "r") as f:
            self.json_data = json.load(f)
        self.all_length = len(self.json_data)
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.task_type = task_type
        self.rng = random.Random(seed)
        self.neg_prompt_pt, self.neg_byt5_pt = _load_neg_prompts(
            neg_prompt_path, neg_byt5_path, json_path
        )

    def __len__(self) -> int:
        return self.all_length

    def __getitem__(self, idx: int) -> dict:
        while True:
            try:
                latent_pt_path = self.json_data[idx]["latent_path"]
                latent_pt = torch.load(latent_pt_path, map_location="cpu", weights_only=True)
                latent = latent_pt["latent"][0]  # (C, T_lat, H_lat, W_lat)
                latent_length = latent.shape[1]
                if latent_length < self.window_frames:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue

                max_length = min(latent_length // 4 * 4, self.window_frames)
                latent = latent[:, :max_length, ...]
                T_lat, H_lat, W_lat = latent.shape[1], latent.shape[2], latent.shape[3]

                intrinsics = latent_pt["intrinsics"].numpy()  # (4,)
                if intrinsics[0] <= 0 or intrinsics[1] <= 0:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                poses_lat = latent_pt["poses"].numpy()[:T_lat]  # (T_lat, 7)
                viewmats_np, Ks_np = build_viewmats_and_Ks(intrinsics, poses_lat)

                action = torch.from_numpy(discretize_poses_to_actions(viewmats_np))  # (T_lat,)
                viewmats = torch.from_numpy(viewmats_np)  # (T_lat, 4, 4)
                Ks = torch.from_numpy(Ks_np)  # (T_lat, 3, 3)

                text = self._select_text(latent_pt)
                image_cond, vision_states = self._image_conditioning(latent_pt, H_lat, W_lat)

                return {
                    "latent": latent,
                    "viewmats": viewmats,
                    "Ks": Ks,
                    "action": action,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "i2v_mask": torch.ones_like(latent),
                    "video_path": latent_pt_path,
                    "select_window_out_flag": 0,
                    **text,
                }
            except Exception as e:  # noqa: BLE001 - skip corrupt samples, resample
                print("error:", e, self.json_data[idx].get("latent_path"), flush=True)
                idx = self.rng.randint(0, self.all_length - 1)

    def _select_text(self, latent_pt: dict) -> dict:
        """Return prompt embeddings, swapped for the negative prompt at cfg_rate."""
        if self.rng.random() < self.cfg_rate:
            return {
                "prompt_embed": self.neg_prompt_pt["negative_prompt_embeds"][0],
                "prompt_mask": self.neg_prompt_pt["negative_prompt_mask"][0],
                "byt5_text_states": self.neg_byt5_pt["byt5_text_states"][0],
                "byt5_text_mask": self.neg_byt5_pt["byt5_text_mask"][0],
            }
        return {
            "prompt_embed": latent_pt["prompt_embeds"][0],
            "prompt_mask": latent_pt["prompt_mask"][0],
            "byt5_text_states": latent_pt["byt5_text_states"][0],
            "byt5_text_mask": latent_pt["byt5_text_mask"][0],
        }

    def _image_conditioning(
        self, latent_pt: dict, h_lat: int, w_lat: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image_cond, vision_states)``; zeros for the t2v path."""
        if self.task_type == "i2v":
            return latent_pt["image_cond"][0], latent_pt["vision_states"][0]
        image_cond = torch.zeros(_T2V_IMAGE_COND_CHANNELS, 1, h_lat, w_lat)
        vision_states = torch.zeros(*_T2V_VISION_STATES_SHAPE)
        return image_cond, vision_states


class CausalODEDataset(CameraPluckerDataset):
    """Pre-encoded ODE-trajectory dataset with PRoPE camera conditioning.

    Like :class:`CameraPluckerDataset` but each ``.pt`` also stores an
    ``ode_trajectory`` (the solver outputs the student regresses against). When
    the ``.pt`` carries pre-built ``viewmats``/``Ks`` (newer ODE outputs) they
    are used directly; otherwise they are built from the stored intrinsics and
    poses, matching :class:`CameraPluckerDataset`.

    Each sample adds an ``ode_trajectory`` key to the
    :class:`CameraPluckerDataset` contract.
    """

    def __getitem__(self, idx: int) -> dict:
        while True:
            try:
                latent_pt_path = self.json_data[idx]["latent_path"]
                latent_pt = torch.load(latent_pt_path, map_location="cpu", weights_only=True)
                latent = latent_pt["latent"][0]  # (C, T_lat, H_lat, W_lat)
                latent_length = latent.shape[1]
                if latent_length < self.window_frames:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue

                latent = latent[:, : self.window_frames, ...]
                H_lat, W_lat = latent.shape[2], latent.shape[3]

                text = self._select_text(latent_pt)
                image_cond, vision_states = self._image_conditioning(latent_pt, H_lat, W_lat)
                viewmats, Ks = self._camera(latent_pt, latent.shape[1])
                action = torch.from_numpy(discretize_poses_to_actions(viewmats))  # (T_lat,)

                return {
                    "i2v_mask": torch.ones_like(latent),
                    "latent": latent,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "video_path": latent_pt_path,
                    "select_window_out_flag": 0,
                    "ode_trajectory": latent_pt["ode_trajectory"][0],
                    "viewmats": viewmats,
                    "Ks": Ks,
                    "action": action,
                    **text,
                }
            except Exception as e:  # noqa: BLE001 - skip corrupt samples, resample
                print("error:", e, self.json_data[idx].get("latent_path"), flush=True)
                idx = self.rng.randint(0, self.all_length - 1)

    def _camera(self, latent_pt: dict, t_lat: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Use pre-built viewmats/Ks if present, else build from poses."""
        if latent_pt.get("viewmats") is not None:
            return latent_pt["viewmats"][0], latent_pt["Ks"][0]
        intrinsics = latent_pt["intrinsics"].numpy()  # (4,)
        poses_lat = latent_pt["poses"].numpy()[:t_lat]  # (T_lat, 7)
        viewmats_np, Ks_np = build_viewmats_and_Ks(intrinsics, poses_lat)
        return torch.from_numpy(viewmats_np).float(), torch.from_numpy(Ks_np).float()
