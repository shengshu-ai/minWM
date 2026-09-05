#!/usr/bin/env python3
"""
Build camera-aware LMDB from WorldPlayGen data for Wan + PRoPE training.

Input:
    - input_json:  preencode_input.json with [image_path, caption, pose_json_path, pose_str]
    - video_dir:   {index}_{pose_str}/gen.mp4 (77 frames, 480x832)

Output LMDB keys (same format as build_dl3dv_lmdb.py):
    latents    -- (N, 20, 16, 60, 104) float16   Wan VAE latent
    prompts    -- (N,) str                        caption text
    intrinsics -- (N, 4) float32                  [fx/W, fy/H, cx/W, cy/H]
    poses      -- (N, 20, 7) float32              [tx,ty,tz, qx,qy,qz,qw]

Each 77-frame video -> 1 segment -> 20 latent frames.
Memory-safe: each rank streams to its own LMDB shard, rank 0 merges at end.

Usage:
    torchrun --nproc_per_node=8 tools/data/wan21/build_worldplaygen_lmdb.py \
        --input_json /path/to/preencode_input.json \
        --video_dir  /path/to/videos \
        --output_dir /path/to/output_lmdb \
        --vae_path   wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
"""

import argparse
import json
import os
import re
import shutil
import time

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from minwm.data.preprocessing.trajectory import generate_camera_trajectory_local
from minwm.modeling.wan21.vae import _video_vae

MAX_FRAMES = 77
N_LATENT = (MAX_FRAMES - 1) // 4 + 1  # 20
# bytes per sample for LMDB map_size estimation
PER_SAMPLE_BYTES = 20 * 16 * 60 * 104 * 2 + 4 * 4 + 20 * 7 * 4 + 2000

try:
    import decord

    decord.bridge.set_bridge("torch")
    USE_DECORD = True
except ImportError:
    USE_DECORD = False
    import cv2


def load_video_frames(video_path, target_h=480, target_w=832):
    """Load all 77 frames from video, resize, return [C, F, H, W] in [-1, 1]."""
    if USE_DECORD:
        vr = decord.VideoReader(video_path)
        if len(vr) < MAX_FRAMES:
            return None
        frames = vr.get_batch(list(range(MAX_FRAMES)))
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    else:
        cap = cv2.VideoCapture(video_path)
        buf = []
        for fi in range(MAX_FRAMES):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        tensor = torch.from_numpy(np.stack(buf)).float().permute(3, 0, 1, 2)
    if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(tensor.shape[1], target_h, target_w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    tensor = (tensor / 255.0 - 0.5) * 2.0
    return tensor


def _parse_pose_string(pose_string):
    """Parse pose string like 'down-4, up-4, w-4, a-7' into motions list.

    Exact copy of preencode_generated_wdplay.py to ensure alignment.
    """
    forward_speed = 0.08
    yaw_speed = np.deg2rad(3)
    pitch_speed = np.deg2rad(3)
    motions = []
    for cmd in [c.strip() for c in pose_string.split(",")]:
        if not cmd:
            continue
        parts = cmd.split("-")
        action = parts[0].strip()
        num_frames = int(float(parts[1].strip()))
        for _ in range(num_frames):
            if action == "w":
                motions.append({"forward": forward_speed})
            elif action == "s":
                motions.append({"forward": -forward_speed})
            elif action == "a":
                motions.append({"right": -forward_speed})
            elif action == "d":
                motions.append({"right": forward_speed})
            elif action == "up":
                motions.append({"pitch": pitch_speed})
            elif action == "down":
                motions.append({"pitch": -pitch_speed})
            elif action == "left":
                motions.append({"yaw": -yaw_speed})
            elif action == "right":
                motions.append({"yaw": yaw_speed})
            else:
                raise ValueError(f"Unknown action: {action}")
    return motions


def poses_from_pose_str(pose_str):
    """Convert pose_str to (intrinsics, poses) matching the .pt pipeline.

    Regenerates c2w from pose_str via generate_camera_trajectory_local,
    then inverts to w2c. This is the same code path as
    preencode_generated_wdplay.py:poses_from_pose_str.

    Returns:
        intrinsics: (4,) float32 — [fx_norm, fy_norm, cx_norm, cy_norm]
        poses:      (N_LATENT, 7) float32 — [tx,ty,tz, qx,qy,qz,qw] w2c
    """
    motions = _parse_pose_string(pose_str)
    c2w_list = generate_camera_trajectory_local(motions)
    assert (
        len(c2w_list) >= N_LATENT
    ), f"pose_str '{pose_str}' produces {len(c2w_list)} frames, need {N_LATENT}"

    # Intrinsics: WorldPlayGen default (1920x1080, f~969.7)
    fx_norm = 969.6969696969696 / 1920.0
    fy_norm = 969.6969696969696 / 1080.0
    intrinsics = np.array([fx_norm, fy_norm, 0.5, 0.5], dtype=np.float32)

    # Poses: c2w → w2c → [tx,ty,tz, qx,qy,qz,qw]
    poses = np.zeros((N_LATENT, 7), dtype=np.float32)
    for i in range(N_LATENT):
        c2w = np.array(c2w_list[i])
        w2c = np.linalg.inv(c2w)
        poses[i, :3] = w2c[:3, 3]
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()

    return intrinsics, poses


def pose_str_to_dir_suffix(pose_str):
    return re.sub(r"[^a-z0-9]", "", pose_str.lower().replace(" ", ""))


class Wan21VAE:
    def __init__(self, vae_path, device):
        self.device = device
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.model = (
            _video_vae(pretrained_path=vae_path, z_dim=16).eval().requires_grad_(False).to(device)
        )

    @torch.no_grad()
    def encode(self, pixel):
        scale = [self.mean.to(self.device), 1.0 / self.std.to(self.device)]
        z = self.model.encode(pixel.to(self.device), scale).float()
        z = z.permute(0, 2, 1, 3, 4)  # (1,16,F,H,W) -> (1,F,16,H,W)
        return z.half()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vae_path", default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--target_h", type=int, default=480)
    p.add_argument("--target_w", type=int, default=832)
    return p.parse_args()


def main():
    args = parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        import datetime
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=4))
        global_rank = dist.get_rank()
    else:
        global_rank = 0
    device = torch.device(f"cuda:{local_rank}")

    # Load input JSON
    with open(args.input_json) as f:
        data_list = json.load(f)
    if global_rank == 0:
        print(f"Loaded {len(data_list)} entries from {args.input_json}")

    # Build video paths and validate
    valid_list = []
    for i, entry in enumerate(data_list):
        suffix = pose_str_to_dir_suffix(entry["pose_str"])
        vp = os.path.join(args.video_dir, f"{i:06d}_{suffix}", "gen.mp4")
        if os.path.exists(vp):
            valid_list.append(
                {
                    "index": i,
                    "video_path": vp,
                    "pose_str": entry["pose_str"],
                    "caption": entry["caption"],
                }
            )
    if global_rank == 0:
        print(f"Valid videos: {len(valid_list)} / {len(data_list)}")

    # Shard across GPUs
    if world_size > 1:
        per_gpu = (len(valid_list) + world_size - 1) // world_size
        shard = valid_list[global_rank * per_gpu : (global_rank + 1) * per_gpu]
    else:
        shard = valid_list
    print(f"GPU{local_rank}: {len(shard)} samples to encode")

    # Init VAE
    vae = Wan21VAE(args.vae_path, device)

    # Each rank streams into its own LMDB shard (no memory buildup)
    os.makedirs(args.output_dir, exist_ok=True)
    rank_dir = os.path.join(args.output_dir, f".rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)
    rank_map = int(len(shard) * PER_SAMPLE_BYTES * 1.3) + 100_000_000
    rank_env = lmdb.open(rank_dir, map_size=rank_map, subdir=True)

    count = 0
    errors = 0
    first_shape = None
    t0 = time.time()

    pbar = tqdm(shard, desc=f"GPU{local_rank}", disable=(global_rank != 0), dynamic_ncols=True)
    for idx, item in enumerate(pbar):
        video_tensor = None
        pixel = None
        try:
            video_tensor = load_video_frames(item["video_path"], args.target_h, args.target_w)
            if video_tensor is None:
                errors += 1
                continue
            pixel = video_tensor.unsqueeze(0).to(device)
            del video_tensor
            video_tensor = None
            latent_np = vae.encode(pixel).cpu().numpy()[0]
            del pixel
            pixel = None
            intrinsics, poses = poses_from_pose_str(item["pose_str"])
        except Exception as e:
            print(f"Error #{item['index']}: {e}")
            errors += 1
            continue
        finally:
            del video_tensor, pixel
            torch.cuda.empty_cache()

        # Stream write to per-rank LMDB
        with rank_env.begin(write=True) as txn:
            txn.put(f"latents_{count}_data".encode(), latent_np.tobytes())
            txn.put(f"prompts_{count}_data".encode(), item["caption"].encode())
            txn.put(f"intrinsics_{count}_data".encode(), intrinsics.tobytes())
            txn.put(f"poses_{count}_data".encode(), poses.tobytes())

        if first_shape is None:
            first_shape = (latent_np.shape, intrinsics.shape, poses.shape)
        if count == 0 and local_rank == 0:
            print(
                f"\nFirst sample: latent={latent_np.shape} "
                f"intrinsics={intrinsics} poses={poses.shape}"
            )

        del latent_np, intrinsics, poses
        count += 1

        elapsed = time.time() - t0
        speed = (idx + 1) / elapsed if elapsed > 0 else 0
        pbar.set_postfix(ok=count, err=errors, speed=f"{speed:.1f}it/s")

    # Write rank metadata
    with rank_env.begin(write=True) as txn:
        txn.put(b"__count__", str(count).encode())
        if first_shape:
            txn.put(b"__lat_shape__", " ".join(map(str, first_shape[0])).encode())
            txn.put(b"__intr_shape__", " ".join(map(str, first_shape[1])).encode())
            txn.put(b"__poses_shape__", " ".join(map(str, first_shape[2])).encode())
    rank_env.sync()
    rank_env.close()
    print(f"GPU{local_rank}: {count} OK, {errors} errors, " f"{time.time() - t0:.0f}s elapsed")

    # ---- Phase 2: Rank 0 merges per-rank LMDBs (streaming, constant mem) ----
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    if global_rank == 0:
        total = 0
        lat_shape = intr_shape = poses_shape = None
        for r in range(world_size):
            rd = os.path.join(args.output_dir, f".rank_{r}")
            if not os.path.exists(rd):
                continue
            renv = lmdb.open(rd, readonly=True, lock=False)
            with renv.begin() as txn:
                total += int(txn.get(b"__count__").decode())
                if lat_shape is None:
                    ls = txn.get(b"__lat_shape__")
                    if ls:
                        lat_shape = tuple(map(int, ls.decode().split()))
                        intr_shape = tuple(map(int, txn.get(b"__intr_shape__").decode().split()))
                        poses_shape = tuple(map(int, txn.get(b"__poses_shape__").decode().split()))
            renv.close()

        if total == 0:
            print("No valid samples.")
        else:
            print(f"Merging {total} samples from {world_size} ranks ...")
            final_dir = os.path.join(args.output_dir, "data")
            os.makedirs(final_dir, exist_ok=True)
            fmap = int(total * PER_SAMPLE_BYTES * 1.3) + 1_000_000_000
            env = lmdb.open(final_dir, map_size=fmap, subdir=True)

            gi = 0
            for r in tqdm(range(world_size), desc="Merge ranks"):
                rd = os.path.join(args.output_dir, f".rank_{r}")
                if not os.path.exists(rd):
                    continue
                renv = lmdb.open(rd, readonly=True, lock=False)
                with renv.begin() as rtxn:
                    rc = int(rtxn.get(b"__count__").decode())
                    for j in range(rc):
                        lat = rtxn.get(f"latents_{j}_data".encode())
                        cap = rtxn.get(f"prompts_{j}_data".encode())
                        intr = rtxn.get(f"intrinsics_{j}_data".encode())
                        pos = rtxn.get(f"poses_{j}_data".encode())
                        with env.begin(write=True) as wtxn:
                            wtxn.put(f"latents_{gi}_data".encode(), lat)
                            wtxn.put(f"prompts_{gi}_data".encode(), cap)
                            wtxn.put(f"intrinsics_{gi}_data".encode(), intr)
                            wtxn.put(f"poses_{gi}_data".encode(), pos)
                        gi += 1
                renv.close()
                shutil.rmtree(rd)

            with env.begin(write=True) as txn:
                txn.put(b"latents_shape", f"{total} {' '.join(map(str, lat_shape))}".encode())
                txn.put(b"prompts_shape", f"{total}".encode())
                txn.put(b"intrinsics_shape", f"{total} {' '.join(map(str, intr_shape))}".encode())
                txn.put(b"poses_shape", f"{total} {' '.join(map(str, poses_shape))}".encode())
            env.sync()
            env.close()
            print(f"Done! {total} samples -> {final_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
