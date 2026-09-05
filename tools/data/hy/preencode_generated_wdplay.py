#!/usr/bin/env python3
"""
Preencode WorldPlay generated (distilled) videos into CameraPluckerDataset-compatible .pt.

Input:
    - preencode_input.json: [{image_path, caption, pose_str, pose_json_path}, ...]
    - Video dir: {video_root}/{idx:06d}_{pose_suffix}/gen.mp4

Output .pt keys (matching dl3dv preencode format):
    latent (1,32,20,H,W), image_cond (1,32,1,H,W),
    prompt_embeds (1,1000,3584), prompt_mask (1,1000),
    vision_states (1,729,1152), byt5_text_states (1,256,1472), byt5_text_mask (1,256),
    intrinsics (4,), poses (20,7), camera_indices (20,),
    video_frame_indices (77,), num_video_frames (int), video_id (str)

Usage:
    torchrun --nproc_per_node=8 tools/data/hy/preencode_generated_wdplay.py \
        --input_json /path/to/preencode_input.json \
        --video_root /path/to/videos \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5 \
        --skip_existing
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from minwm.data.preprocessing.hy import VideoLatentExtractor
from minwm.data.preprocessing.trajectory import generate_camera_trajectory_local

MAX_FRAMES = 77
N_LATENT = (MAX_FRAMES - 1) // 4 + 1  # 20


# ── Pose string → camera data (WorldPlay pose-string DSL, specific to this script) ──


def parse_pose_string(pose_string):
    """Parse pose string like 'w-3, right-8, d-4' into motions list."""
    forward_speed = 0.08
    yaw_speed = np.deg2rad(3)
    pitch_speed = np.deg2rad(3)

    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    for cmd in commands:
        if not cmd:
            continue
        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pose command: {cmd}")
        action = parts[0].strip()
        num_frames = int(float(parts[1].strip()))

        if action == "w":
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == "s":
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == "a":
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == "d":
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == "up":
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == "down":
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == "left":
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == "right":
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(f"Unknown action: {action}")

    return motions


def pose_string_to_json(pose_string):
    """Convert pose string to pose JSON with extrinsic (c2w 4x4) and K (3x3)."""
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)

    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    pose_json = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}
    return pose_json


def poses_from_pose_str(pose_str, latent_num=N_LATENT):
    """
    Convert pose_str to (intrinsics, poses) matching CameraPluckerDataset format.

    Returns:
        intrinsics: (4,) float32 — [fx_norm, fy_norm, cx_norm, cy_norm]
        poses:      (latent_num, 7) float32 — [tx,ty,tz, qx,qy,qz,qw] w2c
    """
    pose_json = pose_string_to_json(pose_str)
    pose_keys = sorted(pose_json.keys(), key=lambda x: int(x))
    assert (
        len(pose_keys) >= latent_num
    ), f"pose_str produces {len(pose_keys)} frames, need {latent_num}"

    # Intrinsics: normalize like pose_to_input (generate.py lines 218-221)
    K = np.array(pose_json[pose_keys[0]]["K"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsics = np.array([fx / (cx * 2), fy / (cy * 2), 0.5, 0.5], dtype=np.float32)

    # Poses: c2w → w2c → [tx,ty,tz, qx,qy,qz,qw]
    poses = np.zeros((latent_num, 7), dtype=np.float32)
    for i in range(latent_num):
        c2w = np.array(pose_json[pose_keys[i]]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        poses[i, :3] = w2c[:3, 3]
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # (qx,qy,qz,qw)

    return intrinsics, poses


def pose_str_to_suffix(pose_str):
    """Convert pose_str like 'right-8, a-11' to compact suffix 'right8a11'."""
    return re.sub(r"[\s,]+", "", pose_str).replace("-", "")


# ── Video encoding ───────────────────────────────────────────────────────────

# Camera indices for 77-frame videos: latent frame i corresponds to video frame 4*i
# This gives [0, 4, 8, ..., 76] — 20 entries matching N_LATENT.
OPENVID_CAMERA_INDICES = list(range(0, MAX_FRAMES, 4))
assert len(OPENVID_CAMERA_INDICES) == N_LATENT


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Preencode WorldPlay generated videos")
    p.add_argument("--input_json", type=str, required=True, help="Path to preencode_input.json")
    p.add_argument(
        "--video_root", type=str, required=True, help="Root dir containing {idx}_{suffix}/gen.mp4"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for .pt files and index JSON",
    )
    p.add_argument(
        "--hunyuan_checkpoint_path",
        type=str,
        required=True,
        help="Path to HunyuanVideo-1.5 checkpoint",
    )
    p.add_argument("--skip_existing", action="store_true", help="Skip if output .pt already exists")
    p.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (-1 = all)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Distributed setup ──
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.distributed.init_process_group("nccl")

    # ── Load task list ──
    with open(args.input_json, "r") as f:
        all_tasks = json.load(f)
    total = len(all_tasks)
    if args.max_samples > 0:
        all_tasks = all_tasks[: args.max_samples]
        total = len(all_tasks)

    # ── Build (idx, task) with video_id, filter to existing videos ──
    tasks_with_video = []
    for idx, task in enumerate(all_tasks):
        suffix = pose_str_to_suffix(task["pose_str"])
        video_id = f"{idx:06d}_{suffix}"
        video_path = os.path.join(args.video_root, video_id, "gen.mp4")
        if os.path.isfile(video_path):
            tasks_with_video.append((idx, task, video_id, video_path))

    if rank == 0:
        print(f"Total tasks: {total}, with existing videos: {len(tasks_with_video)}")

    # ── Shard across GPUs ──
    my_tasks = [t for i, t in enumerate(tasks_with_video) if i % world_size == rank]
    if rank == 0:
        print(f"Rank {rank}/{world_size}: processing {len(my_tasks)} samples")

    # ── Output dirs ──
    latent_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latent_dir, exist_ok=True)

    # ── Initialize encoder ──
    extractor = VideoLatentExtractor(args.hunyuan_checkpoint_path, device=f"cuda:{local_rank}")

    # ── Process ──
    index_entries = []
    pbar = tqdm(my_tasks, desc=f"[rank {rank}]", disable=(rank != 0))

    for global_idx, task, video_id, video_path in pbar:
        out_path = os.path.join(latent_dir, f"{video_id}.pt")

        if args.skip_existing and os.path.isfile(out_path):
            index_entries.append({"latent_path": out_path})
            continue

        try:
            # Camera parameters from pose_str
            intrinsics, poses = poses_from_pose_str(task["pose_str"], N_LATENT)

            # Encode video + text + vision using extractor.extract()
            data = extractor.extract(
                video_path,
                task["caption"],
                camera_indices=OPENVID_CAMERA_INDICES,
                max_frames=MAX_FRAMES,
            )
            if data is None:
                print(f"[rank {rank}] SKIP {video_id}: extract returned None", flush=True)
                continue
            data.pop("frame_indices", None)  # not needed in output

            # Assemble .pt
            data["intrinsics"] = torch.from_numpy(intrinsics)  # (4,)
            data["poses"] = torch.from_numpy(poses)  # (20, 7)
            data["camera_indices"] = torch.arange(N_LATENT, dtype=torch.int64)
            data["video_frame_indices"] = torch.arange(MAX_FRAMES, dtype=torch.int64)
            data["num_video_frames"] = MAX_FRAMES
            data["video_id"] = video_id

            torch.save(data, out_path)
            index_entries.append({"latent_path": out_path})

        except Exception as e:
            print(f"[rank {rank}] ERROR {video_id}: {e}", flush=True)
            continue

    # ── Save per-rank index ──
    rank_json = os.path.join(args.output_dir, f"index_rank{rank}.json")
    with open(rank_json, "w") as f:
        json.dump(index_entries, f)
    print(f"[rank {rank}] Done: {len(index_entries)} entries -> {rank_json}")

    # ── Merge on rank 0 ──
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merged = []
        for r in range(world_size):
            rj = os.path.join(args.output_dir, f"index_rank{r}.json")
            if os.path.isfile(rj):
                merged.extend(json.load(open(rj)))
        merged_path = os.path.join(args.output_dir, "train_index.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged index: {len(merged)} entries -> {merged_path}")


if __name__ == "__main__":
    main()
