#!/usr/bin/env python3
"""Preencode HunyuanVideo latents with raw camera data (camera-aligned sampling).

Like ``preencode_video.py`` but samples frames aligned to the VAE's 4x temporal
downsampling (via the extractor's ``camera_indices`` path) and stores the raw
intrinsics/poses alongside the latents for plucker-embedding construction in the
dataloader. The encode core lives in
:class:`minwm.data.preprocessing.hy.VideoLatentExtractor`.

Input JSON (from prepare_camera_json.py):
    [{"video_path": ..., "caption_path": ..., "intrinsics_path": ...,
      "poses_path": ..., "camera_indices": [...], "num_frames": ..., ...}]

Output .pt adds: intrinsics (4,), poses (N_cam, 7), camera_indices (N_cam,),
video_frame_indices (max_frames,), num_video_frames (int), video_id (str).

Usage:
    torchrun --nproc_per_node=8 tools/data/hy/preencode_camera_video.py \
        --input_json /path/to/train_camera.json \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/HunyuanVideo-1.5 \
        --max_frames 77
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from minwm.data.preprocessing.hy import VideoLatentExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hunyuan_checkpoint_path", type=str, required=True)
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_width", type=int, default=832)
    parser.add_argument("--max_frames", type=int, default=77)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2),
        )
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    device = torch.cuda.current_device()

    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)

    with open(args.input_json) as f:
        raw_list = json.load(f)

    data_list = []
    for m in raw_list:
        caption = ""
        if m.get("caption_path"):
            try:
                with open(m["caption_path"]) as f:
                    caption = json.load(f).get("SceneSummary", "")
            except Exception:  # noqa: BLE001 - missing/garbled caption is non-fatal
                pass
        data_list.append({**m, "caption": caption})

    if global_rank == 0:
        print(f"Loaded {len(data_list)} items from {args.input_json}")

    total_items = len(data_list)
    if world_size > 1:
        per_gpu = (total_items + world_size - 1) // world_size
        start_idx = global_rank * per_gpu
        end_idx = min(start_idx + per_gpu, total_items)
        data_list = data_list[start_idx:end_idx]
    else:
        start_idx = 0

    print(f"GPU {local_rank}: Processing {len(data_list)} items")

    extractor = VideoLatentExtractor(
        args.hunyuan_checkpoint_path,
        device,
        (args.target_height, args.target_width),
    )

    output_items = []
    for idx, item in enumerate(tqdm(data_list, desc=f"GPU {local_rank}")):
        video_path = item.get("video_path")
        caption = item.get("caption", "")

        if not video_path or not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        global_idx = start_idx + idx
        video_id = item.get("video_id", f"{global_idx:06d}")
        output_filename = f"{video_id}.pt"
        output_path = os.path.join(latents_dir, output_filename)
        output_path_abs = os.path.abspath(output_path)

        if args.skip_existing and os.path.exists(output_path):
            output_items.append({"latent_path": output_path_abs})
            continue

        try:
            latents = extractor.extract(
                video_path,
                caption,
                camera_indices=item["camera_indices"],
                max_frames=args.max_frames,
            )

            if latents is None:
                print(f"Skipped (not enough camera frames): {video_path}")
                continue

            intrinsics_raw = np.load(item["intrinsics_path"]).astype(np.float32)
            poses_raw = np.load(item["poses_path"]).astype(np.float32)
            latents["intrinsics"] = torch.from_numpy(intrinsics_raw[0])  # (4,)
            latents["poses"] = torch.from_numpy(poses_raw)  # (N_cam, 7)
            latents["camera_indices"] = torch.tensor(item["camera_indices"])  # (N_cam,)
            latents["video_frame_indices"] = torch.tensor(
                latents.pop("frame_indices"), dtype=torch.long
            )  # (max_frames,)
            latents["num_video_frames"] = item["num_frames"]  # int
            latents["video_id"] = item.get("video_id", output_filename)

            torch.save(latents, output_path)
            output_items.append({"latent_path": output_path_abs})

            if idx == 0 and local_rank == 0:
                print("\nFirst item shapes:")
                for key, value in latents.items():
                    if hasattr(value, "shape"):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {value}")
        except Exception as e:  # noqa: BLE001 - log and skip bad samples
            print(f"Error processing {video_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    output_json = os.path.join(args.output_dir, f"train_index_rank{global_rank}.json")
    with open(output_json, "w") as f:
        json.dump(output_items, f, indent=2)
    print(f"GPU {global_rank}: Saved {len(output_items)} items to {output_json}")

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
        if global_rank == 0:
            all_items = []
            for rank in range(world_size):
                rank_json = os.path.join(args.output_dir, f"train_index_rank{rank}.json")
                if os.path.exists(rank_json):
                    with open(rank_json) as f:
                        all_items.extend(json.load(f))
                    os.remove(rank_json)
            all_items.sort(key=lambda x: x["latent_path"])
            final_json = os.path.join(args.output_dir, "train_index.json")
            with open(final_json, "w") as f:
                json.dump(all_items, f, indent=2)
            print(f"Merged {len(all_items)} items to {final_json}")
        dist.barrier()
        dist.destroy_process_group()

    print(f"GPU {local_rank}: Done!")


if __name__ == "__main__":
    main()
