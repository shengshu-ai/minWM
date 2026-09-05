#!/usr/bin/env python3
"""Preencode HunyuanVideo latents from raw videos (sequential frame sampling).

The encode core lives in :class:`minwm.data.preprocessing.hy.VideoLatentExtractor`;
this script owns only argument parsing and the distributed sharding/merge loop.

``image_cond`` is taken from the first frame of the VAE-encoded latents (not a
separate image encode), matching the training pipeline.

Input JSON: ``[{"video_path": ..., "caption": ...}, ...]``.
Output: ``<output_dir>/latents/<idx>.pt`` plus a merged ``train_index.json``.

Usage:
    python tools/data/hy/preencode_video.py \
        --input_json /path/to/videos.json \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/HunyuanVideo-1.5
"""

import argparse
import json
import os

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
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    device = torch.cuda.current_device()

    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)

    with open(args.input_json, "r") as f:
        data_list = json.load(f)

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
        output_path = os.path.join(latents_dir, f"{global_idx:06d}.pt")
        output_path_abs = os.path.abspath(output_path)

        if args.skip_existing and os.path.exists(output_path):
            output_items.append({"latent_path": output_path_abs})
            continue

        try:
            latents = extractor.extract(video_path, caption, max_frames=args.max_frames)
            torch.save(latents, output_path)
            output_items.append({"latent_path": output_path_abs})

            if idx == 0 and local_rank == 0:
                print("\nFirst item shapes:")
                for key, value in latents.items():
                    if hasattr(value, "shape"):
                        print(f"  {key}: {value.shape}")
        except Exception as e:  # noqa: BLE001 - log and skip bad videos
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
                    with open(rank_json, "r") as f:
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
