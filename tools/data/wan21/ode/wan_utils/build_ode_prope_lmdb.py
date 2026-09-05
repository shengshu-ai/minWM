"""
Convert .pt files (from get_causal_ode_data_prope.py) into a single LMDB
suitable for CameraODERegressionLMDBDataset.

Each .pt file contains:
    - "prompt":   str
    - "latents":  (1, S, F, C, H, W) float16/float32
    - "viewmats": (1, F, 4, 4) float32
    - "Ks":       (1, F, 3, 3) float32

The output LMDB stores:
    - latents_{i}_data:  bytes (float16, row shape S,F,C,H,W)
    - prompts_{i}_data:  utf-8 encoded string
    - viewmats_{i}_data: bytes (float32, row shape F,4,4)
    - Ks_{i}_data:       bytes (float32, row shape F,3,3)
    - latents_shape:     "N S F C H W"
    - prompts_shape:     "N"
    - viewmats_shape:    "N F 4 4"
    - Ks_shape:          "N F 3 3"

Usage:
    python wan_utils/build_ode_prope_lmdb.py \
        --input_dir /path/to/pt_files \
        --output_dir /path/to/output_lmdb \
        [--map_size_gb 50]
"""

import argparse
import glob
import os

import lmdb
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing .pt files"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output LMDB directory")
    parser.add_argument("--map_size_gb", type=float, default=50, help="LMDB map size in GB")
    args = parser.parse_args()

    pt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    print(f"Found {len(pt_files)} .pt files in {args.input_dir}")
    if len(pt_files) == 0:
        print("No .pt files found. Exiting.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    map_size = int(args.map_size_gb * (1024**3))
    env = lmdb.open(
        args.output_dir, map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False
    )

    # Determine shapes from first file
    first = torch.load(pt_files[0], map_location="cpu")
    latents_0 = first["latents"].numpy().astype(np.float16)
    if latents_0.ndim == 6:
        # (1, S, F, C, H, W) -> (S, F, C, H, W)
        latents_0 = latents_0[0]
    lat_row_shape = latents_0.shape  # (S, F, C, H, W)

    viewmats_0 = first["viewmats"].numpy().astype(np.float32)
    if viewmats_0.ndim == 4:
        viewmats_0 = viewmats_0[0]  # (F, 4, 4)
    vm_row_shape = viewmats_0.shape

    Ks_0 = first["Ks"].numpy().astype(np.float32)
    if Ks_0.ndim == 4:
        Ks_0 = Ks_0[0]  # (F, 3, 3)
    ks_row_shape = Ks_0.shape

    print(f"Latents row shape: {lat_row_shape}")
    print(f"Viewmats row shape: {vm_row_shape}")
    print(f"Ks row shape: {ks_row_shape}")

    N = len(pt_files)
    BATCH = 256

    for batch_start in tqdm(range(0, N, BATCH), desc="Writing LMDB"):
        batch_end = min(batch_start + BATCH, N)
        while True:
            try:
                with env.begin(write=True) as txn:
                    for i in range(batch_start, batch_end):
                        data = torch.load(pt_files[i], map_location="cpu")

                        # Latents
                        lat = data["latents"].numpy().astype(np.float16)
                        if lat.ndim == 6:
                            lat = lat[0]
                        assert (
                            lat.shape == lat_row_shape
                        ), f"Shape mismatch at {pt_files[i]}: {lat.shape} vs {lat_row_shape}"
                        txn.put(f"latents_{i}_data".encode(), np.ascontiguousarray(lat).tobytes())

                        # Prompts
                        prompt = data["prompt"]
                        if isinstance(prompt, list):
                            prompt = prompt[0]
                        txn.put(f"prompts_{i}_data".encode(), prompt.encode("utf-8"))

                        # Viewmats
                        vm = data["viewmats"].numpy().astype(np.float32)
                        if vm.ndim == 4:
                            vm = vm[0]
                        txn.put(f"viewmats_{i}_data".encode(), np.ascontiguousarray(vm).tobytes())

                        # Ks
                        ks = data["Ks"].numpy().astype(np.float32)
                        if ks.ndim == 4:
                            ks = ks[0]
                        txn.put(f"Ks_{i}_data".encode(), np.ascontiguousarray(ks).tobytes())
                break
            except lmdb.MapFullError:
                cur = env.info()["map_size"]
                env.set_mapsize(int(cur * 1.5) + (1 << 30))

    # Write shape metadata
    with env.begin(write=True) as txn:
        txn.put(b"latents_shape", " ".join(map(str, (N, *lat_row_shape))).encode())
        txn.put(b"prompts_shape", str(N).encode())
        txn.put(b"viewmats_shape", " ".join(map(str, (N, *vm_row_shape))).encode())
        txn.put(b"Ks_shape", " ".join(map(str, (N, *ks_row_shape))).encode())

    env.sync()
    env.close()
    print(f"Done. Wrote {N} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
