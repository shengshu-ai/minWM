#!/usr/bin/env python3
"""Generate a ``train_index.json`` from a directory of ``.pt`` latent files.

Backbone-agnostic: scans a directory for pre-encoded ``.pt`` latents and writes
a JSON index of absolute paths consumed by the training dataloaders. An existing
index at the output path is overwritten.

Usage:
    python tools/data/create_train_index.py /path/to/latents_dir
    python tools/data/create_train_index.py /path/to/latents_dir -o /path/to/train_index.json
"""

import argparse
import glob
import json
import os


def build_index(data_dir: str, recursive: bool = False) -> list[dict[str, str]]:
    """Scan ``data_dir`` for ``.pt`` files and build an index of absolute paths.

    Args:
        data_dir (str): directory containing ``.pt`` latent files.
        recursive (bool): if True, also search subdirectories.

    Returns:
        list[dict[str, str]]: one ``{"latent_path": <abs path>}`` entry per file,
            sorted by path.

    Raises:
        FileNotFoundError: if ``data_dir`` is not a directory or has no ``.pt`` files.
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pattern = os.path.join(data_dir, "**/*.pt") if recursive else os.path.join(data_dir, "*.pt")
    pt_files = sorted(glob.glob(pattern, recursive=recursive))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    return [{"latent_path": os.path.abspath(p)} for p in pt_files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train_index.json from .pt files")
    parser.add_argument("data_dir", help="Directory containing .pt latent files")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for train_index.json (default: <data_dir>/train_index.json)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively search subdirectories",
    )
    args = parser.parse_args()

    index = build_index(args.data_dir, recursive=args.recursive)
    output_path = args.output or os.path.join(os.path.abspath(args.data_dir), "train_index.json")
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Created {output_path} with {len(index)} samples")


if __name__ == "__main__":
    main()
