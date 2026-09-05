#!/usr/bin/env python3
"""Browse and inspect a dataset defined in a minwm config file.

Usage::

    python tools/browse_dataset.py --config configs/hy/action2v/train/stage3_ar_dmd_mock.py
    python tools/browse_dataset.py --config configs/hy/action2v/train/stage3_ar_dmd_mock.py --key data.dataset
    python tools/browse_dataset.py --config configs/hy/action2v/train/stage3_ar_dmd_mock.py --n 5 --no-data

The script resolves the ``dataset`` config node (via ``type``), prints dataset
metadata, and iterates the first ``--n`` samples showing key/shape/dtype.

Arguments
---------
--config     Path to a .py or .yaml config file.
--key        Dot-path to the dataset config node (default: ``data.dataset``).
             Set to ``data`` to invoke ``build_dataloader`` instead.
--n          Number of samples to inspect (default: 3; 0 = metadata only).
--no-data    Print metadata but skip sample iteration.
"""

import argparse
import functools
import operator
from typing import Any

import torch


def _get_nested(d: dict, dotpath: str) -> Any:
    return functools.reduce(operator.getitem, dotpath.split("."), d)


def _fmt_value(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        return f"Tensor{list(v.shape)} {v.dtype}"
    if isinstance(v, str) and len(v) > 80:
        return repr(v[:77] + "...")
    return repr(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to .py or .yaml config file")
    parser.add_argument("--key", default="data.dataset", help="Dot-path to the dataset config node")
    parser.add_argument(
        "--n", type=int, default=3, help="Number of samples to inspect (0 = metadata only)"
    )
    parser.add_argument("--no-data", action="store_true", help="Skip sample iteration")
    args = parser.parse_args()

    from minwm.config import Data, load
    from minwm.data import build_dataloader, build_dataset

    cfg = load(args.config)

    try:
        node = _get_nested(cfg, args.key)
    except KeyError as e:
        parser.error(f"Key {args.key!r} not found in config: {e}")

    # Decide whether to build a dataset or a full dataloader
    if "dataset" in node:
        print(f"[browse_dataset] Building dataloader from key={args.key!r}")
        loader = build_dataloader(Data(**node))
        dataset = loader.dataset
        print(f"  DataLoader  batch_size={loader.batch_size}, num_workers={loader.num_workers}")
    else:
        print(f"[browse_dataset] Building dataset from key={args.key!r}")
        dataset = build_dataset(node)
        loader = None

    print(f"  Dataset     {type(dataset).__name__}")
    print(f"  Length      {len(dataset)}")

    n = 0 if args.no_data else args.n
    if n <= 0:
        return

    print(f"\n--- First {n} sample(s) ---")
    if loader is not None:
        it = iter(loader)
        for i in range(n):
            try:
                batch = next(it)
            except StopIteration:
                break
            print(f"\nBatch {i}:")
            for k, v in batch.items():
                print(f"  {k}: {_fmt_value(v)}")
    else:
        for i in range(min(n, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            if isinstance(sample, dict):
                for k, v in sample.items():
                    print(f"  {k}: {_fmt_value(v)}")
            else:
                print(f"  {_fmt_value(sample)}")


if __name__ == "__main__":
    main()
