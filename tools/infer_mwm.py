"""Generic inference entry point for minWM.

The benchmark to run is a config value: ``inference.benchmark`` points at a JSON
file listing items, each carrying a required ``id`` (used to name the output
``{id}.mp4``) and a ``caption``, plus an optional ``trajectory``; an ``image``
makes the item image-to-video (HY), and omitting it makes it text-to-video (Wan).

::

    python tools/infer_mwm.py --config-file configs/wan21/action2v/infer/stage0_bi_sft.py

Every config value is overridable inline via dotlist ``opts``, e.g.::

    ... inference.benchmark=./prompts/t2v.json inference.output_dir=./out inference.sp_size=8
"""

import argparse

from minwm.config import apply_overrides, load
from minwm.engine import BaseInferencer

# Re-exported so callers/tests importing ``read_benchmark`` from the entry point
# keep working; the benchmark loading itself now lives in the engine.
from minwm.engine.inferencer import read_benchmark  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minWM inference.")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Dotlist overrides for any config value, e.g. "
        "inference.benchmark=./prompts/t2v.json inference.sp_size=8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load(args.config_file), [o for o in (args.opts or []) if o])
    # The engine owns the data source (benchmark), mirroring the trainer's
    # build_dataloader(cfg.data): the CLI only names the config.
    inferencer = BaseInferencer(cfg)
    inferencer.run()


if __name__ == "__main__":
    main()
