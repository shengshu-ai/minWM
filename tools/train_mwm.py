"""Training entry point for minwm.

Usage (single GPU):
    python tools/train_mwm.py --config-file configs/wan21/action2v/train/stage0_bi_sft.py \\
        --output-dir outputs/my-exp training.max_steps=5000

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=8 tools/train_mwm.py \\
        --config-file configs/wan21/action2v/train/stage0_bi_sft.py \\
        training.sp_size=8
"""

import argparse
import os

from minwm.config import MWMConfig, apply_overrides, load, locate
from minwm.engine import BaseTrainer
from minwm.engine.paths import log_dir
from minwm.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minwm model.")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Dotlist overrides: training.max_steps=10000 model.dim=1024 …",
    )
    return parser.parse_args()


def build_trainer(cfg: dict) -> BaseTrainer:
    """Instantiate the trainer class named by ``trainer.type`` (default: BaseTrainer).

    The ``trainer`` selector is a non-schema key read off the raw config dict
    (before it is parsed into :class:`MWMConfig`); the resolved class is then
    handed the typed config via ``trainer_cls(MWMConfig.from_dict(cfg))``.

    Args:
        cfg (dict): the loaded + override-applied raw config dict.

    Returns:
        BaseTrainer: the constructed trainer.
    """
    trainer_cfg = cfg.get("trainer", {})
    trainer_cls = locate(trainer_cfg["type"]) if "type" in trainer_cfg else BaseTrainer
    return trainer_cls(MWMConfig.from_dict(cfg))


def main() -> None:
    args = parse_args()
    cfg = load(args.config_file)

    overrides = [o for o in (args.opts or []) if o]
    cfg = apply_overrides(cfg, overrides)

    if args.output_dir is not None:
        cfg.setdefault("training", {})["output_dir"] = args.output_dir
    if args.resume:
        cfg.setdefault("checkpoint", {})["resume"] = True

    rank = int(os.environ.get("RANK", 0))
    output_dir = cfg.get("training", {}).get("output_dir") or None
    setup_logger(output=log_dir(output_dir), distributed_rank=rank)

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
