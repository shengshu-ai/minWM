"""Profile CUDA memory lifetime for one minWM training step.

Run under the same ``torchrun`` command as training. The tool writes a Chrome
trace, a profiler memory/operator summary, and (when supported by the installed
PyTorch build) a CUDA allocator snapshot and memory timeline.

Example::

    CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=. python -m torch.distributed.run \\
        --nproc_per_node=2 tools/profile_activation_memory.py \\
        --config-file /tmp/wan_bench.py \\
        --output-dir /tmp/wan-memory-profile \\
        training.activation_offload=true training.max_steps=1
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from minwm.config import MWMConfig, apply_overrides, load
from minwm.engine import BaseTrainer
from minwm.utils import comm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one minWM CUDA training step.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _build_trainer(config_file: str, overrides: list[str]) -> BaseTrainer:
    """Build a trainer from a config and dotlist overrides.

    Args:
        config_file (str): Python or YAML training config.
        overrides (list[str]): OmegaConf-style config overrides.

    Returns:
        BaseTrainer: initialized trainer whose model and dataloader are ready.
    """
    cfg = apply_overrides(load(config_file), [item for item in overrides if item])
    cfg.setdefault("training", {})["max_steps"] = 1
    cfg["training"]["ckpt_interval"] = 0
    cfg["training"]["output_dir"] = ""
    return BaseTrainer(MWMConfig.from_dict(cfg))


def _start_memory_history() -> bool:
    """Start the CUDA allocator history when the PyTorch build supports it."""
    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
    if recorder is None:
        return False
    try:
        recorder(enabled="all", stacks="all")
    except (RuntimeError, TypeError):
        try:
            recorder(enabled=True)
        except RuntimeError:
            return False
    return True


def _dump_memory_artifacts(output_dir: Path, profiler: torch.profiler.profile) -> None:
    """Write profiler and allocator artifacts for the current rank.

    Args:
        output_dir (Path): destination directory.
        profiler (torch.profiler.profile): completed CUDA profiler.
    """
    rank = comm.get_rank()
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output_dir / f"trace.rank{rank}.json"))
    summary = profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=80)
    (output_dir / f"memory_ops.rank{rank}.txt").write_text(summary)
    try:
        profiler.export_memory_timeline(str(output_dir / f"memory_timeline.rank{rank}.html"))
    except (AttributeError, RuntimeError, ValueError):
        pass
    dumper = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dumper is not None:
        try:
            dumper(str(output_dir / f"allocator_snapshot.rank{rank}.pickle"))
        except (RuntimeError, OSError):
            pass


def main() -> None:
    """Profile one forward/backward/optimizer step and report memory peaks."""
    args = _parse_args()
    trainer = _build_trainer(args.config_file, args.opts or [])
    if not torch.cuda.is_available():
        raise RuntimeError("profile_activation_memory.py requires CUDA")

    torch.cuda.reset_peak_memory_stats()
    history_started = _start_memory_history()
    profiler_kwargs: dict[str, Any] = {
        "activities": [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": True,
    }
    with torch.profiler.profile(**profiler_kwargs) as profiler:
        batch = trainer._next_batch()
        trainer.recipe.train_one_step(
            trainer.model,
            batch,
            trainer.optimizers,
            0,
            auxiliary_models=trainer.auxiliary_models,
        )
        torch.cuda.synchronize()

    if comm.is_main_process:
        output_dir = Path(args.output_dir)
        _dump_memory_artifacts(output_dir, profiler)
        controller = getattr(trainer.model, "_activation_offload_controller", None)
        stats = controller.stats if controller is not None else {}
        (output_dir / "activation_offload_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n"
        )
        print("activation_offload_stats", json.dumps(stats, sort_keys=True), flush=True)
        print(
            f"memory_profile history={history_started} "
            f"allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f} "
            f"reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}",
            flush=True,
        )
    trainer.close()


if __name__ == "__main__":
    main()
