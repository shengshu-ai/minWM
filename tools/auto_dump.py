"""Watch a training run and dump DCP checkpoints to single-file weights.

The first half of the train -> dump -> sample pipeline. FSDP2 training writes
checkpoints as sharded :mod:`torch.distributed.checkpoint` (DCP) directories
(``<ckpts>/checkpoint_{step}/``) that the inference engine cannot load directly.
This tool polls the run's ``ckpts/`` tree for the requested steps and, as each
lands, consolidates it into a single ``model_{step}.{pt,safetensors}`` (or a
diffusers dir) via the same path as ``tools/export_checkpoint.py``. The dumped
files are what ``tools/auto_sample.py`` then waits on.

DCP writes its ``.metadata`` file last, after every shard is durable, so its
presence is the "save finished" signal — the tool waits for it (local or remote)
before dumping, and never consolidates a half-written checkpoint. Both the
``ckpts/`` tree and the dump destination may be local or an object store
(``s3://`` / ``oss://``); the URL scheme decides, via ``minwm``'s ``Storage``.

Consolidation is a CPU-only, single-process, memory-heavy step (it rebuilds the
whole model state dict in RAM). By default each step's dump is submitted as its
own cluster job (``<launch-tool> submit -- scripts/dump.sh ...``) so the login node
only polls; ``--local`` runs the consolidation in-process instead. The submit tool
defaults to the ``<launch-tool>`` placeholder — set ``MINWM_LAUNCH_TOOL`` to your
scheduler's binary (and adapt the submit CLI in ``tools/_cluster.py`` to match)
or pass ``--local``.

Usage (submit each dump to the cluster — the default)::

    python tools/auto_dump.py \
        --output-dir outputs/wan_action2v_bidirectional \
        --ckpt-steps 10000 15000 20000 \
        --format safetensors \
        -c <cluster> -q <queue>

Local, in-process consolidation (no cluster; needs enough RAM)::

    python tools/auto_dump.py \
        --output-dir outputs/wan_action2v_bidirectional \
        --ckpt-steps 10000 \
        --local

DMD generator EMA weights (the ``aux/generator_ema`` DCP entry)::

    python tools/auto_dump.py \
        --output-dir outputs/wan_action2v_dmd \
        --ckpt-steps 200 400 \
        --model-key aux/generator_ema \
        --format pt -c <cluster> -q <queue>

Checkpoints on an object store (``checkpoint.remote_root`` used in training)::

    python tools/auto_dump.py \
        --output-dir outputs-sft/wan_action2v_bidirectional \
        --remote-root s3://<bucket>/<prefix> \
        --ckpt-steps 10000 \
        --export-dir ./ckpts/Wan21/Action2V/sft --local

Args:
    --output-dir: training ``training.output_dir`` (run name); ``ckpts/`` lives under it.
    --remote-root: object-store prefix if training used ``checkpoint.remote_root``;
        the ``ckpts/`` tree is then read from ``<remote-root>/<output-dir>/ckpts``.
    --ckpt-steps: checkpoint steps to dump.
    --format: output format — ``safetensors`` (default), ``pt``, or ``diffusers``.
    --model-key: DCP model entry to dump (default ``model``; ``aux/generator_ema`` for DMD EMA).
    --config: config file defining the model architecture (required for ``diffusers``).
    --export-dir: where dumped weights are written (default ``<ckpts>/exported``).
    --local: consolidate in-process instead of submitting a cluster job.
    --cluster/-c, --queue/-q, --nodes: cluster target for the launch tool submit (non-local).
    --wait-interval: seconds between polls (default 60).
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.checkpoint.storage import Storage, is_remote, join  # noqa: E402
from minwm.engine.paths import ckpt_dir as resolve_ckpt_dir  # noqa: E402
from tools._cluster import build_safe_job_name, build_submit_argv  # noqa: E402

# import the single-shot export helpers so the two tools stay one code path
from tools.export_checkpoint import (  # noqa: E402
    _load_dcp,
    export_diffusers,
    export_pt,
    export_safetensors,
)

DCP_SENTINEL = ".metadata"
DUMP_SCRIPT = "scripts/dump.sh"
_EXT = {"safetensors": "safetensors", "pt": "pt", "diffusers": ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-dump minWM DCP checkpoints as they land.")
    parser.add_argument("--output-dir", required=True, help="training output_dir (run name)")
    parser.add_argument(
        "--remote-root",
        default=None,
        help="object-store prefix if training used checkpoint.remote_root (s3:// / oss://)",
    )
    parser.add_argument(
        "--ckpt-steps",
        nargs="+",
        type=int,
        required=True,
        help="checkpoint steps to dump",
    )
    parser.add_argument(
        "--format",
        choices=["safetensors", "pt", "diffusers"],
        default="safetensors",
        help="output format (default: safetensors)",
    )
    parser.add_argument(
        "--model-key",
        default="model",
        help="DCP model entry to dump ('aux/generator_ema' for DMD EMA); default 'model'",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config file defining the model architecture (required for --format diffusers)",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="destination for dumped weights (default: <ckpts>/exported)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="consolidate in-process instead of submitting a cluster job",
    )
    parser.add_argument("--cluster", "-c", default=None, help="cluster for the launch tool submit")
    parser.add_argument("--queue", "-q", default=None, help="queue for the launch tool submit")
    parser.add_argument(
        "--nodes", "-n", type=int, default=1, help="nodes for the launch tool submit"
    )
    parser.add_argument("--wait-interval", type=int, default=60, help="seconds between polls")
    return parser.parse_args()


def dcp_dir_for(ckpts_root: str, step: int) -> str:
    """Path of the DCP directory a run writes for ``step``."""
    return join(ckpts_root, f"checkpoint_{step}")


def output_path_for(export_dir: str, step: int, fmt: str) -> str:
    """Path of the dumped weight for ``step`` in the requested format.

    Args:
        export_dir (str): destination directory for dumped weights.
        step (int): checkpoint step.
        fmt (str): ``safetensors``, ``pt``, or ``diffusers``.

    Returns:
        str: ``<export_dir>/model_{step}.{ext}`` for single-file formats, or the
        directory ``<export_dir>/model_{step}`` for ``diffusers``.
    """
    ext = _EXT[fmt]
    name = f"model_{step}" if not ext else f"model_{step}.{ext}"
    return join(export_dir, name)


def is_exported(storage: Storage, out_path: str, fmt: str) -> bool:
    """True if a prior run already produced the dump for this step."""
    if fmt == "diffusers":
        return storage.exists(join(out_path, "config.json"))
    return storage.exists(out_path)


def wait_for_dcp(storage: Storage, dcp_dir: str, step: int, wait_interval: int) -> None:
    """Block until the DCP ``.metadata`` sentinel exists for ``step``.

    ``.metadata`` is written last by DCP, after every shard is durable, so its
    presence means the checkpoint is complete and safe to consolidate.

    Args:
        storage (Storage): backend used to probe the sentinel (local or remote).
        dcp_dir (str): the ``checkpoint_{step}`` directory to wait on.
        step (int): checkpoint step (for logging).
        wait_interval (int): seconds to sleep between probes.
    """
    sentinel = join(dcp_dir, DCP_SENTINEL)
    while not storage.exists(sentinel):
        print(f"[auto_dump] step {step}: waiting for {sentinel} …", flush=True)
        time.sleep(wait_interval)


def build_export_args(dcp_dir: str, out_path: str, args: argparse.Namespace) -> list[str]:
    """Assemble the tools/export_checkpoint.py CLI args (shared local + cluster)."""
    export_args = [
        "--checkpoint",
        dcp_dir,
        "--output",
        out_path,
        "--format",
        args.format,
        "--model-key",
        args.model_key,
    ]
    if args.config:
        export_args += ["--config", args.config]
    return export_args


def dump_local(step: int, dcp_dir: str, out_path: str, args: argparse.Namespace, storage: Storage):
    """Consolidate one DCP checkpoint in-process and write it in the requested format."""
    weights = _load_dcp(dcp_dir, model_key=args.model_key, storage=storage)
    print(f"[auto_dump] step {step}: loaded {len(weights)} tensors", flush=True)

    if args.format == "diffusers":
        export_diffusers(weights, out_path, args.config)
    elif args.format == "pt":
        export_pt(weights, out_path)
    else:
        export_safetensors(weights, out_path)


def dump_cluster(step: int, dcp_dir: str, out_path: str, args: argparse.Namespace, exp_name: str):
    """Submit one step's consolidation as a single-node cluster job.

    The dump is CPU-only and single-process, so it is submitted as a one-node
    job (``-n``); it needs a node with RAM >= model size but no GPU work. The
    submit tool is ``$MINWM_LAUNCH_TOOL`` (see :mod:`tools._cluster`); with none
    set the call bails out asking you to configure it or pass ``--local``.
    """
    job_name = build_safe_job_name(exp_name, step, "dump")
    submit = build_submit_argv(job_name, args.nodes, args.cluster, args.queue)
    submit += ["--", DUMP_SCRIPT, *build_export_args(dcp_dir, out_path, args)]
    print(f"[auto_dump] submit: {' '.join(shlex.quote(c) for c in submit)}", flush=True)
    subprocess.run(submit, check=True)
    print(f"[auto_dump] step {step}: submitted job {job_name}", flush=True)


def main() -> None:
    args = parse_args()

    if args.format == "diffusers" and args.config is None:
        print("--format diffusers requires --config", file=sys.stderr)
        sys.exit(1)

    storage = Storage()
    ckpts_root = resolve_ckpt_dir(args.output_dir, args.remote_root)
    if not ckpts_root:
        print("could not resolve ckpts dir; is --output-dir empty?", file=sys.stderr)
        sys.exit(1)
    export_dir = args.export_dir or join(ckpts_root, "exported")
    exp_name = os.path.basename(str(args.output_dir).rstrip("/"))

    # local dump destinations must exist before save_file / torch.save writes
    if args.local and not is_remote(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    print(f"[auto_dump] ckpts:    {ckpts_root}")
    print(f"[auto_dump] exports:  {export_dir}")
    print(
        f"[auto_dump] steps:    {args.ckpt_steps}  "
        f"(format={args.format}, key={args.model_key}, local={args.local})"
    )

    for step in args.ckpt_steps:
        out_path = output_path_for(export_dir, step, args.format)
        if is_exported(storage, out_path, args.format):
            print(f"[auto_dump] step {step}: already dumped at {out_path}, skipping", flush=True)
            continue

        dcp_dir = dcp_dir_for(ckpts_root, step)
        wait_for_dcp(storage, dcp_dir, step, args.wait_interval)
        print(f"[auto_dump] step {step}: DCP ready at {dcp_dir}", flush=True)
        if args.local:
            dump_local(step, dcp_dir, out_path, args, storage)
            print(f"[auto_dump] step {step}: done -> {out_path}", flush=True)
        else:
            dump_cluster(step, dcp_dir, out_path, args, exp_name)


if __name__ == "__main__":
    main()
