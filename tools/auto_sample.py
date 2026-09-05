"""Watch for dumped weights and run inference on each as it lands.

The second half of the train -> dump -> sample pipeline, paired with
``tools/auto_dump.py``. For each requested step it blocks until the dumped
weight (``model_{step}.{pt,safetensors}``, produced by ``auto_dump``) is fully
written, then launches inference on it — either locally via
``scripts/infer.sh`` (torchrun) or as a cluster job via ``<launch-tool> submit``
(your cluster's submit command; see :mod:`tools._cluster`).
Samples for each step land in ``<output-dir>/<sample-name>/{step}-{exp_name}/``.

A ``.safetensors`` file is only run once its header math proves every tensor's
bytes are present (a partially flushed file has a valid size prefix but too few
bytes); a ``.pt`` file is run once its size has been stable across a few probes.
This keeps inference from starting on a half-flushed weight.

``infer_mwm.py`` takes only ``--config-file`` plus dotlist overrides, so the
per-step values this launcher computes are forwarded as ``inference.checkpoint=``
/ ``inference.output_dir=`` assignments. The benchmark to run normally comes from
the config (``inference.benchmark``); a camera trajectory is not a launcher flag
at all, it rides on each benchmark item.

Usage (cluster submission — the default; needs a launch tool, see below)::

    python tools/auto_sample.py \
        --output-dir outputs/wan_action2v_dmd \
        --ckpt-steps 200 400 \
        --config-file configs/wan21/action2v/infer/stage3_ar_dmd.py \
        -c <cluster> -q <queue> --nodes 1

Local run (torchrun via ``scripts/infer.sh``, overriding the benchmark)::

    python tools/auto_sample.py \
        --output-dir outputs/wan_action2v_bidirectional \
        --ckpt-steps 10000 15000 \
        --config-file configs/wan21/action2v/infer/stage0_bi_sft.py \
        --benchmark ./prompts/t2v.json \
        --local

Args:
    --output-dir: run dir; weights read from ``<output-dir>/<export-name>``,
        samples written to ``<output-dir>/<sample-name>/{step}-{exp_name}``.
    --ckpt-steps: checkpoint steps to sample.
    --config-file: inference config passed to ``tools/infer_mwm.py``.
    --format: dumped weight format to wait on — ``safetensors`` (default) or ``pt``.
    --benchmark: benchmark JSON override; omit to use the config's ``inference.benchmark``.
    --seed / --ema: forwarded as ``inference.seed`` / ``inference.prefer_ema``.
    --export-name: subdir under ``output-dir`` with dumped weights (default ``ckpts/exported``).
    --sample-name: subdir under ``output-dir`` for samples (default ``samples``).
    --local: run locally via ``scripts/infer.sh`` instead of submitting a cluster job.
    --cluster/-c, --queue/-q, --nodes: cluster target for the launch tool (non-local).
    --wait-interval: seconds between polls (default 60).

The cluster submit command defaults to the ``<launch-tool>`` placeholder; set
``MINWM_LAUNCH_TOOL`` to your scheduler's binary (and adapt the submit CLI in
``tools/_cluster.py`` to match) or pass ``--local``.
"""

import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.checkpoint.storage import is_remote  # noqa: E402
from tools._cluster import build_safe_job_name, build_submit_argv  # noqa: E402

STABLE_CHECK_INTERVAL = 30
STABLE_CHECK_ROUNDS = 3
INFER_SCRIPT = "scripts/infer.sh"
OOM_MAX_RETRIES = 3
OOM_BACKOFF_SECONDS = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-run minWM inference on exported weights.")
    parser.add_argument(
        "--output-dir", required=True, help="run dir (weights + samples live under it)"
    )
    parser.add_argument("--ckpt-steps", nargs="+", type=int, required=True, help="steps to sample")
    parser.add_argument(
        "--config-file", required=True, help="inference config for tools/infer_mwm.py"
    )
    parser.add_argument(
        "--format",
        choices=["safetensors", "pt"],
        default="safetensors",
        help="exported weight format to wait on (default: safetensors)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="benchmark JSON override (default: the config's inference.benchmark)",
    )
    parser.add_argument("--seed", type=int, default=None, help="inference seed override")
    parser.add_argument("--ema", action="store_true", help="prefer generator_ema weights")
    parser.add_argument(
        "--export-name",
        default="ckpts/exported",
        help="subdir under output-dir holding exported weights (default: ckpts/exported)",
    )
    parser.add_argument(
        "--sample-name",
        default="samples",
        help="subdir under output-dir for samples (default: samples)",
    )
    parser.add_argument("--local", action="store_true", help="run locally via scripts/infer.sh")
    parser.add_argument("--cluster", "-c", default=None, help="cluster for the launch tool submit")
    parser.add_argument(
        "--queue", "-q", default=None, help="cluster queue for the launch tool submit"
    )
    parser.add_argument(
        "--nodes", "-n", type=int, default=1, help="nodes for the launch tool submit (non-local)"
    )
    parser.add_argument("--wait-interval", type=int, default=60, help="seconds between polls")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="extra dotlist overrides forwarded to infer_mwm.py",
    )
    return parser.parse_args()


def is_safetensors_fully_written(path: str) -> bool:
    """Whether a ``.safetensors`` file is fully flushed to disk.

    safetensors layout: 8-byte little-endian header length | JSON header | tensor
    data. The header records each tensor's byte offsets; the largest end offset
    plus the 8-byte prefix plus the header length must fit within the file size.

    Args:
        path (str): local path to the ``.safetensors`` file.

    Returns:
        bool: True if the file is present and every tensor's bytes are on disk.
    """
    try:
        file_size = os.path.getsize(path)
        if file_size < 8:
            return False
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if file_size < 8 + header_len:
                return False
            header = json.loads(f.read(header_len))
        max_end = 0
        for k, v in header.items():
            if k == "__metadata__":
                continue
            offsets = v.get("data_offsets")
            if offsets and len(offsets) == 2:
                max_end = max(max_end, offsets[1])
        return file_size >= 8 + header_len + max_end
    except Exception:
        return False


def wait_for_stable_file(path: str, fmt: str, wait_interval: int) -> None:
    """Block until ``path`` exists and is fully written for its format.

    safetensors is validated by header math; pt (opaque pickle) is validated by
    a stable, non-zero size across :data:`STABLE_CHECK_ROUNDS` probes.

    Args:
        path (str): local path to the exported weight.
        fmt (str): ``safetensors`` or ``pt``.
        wait_interval (int): seconds to sleep while the file is still missing.
    """
    last_size = -1
    stable_rounds = 0
    while True:
        if not os.path.exists(path):
            print(f"[auto_sample] waiting for {path} …", flush=True)
            last_size, stable_rounds = -1, 0
            time.sleep(wait_interval)
            continue

        if fmt == "safetensors":
            if is_safetensors_fully_written(path):
                return
            print(f"[auto_sample] {path} present but header incomplete, waiting …", flush=True)
            time.sleep(STABLE_CHECK_INTERVAL)
            continue

        cur_size = os.path.getsize(path)
        if cur_size == last_size and cur_size > 0:
            stable_rounds += 1
        else:
            stable_rounds, last_size = 0, cur_size
        if stable_rounds >= STABLE_CHECK_ROUNDS:
            return
        time.sleep(STABLE_CHECK_INTERVAL)


def build_infer_args(args: argparse.Namespace, model_path: str, output_dir: str) -> list[str]:
    """Assemble the tools/infer_mwm.py args shared by local and cluster launch.

    The entry point takes ``--config-file`` plus dotlist overrides only, so the
    per-step weight and sample dir this launcher computes go out as
    ``inference.*`` assignments.

    Args:
        args (argparse.Namespace): parsed launcher options.
        model_path (str): the dumped weight to run this step on.
        output_dir (str): where this step's samples are written.

    Returns:
        list[str]: the argument list for ``tools/infer_mwm.py``.
    """
    infer_args = [
        "--config-file",
        args.config_file,
        f"inference.checkpoint={model_path}",
        f"inference.output_dir={output_dir}",
    ]
    if args.benchmark:
        infer_args.append(f"inference.benchmark={args.benchmark}")
    if args.seed is not None:
        infer_args.append(f"inference.seed={args.seed}")
    if args.ema:
        infer_args.append("inference.prefer_ema=True")
    # argparse.REMAINDER keeps a leading "--"; drop it so it isn't forwarded.
    extra = [o for o in (args.opts or []) if o and o != "--"]
    return infer_args + extra


def _run_streaming(cmd: list[str]) -> tuple[int, bool]:
    """Run ``cmd`` streaming stderr live to our stderr; report exit code + OOM.

    Streams rather than buffering (``capture_output``) so a long inference run's
    output is visible as it happens instead of only at exit, and nothing is held
    in memory. stderr is tee'd to the terminal while scanned for the CUDA OOM
    marker; stdout is left on its own fd (inherited).

    Args:
        cmd (list[str]): the command to run.

    Returns:
        tuple[int, bool]: (exit code, whether a CUDA OOM message was seen).
    """
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    saw_oom = False
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
        if "CUDA out of memory" in line:
            saw_oom = True
    proc.wait()
    return proc.returncode, saw_oom


def launch(job_name: str, infer_args: list[str], output_dir: str, args: argparse.Namespace) -> None:
    """Run inference locally (scripts/infer.sh) or submit it as a cluster job.

    Local runs stream output and retry on CUDA OOM up to :data:`OOM_MAX_RETRIES`
    times (``OOM_BACKOFF_SECONDS`` apart) so a transient OOM recovers but a
    genuinely too-big job fails instead of spinning forever. Cluster runs
    fire-and-forget one launch-tool job.

    Args:
        job_name (str): cluster job name (unused for local runs).
        infer_args (list[str]): tools/infer_mwm.py args.
        output_dir (str): this step's sample dir, for the completion message.
        args (argparse.Namespace): parsed options (``local`` / ``nodes`` / ``queue``).

    Raises:
        subprocess.CalledProcessError: if a local run exits non-zero for a reason
            other than OOM, or keeps hitting OOM past the retry limit.
    """
    if args.local:
        cmd = ["bash", INFER_SCRIPT, *infer_args]
        print(f"[auto_sample] local: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        for attempt in range(OOM_MAX_RETRIES + 1):
            code, saw_oom = _run_streaming(cmd)
            if code == 0:
                print(f"[auto_sample] sample complete: {output_dir}")
                return
            if saw_oom and attempt < OOM_MAX_RETRIES:
                print(
                    f"[auto_sample] CUDA OOM (attempt {attempt + 1}/{OOM_MAX_RETRIES}), "
                    f"retrying in {OOM_BACKOFF_SECONDS}s …",
                    flush=True,
                )
                time.sleep(OOM_BACKOFF_SECONDS)
                continue
            raise subprocess.CalledProcessError(code, cmd)
    else:
        submit = build_submit_argv(job_name, args.nodes, args.cluster, args.queue)
        submit += ["--", INFER_SCRIPT, *infer_args]
        print(f"[auto_sample] submit: {' '.join(shlex.quote(c) for c in submit)}", flush=True)
        subprocess.run(submit, check=True)
        print(f"[auto_sample] submitted job: {job_name}", flush=True)


def main() -> None:
    args = parse_args()
    # Weights are watched with the local filesystem (os.path), and inference runs
    # on a local/cluster node that mounts the run dir — a remote (s3://oss://)
    # output-dir would silently never appear. Reject it up front rather than
    # block forever. (auto_dump can write remote; pull those down first, or point
    # --output-dir at the mounted copy.)
    if is_remote(args.output_dir):
        print(
            f"--output-dir must be a local/mounted path, got remote {args.output_dir!r}. "
            "auto_sample watches weights via the local filesystem; localize them first.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    exp_name = os.path.basename(output_dir.rstrip("/"))
    export_dir = os.path.join(output_dir, args.export_name)
    sample_root = os.path.join(output_dir, args.sample_name)
    ext = "safetensors" if args.format == "safetensors" else "pt"

    print(f"[auto_sample] weights: {export_dir}")
    print(f"[auto_sample] samples: {sample_root}")
    print(f"[auto_sample] steps:   {args.ckpt_steps}  (local={args.local})")

    for step in args.ckpt_steps:
        model_path = os.path.join(export_dir, f"model_{step}.{ext}")
        step_out = os.path.join(sample_root, f"{step}-{exp_name}")
        print(f"[auto_sample] step {step}: waiting on {model_path}", flush=True)
        wait_for_stable_file(model_path, args.format, args.wait_interval)
        print(f"[auto_sample] step {step}: weight ready, launching inference", flush=True)
        infer_args = build_infer_args(args, model_path, step_out)
        launch(build_safe_job_name(exp_name, step, "sample"), infer_args, step_out, args)


if __name__ == "__main__":
    main()
