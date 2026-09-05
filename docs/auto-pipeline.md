# Auto Dump & Sample Pipeline

Automates the `train → dump → sample` loop: as a training run writes
checkpoints, [`tools/auto_dump.py`](https://github.com/shengshu-ai/minWM/blob/main/tools/auto_dump.py) consolidates each
sharded DCP checkpoint into a single-file weight, and
[`tools/auto_sample.py`](https://github.com/shengshu-ai/minWM/blob/main/tools/auto_sample.py) runs inference on each dumped
weight as it appears — locally or as a cluster job.

```
training run                auto_dump.py                   auto_sample.py
  ckpts/checkpoint_N/  ──▶   model_N.safetensors    ──▶    samples/N-<exp>/*.mp4
  (DCP shards)               (bare-model weights)          (torchrun / launch-tool)
```

The two tools are decoupled through the filesystem (or object store): run them
as two processes. `auto_dump` gates on the DCP `.metadata` sentinel (written
last, after every shard is durable); `auto_sample` gates on the dumped file
being fully flushed. Both poll and block, so start them any time — before,
during, or after training.

Both default to **submitting one cluster job per step** via `<launch-tool> submit`,
so the login node only polls and dispatches. `--local` runs the work in the
current process instead (consolidation for `auto_dump`, torchrun for
`auto_sample`).

> **`<launch-tool>` is a placeholder for your cluster's job-submission command**
> (Slurm `sbatch`, a k8s wrapper, an in-house tool, …). This repo ships no such
> tool. Point `MINWM_LAUNCH_TOOL` at your binary and adapt the submit CLI in
> [`tools/_cluster.py`](https://github.com/shengshu-ai/minWM/blob/main/tools/_cluster.py) — the `submit -j <name> -n <nodes>
> [-c <cluster>] [-q <queue>] --no-log -- <script>` form shown throughout this
> doc is **one tool's syntax, used here only as an example**. Or pass `--local`
> to skip cluster submission entirely.

## Cluster environment

Cluster jobs run in whatever image your launch tool selects, and the launch
scripts (`scripts/dump.sh`, `scripts/infer.sh`) need an interpreter with the full
minWM stack. The recommended setup points the launch tool at an image built from
[`docker/Dockerfile`](https://github.com/shengshu-ai/minWM/blob/main/docker/Dockerfile) — it bakes the pinned
`requirements/base.txt` + flash-attn into a venv at `/opt/venv` and puts it on
`PATH`. Keep your launcher config **out of the tree** (this repo intentionally
ships none, to keep private registry addresses uncommitted); a git-ignored config
for a Slurm-style tool might look like:

```yaml
# <repo>/.<launch-tool>/config.yaml  (git-ignored; yours, not committed)
image_url: <your-registry>/minwm:<tag>
conda_bin_path: /opt/venv/bin
```

`scripts/_env.sh` then resolves the interpreter as `$MINWM_PYTHON` →
`$CONDA_PREFIX` → `/opt/venv/bin/python` → `python`, and always prepends the repo
root to `PYTHONPATH` (the image ships the deps but not the `minwm` package
itself, so this is what makes `import minwm` resolve the mounted checkout). For a
**local** run, build an env per [`INSTALL.md`](https://github.com/shengshu-ai/minWM/blob/main/INSTALL.md) and either
`conda activate` it or set `$MINWM_PYTHON`.

## auto_dump

Polls a run's `ckpts/` tree for the requested steps. As each `checkpoint_N/`
DCP directory becomes complete, it consolidates **only the model weights**
(optimizer / RNG state is skipped — ~2/3 of a training checkpoint's bytes) into
`<ckpts>/exported/model_N.{safetensors,pt}` (or a diffusers dir). Consolidation
is CPU-only, single-process, and RAM-heavy, so by default each step is submitted
as its own cluster job (`scripts/dump.sh`).

```bash
# cluster submission (default): one dump job per step
python tools/auto_dump.py \
    --output-dir outputs/wan_action2v_bidirectional \
    --ckpt-steps 10000 15000 20000 \
    --format safetensors \
    -c <cluster> -q <queue>

# local, in-process consolidation (no cluster; needs RAM >= model size)
python tools/auto_dump.py \
    --output-dir outputs/wan_action2v_bidirectional \
    --ckpt-steps 10000 \
    --local

# DMD generator-EMA weights (the aux/generator_ema DCP entry), pt output
python tools/auto_dump.py \
    --output-dir outputs/wan_action2v_dmd \
    --ckpt-steps 200 400 \
    --model-key aux/generator_ema \
    --format pt -c <cluster> -q <queue>

# checkpoints on an object store (training used checkpoint.remote_root)
python tools/auto_dump.py \
    --output-dir outputs-sft/wan_action2v_bidirectional \
    --remote-root s3://<bucket>/<prefix> \
    --ckpt-steps 10000 \
    --export-dir ./ckpts/Wan21/Action2V/sft --local
```

Key flags: `--format {safetensors,pt,diffusers}` (diffusers needs `--config`),
`--model-key` (`model` default; `aux/generator_ema` for DMD EMA), `--export-dir`
(default `<ckpts>/exported`), `--remote-root`, `--local`, `--cluster/-c` /
`--queue/-q` / `--nodes`, `--wait-interval`.

## auto_sample

For each requested step, blocks until the dumped weight is fully written, then
launches inference on it. Samples land in `<output-dir>/<sample-name>/{step}-<exp>/`.

```bash
# cluster submission (default): one sample job per step
python tools/auto_sample.py \
    --output-dir outputs/wan_action2v_dmd \
    --ckpt-steps 200 400 \
    --config-file configs/wan21/action2v/infer/stage3_ar_dmd.py \
    -c <cluster> -q <queue> --nodes 1

# local (torchrun via scripts/infer.sh), overriding the config's benchmark
python tools/auto_sample.py \
    --output-dir outputs/wan_action2v_bidirectional \
    --ckpt-steps 10000 15000 \
    --config-file configs/wan21/action2v/infer/stage0_bi_sft.py \
    --benchmark ./prompts/t2v.json \
    --local
```

The prompts to run come from the config (`inference.benchmark`), so no input
flag is required; `--benchmark` only overrides it. A camera trajectory is not a
launcher flag — it rides on each benchmark item.

Key flags: `--format {safetensors,pt}` (must match `auto_dump`), `--benchmark`,
`--seed`, `--ema`, `--local` (else `<launch-tool> submit`), `--cluster/-c` /
`--queue/-q` / `--nodes`, `--export-name` (default `ckpts/exported`),
`--sample-name` (default `samples`). The per-step weight and sample dir are
forwarded to `infer_mwm.py` as `inference.checkpoint=` / `inference.output_dir=`
dotlist overrides, and trailing overrides after a `--` are forwarded too.

`scripts/infer.sh` is the shared launcher: it derives the torchrun topology from
the cluster env (`WORLD_SIZE` / `RANK` / `MASTER_ADDR` / `MASTER_PORT`) exactly
like the training scripts, so the same script serves a local run and a
`<launch-tool> submit -- scripts/infer.sh ...` job. `scripts/dump.sh` is the matching
launcher for the CPU-only consolidation.

## Same for every model

`auto_dump` / `auto_sample` are **model-agnostic** — one validated run covers
all of them:

- **dump**: every model saves the same DCP shape `{"app": {"model", "aux/...",
  "opt/..."}}`; consolidation differs only by `--model-key` (`model` vs
  `aux/generator_ema`).
- **sample**: the model-specific pipeline (bidirectional / AR-DMD / …) lives
  entirely inside `tools/infer_mwm.py`, selected by `--config-file`;
  `auto_sample` only waits for the weight and launches inference.

## Properties, costs, and limits

Know these before relying on the pipeline — they are behavioral contracts, not
implementation details.

- **Model-only dump.** `auto_dump` consolidates only the `--model-key` subtree.
  The dumped weight cannot resume training (no optimizer/RNG state); it is an
  inference/eval artifact. Resuming still uses the original DCP dir.
- **Peak memory ≈ full model on one process.** Consolidation rebuilds the whole
  model state dict in one CPU process (the 14B Wan model peaks ~6 GB RSS and
  takes several minutes — single-threaded, I/O-bound DCP reads, no GPU). In
  cluster mode this runs on a cluster node; `--local` runs it on the box you
  launched from, which must have RAM ≥ model size.
- **Readiness gates, not locks.** `auto_dump` trusts DCP's `.metadata` sentinel;
  `auto_sample` trusts safetensors header math (or a stable `.pt` size). Neither
  takes a lock — if a writer deletes/rewrites a checkpoint in place mid-read,
  behavior is undefined. Checkpoints are write-once in practice, so this is safe
  for the normal flow.
- **Idempotent, resumable.** Both tools skip steps whose output already exists,
  so re-running after an interruption resumes rather than redoing work. To force
  a re-dump, delete the dumped file first.
- **`--format` must agree.** `auto_sample` waits on `model_N.<ext>` for the ext
  implied by its `--format`; if `auto_dump` wrote a different format the wait
  never completes. `diffusers` is dump-only here — `auto_sample` consumes
  single-file weights (`safetensors` / `pt`).
- **Shared storage for cluster mode.** Cluster jobs read the DCP dir and write
  the dumped weight / samples through the job's `working_dir` (the launch dir).
  That path must be on storage the cluster nodes can see (e.g. a shared network
  mount), or use `--remote-root` / an `s3://` export dir.
- **Remote storage needs the fsspec backend.** `s3://` / `oss://` paths require
  the matching backend installed (`s3fs` / `ossfs`); otherwise reads fail hard.
  Credentials/endpoint are resolved by `minwm`'s `Storage` (env + `~/.aws`).
- **Cluster mode fire-and-forgets.** Both tools submit one job per
  step and return; they do not track job success. Watch the cluster for results.
  `auto_sample --local` runs synchronously and retries once on CUDA OOM (600 s
  backoff), failing hard on any other non-zero exit.
- **Blocking, unbounded wait.** A step whose checkpoint never appears blocks
  that tool forever (by design, for the streaming train→sample case). Only pass
  steps you expect the run to reach.
