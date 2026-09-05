# Getting Started

## Installation

```bash
conda create -n minwm python=3.12 -y
conda activate minwm
pip install -r requirements/base.txt
pip install flash-attn --no-build-isolation
pip install -e .          # editable install: makes `import minwm` resolve, no PYTHONPATH
```

`minwm` is installed editable so `import minwm` resolves without `PYTHONPATH`.
The launcher scripts under `scripts/` additionally put the repo root on
`PYTHONPATH` themselves (via `scripts/_env.sh`), since the cluster image ships
the third-party dependencies but not `minwm` itself.

!!! note
    Full requirements, verification, developer setup, and troubleshooting live in
    [`INSTALL.md`](https://github.com/shengshu-ai/minWM/blob/main/INSTALL.md).

## Inference

The fastest path: install, download a base model plus a 4-step DMD checkpoint,
run one command. All weights live under `./ckpts/` after download.

### 1. Download the demo checkpoints

```bash
# Wan base (T2V-1.3B)
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B

# The ODE data-curation tool loads the base from `wan_models/` relative to the
# repo root, so mirror it there (inference reads ./ckpts/Wan2.1-T2V-1.3B).
mkdir -p wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" wan_models/Wan2.1-T2V-1.3B

# 4-step DMD checkpoints
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/dmd/*"
hf download MIN-Lab/minWM --local-dir ./ckpts --include "HY15/Action2V/dmd/*"

# The HF repo publishes stage weights under release names (`dmd`, `causal_cd`, …)
# while the configs load the `stage{N}_*` names. Link them so the configs resolve
# with no extra flags (relative + idempotent; safe to re-run).
ln -sfnT dmd ./ckpts/Wan21/Action2V/stage3_ar_dmd
ln -sfnT dmd ./ckpts/HY15/Action2V/stage3_ar_dmd
```

HY pipelines also need the HunyuanVideo-1.5 base plus its text/vision encoders;
see the [README](https://github.com/shengshu-ai/minWM#inference) for the full
download block.

??? note "Checkpoint naming: release names → config names"
    The two naming schemes differ, so each downloaded stage needs one
    directory-level symlink. The links are relative (so `./ckpts/` stays movable)
    and re-running is a no-op. Wan configs load `<stage>/model.pt`; HY configs
    load `<stage>/` as a diffusers directory — one link per stage satisfies both.

    Keep the `-T`: without it, if the `stage{N}_*` path already exists as a **real**
    directory (e.g. you exported your own checkpoint there), `ln` would quietly
    create a nested link *inside* it and the config would keep loading the old
    weights. With `-T` you get a loud `cannot overwrite directory` instead.

    | Release name (on HF) | Config name (symlink) | Stage |
    | --- | --- | --- |
    | `bidirectional` | `stage0_bi_sft` | Phase 1 bidirectional SFT |
    | `ar_diffusion_tf` | `stage1_ar_tf` | Phase 2 Stage 1 teacher forcing |
    | `causal_ode` | `stage2_ar_ode` | Phase 2 Stage 2(a) ODE distillation |
    | `causal_cd` | `stage2_ar_cd` | Phase 2 Stage 2(b) consistency distillation |
    | `dmd` | `stage3_ar_dmd` | Phase 2 Stage 3 DMD (4-step) |

    To link every stage you downloaded, for either model line:

    ```bash
    for line in Wan21/Action2V HY15/Action2V; do
      ( cd ./ckpts/$line 2>/dev/null || exit 0
        ln -sfnT bidirectional    stage0_bi_sft
        ln -sfnT ar_diffusion_tf  stage1_ar_tf
        ln -sfnT causal_ode       stage2_ar_ode
        ln -sfnT causal_cd        stage2_ar_cd
        ln -sfnT dmd              stage3_ar_dmd )
    done
    ```

    Dangling links for stages you did not download are harmless — nothing reads
    them. Alternatively, skip the links and pass the release path explicitly:
    `inference.checkpoint=./ckpts/Wan21/Action2V/dmd/model.pt`.

### 2. Run the demos

One entrypoint — `tools/infer_mwm.py` — drives every model line and stage. The
loop, sampler, guidance and step count all come from `--config-file`, so
switching stage means switching the config; the command shape never changes.

The input is a benchmark JSON named by the config key `inference.benchmark`: a
list of `{id, caption, trajectory}` items (extra keys ignored) where each output
is named `{id}.mp4`. An `image` field makes the item image-to-video (HY);
omitting it makes it text-to-video (Wan).

```bash
# 2.1  Wan Action2V (4-step DMD, camera control)
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/wan21/action2v/infer/stage3_ar_dmd.py \
    inference.output_dir=./outputs/quickstart_wan_action2v

# 2.2  HY Action2V (4-step DMD, camera control)
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage3_ar_dmd.py \
    inference.benchmark=assets/example.json inference.limit=2 \
    inference.output_dir=./outputs/quickstart_hy_action2v
```

Each run writes one `.mp4` per sample (Wan Action2V: 832×480, 77 frames @ 16 fps)
plus a `manifest.json` holding the resolved config and per-item
`{index, id, prompt, trajectory, seed, video}`.

Every config value is overridable inline as a dotlist `key=value`: e.g.
`inference.checkpoint=` / `inference.output_dir=` beat the config's values,
`inference.limit=N` runs only the first N items, and `inference.sp_size=N` (with
a matching `--nproc_per_node=N`) enables sequence-parallel sampling.

!!! tip "Camera trajectories"
    Format is `key*N` segments joined by commas — e.g. `d*8,i*5,l*6` = pan right 8,
    tilt up 5, pan left 6. `w/s/a/d` translate, `i/k/j/l` rotate; for 20 latent
    frames the segment counts sum to 19. The trajectory is per-sample, read from
    each benchmark item's `"trajectory"` field.

### 3. Optional: overlay the key indicator

Draws the WASD/KIJL presses onto each clip and concatenates them into a single
overview. It reads `manifest.json`, so it works on any output directory above:

```bash
python demos/overlay_from_manifest.py \
    --input-dir ./outputs/quickstart_wan_action2v \
    --output final_with_keys.mp4
```

!!! note
    Needs `ffmpeg` / `ffprobe` on `PATH`. Cluster images often lack them — run this
    step locally against the (shared-filesystem) output directory instead.

## Full reproduction

Two model lines × two phases × the Phase-2 stage ladder, each documented as
**(1) Model download → (2) Data preparation → (3) Training script →
(4) Validation**. The full guides are split by backbone:

- [`configs/wan21/action2v/`](https://github.com/shengshu-ai/minWM/blob/main/configs/wan21/action2v/README.md) — Wan 2.1 backbone
- [`configs/hy/action2v/`](https://github.com/shengshu-ai/minWM/blob/main/configs/hy/action2v/README.md) — HY1.5-8B backbone

The stage ladder is `stage0_bi_sft` → `stage1_ar_tf` → `stage2_ar_ode` →
`stage2_ar_cd` → `stage3_ar_dmd`; each stage has a matching config under
`configs/<backbone>/action2v/{train,infer}/`.
