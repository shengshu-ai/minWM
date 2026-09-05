# Wan2.1 Action2V

> 中文版 / Chinese: [README_cn.md](README_cn.md)

data encode → SFT → TF → ODE data curation → ODE → CD → DMD → inference → key overlay.


**Stage chain** (each stage's output is the next stage's input, must run serially):

```
encode ─→ SFT ─→ TF ─┬─→ ODE curation ─→ ODE ─┐
                     └─→ CD ───────────────────┴─→ DMD
```

Both `ODE` and `CD` start from the TF weights and are independent of each other; `DMD` needs the
CD (student+ema) and SFT (teacher+critic) weights. After each stage finishes training you must
export the DCP directory into a single-file `.pt` (§3.1); it is both **this stage's inference
checkpoint** and **the next stage's `checkpoint.pretrained`**.

---

## 0. Prerequisites

<details>
<summary><big><b>0.1 ⭐ Config block (edit here first; all later commands reference it)</b></big></summary>

**Every command later in this document assumes you have already sourced this block.** Only this
part needs to be changed for your environment; all other sections use variables throughout, so you
can copy them verbatim. Recommended to save it as `env.sh` outside the repo and `source` it once
per new shell.

```bash
# ---- (1) Repo and Python environment -------------------------------------------------
cd /path/to/minWM                      # ← change to your checkout path
export PROJECT_ROOT="$PWD"
conda activate minwm                   # ← see INSTALL.md

# ---- (2) Dataset short name: the unique identifier threaded through all output paths -----------------------------
# Pick anything, as long as it is unique on your machine. All dataset/ outputs/ paths derive from it,
# so switching to a different dataset only changes this one line, and the runs naturally don't collide.
export DS=my_dataset                   # ← change to your dataset name

# ---- (3) Source data (only these two are absolute paths outside the repo)------------------------------
# The video directory layout must be <VIDEO_DIR>/<idx:06d>_<suffix>/gen.mp4, see §1.
export SRC_JSON=/abs/path/to/preencode_input.json   # ← caption + pose_str
export SRC_VIDEOS=/abs/path/to/videos               # ← video root directory

# ---- (4) Derived paths (no need to change)--------------------------------------------------
export BASE="$PROJECT_ROOT/ckpts/Wan2.1-T2V-1.3B"   # Wan2.1 base (download in §0.2)
export DATA_ROOT="dataset/Wan21/Action2V_$DS"       # encode/curation outputs
export LMDB="$DATA_ROOT/data"                       # LMDB read by SFT/TF/CD/DMD
export ODE_LATENTS="$DATA_ROOT/ode_latents"         # ODE per-clip .pt
export ODE_LMDB="$DATA_ROOT/ode_lmdb"               # LMDB read by ODE training
export CKPT_ROOT="ckpts/Wan21/Action2V"             # exported single-file weights
export NPROC=8                                      # number of GPUs

# ---- (5) Benchmark JSON for evaluation (optional, §3.2.5)-------------------------------
# Format [{id, caption, trajectory}]; leave empty to use the default benchmark bundled with each infer config.
export BENCH="assets/example_t2v.json"              # ← benchmark JSON for evaluation

# ---- (6) Runtime environment --------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
```

> Set `$NPROC` to your actual GPU count (`nvidia-smi -L | wc -l`). **The GPU count must be divisible
> by `training.sp_size`**, otherwise it won't start — see the table in §0.3.
>
> `$DS` is the single variable of the whole path scheme: `DATA_ROOT` / training outputs / inference
> outputs all derive from it, so running multiple datasets on the same machine only requires
> changing `DS`, and the outputs are naturally isolated.

</details>

<details>
<summary><big><b>0.2 base model</b></big></summary>

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$BASE" \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" \
              "diffusion_pytorch_model.safetensors" "config.json"
```

All train / infer configs write `_BASE = "./ckpts/Wan2.1-T2V-1.3B"` (a repo-relative path), so if
`$BASE` points elsewhere you must either symlink it over or add `model._from_pretrained=$BASE` to
the command. In addition, the ODE curation tool hardcodes the base path as
`wan_models/Wan2.1-T2V-1.3B/`, so be sure to create this symlink:

```bash
mkdir -p wan_models && ln -sfn "$BASE" wan_models/Wan2.1-T2V-1.3B
```

</details>

<details>
<summary><big><b>0.3 Topology and parallelism (choose `sp_size` by your GPU count)</b></big></summary>

`total GPUs = sp_size × DP`, `GBS = DP × data.batch_size`. There is only one constraint: **`$NPROC`
must be divisible by `sp_size`**.

The commands in this document uniformly write `training.sp_size=2` (a safe choice for 1.3B at 20
latent frames). A smaller `sp_size` means larger DP/GBS and higher memory pressure; a larger one
slices each sample more finely, saving memory but adding communication.

</details>

<details>
<summary><big><b>0.4 Path conventions</b></big></summary>

Everything derives from `$DS` in §0.1, **switching datasets only changes `DS` in one place**, and
each run's outputs are naturally isolated:

| Role | Path | Variable |
|---|---|---|
| Source JSON / videos | Outside the repo, your own data | `$SRC_JSON` / `$SRC_VIDEOS` |
| SFT/TF/CD/DMD LMDB | `dataset/Wan21/Action2V_$DS/data` | `$LMDB` |
| ODE per-clip `.pt` | `dataset/Wan21/Action2V_$DS/ode_latents` | `$ODE_LATENTS` |
| ODE LMDB | `dataset/Wan21/Action2V_$DS/ode_lmdb` | `$ODE_LMDB` |
| Training outputs | `outputs/wan_action2v_{stage}_$DS/` | — |
| Exported single-file weights | `ckpts/Wan21/Action2V/{stage}/model.pt` | `$CKPT_ROOT/{stage}/model.pt` |
| Inference outputs | `outputs/infer_{stage}_$DS/{step}/` | — |

`{stage}` ∈ `stage0_bi_sft` / `stage1_ar_tf` / `stage2a_ar_ode` / `stage2_ar_cd` / `stage3_ar_dmd`.

> **The weights path does not include `$DS`**: `$CKPT_ROOT/{stage}/model.pt` is each config's
> default value (hardcoded in the config), so when running multiple datasets on the same machine
> **these exported weights will overwrite each other**. To keep them side by side, add a suffix
> yourself, e.g. `--output "$CKPT_ROOT/stage0_bi_sft/model_$DS.pt"`, and point `--checkpoint` at the
> same one during inference.
>
> In the configs the data path defaults to `./dataset/Wan21/Action2V/{data,ode_lmdb}` (without
> `$DS`), so the commands in this document always explicitly override `data.dataset.data_path=`; you
> can also do the reverse with a symlink:
> `ln -sfn "$PROJECT_ROOT/$LMDB" dataset/Wan21/Action2V/data`.

</details>

<details>
<summary><big><b>0.5 Monitoring</b></big></summary>

**wandb is off by default**: don't pass `monitor.*`, and metrics only go to STDOUT
(`training.log_interval=10`, one line every 10 steps: loss + steps/sec + ms/step + peak memory). To
use wandb, configure it yourself, and **keep the key only in the shell / a file outside the repo,
never write it into any file that goes into git**:

```bash
export WANDB_API_KEY='your key'
export WANDB_ENTITY='your entity'      # if unset, uses your account's default entity
# Append at the end of the training command (pick a run name yourself, ideally with $DS and sp_size for easy distinction):
#   monitor.backends="['wandb']" monitor.wandb_project=wan21 \
#   monitor.wandb_run_name="sft-$DS-sp4"
```

</details>

---

## 1. Data encode (VAE → LMDB)

<details>
<summary><big><b>1.1 Data encoding</b></big></summary>

Encode the WorldPlayGen videos + captions + `pose_str` into a merged LMDB of Wan VAE latents. Each
rank writes its own `.rank_{r}` shard → rank0 streams the merge into `data/` and deletes the shards
(bounded memory).

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29700 \
    tools/data/wan21/build_worldplaygen_lmdb.py \
    --input_json "$SRC_JSON" \
    --video_dir  "$SRC_VIDEOS" \
    --output_dir "$DATA_ROOT" \
    --vae_path   "$BASE/Wan2.1_VAE.pth" \
    --target_h 480 --target_w 832
```

Produces `$LMDB` (= `$DATA_ROOT/data`, an LMDB directory, `data.mdb` + `lock.mdb`).

**LMDB key contract**:

```
latents    (N, 20, 16, 60, 104) float16   Wan VAE latent (77 pixel frames → 20 latent frames)
prompts    (N,)                 str       caption
intrinsics (N, 4)               float32   [fx/W, fy/H, cx/W, cy/H]
poses      (N, 20, 7)           float32   [tx,ty,tz, qx,qy,qz,qw] (w2c)
```

`N` = number of valid clips; after it finishes, check rank0's merge log (reference magnitude: for
one 14975-clip dataset, 0 lost).

**Input format requirements** (prepare per this when switching datasets):

- `$SRC_JSON`: each entry has `caption` + `pose_str`.
- `$SRC_VIDEOS`: layout must be `<$SRC_VIDEOS>/<idx:06d>_<suffix>/gen.mp4`, where `suffix` = that
  entry's `pose_str` lowercased with all non-`[a-z0-9]` characters removed; `idx` is its index in
  the JSON.
- Videos are 77 frames, 480×832 (`--target_h/--target_w` will resize; frame-count mismatches are
  skipped).

**Notes**:

- The multi-machine version merges shards via a shared disk + NCCL barrier; single-machine is just
  `$NPROC` shards merged locally, no shared disk needed.
- For the internal encoding details (Wan21VAE z_dim=16 normalization, `pose_str`→camera trajectory
  synthesis, key contract) see
  [`tools/data/wan21/README.md`](../../../tools/data/wan21/README.md).

</details>

<details>
<summary><big><b>1.2 Data sanity check (optional but recommended)</b></big></summary>

Before training, confirm the camera trajectory / caption / frames all line up. Decode the latents
back to pixels, and per clip produce an `[RGB | BEV]` side-by-side mp4 (top-down X-Z trajectory +
yaw arrow + pitch dial + path length + caption bar), and run a full-dataset scale census
(`trans_span` / `path_len` / `rot_span` histograms + outlier list):

```bash
python tools/data/wan21/check_dataset_bev.py \
    --data_path "$LMDB" \
    --vae "$BASE/Wan2.1_VAE.pth" \
    --num_videos 8 --concat 10 --gpu 0
```

For only the scale census plots and no videos (no GPU needed): add `--scale_plot`. For the
per-flag description see [`tools/data/wan21/README.md`](../../../tools/data/wan21/README.md). Pose
normalization is exactly the same as in training, so what you see is what the model sees.

---

</details>

## 2. Training pipeline

<details>
<summary><big><b>2.1 Phase-1 Bidirectional SFT (`stage0_bi_sft`)</b></big></summary>

Bidirectional + camera (PRoPE) supervised fine-tuning. `Wan21Model` + `BiSFTRecipe` +
`FlowMatchingLoss`, reading the clean-latent camera LMDB from §1 (viewmats / Ks).

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29701 \
    tools/train_mwm.py \
    --config-file configs/wan21/action2v/train/stage0_bi_sft.py \
    --output-dir "outputs/wan_action2v_sft_$DS" \
    training.sp_size=4 \
    training.max_steps=20000 \
    training.ckpt_interval=2500 \
    training.log_interval=10 \
    data.dataset.data_path="$LMDB"
```

- Initialization: `_from_pretrained=./ckpts/Wan2.1-T2V-1.3B` (non-strict load of the base weights).
  **No `checkpoint.pretrained`** — SFT is the start of the chain.
- Hyperparameters (config defaults): `lr=2e-6`, `betas=(0.0,0.999)`, `weight_decay=0.01`,
  `batch_size=1`, `num_workers=4`, `activation_checkpointing=True`.
- The config defaults are `max_steps=10000` / `ckpt_interval=1000`; above they are overridden to the
  cluster run's `20000` / `2500`. Smoke run: `training.max_steps=100 training.ckpt_interval=50`.
- Output: `outputs/wan_action2v_sft_$DS/{ckpts/checkpoint_{step}/, logs/}`, with `ckpts/latest.txt`
  pointing at the latest. **The checkpoint is an FSDP2 DCP shard directory, not a single `.pt`** (see
  §3.1).

**Normal first-step warnings**: a large batch of
`blocks.*.self_attn.prope_o.* newly initialized` + `You should probably TRAIN this model...` —
the PRoPE camera projection parameters don't exist in the base and are newly initialized, **as
expected**.

**A speedup to try when memory is loose**: `model.gradient_checkpointing=False` (backward is ~70% of
a single step, part of which is recompute overhead). Doesn't change GBS — trading memory for speed.

**Verify the overrides took effect** (without starting training):

```bash
python -c "
import os
from minwm.config import load, apply_overrides
c = apply_overrides(load('configs/wan21/action2v/train/stage0_bi_sft.py'),
                    ['training.sp_size=4', 'training.ckpt_interval=2500',
                     f'data.dataset.data_path={os.environ[\"LMDB\"]}'])
print(c['training']['sp_size'], c['data']['dataset']['data_path'])
"
```

After training, pick the best step and export (§3.1) → `$CKPT_ROOT/stage0_bi_sft/model.pt`. It is
both the checkpoint for SFT inference and the **starting point for TF training** + the **teacher/critic
seed for DMD**.

---

</details>

<details>
<summary><big><b>2.2 Phase-2 Stage-1 Teacher-Forcing AR (`stage1_ar_tf`)</b></big></summary>

**If you skip the previous stage, you can download the officially provided previous-stage checkpoint:**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/bidirectional/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT bidirectional stage0_bi_sft
)
```

After downloading, the checkpoint is at `./ckpts/Wan21/Action2V/bidirectional/`, and the symlink
above maps it to the `$CKPT_ROOT/stage0_bi_sft/model.pt` path used in this document; you can also
explicitly override `checkpoint.pretrained` on the command line.

Convert the bidirectional SFT model into a causal + teacher-forcing AR diffusion. `CausalWan21Model`
(`num_frame_per_block=4`, `local_attn_size=20`) + `ARTFRecipe` + `FlowMatchingLoss`, a
`causal=True` `Wan21Adapter`, `use_prope=True` retaining the PRoPE parameters. **Same LMDB as SFT**.

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29702 \
    tools/train_mwm.py \
    --config-file configs/wan21/action2v/train/stage1_ar_tf.py \
    --output-dir "outputs/wan_action2v_tf_$DS" \
    training.sp_size=4 \
    training.max_steps=20000 \
    training.ckpt_interval=2500 \
    training.log_interval=10 \
    data.dataset.data_path="$LMDB" \
    checkpoint.pretrained="$CKPT_ROOT/stage0_bi_sft/model.pt"
```

- **Requires the SFT export first**: `$CKPT_ROOT/stage0_bi_sft/model.pt` (§3.1). The trainer's loader
  automatically strips the `generator` / `model.` prefixes.
- Hyperparameters same as SFT (`lr=2e-6`, `betas=(0.0,0.999)`). TF is about 2.3× slower than SFT.
- Output: `outputs/wan_action2v_tf_$DS/`.

After training, export → `$CKPT_ROOT/stage1_ar_tf/model.pt`. It is the **teacher for ODE curation**,
the **starting point for ODE training**, and the **student/teacher/ema three-way seed for CD
training**.

---

</details>

<details>
<summary><big><b>2.3 Stage-2(a) ODE data curation (data preparation, not training)</b></big></summary>

**TF teacher checkpoint (output of 2.2; if you skip the TF stage, you can download the officially provided checkpoint):**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/ar_diffusion_tf/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT ar_diffusion_tf stage1_ar_tf
)
```

After downloading, the checkpoint is at `./ckpts/Wan21/Action2V/ar_diffusion_tf/`, and the symlink
above maps it to `$CKPT_ROOT/stage1_ar_tf/model.pt`, which the ODE curation command below uses by
default; if the path differs, you can also explicitly override `--generator_ckpt`.

Using the **frozen TF teacher**, **pre-solve one 6-point ODE trajectory** for each clip in the SFT
LMDB and freeze them into a new LMDB, for the `ARODERecipe` in §2.4 to fit by regression. **No
config, pure CLI tools**, two steps.

#### 2.3.1 Sampling (48-step CFG flow → per-clip `.pt`)

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29703 \
    tools/data/wan21/ode/get_causal_ode_data_prope.py \
    --generator_ckpt $CKPT_ROOT/stage1_ar_tf/model.pt \
    --rawdata_path   "$LMDB" \
    --output_folder  "$ODE_LATENTS" \
    --guidance_scale 6.0
```

- Loads the TF causal model (takes `state_dict["generator"]`, strips the FSDP/compile prefixes,
  `strict=True`) + umt5 T5. For each clip, conditioned on its `clean_latent` + camera
  `viewmats`/`Ks`, runs a **48-step CFG flow** (`shift=5.0`, `sigma_min=0.0`, `extra_one_step`,
  guidance 6.0), keeping indices `[0,12,24,36,-2,-1]` → **6 points** = 4 denoising anchors
  (t≈1000/750/500/250) + the 48-step target + clean.
- Sharded by `index * world_size + rank`; each clip writes `{idx:05d}.pt`:
  `{prompt, latents(1,6,20,16,60,104), viewmats(1,20,4,4), Ks(1,20,3,3)}`.
- **Must be run from the repo root**: the T5 / model `config.json` paths are hardcoded in
  `wan_models/Wan2.1-T2V-1.3B/` (§0.2 symlink).
- **Requires a flash-attn kernel of sm_80 or higher** (A100/A800 and the like ✓; in practice B200
  lacks the corresponding kernel). If unsure, run a small batch first (the `--num_videos` kind of
  probe doesn't apply here; instead just check whether the first few `.pt` land on disk).
- **This is the heaviest part of the whole pipeline**: each clip runs 48 steps × 2 forwards (CFG),
  amortized to `ceil(N / $NPROC)` clips per GPU (`N` = number of LMDB entries). For a sense of scale:
  one ~15k-clip dataset on 128 GPUs is ~117 clips/GPU; with fewer GPUs, scale up proportionally.
  **Test the flow on a small dataset before going full scale.**
- Track progress via the increasing count of `.pt` files on disk, don't trust tqdm (the `\r` frames
  don't land on disk after redirection):
  ```bash
  ls "$ODE_LATENTS" | wc -l        # target = number of LMDB entries N
  ```

#### 2.3.2 Merge (per-clip `.pt` → single LMDB)

Single process, pure CPU. The `dist.barrier()` at the end of torchrun guarantees all ranks' `.pt`
have landed on disk.

```bash
python tools/data/wan21/ode/wan_utils/build_ode_prope_lmdb.py \
    --input_dir  "$ODE_LATENTS" \
    --output_dir "$ODE_LMDB" \
    --map_size_gb 10000
```

Produces `ode_lmdb/{data.mdb,lock.mdb}`, with keys
`latents_{i}_data` / `prompts_{i}_data` / `viewmats_{i}_data` / `Ks_{i}_data`
+ `latents_shape="N 6 20 16 60 104"`. Expected around 46 GB (`map_size_gb=10000` has ample
headroom).

**To re-run only the merge** (the `.pt` are already there, skip 2.3.1): just run this 2.3.2 command.

#### 2.3.3 How to skip 2.3.1: download ready-made ODE latents

What is released on HF is the **unmerged `.pt`** (exactly the output of 2.3.1); after downloading,
locally run only the 2.3.2 merge:

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset --include "ODE_data/Wan21/Action2V/**"

python tools/data/wan21/ode/wan_utils/build_ode_prope_lmdb.py \
    --input_dir  ./dataset/ODE_data/Wan21/Action2V \
    --output_dir "$ODE_LMDB" \
    --map_size_gb 10000
```

---

</details>

<details>
<summary><big><b>2.4 Stage-2(a) Causal ODE Distillation (`stage2_ar_ode`)</b></big></summary>

**Initialization checkpoint:** use the TF checkpoint prepared in `2.3`
(`$CKPT_ROOT/stage1_ar_tf/model.pt`). If you skip `2.3` and directly use pre-generated ODE latents,
you still need to first download or prepare that TF checkpoint per the instructions at the start of
`2.3`.

Using the 6-point trajectory LMDB pre-solved in §2.3, fit the causal student by regression.
`ODETrajectorySample` samples one denoising anchor per step (`denoising_step_list=[1000,750,500,250]`,
`warp_denoising_step=True` maps to the shifted schedule times), and `ODERegressionLoss` regresses
the near-clean target in **x0 space**. The model is the same `CausalWan21Model` + PRoPE camera flow
as TF, only the training signal changes from flow-matching to ODE regression.

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29704 \
    tools/train_mwm.py \
    --config-file configs/wan21/action2v/train/stage2_ar_ode.py \
    --output-dir "outputs/wan_action2v_ode_$DS" \
    training.sp_size=4 \
    training.max_steps=10000 \
    training.ckpt_interval=1000 \
    training.log_interval=10 \
    data.dataset.data_path="$ODE_LMDB" \
    checkpoint.pretrained=$CKPT_ROOT/stage1_ar_tf/model.pt
```

- **Data is `ode_lmdb`, not SFT's `data`** (read by `CameraODERegressionLMDBDataset`). The config
  defaults to `./dataset/Wan21/Action2V/ode_lmdb`, so this override is required.
- Initialization: the TF promoted weights, strict load retaining the PRoPE parameters.
- Hyperparameters (config defaults): `lr=2e-6`, **`betas=(0.9,0.999)`** (note this differs from
  SFT/TF's `(0.0,0.999)`), `weight_decay=0.01`, `timestep_shift=5.0`, `sigma_min=0.0`,
  `extra_one_step=True`.
- Output: `outputs/wan_action2v_ode_$DS/`.

**Verify the overrides took effect**:

```bash
python -c "
import os
from minwm.config import load, apply_overrides
c = apply_overrides(load('configs/wan21/action2v/train/stage2_ar_ode.py'),
                    ['training.sp_size=4',
                     f'data.dataset.data_path={os.environ[\"ODE_LMDB\"]}',
                     f'checkpoint.pretrained={os.environ[\"CKPT_ROOT\"]}/stage1_ar_tf/model.pt'])
print(c['recipe']['type'], c['checkpoint']['pretrained'], c['data']['dataset']['data_path'])
"
```

After training, export → `$CKPT_ROOT/stage2a_ar_ode/model.pt`. ODE is a **single-model recipe** (no
`generator_ema`), export with the default `--model-key model`. It is the checkpoint for ODE
few-step inference, and also an alternative starting point for CD.

---

</details>

<details>
<summary><big><b>2.5 Stage-2(b) Causal Consistency Distillation (`stage2_ar_cd`)</b></big></summary>

**Initialization checkpoint:** both the CD and ODE branches start from the TF checkpoint, using the
`$CKPT_ROOT/stage1_ar_tf/model.pt` prepared in `2.3`. If you skip `2.2` and `2.3`, download or
prepare that checkpoint per the instructions at the start of `2.3`.

Distill the causal TF model into a few-step consistency model: the frozen teacher takes one CFG
Euler step `t→t_next`, the student predicts `x0` at `t`, the EMA network predicts `x0` at `t_next`,
and the loss is the MSE of the two `x0`s.

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29705 \
    tools/train_mwm.py \
    --config-file configs/wan21/action2v/train/stage2_ar_cd.py \
    --output-dir "outputs/wan_action2v_cd_$DS" \
    training.sp_size=4 \
    training.max_steps=10000 \
    training.ckpt_interval=1000 \
    training.log_interval=10 \
    data.dataset.data_path="$LMDB" \
    checkpoint.pretrained=$CKPT_ROOT/stage1_ar_tf/model.pt
```

- **Initialization (key)**: **all three 1.3B copies — student / teacher / ema — are seeded from the
  same TF weights**. Only one `checkpoint.pretrained` is given, and `ARCDRecipe` uses `copy_params`
  at step 0 to copy the student parameters into teacher / ema. teacher + EMA are declared as frozen
  `auxiliary_models` in the config.
- **Data returns to SFT's clean-latent LMDB** (CD doesn't need pre-solved trajectories, it uses
  clean latents directly).
- Hyperparameters (config defaults): `lr=2e-6`, `guidance_scale=3.0`, `discrete_cd_n=50`,
  `ema_decay=0.99`, `activation_checkpointing=True` (three 1.3B copies, memory is fairly tight,
  don't turn it off).
- Output: `outputs/wan_action2v_cd_$DS/`.

**Verify the overrides + aux models**:

```bash
python -c "
import os
from minwm.config import load, apply_overrides
c = apply_overrides(load('configs/wan21/action2v/train/stage2_ar_cd.py'),
                    ['training.sp_size=4',
                     f'checkpoint.pretrained={os.environ[\"CKPT_ROOT\"]}/stage1_ar_tf/model.pt'])
print(c['checkpoint']['pretrained'], list(c['auxiliary_models'].keys()))
"
```

After training, export → `$CKPT_ROOT/stage2_ar_cd/model.pt` (the checkpoint for CD few-step
inference + the **generator / generator_ema seed for DMD**).

---

</details>

<details>
<summary><big><b>2.6 Stage-3 Asymmetric DMD with Self Rollout (`stage3_ar_dmd`)</b></big></summary>

**If you skip the earlier initialization stages, you can download the officially provided ODE or CD checkpoint:**

```bash
# Default ODE initialization
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/causal_ode/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT causal_ode stage2_ar_ode
)

# Or use CD initialization
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/causal_cd/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT causal_cd stage2_ar_cd
)
```

After downloading, the checkpoints are at `./ckpts/Wan21/Action2V/causal_ode/` and
`./ckpts/Wan21/Action2V/causal_cd/` respectively, and the symlinks above map them to `stage2_ar_ode`
and `stage2_ar_cd`. DMD also needs the bidirectional SFT checkpoint as the seed for `real_score` /
`fake_score`; if you don't have it locally, download it per the command in 2.2, or explicitly
override the corresponding `checkpoint.*` paths on the command line.

Distribution Matching Distillation distills the causal generator into a 4-step model. **Four 1.3B
copies**:

| aux name | role | seed |
|---|---|---|
| `generator` (main model) | student, trainable, causal | CD `model.pt` |
| `generator_ema` | frozen EMA, causal | CD `model.pt` |
| `real_score` | frozen teacher, **bidirectional** | SFT `model.pt` |
| `fake_score` | trainable critic, **bidirectional** | SFT `model.pt` |

The cluster script does a two-step "rank-0 export + all-node FS barrier" preprocessing for this.
**Locally you don't need the barrier** — first export both `.pt` (§3.1), then launch a single
torchrun:

```bash
# Prerequisite: both source weights are already exported as single files (idempotent, skipped if they exist)
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_cd_$DS/ckpts/checkpoint_6000" \
    --output     $CKPT_ROOT/stage2_ar_cd/model.pt
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_sft_$DS/ckpts/checkpoint_15000" \
    --output     $CKPT_ROOT/stage0_bi_sft/model.pt

CD_PT=$CKPT_ROOT/stage2_ar_cd/model.pt
SFT_PT=$CKPT_ROOT/stage0_bi_sft/model.pt

torchrun --nproc_per_node="$NPROC" --master_port=29706 \
    tools/train_mwm.py \
    --config-file configs/wan21/action2v/train/stage3_ar_dmd.py \
    --output-dir "outputs/wan_action2v_dmd_$DS" \
    training.sp_size=4 \
    training.max_steps=10000 \
    training.ckpt_interval=1000 \
    training.log_interval=10 \
    training.activation_offload=True \
    data.dataset.data_path="$LMDB" \
    checkpoint.pretrained="$CD_PT" \
    checkpoint.auxiliary_pretrained.generator_ema="$CD_PT" \
    checkpoint.auxiliary_pretrained.real_score="$SFT_PT" \
    checkpoint.auxiliary_pretrained.fake_score="$SFT_PT"
```

- The `checkpoint_6000` / `checkpoint_15000` above are the best steps selected from the cluster run;
  swap them per your own validation results. **When both `.pt` already exist, the export is skipped**
  (to re-export a different step, delete the `.pt` first).
- **⚠️ All four `checkpoint.*` overrides are required (verified in practice)**: the config in
  `stage3_ar_dmd.py` defaults the generator / generator_ema seed to the **ODE** weights
  (`_ODE = ./$CKPT_ROOT/stage2_ar_ode/model.pt`); starting from ODE rather than CD is another valid
  choice — in that case replace `$CD_PT` with `$CKPT_ROOT/stage2a_ar_ode/model.pt`.
- **The data is only used to take latent shape + prompt + camera as conditioning**, the clean latent
  does not enter the loss directly (self rollout, no real-video supervision).
- Hyperparameters (config defaults): generator `lr=2e-6`, critic `lr=4e-7`, `dfake_gen_update_ratio=5`
  (generator updates once every 5 steps), `guidance_scale=3.0`,
  `denoising_step_list=[1000,750,500,250]`, `generator_ema_decay=0.99`,
  `activation_checkpointing=True` (**four 1.3B copies, turning it off will OOM for sure**).
- Output: `outputs/wan_action2v_dmd_$DS/`. The DCP directory contains 4 models, about 46 GB.

After training, export → `$CKPT_ROOT/stage3_ar_dmd/model.pt`, **must use
`--model-key aux/generator_ema`** (see §3.1.2).

---

</details>

## 3. Inference pipeline

<details>
<summary><big><b>3.1 Export checkpoint (DCP directory → single-file `.pt`)</b></big></summary>

What training saves is an **FSDP2 DCP shard directory** `ckpts/checkpoint_{step}/` (`__*.distcp` +
`.metadata`), which `torch.load` cannot read and `cp` cannot "promote". `tools/export_checkpoint.py`
merges the shards, strips the FSDP / compile prefixes, and writes out a single file
`{"<model_key>": state_dict}`.

```bash
BEST_STEP=15000    # chosen by validation, must be a multiple of ckpt_interval
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_sft_$DS/ckpts/checkpoint_${BEST_STEP}" \
    --output     $CKPT_ROOT/stage0_bi_sft/model.pt \
    --format     pt
```

Single process, pure CPU, about 2 min; for 1.3B it comes out ≈5.96 GB / 885 tensors. The parent
directory of `--output` is created automatically. `--format` can be `pt` / `safetensors` /
`diffusers` (the latter needs `--config`); if not given, it is inferred from the `--output` suffix.
It also supports `s3://` / `oss://` DCP directories.


#### 3.1.1 The `--model-key` per stage

| stage | `--model-key` | target path |
|---|---|---|
| SFT | `model` (default) | `$CKPT_ROOT/stage0_bi_sft/model.pt` |
| TF | `model` (default) | `$CKPT_ROOT/stage1_ar_tf/model.pt` |
| ODE | `model` (default) | `$CKPT_ROOT/stage2a_ar_ode/model.pt` |
| CD | `model` (default) | `$CKPT_ROOT/stage2_ar_cd/model.pt` |
| **DMD** | **`aux/generator_ema`** | `$CKPT_ROOT/stage3_ar_dmd/model.pt` |

SFT / TF / ODE are all single-model recipes (no EMA), so the default key is correct. CD's main model
is also `model`.

> **⚠️ The ODE path is inconsistent (verified in practice)**: the default `inference.checkpoint` in
> `infer/stage2_ar_ode.py` writes `./$CKPT_ROOT/stage2_ar_ode/model.pt`, while this document / the
> existing outputs use **`stage2a_ar_ode/`** (the `stage2_ar_ode/` directory does not exist on
> disk). So ODE inference **must explicitly give `--checkpoint`** (§3.2.4 already writes it this
> way), otherwise it will look for a non-existent default path. To skip this flag, make a symlink:
> `ln -sfn "$PWD/$CKPT_ROOT/stage2a_ar_ode" $CKPT_ROOT/stage2_ar_ode`.

#### 3.1.2 DMD must export EMA

```bash
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_dmd_$DS/ckpts/checkpoint_1000" \
    --output     $CKPT_ROOT/stage3_ar_dmd/model.pt \
    --model-key  aux/generator_ema
```

The DMD checkpoint has four models, and inference uses the **EMA generator**; using the default
`--model-key model` would export the bare student. The DMD inference config sets `prefer_ema=True`
to match.

#### 3.1.3 The dual use of the single-file weights

`$CKPT_ROOT/{stage}/model.pt` is both **this stage's inference `inference.checkpoint`** and **the
next stage's training `checkpoint.pretrained`** — exported once, shared by both.

#### 3.1.4 Export-while-training + auto inference (optional)

`tools/auto_dump.py` polls `ckpts/`, waits for the DCP `.metadata` to appear (DCP writes it **last**,
so it is the "fully saved" signal, and it will never merge a half-finished one), then merges
automatically; `tools/auto_sample.py` then runs inference automatically. **Locally you must add
`--local`** (otherwise it uses `<launch-tool> submit` to submit a cluster job; `<launch-tool>` is a
placeholder for the cluster submission tool, pointed at your own tool via `MINWM_LAUNCH_TOOL`, see
`tools/_cluster.py`):

```bash
# Local in-process merge (uses RAM ≥ model size)
python tools/auto_dump.py \
    --output-dir "outputs/wan_action2v_sft_$DS" \
    --ckpt-steps 10000 15000 20000 --format safetensors --local

# DMD needs the EMA weights
python tools/auto_dump.py \
    --output-dir "outputs/wan_action2v_dmd_$DS" \
    --ckpt-steps 1000 --model-key aux/generator_ema --format pt --local

# Auto inference (prompts come from the config's inference.benchmark; --benchmark just overrides it)
python tools/auto_sample.py \
    --output-dir "outputs/wan_action2v_sft_$DS" \
    --ckpt-steps 10000 15000 \
    --config-file configs/wan21/action2v/infer/stage0_bi_sft.py \
    --benchmark "$BENCH" \
    --local
```

Samples land in `<output-dir>/<sample-name>/{step}-{exp_name}/`. This pipeline only automates
"export + inference"; **the key overlay (§4.1) still has to be run separately**. See
[`docs/auto-pipeline.md`](../../../docs/auto-pipeline.md) for details.

---

</details>

<details>
<summary><big><b>3.2 Inference</b></big></summary>

#### 3.2.1 Unified entry point (config + benchmark JSON driven)

The five stages share `tools/infer_mwm.py`, and loop / sampler / guidance / steps are **entirely
determined by `--config-file`**, so switching stages only requires switching config + checkpoint.
The command line has **only** `--config-file` plus a trailing dotlist (`inference.*=` / `model.*=`)
— value flags (`--checkpoint` / `--output-dir` / `--prompts` / `--input-json` / `--limit` …) have
all been removed.

The input is a benchmark JSON (`inference.benchmark`, format `[{id, caption, trajectory}]`), each
entry producing one `{id}.mp4`; if the `image` field is present it's i2v (HY), if absent it's t2v
(Wan).

```bash
# First set these three per stage
CFG=configs/wan21/action2v/infer/stage0_bi_sft.py    # that stage's infer config
PT="$CKPT_ROOT/stage0_bi_sft/model.pt"              # exported single-file weights
OUT="outputs/infer_sft_$DS/15000"                   # output directory, 15000 is just an example

torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file "$CFG" \
    inference.checkpoint="$PT" \
    inference.benchmark="$BENCH" \
    inference.output_dir="$OUT" \
    inference.num_inference_steps=50
```

`inference.benchmark` already has a default value in each stage config (the 50-entry official
benchmark), so if you just want to run the default evaluation that line can be omitted.

Common overrides: `inference.limit=N` (only run the first N entries, for smoke tests),
`inference.seed=`, `inference.dtype=`, `inference.prefer_ema=True`, `inference.sp_size=N` (pair with
`--nproc_per_node=N` to enable sequence parallelism).

**Config differences per stage** (all under `configs/wan21/action2v/infer/`):

| stage | config | `inference.loop` | `sampler.solver` | guidance | steps |
|---|---|---|---|---|---|
| SFT | `stage0_bi_sft.py` | `BidirectionalGenerationLoop` | `UniPCSolver` | 8.0 | 50 |
| TF | `stage1_ar_tf.py` | `ARGenerationLoop` | `UniPCSolver` | 3.0 | 50 |
| ODE | `stage2_ar_ode.py` | `ARGenerationLoop` | `CMSolver` | 1.0 | 4 |
| CD | `stage2_ar_cd.py` | `ARGenerationLoop` | `CMSolver` | 1.0 | 4 |
| DMD | `stage3_ar_dmd.py` | `ARGenerationLoop` | `CMSolver` | 1.0 | 4 |

- `loop` / `solver` are both **class-name strings**, resolved by name in code (`ARGenerationLoop` in
  `minwm/engine/inference/loop.py`, `CMSolver` in `.../samplers.py`), so switching loop / switching
  sampler is all in the config; the code has no `if pipeline == ...` branch.
- SFT/TF use 50-step flow-UniPC. **TF is an AR loop but still 50 steps** — few-step=4 only applies to
  ODE/CD/DMD, don't confuse them.
- The `CMSolver` for ODE/CD/DMD uses `denoising_step_list=[1000,750,500,250]`, so whatever you pass
  for `inference.num_inference_steps` is harmless (the actual number of steps is determined by
  `denoising_step_list`).
- Common defaults: `num_frames=20` (→ 77 pixel frames), `latent_shape=(16,60,104)` → 832×480,
  `fps=16`, `seed=0`, `sp_size=1`. **The trajectory is per-entry**, coming from the benchmark item's
  `trajectory` field (there is no longer a global `inference.trajectory` knob).
- checkpoint: `checkpoint_key="auto"` + `prefer_ema=True`, searching in order `generator_ema` →
  `generator` → `model`, so the SFT/TF/ODE/CD exports containing only `model` also load fine.

**The three fixed inference steps**: ① export DCP → single `.pt` (§3.1, idempotent) ②
`infer_mwm.py` sampling ③ `overlay_from_manifest.py` key overlay + concatenation (§4.1, pure CPU).
All single-GPU.

#### 3.2.2 SFT inference (50-step bidirectional)

```bash
STEP=15000
OUT="outputs/infer_sft_$DS/${STEP}"

# ① Export (skipped if it exists; to re-export a different step, delete model.pt first)
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_sft_$DS/ckpts/checkpoint_${STEP}" \
    --output     $CKPT_ROOT/stage0_bi_sft/model.pt --format pt

# ② Sampling
mkdir -p "$OUT"
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/wan21/action2v/infer/stage0_bi_sft.py \
    inference.checkpoint=$CKPT_ROOT/stage0_bi_sft/model.pt \
    inference.benchmark="$BENCH" \
    inference.output_dir="$OUT" \
    inference.num_inference_steps=50

# ③ Key overlay + concatenation (see §4.1)
python demos/overlay_from_manifest.py --input-dir "$OUT" --output final_with_keys_${STEP}.mp4
```

Smoke (1 entry): add `inference.limit=1` in ②.

#### 3.2.3 TF inference (causal AR, still 50 steps)

Isomorphic to 3.2.2, only swapping config / DCP directory / output directory:

```bash
STEP=7500
OUT="outputs/infer_tf_$DS/${STEP}"

python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_tf_$DS/ckpts/checkpoint_${STEP}" \
    --output     $CKPT_ROOT/stage1_ar_tf/model.pt --format pt

mkdir -p "$OUT"
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/wan21/action2v/infer/stage1_ar_tf.py \
    inference.checkpoint=$CKPT_ROOT/stage1_ar_tf/model.pt \
    inference.benchmark="$BENCH" \
    inference.output_dir="$OUT" \
    inference.num_inference_steps=50

python demos/overlay_from_manifest.py --input-dir "$OUT" --output final_with_keys_${STEP}.mp4
```

#### 3.2.4 few-step inference (ODE / CD / DMD — the same code path)

All three **belong to the same `few_step` + `causal_ar` code path**, differing only in config and
weights, so don't write a separate one for each:

```bash
# --- CD ---
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_cd_$DS/ckpts/checkpoint_6000" \
    --output     $CKPT_ROOT/stage2_ar_cd/model.pt
CFG=configs/wan21/action2v/infer/stage2_ar_cd.py
PT=$CKPT_ROOT/stage2_ar_cd/model.pt
OUT="outputs/infer_cd_$DS/cd_bench50"

# --- ODE (just change these three lines)---
# CFG=configs/wan21/action2v/infer/stage2_ar_ode.py
# PT=$CKPT_ROOT/stage2_ar_ode/model.pt
# OUT="outputs/infer_ode_$DS/6000_bench50"

# --- DMD (note: export uses --model-key aux/generator_ema, see §3.1.2)---
# CFG=configs/wan21/action2v/infer/stage3_ar_dmd.py
# PT=$CKPT_ROOT/stage3_ar_dmd/model.pt
# OUT="outputs/infer_dmd_$DS/1000_bench50"

mkdir -p "$OUT"
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file "$CFG" \
    inference.checkpoint="$PT" \
    inference.benchmark="$BENCH" \
    inference.output_dir="$OUT" \
    inference.num_inference_steps=4

python demos/overlay_from_manifest.py --input-dir "$OUT" --output "final_with_keys_$(basename "$OUT").mp4"
```

- **If there is no `$BENCH`** (left empty in §0.1), delete the `inference.benchmark=` line and use
  the default benchmark bundled with the config (the 50-entry official benchmark).
- **To specify which GPU**: prefix with `CUDA_VISIBLE_DEVICES=<id>` (inference is single-process, and
  when multiple GPUs are idle you can run several stages in parallel, each on one GPU).
- Time scale: 50 entries, ~20 min on a single GPU.

#### 3.2.5 benchmark JSON (the only input format)

Trajectory string semantics: `token*n` segments joined with commas, e.g. `d*8,i*5,l*6` = move right
8 + pitch up 5 + move left 6, **19 action segments total for 20 latent frames**. The tokens are WASD
(translation) + KIJL (rotate view).

benchmark JSON (`inference.benchmark`, the `$BENCH` above): a JSON array, each entry
`{id, caption, trajectory}` (`image` optional). Just build one yourself:

```json
[
  {"id": "0001", "caption": "a drone shot over a forest lake", "trajectory": "d*8,i*5,l*6"},
  {"id": "0002", "caption": "walking through a narrow alley at dusk", "trajectory": "w*19"}
]
```

- `id` determines the output filename `{id}.mp4`, so it must be unique.
- `trajectory` follows the same segment semantics as above, and **the sum of segments should be 19**
  (aligned to 20 latent frames; the key overlay in §4.1 converts based on this).
- The `image` field is **silently ignored** during Wan21 inference (Wan's preprocessor doesn't use
  it), degrading to pure T2V without error — so the i2v benchmark files from the HY side can be
  reused directly.
- If you want a fixed evaluation set, maintain a few size tiers yourself (1 entry smoke / 20 entry
  small batch / 50 entry standard / 200 entry large batch) and switch with `$BENCH`. If your team
  already has a shared benchmark, ask where it is, `chmod o+r` it, then fill it into `$BENCH`.

**Don't overwrite old outputs**: give the output directory a purpose suffix (e.g. `_bench50` /
`_smoke`), see the `OUT=` in §3.2.4.

#### 3.2.6 Output directory contract

Under `outputs/infer_{stage}_$DS/{step}/`:

- One mp4 per entry: **832×480 / 77 frames / 16 fps**, filename `{id}.mp4` (`id` comes from the
  benchmark item).
- `manifest.json` — `{inference: <config>, items: [{index, id, prompt, trajectory, seed, video}]}`.
- `final_with_keys_{...}.mp4` — the concatenated overview with WASD/KIJL overlaid (§4.1).
- `overlay_config.json` — per entry `{input, trajectory, sequence}` (written by §4.1).

---

</details>

## 4. Auxiliary tools and operations

<details>
<summary><big><b>4.1 Key overlay + concatenation (step ③)</b></big></summary>

Reads each `trajectory` from `manifest.json`, converts the per-segment pixel frame counts, overlays
a WASD/KIJL indicator on each mp4, then concatenates them into one overview video. Pure CPU, no GPU.

```bash
python demos/overlay_from_manifest.py \
    --input-dir "outputs/infer_sft_$DS/15000" \
    --output    final_with_keys_15000.mp4
```

Outputs land under `--input-dir`: `final_with_keys_*.mp4` + `overlay_config.json`. `--keep-temp`
keeps the intermediate per-segment overlaid files.

- **Depends on ffmpeg / ffprobe + PIL + numpy**. The local minwm environment has `/usr/bin/ffmpeg`,
  so this step runs locally; the cluster image **does not** have ffmpeg, so running it there is bound
  to fail (which is also the original reason it was split off into an independent step ③).
- **It reads the manifest's `trajectory` field directly, without parsing filenames**, so it also
  works for the concise `{id}.mp4` naming. (The `demos/batch_overlay.py` in the same directory is the
  old version that relies on parsing filenames, which doesn't work for `{id}.mp4`.)
- **Frame count conversion**: the model produces 20 latent frames, which the VAE temporally decodes
  into **77 pixel frames** (first frame 1 frame, the rest 4 frames per latent frame). The 19 action
  segments of the trajectory are allocated pixel frames by segment position: first segment
  `1+4*(n-1)`, last segment `4+4*n`, middle segment `4*n`, single segment `1+4*(n-1)+4`, and the
  segment sums always total 77.
- The sum of trajectory segments should be 19. bench50 has been verified: all 50 entries are 832×480
  / 77 frames / 16fps, with segment sums of 19.

---

</details>

<details>
<summary><big><b>4.2 Resume training and re-runs</b></big></summary>

#### 4.2.1 RESUME (continue after a crash / manual kill)

`ckpts/checkpoint_{step}/` is the **complete DCP training state** (model + optimizers + aux + step,
not just the weights), and `ckpts/latest.txt` points at the latest. In the trainer,
**`checkpoint.resume=True` takes precedence over `checkpoint.pretrained`**: resume restores the full
training state, continues from the checkpoint step, then `return`s and skips pretrained. Just replace
`checkpoint.pretrained=...` with `checkpoint.resume=True` (**don't give both at once**, it's easy to
misread):

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29702 \
    tools/train_mwm.py \
    --config-file configs/wan21/action2v/train/stage1_ar_tf.py \
    --output-dir "outputs/wan_action2v_tf_$DS" \
    training.sp_size=4 \
    training.max_steps=20000 \
    training.ckpt_interval=2500 \
    data.dataset.data_path="$LMDB" \
    checkpoint.resume=True
```

(`tools/train_mwm.py` also has an equivalent `--resume` flag.)

**Verify it really resumed**: the log shows `loaded checkpoint .../checkpoint_{step}`, and the step
of the first monitor line is **the next log_interval after the checkpoint** (e.g. resuming from 2500
→ first line `step 2510`, **not step 10**). When `latest.txt` cannot be found it **errors loudly**,
it will not silently fall back to fresh init / pretrained — to avoid starting training from wrong
weights.

Other hyperparameters (`max_steps` / `sp_size` / `ckpt_interval`) stay unchanged when resuming.

#### 4.2.2 The overwrite trap of retraining from scratch

Saving uses `checkpoint_{step}` naming + rewrites `latest.txt`. **Reusing an existing run's
`--output-dir` to rerun from step 0 will silently overwrite the old `checkpoint_2500` and rewrite
`latest.txt` when the step climbs to 2500.**


```bash
# Recommended: use a new output-dir, zero risk, doesn't touch the old run (pick a suffix yourself, e.g. _v2 / _lr2e6)
torchrun --nproc_per_node="$NPROC" ... \
    --output-dir "outputs/wan_action2v_tf_${DS}_v2"

# Or switch to a different DS (isolates the data outputs too, see §0.4)
```

To overwrite the old run in place, first confirm yourself that that ckpt can really be discarded:
`ls "outputs/wan_action2v_tf_$DS/ckpts"` to see clearly before `rm -rf`ing that `ckpts` directory.

> For TF/ODE/CD/DMD, "from scratch" = starting from **step 0** of the previous stage's `pretrained`
> (true random initialization is meaningless for a causal model, you must first convert from the
> upstream weights).

---

</details>

<details>
<summary><big><b>4.3 Full-pipeline serial quick reference</b></big></summary>

Just run through them in order (each stage's `<BEST_STEP>` is chosen by your own validation and must
be a multiple of `ckpt_interval`):

First source the config block in §0.1, then advance in order (each stage's `BEST_STEP` is chosen by
your own validation and must be a multiple of that stage's `ckpt_interval`):

```bash
# Do this once in every new shell (§0.1)
source /path/to/your/env.sh        # or just paste the §0.1 block in

# ① encode                                      → $LMDB                        (§1)
# ② SFT training                                → export $CKPT_ROOT/stage0_bi_sft/model.pt   (§2.1, §3.1)
# ③ TF  training (pretrained=stage0_bi_sft)     → export $CKPT_ROOT/stage1_ar_tf/model.pt    (§2.2, §3.1)
# ④ ODE curation: sampling + merge              → $ODE_LMDB                    (§2.3)
# ⑤ ODE training (pretrained=stage1, data=$ODE_LMDB) → export $CKPT_ROOT/stage2a_ar_ode/model.pt  (§2.4, §3.1)
# ⑥ CD  training (pretrained=stage1, data=$LMDB)     → export $CKPT_ROOT/stage2_ar_cd/model.pt    (§2.5, §3.1)
# ⑦ DMD training (CD seeds generator/ema + SFT seeds real/fake_score)
#                                                → export $CKPT_ROOT/stage3_ar_dmd/model.pt
#                                                  --model-key aux/generator_ema            (§2.6, §3.1.2)
# ⑧ Per-stage inference + key overlay                                          (§3.2, §4.1)
```

④⑤ and ⑥ depend only on ③, independent of each other; ⑦ needs both the ⑥ and ② weights (see the
stage chain diagram at the top). Each stage's `--master_port` is staggered (29700–29706), so running
multiple stages at once doesn't collide on ports.

---

</details>

<details>
<summary><big><b>4.4 Troubleshooting</b></big></summary>

| Symptom | Cause / handling |
|---|---|
| `unrecognized arguments: --checkpoint / --prompts / --input-json ...` | You copied the old flag style. `infer_mwm.py` now **only accepts** `--config-file` + trailing dotlist (`inference.checkpoint=` / `inference.benchmark=` / `inference.output_dir=`), see §3.2.1. |
| `unknown key(s) '...' in Inference` | There is a misspelled key in the `inference` block. That block is a strict schema (`minwm/config/schema.py:Inference`); a wrong key errors directly instead of silently using the default; fix it against the legal key names listed in the error. |
| `blocks.*.prope_o.* newly initialized` + `You should probably TRAIN this model` | **Normal**. The PRoPE camera parameters don't exist in the base and are newly initialized. The same batch of warnings is also printed when loading the exported weights for inference. |
| DCP directory can't be read by `torch.load` | A shard directory is not a single file, you must merge it first with `tools/export_checkpoint.py` (§3.1). |
| DMD inference results clearly wrong | You forgot `--model-key aux/generator_ema` during export and exported the bare student (§3.1.2). |
| ODE curation fails to load base / hangs | The proxy wasn't unset. `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` (§0.1); and it must be run from the repo root, with the `wan_models/Wan2.1-T2V-1.3B` symlink present. |
| ODE curation flash-attn kernel error | Needs an sm_80 or higher kernel; found missing on B200 in practice (§2.3.1). |
| ODE training reports a data key not found | The data path was given as `$LMDB`. ODE must use `$ODE_LMDB` (§2.4). |
| Something like `world_size not divisible by sp_size` won't start | `$NPROC` is not divisible by `training.sp_size`. Adjust `sp_size` per the table in §0.3. |
| An empty variable makes the path become `dataset/Wan21/Action2V_/data` | You didn't source the §0.1 config block (a new shell / new terminal must re-source it). Self-check with `echo "$DS $LMDB"`. |
| DMD OOM | Don't turn off `gradient_checkpointing` (four 1.3B copies). You can also increase `training.sp_size` to spread out the activations. |
| tqdm progress bar stuck on the first frame | When redirected to a file the `\r` frames don't land on disk, which **does not mean it's stuck**. Look at the `log_interval` print lines, or count the files on disk (`.rank_*` shards / `$ODE_LATENTS/*.pt`). |
| Training silently overwrote an old checkpoint | You reused the same `--output-dir` to run from scratch (§4.2.2). Use a new directory or switch `$DS`. |
| Switched datasets but the weights got overwritten | The export weights path doesn't include `$DS` (the reminder in §0.4). Add a suffix or switch `$CKPT_ROOT`. |
| Key overlay reports ffmpeg not found | This step needs ffmpeg/ffprobe, install it (`conda install -c conda-forge ffmpeg` or your system package manager). |
| Out of disk | First `df -h .`. Magnitude reference: ODE LMDB ≈46 GB, DMD DCP ≈46 GB/step, exported single weights ≈6 GB. |

---

</details>

<details>
<summary><big><b>4.5 References</b></big></summary>

- Per-stage configs: [`train/`](train/) + [`infer/`](infer/) (`stage0_bi_sft` / `stage1_ar_tf` /
  `stage2_ar_ode` / `stage2_ar_cd` / `stage3_ar_dmd`).
- Data tools and key contract: [`tools/data/wan21/README.md`](../../../tools/data/wan21/README.md).
- Auto export / auto sample: [`docs/auto-pipeline.md`](../../../docs/auto-pipeline.md).
- Installation and model-line overview: [`configs/wan21/README.md`](../README.md);
  HunyuanVideo backbone: [`configs/hy/`](../../hy/README.md).
- Wan's CFG negative prompt is in the code (`DEFAULT_NEGATIVE_PROMPT`,
  `minwm/modeling/wan21/adapter.py`), referenced by the configs that need it, and **does not need
  `.pt` pre-encoding**.

</details>

