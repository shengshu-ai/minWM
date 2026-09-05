# HY Action2V Training

> 中文版 / Chinese: [README_cn.md](README_cn.md)

Data encode → SFT → TF → ODE → CD → DMD, full-pipeline training (FSDP2 + sequence parallelism, single node with 8 GPUs by default).
This document assumes you have already prepared the base models per §0.2 and placed the **raw video data** at the default relative paths (or provide your own paths by editing the two lines in §0.1)—encoding is done in §1, and its products are the training data read by all subsequent stages.


**Stage chain** (each segment's product is the next segment's input):

```
encode ─→ SFT ─→ TF ─┬─→ ODE
                     └─→ CD ─→ DMD
```

Both `ODE` and `CD` start from the TF weights and are independent of each other; `DMD` needs the CD (generator) and SFT
(real/fake score) weights. After each segment finishes training, export its DCP directory into a diffusers directory (§3.1), which is both the
**checkpoint for this segment's inference** and the **`_from_pretrained` for the next segment's training**—so you must advance serially per the diagram above.

---

## 0. Prerequisites

<details>
<summary><big><b>0.1 ⭐ Config block (change this first; all later commands reference it)</b></big></summary>

**Every command later in this document assumes you have already sourced this block.** Only the repo path and raw data path need to be changed for your
environment; everything else uses variables, so copy them as-is. It is recommended to save this as `env.sh` outside the repo and `source` it once per new shell.

```bash
# ---- (1) Repo and Python environment -------------------------------------------------
cd /path/to/minWM                      # ← change to your checkout path
export PROJECT_ROOT="$PWD"
conda activate minwm                   # ← see INSTALL.md
pip install -e .                       # after editable install, import minwm works, no PYTHONPATH needed

# ---- (2) Weights --------------------------------------------------------------
export HYCKPT="./ckpts/HunyuanVideo-1.5"                     # HY1.5 base + VAE/encoders (used for encoding + all stages)
export BASE="$HYCKPT/transformer/480p_i2v"                   # base transformer that SFT initializes from
export CKPT_ROOT="./ckpts/HY15/Action2V"                     # root for each stage's diffusers weights

# ---- (3) Raw video data (before encoding; change these two lines for your own data)--------------------------
export SRC_JSON="./dataset/preencode_input.json"            # raw caption + pose_str (§1)
export SRC_VIDEOS="./dataset/videos"                        # raw video root directory (§1)
export NEG_PROMPT="./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"      # CFG negative embedding
export NEG_BYT5="./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"
export BENCH="assets/example.json"                           # benchmark JSON for evaluation (§3.2)

# ---- (4) Topology and runtime environment --------------------------------------------------
export NPROC=8                         # number of GPUs (single node with 8 GPUs by default)
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True         # HY memory fragmentation is significant, don't skip
```

> Set `$NPROC` to your actual GPU count (`nvidia-smi -L | wc -l`). **The GPU count must be divisible by `training.sp_size`**,
> otherwise it won't start—see §0.4.
>
> **The path variables for post-encoding / ODE data are exported at their respective stages**: `$INDEX` (encoding product) in §1,
> `$ODE_INDEX` (ODE preprocessing product) in §2.3, given alongside that step.

</details>

<details>
<summary><big><b>0.2 Base models</b></big></summary>

HY training and inference need HunyuanVideo 1.5's VAE, scheduler, transformer, plus separate text and vision encoders:

```bash
hf download tencent/HunyuanVideo-1.5 --local-dir "$HYCKPT" \
    --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$HYCKPT/text_encoder/llm"
hf download google/byt5-small --local-dir "$HYCKPT/text_encoder/byt5-small"
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir "$HYCKPT/text_encoder/Glyph-SDXL-v2"
hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir "$HYCKPT/vision_encoder/siglip" --token <your_hf_token>
```

The `$HYCKPT` above defaults to `./ckpts/HunyuanVideo-1.5`, matching the paths in each config; if the models are stored
elsewhere, just modify `$HYCKPT` / `$BASE` in §0.1, or override the corresponding `_from_pretrained` on the command line.

</details>

<details>
<summary><big><b>0.3 Default path conventions</b></big></summary>

The default paths hardcoded in each stage config are all **repo-internal relative paths**, corresponding one-to-one with this document's variables. Put things in the right place, and
the `$INDEX` / `$NEG_*` / `$CKPT_ROOT/...` in the commands will hit directly:

| Role | Default relative path | Variable | When produced |
|---|---|---|---|
| HY1.5 base + encoders | `./ckpts/HunyuanVideo-1.5/` | `$HYCKPT` / `$BASE` | download |
| each stage's weights | `./ckpts/HY15/Action2V/{stage}/` (diffusers directory) | `$CKPT_ROOT/{stage}` | exported after each segment finishes training (§3.1) |
| raw video + JSON | `./dataset/{videos, preencode_input.json}` | `$SRC_VIDEOS` / `$SRC_JSON` | you provide |
| post-encoding training data | `./dataset/HY15/Action2V/{latents, train_index.json}` | `$INDEX` (§1) | §1 encoding |
| ODE data | `./dataset/HY15/Action2V_ode/{latents, train_index.json}` | `$ODE_INDEX` (§2.3) | §2.3 preprocessing |
| CFG negative embedding | `./dataset/others/HY/Action2V/*.pt` | `$NEG_PROMPT` / `$NEG_BYT5` | §1 generated or downloaded |

`{stage}` ∈ `stage0_bi_sft` / `stage1_ar_tf` / `stage2_ar_ode` / `stage2_ar_cd` / `stage3_ar_dmd`.

> **When weights come from HF the names don't match**: the HF release names are `bidirectional` / `ar_diffusion_tf` / … ,
> while the config loads `stage{N}_*`. Just create a symlink layer per the commands in each training subsection; this document always uses the config names.

</details>

<details>
<summary><big><b>0.4 Topology and parallelism (choose `sp_size` by your GPU count)</b></big></summary>

`$NPROC = sp_size × DP`, `GBS = DP × data.batch_size`. There is only one constraint: **`$NPROC` must be divisible by
`sp_size`**, otherwise it won't start.

The commands in this document uniformly use `training.sp_size=2` (single node with 8 GPUs by default → DP=4). The smaller `sp_size`, the larger DP/GBS and
the higher the memory pressure; the larger it is, the more finely each sample is sliced, saving memory but incurring more communication. See §4.2 for each stage's memory pressure.

</details>

<details>
<summary><big><b>0.5 Monitoring</b></big></summary>

**wandb is not used by default**: metrics go only to STDOUT. Logs are in `outputs/<run>/logs/log.txt` (rank0) +
`log.txt.rank{N}`, with `log_interval=10` writing one line every 10 steps (loss + steps/sec + ms/step + peak memory);
the DMD line is `generator_forward/backward` + `critic_forward/backward` + `generator_loss` /
`critic_loss`, four groups. If you want wandb, configure it yourself, **keeping the key only in the shell / a file outside the repo, never writing it into any file that goes into
git**:

```bash
export WANDB_API_KEY='your key'
# append to the tail of the training command (choose your own run name):
#   'monitor.backends=["wandb"]' monitor.wandb_project=<project> \
#   monitor.wandb_run_name=<run-name>
```

</details>

---

## 1. Data encode (raw video → latent + train_index)

<details>
<summary><big><b>1.1 Data encoding</b></big></summary>

Encode raw videos + caption + `pose_str` into `.pt` latent shards read directly by each training stage
(VAE + SigLIP vision + LLM text + byT5 glyph), where each rank writes its own shard and then rank 0 merges out
`train_index.json`. Uses the WorldPlay distillation video pipeline (camera poses synthesized from the `pose_str` DSL):

```bash
export INDEX="./dataset/HY15/Action2V/train_index.json"    # ← product of this step, read later by SFT/TF/CD/DMD

torchrun --nproc_per_node="$NPROC" \
    tools/data/hy/preencode_generated_wdplay.py \
    --input_json "$SRC_JSON" \
    --video_root "$SRC_VIDEOS" \
    --output_dir ./dataset/HY15/Action2V \
    --hunyuan_checkpoint_path "$HYCKPT" \
    --skip_existing
```

- Products: `./dataset/HY15/Action2V/{latents/*.pt, train_index.json}` (= `$INDEX`). The `.pt` key convention
  is in [`tools/data/hy/README.md`](../../../tools/data/hy/README.md) (`latent (1,32,20,H,W)` +
  `prompt_embeds` + `vision_states` + `byt5_*` + camera `intrinsics`/`poses`/`camera_indices`).
- If you have your own ready camera poses `.npy` (not the `pose_str` DSL), switch to `preencode_camera_video.py`,
  whose inputs and output contract are the same as above; see that README for details.

**Input layout** (raw data, before encoding)—each `$SRC_JSON` entry contains `caption` / `pose_str` (the first frame is provided by the video
`gen.mp4` frame 0, with no `image_path` in the JSON); videos are named `<idx:06d>_<suffix>`, where `suffix` is
`pose_str` with whitespace/commas removed and dashes deleted (`right-8, a-11` → `right8a11`):

```
./dataset/
├── preencode_input.json                # = $SRC_JSON
└── videos/                             # = $SRC_VIDEOS
    ├── 000000_right8a11/gen.mp4
    ├── 000001_w10d9/gen.mp4
    └── ...
```

</details>

<details>
<summary><big><b>1.2 CFG negative embedding</b></big></summary>

**CFG negative embedding** (generated once per HY checkpoint, shared by all stages):

```bash
# You can also directly download the officially provided embedding
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/Action2V/**"

# If you don't download the ready file, generate it using the local HY base model
python tools/data/hy/generate_negative_prompts.py \
    --hunyuan_checkpoint_path "$HYCKPT" \
    --output_dir ./dataset/others/HY/Action2V     # → $NEG_PROMPT / $NEG_BYT5
```

> If you already downloaded `others/HY/Action2V/*.pt` from HF (main README), skip this step.

**Product layout** (after encoding; the `.pt` key convention is in [`tools/data/hy/README.md`](../../../tools/data/hy/README.md):
`latent (1,32,20,H,W)` + `prompt_embeds` + `vision_states` + `byt5_*` + camera
`intrinsics`/`poses`/`camera_indices`):

```
./dataset/
├── HY15/Action2V/                      # encoding product
│   ├── latents/                        # one .pt per video
│   └── train_index.json                # = $INDEX, read by SFT/TF/CD/DMD
└── others/HY/Action2V/                 # CFG negative embedding
    ├── hunyuan_neg_prompt.pt           # = $NEG_PROMPT
    ├── hunyuan_neg_byt5_prompt.pt      # = $NEG_BYT5
    └── negative_prompt.pt
```

</details>

---

## 2. Training pipeline

<details>
<summary><big><b>2.1 Phase-1 Bidirectional SFT (`stage0_bi_sft`)</b></big></summary>

Bidirectional + camera (PRoPE) supervised fine-tuning. `ARHunyuanVideo_1_5_DiffusionTransformer` + `BiSFTRecipe` +
`FlowMatchingLoss`, reading the `$INDEX` encoded in §1 (clean-latent camera data, viewmats / Ks).

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29635 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage0_bi_sft.py \
    --output-dir outputs/hy_action2v_sft \
    training.sp_size=2
```

- Initialization: config defaults to `_from_pretrained=$BASE` (`./ckpts/HunyuanVideo-1.5/transformer/480p_i2v`)
  —SFT is the start of the chain, starting from base. **No `checkpoint.pretrained`**.
- Hyperparameters (config defaults): Muon `lr=2e-5`, `weight_decay=1e-4`, `timestep_shift=3.0`,
  `schedule=linear`, `logit_normal` timestep sampling, `window_frames=20`, `batch_size=1`,
  `activation_checkpointing=True`.
- Steps (config defaults): `max_steps=100000` / `ckpt_interval=1000`; override as needed, e.g.
  `training.max_steps=... training.ckpt_interval=...`.
- Output: `outputs/hy_action2v_sft/{ckpts/checkpoint_{step}/, logs/}`, with `ckpts/latest.txt` pointing to
  the latest. **The checkpoint is an FSDP2 DCP shard directory, not a single `.pt`** (see §3.1).

**Normal warning at the first step**: a large batch of `double_blocks.*.img_attn_prope_proj.* newly initialized` +
`You should probably TRAIN this model...`—the PRoPE camera parameters don't exist in base and are newly initialized,
**as expected**.

After training, export (§3.1) → `$CKPT_ROOT/stage0_bi_sft/`. It is both the checkpoint for SFT inference,
and the **starting point for TF training** + the **real/fake score seed for DMD**.

---

</details>

<details>
<summary><big><b>2.2 Phase-2 Stage-1 Teacher-Forcing AR (`stage1_ar_tf`)</b></big></summary>

**If you skip the previous stage, you can download the officially provided previous-stage checkpoint:**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/bidirectional/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT bidirectional stage0_bi_sft
)
```

After downloading, the checkpoint is at `./ckpts/HY15/Action2V/bidirectional/`, and the symlink above maps it to
the `$CKPT_ROOT/stage0_bi_sft/` path used in this document; you can also explicitly override
`model._from_pretrained` on the command line.

Convert the bidirectional SFT model into causal + teacher forcing AR diffusion. The same
`ARHunyuanVideo_1_5_DiffusionTransformer` + `ARTFRecipe` + `FlowMatchingLoss`,
with `use_prope=True` retaining the PRoPE parameters. **Data is the same `$INDEX` as SFT**.

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29636 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage1_ar_tf.py \
    --output-dir outputs/hy_action2v_ar_tf \
    training.sp_size=2 \
    model._from_pretrained="$CKPT_ROOT/stage0_bi_sft"
```

- **Initialization must be given**: `model._from_pretrained=$CKPT_ROOT/stage0_bi_sft` (the export from §2.1). The config default
  points to base; serial training must start from the SFT weights, so this override is mandatory.
- The only difference from SFT: the recipe becomes `ARTFRecipe`, and the preprocessor chain gets one more
  `CleanContextNoiseAug(max_timestep=0)` at the tail (writing out un-noised clean context, switching the model to
  the block-causal `flex_tf` mask path of teacher-forcing).
- Hyperparameters (config defaults): Muon `lr=1e-5` (lower than SFT's 2e-5), `timestep_shift=3.0`,
  `window_frames=20`; `max_steps=200000` / `ckpt_interval=1000`.

After training, export → `$CKPT_ROOT/stage1_ar_tf/`. It is the **starting point for ODE / CD training**.

---

</details>

<details>
<summary><big><b>2.3 Stage-2(a) ODE data curation (data preparation, not training)</b></big></summary>

**TF teacher checkpoint (product of 2.2; if you skip the TF stage, you can download the officially provided checkpoint):**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/ar_diffusion_tf/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT ar_diffusion_tf stage1_ar_tf
)
```

After downloading, the checkpoint is at `./ckpts/HY15/Action2V/ar_diffusion_tf/`, and the symlink above maps it to
the `$CKPT_ROOT/stage1_ar_tf/` used by the ODE curation command below; if the path differs, you can also explicitly override
`--generator_ckpt`.

Use the frozen TF teacher to pre-solve a 6-point ODE trajectory for the SFT latents, then rebuild the index, for §2.4's
`ARODERecipe` to regression-fit the causal student. **The data is the ODE-specific `$ODE_INDEX`**.

#### 2.3.1 ODE data preprocessing (do first, a training prerequisite)

Use §2.2's TF teacher to run 48-step CFG sampling (guidance 5.0) on the SFT latents, pre-solve the 6-point trajectory,
then rebuild the index. Products go to `./dataset/HY15/Action2V_ode/`, parallel to and non-overlapping with the SFT latents:

```bash
export ODE_INDEX="./dataset/HY15/Action2V_ode/train_index.json"   # ← product of this step, read by §2.4 training

# 1) 48-step CFG sampling (guidance 5.0)—the heaviest step
torchrun --nproc_per_node="$NPROC" --master_port=29640 \
    tools/data/hy/get_causal_ode_data_prope.py \
    --generator_ckpt "$CKPT_ROOT/stage1_ar_tf" \
    --rawdata_path   ./dataset/HY15/Action2V/latents \
    --output_folder  ./dataset/HY15/Action2V_ode/latents \
    --neg_prompt     "$NEG_PROMPT" \
    --neg_byt5       "$NEG_BYT5" \
    --guidance_scale 5.0

# 2) Rebuild absolute-path index
python tools/data/create_train_index.py \
    ./dataset/HY15/Action2V_ode \
    --recursive \
    -o "$ODE_INDEX"
```

- The sampler is minwm-native (`ARHunyuanVideo_1_5_DiffusionTransformer` + `use_prope=True`), requiring
  the flash-attn kernel of the **A series (sm_80)** (A100/A800 class ✓, B200 confirmed missing).
- Sharded by rank, products self-check with shapes: `latent (1,32,20,30,52)` / `ode_trajectory (1,6,32,20,30,52)`
  + prompts/camera. This is the heaviest segment of the whole pipeline.
- HF has ready pre-generated ODE latents, so you can skip sampling and download directly (`ODE_data/HY15/Action2V/**`),
  after which you only run step 2 above to rebuild the index; see the main [`README.md`](../../../README.md) for weights / dataset downloads.

**Product layout** (ODE data is parallel to and non-overlapping with the SFT latents; the negative embedding reuses §1's):

```
./dataset/
├── HY15/
│   ├── Action2V/                       # SFT latents (§1)
│   │   ├── latents/
│   │   └── train_index.json            # = $INDEX
│   └── Action2V_ode/                   # ODE data (this step)
│       ├── latents/                    # one .pt per clip, containing ode_trajectory
│       └── train_index.json            # = $ODE_INDEX
└── others/HY/Action2V/                 # CFG negative embedding (reused)
```

</details>

<details>
<summary><big><b>2.4 Stage-2(a) Causal ODE Distillation (`stage2_ar_ode`)</b></big></summary>

Regression-fit the causal student on the 6-point trajectory pre-solved by the TF teacher. `ARODERecipe` +
`ODERegressionLoss`, with `ODETrajectorySample` sampling one denoising anchor per step
(`[1000,750,500,250]`, `warp_denoising_step=True`).

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29638 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage2_ar_ode.py \
    --output-dir outputs/hy_action2v_ar_ode \
    training.sp_size=2
```

- Initialization: config defaults to `_from_pretrained=$CKPT_ROOT/stage1_ar_tf`—**already points to the TF export, so put the weights
  in the right place and no override is needed**.
- Data: config defaults to reading `$ODE_INDEX` (`CausalODEDataset`); giving a plain `$INDEX` will report a missing
  `ode_trajectory` key.
- Hyperparameters (config defaults): Muon `lr=1e-5`; the scheduler differs from the others—`schedule="shifted"` +
  `timestep_shift=5.0` + `sigma_min=0.0` + `extra_one_step=True`. The trajectory is solved on the σ table at shift=5.0,
  and only `shifted` bakes the shift into `scheduler.timesteps`, so warp can restore the σ that each snapshot
  actually corresponds to (`linear` would mislabel 3 of the 4 anchors' timesteps). These are in the config, no override needed.
  `max_steps=10000` / `ckpt_interval=1000`.

After training, export → `$CKPT_ROOT/stage2_ar_ode/`. It is the checkpoint for ODE few-step inference,
and also an alternative starting point for DMD.

---

</details>

<details>
<summary><big><b>2.5 Stage-2(b) Causal Consistency Distillation (`stage2_ar_cd`)</b></big></summary>

Distill the causal TF model into a few-step consistency model: the frozen teacher takes one CFG Euler step `t→t_next`,
the student predicts `x0` at `t`, the EMA network predicts `x0` at `t_next`, and the loss is the MSE of the two `x0`s.
**Data returns to SFT's `$INDEX`** (no ODE preprocessing needed).

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29637 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage2_ar_cd.py \
    --output-dir outputs/hy_action2v_ar_cd \
    training.sp_size=2 \
    model._from_pretrained="$CKPT_ROOT/stage1_ar_tf" \
    auxiliary_models.teacher._from_pretrained="$CKPT_ROOT/stage1_ar_tf" \
    auxiliary_models.ema._from_pretrained="$CKPT_ROOT/stage1_ar_tf"
```

- **Initialization must be given (three copies)**: **student / teacher / ema all seeded from the same TF weight**. The config default
  points all three to base; serial training must start from the TF weights, so these three overrides are mandatory. `ARCDRecipe` at the first step uses
  `copy_params` to copy the student into teacher / ema, and when all three load from the same directory this copy is a no-op.
  teacher + ema are declared as frozen `auxiliary_models` in the config.
- Negative prompt: `recipe.adapter.neg_*` is already the config default (§1.2), so put it at the right path and no action is needed.
- Hyperparameters (config defaults): Muon `lr=1e-5`, `discrete_cd_n=50`, `timestep_shift=5.0`,
  `guidance_scale=5.0`, `ema_decay=0.999`, `activation_checkpointing=True` (three models, memory is tight,
  don't turn it off); `max_steps=200000` / `ckpt_interval=500`.

After training, export → `$CKPT_ROOT/stage2_ar_cd/` (checkpoint for CD few-step inference +
the **generator seed for DMD**).

---

</details>

<details>
<summary><big><b>2.6 Stage-3 Asymmetric DMD with Self Rollout (`stage3_ar_dmd`)</b></big></summary>

**If you skip the previous initialization stages, you can download the officially provided ODE or CD checkpoint:**

```bash
# Default ODE initialization
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/causal_ode/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT causal_ode stage2_ar_ode
)

# Or use CD initialization
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/causal_cd/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT causal_cd stage2_ar_cd
)
```

After downloading, the checkpoints are at `./ckpts/HY15/Action2V/causal_ode/` and
`./ckpts/HY15/Action2V/causal_cd/` respectively, and the symlinks above map them to `stage2_ar_ode` and
`stage2_ar_cd` respectively. DMD also needs the bidirectional SFT checkpoint as the
seed for `real_score` / `fake_score`; if you don't have it locally, download it per the commands in 2.2, or explicitly
override the corresponding `model._from_pretrained` path on the command line.

Distribution Matching Distillation: the generator self-rolls a fake video from pure noise (AR cm rollout,
single-step truncated BPTT), aligns its distribution to the CFG-guided distribution of the frozen real_score, while the trainable fake_score
critic learns the generator's distribution. **No real-video supervision**—the data is only used to take the condition. Three models:

| aux name | role | seed |
|---|---|---|
| `model` (main model) | student / generator, trainable, causal | CD |
| `real_score` | frozen teacher, **bidirectional** | SFT |
| `fake_score` | trainable critic, **bidirectional** | SFT |

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29639 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage3_ar_dmd.py \
    --output-dir outputs/hy_action2v_ar_dmd \
    training.sp_size=2 \
    model._from_pretrained="$CKPT_ROOT/stage2_ar_cd" \
    auxiliary_models.real_score._from_pretrained="$CKPT_ROOT/stage0_bi_sft" \
    auxiliary_models.fake_score._from_pretrained="$CKPT_ROOT/stage0_bi_sft"
```

- **Initialization must be given (three copies)**: generator ← CD, real/fake score ← SFT. The config default points all three to base,
  and serial training must connect this way, so these three overrides are mandatory. You can also have the generator start from ODE
  (change `model._from_pretrained` to `$CKPT_ROOT/stage2_ar_ode`).
- Negative prompt: `recipe.adapter.neg_*` is already the config default (§1.2).
- Hyperparameters (config defaults): generator Muon `lr=1e-5`, critic `lr=8e-6`, both `weight_decay=0.01` /
  `betas=(0.0,0.999)`; `dfake_gen_update_ratio=5` (generator updates once every 5 steps, critic every step),
  `guidance_scale=5.0`, `denoising_step_list=[1000,750,500,250]`, `num_frame_per_block=4`,
  `activation_checkpointing=True` (three models + rollout, **turning it off will definitely OOM**); `max_steps=200000` /
  `ckpt_interval=500`.

After training, export → `$CKPT_ROOT/stage3_ar_dmd/`, the weights for **4-step real-time inference**.
**HY stage3 has no `generator_ema`** (aux only has `real_score` / `fake_score`), so export uses the default
`--model-key model`, **not** `aux/generator_ema` (that's the Wan-line practice, §3.1).

---

</details>

## 3. Inference pipeline

<details>
<summary><big><b>3.1 Export checkpoint (DCP directory → diffusers directory)</b></big></summary>

Training saves an **FSDP2 DCP shard directory** `ckpts/checkpoint_{step}/` (one `.distcp` per GPU +
one `.metadata`), which `torch.load` cannot read. `tools/export_checkpoint.py` merges the shards, strips the
FSDP / compile prefixes, and writes a diffusers directory that the next segment's training / inference can eat directly.

```bash
BEST_STEP=3000    # selected by validation, must be a multiple of ckpt_interval
# HY inference / next-segment training use the diffusers directory (_from_pretrained), export diffusers format, need --config:
python tools/export_checkpoint.py \
    --checkpoint outputs/hy_action2v_ar_tf/ckpts/checkpoint_${BEST_STEP} \
    --output   "$CKPT_ROOT/stage1_ar_tf" \
    --format   diffusers \
    --config   configs/hy/action2v/train/stage1_ar_tf.py
```

- Single process, pure CPU, no torchrun. `--checkpoint` must point to a real, existing DCP directory.
- **DMD is also `--model-key model` (default)**: HY stage3 has no `generator_ema` (§2.6), so the main model
  `model` is the generator to export. Don't copy Wan README's `aux/generator_ema`.
- `--format` can also be `pt` / `safetensors`; HY's infer / train config uses `_from_pretrained`,
  so use `diffusers`.

The export target is a diffusers directory (`config.json` + safetensors). The 8.55B model > diffusers default
`max_shard_size=10GB`, so `save_pretrained` **automatically shards** into multiple `-0000N-of-M` + one
`.index.json`; **sharding doesn't affect loading**—`from_pretrained` (i.e. the config's `_from_pretrained`) treats
single files and shards alike. Every segment of the whole pipeline exports under `$CKPT_ROOT`, layout:

```
./ckpts/HY15/Action2V/                                    # = $CKPT_ROOT
├── stage0_bi_sft/                                        # SFT export (§2.1)
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00002.safetensors
│   ├── diffusion_pytorch_model-00002-of-00002.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json    # shard index
├── stage1_ar_tf/                                         # TF export (§2.2)
├── stage2_ar_ode/                                        # ODE export (§2.4)
├── stage2_ar_cd/                                         # CD export (§2.5)
└── stage3_ar_dmd/                                        # DMD export (§2.6), 4-step real-time inference weights
```

#### 3.1.1 The dual purpose of a single weight

`$CKPT_ROOT/{stage}/` is both **this stage's inference `_from_pretrained`** and **the next stage's training
`model._from_pretrained`**—one export shared by two places. So export here after each segment finishes training, and the next segment (§2.2/§2.4/§2.5/§2.6's
initialization overrides, and §2.3.1's ODE sampling) references it directly.

---

</details>

<details>
<summary><big><b>3.2 Inference</b></big></summary>

Each of the five training stages has a corresponding infer config (`configs/hy/action2v/infer/`), sharing the same entry
`tools/infer_mwm.py`. loop / sampler / guidance / steps are **all decided by `--config-file`**,
so switching stage only requires switching config—the command shape stays the same. The `inference.checkpoint` in the config already points to
`$CKPT_ROOT/{stage}/` (the directory exported in §3.1), so put it in the right place and just run.

The command line has **only** `--config-file` plus a trailing dotlist (`inference.*=` / `model.*=`); value flags
(`--checkpoint` / `--output-dir` / `--prompts` / `--input-json` / `--limit` …) have all been removed.

**Input is a benchmark JSON** (`inference.benchmark`): `[{id, caption, trajectory}]`, extra keys ignored,
one `{id}.mp4` per entry. **HY is i2v, so every entry must carry `image`** (relative path resolved against the JSON's directory);
entries without `image` degrade to t2v.

If you only do official 4-step DMD inference and skip the training stages, you can directly download the final checkpoint:

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/dmd/**"

# Map the HF release name to the stage name used by the config
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT dmd stage3_ar_dmd
)
```

After downloading, the checkpoint is at `./ckpts/HY15/Action2V/dmd/`, and the symlink maps it to
`$CKPT_ROOT/stage3_ar_dmd/`.

Each stage's differences (config defaults, no manual passing needed):

| stage | config | `inference.loop` | `sampler.solver` | steps | guidance |
|---|---|---|---|---|---|
| SFT | `infer/stage0_bi_sft.py` | `BidirectionalGenerationLoop` | `UniPCSolver` | 50 | 6.0 |
| TF | `infer/stage1_ar_tf.py` | `ARGenerationLoop` | `EulerSolver` | 50 | 6.0 |
| ODE | `infer/stage2_ar_ode.py` | `ARGenerationLoop` | `EulerSolver` | 4 | 1.0 (off) |
| CD | `infer/stage2_ar_cd.py` | `ARGenerationLoop` | `EulerSolver` | 4 | 1.0 (off) |
| DMD | `infer/stage3_ar_dmd.py` | `ARGenerationLoop` | `EulerSolver` | 4 | 1.0 (off) |

`loop` / `solver` are both **class-name strings**, resolved by name (`minwm/engine/inference/{loop,samplers}.py`),
so switching loop / switching sampler is all in the config.

Common defaults: `num_frames=20` (→ 480×832 pixels, 20 latent frames), `fps=16`, `sp_size=1`,
`seed=42`. **The trajectory is per-entry**, from the benchmark item's `trajectory` field.
All five stages set `vae_tiling=False` (HY's 3D convolution full decode; enabling tiling will OOM).

**Inference for the five stages separately (single GPU, each running only the first 2 benchmark entries)**:

**SFT (50-step bidirectional CFG)**:

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage0_bi_sft.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_sft
```

**AR-TF (50-step teacher-forcing)**:

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage1_ar_tf.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_ar_tf
```

**AR-ODE (4-step)**:

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage2_ar_ode.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_ar_ode
```

**AR-CD (4-step)**:

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage2_ar_cd.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_ar_cd
```

**DMD (4-step, final product)**:

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage3_ar_dmd.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_dmd
```

- Common overrides: `inference.output_dir=`, `inference.limit=N` (run only the first N entries), `inference.seed=`,
  `inference.dtype=`, `inference.sp_size=N` (with `--nproc_per_node=N` to enable sequence parallelism).
- Each entry outputs one mp4 (480×832 / 20 latent frames → 77 pixel frames @ 16fps) + `manifest.json`
  (recording the config and per-entry `{index, id, prompt, trajectory, seed, video}`).
- **Trajectory semantics**: `key*N` segments joined with commas, e.g. `d*8,i*5,l*6` = move right 8 + pitch up 5 + move left 6;
  `w/s/a/d` translate, `i/k/j/l` rotate view, 20 latent frames correspond to 19 segments.

**Product layout** (under `inference.output_dir`):

```
outputs/infer_hy_action2v_dmd/       # = inference.output_dir
├── {id}.mp4                          # one per benchmark entry (480×832 / 77 frames @16fps)
├── manifest.json                     # config + per-entry {index, id, prompt, trajectory, seed, video}
└── final_with_keys.mp4               # key-overlay concatenated overview (only after running overlay)
```

---

</details>

## 4. Auxiliary tools and operations

<details>
<summary><big><b>4.1 Key overlay + concatenate</b></big></summary>

**Key overlay indicator (optional, pure CPU)**: overlay the WASD/KIJL keys onto each clip then concatenate into an overview, reading
`manifest.json`, applicable to any of the output directories above:

```bash
python demos/overlay_from_manifest.py \
    --input-dir outputs/infer_hy_action2v_dmd \
    --output final_with_keys.mp4
```

> Requires `ffmpeg` / `ffprobe` on `PATH`; cluster images often lack them, so run this step on a machine with ffmpeg
> (reading the output directory on the shared disk).

</details>

<details>
<summary><big><b>4.2 Training health check, memory, and parallelism</b></big></summary>

After training starts, first confirm the following states:

**① No OOM, loss normal**

```bash
grep "num_ooms: [^0]" outputs/<run>/logs/log.txt          # should be empty
grep -riE "traceback|out of memory" outputs/<run>/logs/    # should be empty
```

Loss is at a reasonable magnitude early and decreases with steps, and `grad_norm` doesn't diverge. For DMD, watch the two lines `generator_loss` /
`critic_loss`, noting the asymmetric updates of `dfake_gen_update_ratio=5`.

**② Checkpoints saving normally**

```bash
grep "saved checkpoint" outputs/<run>/logs/log.txt         # one line per ckpt_interval + one closing line
```

Checkpoints land at `outputs/<run>/ckpts/checkpoint_{step}/`, with `ckpts/latest.txt` pointing to the latest.

**③ Resume**: after a crash / manual kill, continue by adding `checkpoint.resume=True` to the original command (`--output-dir`
unchanged). Resume restores all training state (model + optimizer + aux +
step) from `<output_dir>/ckpts/latest.txt` and continues from the breakpoint, **taking priority over** the initialization `_from_pretrained`; if `latest.txt` is not found it loudly errors out,
not silently falling back to fresh init. All other hyperparameters stay unchanged.

Each stage's memory pressure (relative, under the same `sp_size`):

| stage | memory pressure | note |
|---|---|---|
| SFT | lowest | single model, backward dominates |
| TF / ODE | medium | single model + teacher-forcing / ODE regression |
| CD | high | three models (student + teacher + ema) |
| DMD | **highest** | three models + self rollout, most prone to OOM |

- **DMD is most prone to OOM**. The config's `activation_checkpointing=True` is a life-saving setting, don't turn it off; on real OOM,
  **increase `training.sp_size`** (spread activations across more GPUs).
- **There is only one constraint: `$NPROC` must be divisible by `sp_size`**, otherwise it won't start.
- When `sp_size` is increased, `dp_size = $NPROC / sp_size` shrinks accordingly and GBS shrinks, but each sample is sliced more finely,
  saving memory; the per-step time slightly increases due to SP communication.

---

</details>

<details>
<summary><big><b>4.3 Full-pipeline serial quick reference</b></big></summary>

After each segment finishes training, export the diffusers directory (§3.1), and the next segment references it. `<BEST_STEP>` is selected by validation and must be a
multiple of `ckpt_interval`:

```bash
# ① encode (raw video → latent + index)          → $INDEX                        (§1)
# ② SFT (config default from base)               → export stage0_bi_sft/         (§2.1, §3.1)
# ③ TF  (model._from_pretrained=stage0_bi_sft)   → export stage1_ar_tf/          (§2.2, §3.1)
# ④ ODE preprocessing (TF teacher 48-step pre-solve) → $ODE_INDEX                (§2.3)
# ⑤ ODE (config default from stage1_ar_tf)       → export stage2_ar_ode/         (§2.4, §3.1)
# ⑥ CD  (student/teacher/ema=stage1_ar_tf)       → export stage2_ar_cd/          (§2.5, §3.1)
# ⑦ DMD (gen=stage2_ar_cd, score=stage0_bi_sft)  → export stage3_ar_dmd/         (§2.6, §3.1)
# ⑧ inference (switch infer config + $CKPT_ROOT/{stage})  → outputs/infer_*/ + key overlay  (§3.2, §4.1)
```

④⑤ and ⑥ all start from ③ and are independent of each other; ⑦ needs both ⑥ (generator seed) and ② (real/fake score seed).
ODE preprocessing (④) is the heaviest segment of the whole pipeline, requiring the sm_80-and-above flash-attn kernel (§2.3.1).
Each segment's `--master_port` is already staggered (29635–29640), so running them in sequence needs no port change.

---

</details>

<details>
<summary><big><b>4.4 Troubleshooting</b></big></summary>

| Symptom | Cause / handling |
|---|---|
| `world_size not divisible by sp_size` won't start | `$NPROC` is not divisible by `training.sp_size` (§0.4). |
| `train_index.json` empty / missing entries after encoding | video directory layout wrong: must be `$SRC_VIDEOS/<idx:06d>_<suffix>/gen.mp4`, `suffix` per §1. |
| ODE sampling flash-attn kernel error | requires sm_80 and above (A100/A800 ✓, B200 missing) (§2.3.1). |
| ODE training reports missing `ode_trajectory` key | `json_path` was given a plain `$INDEX`. ODE must use `$ODE_INDEX` (§2.4). |
| TF/CD/DMD trains as if not fine-tuned / starting from base | missed `model._from_pretrained` (and CD/DMD's aux overrides). The config default is base, and serial training must give it (§2.2/§2.5/§2.6). |
| CD / DMD CFG-related error / uncond is empty | negative embedding path wrong. The config default is at `$NEG_PROMPT` / `$NEG_BYT5`, put it right or explicitly override `recipe.adapter.neg_*` (§1.2). |
| DMD OOM | don't turn off `gradient_checkpointing`; or increase `training.sp_size` to spread activations (§4.2). |
| `double_blocks.*.img_attn_prope_proj.* newly initialized` warning | normal. PRoPE camera parameters don't exist in base, newly initialized (§2.1). |
| Exported DMD works wrong / can't find `aux/generator_ema` | HY stage3 has no generator_ema, use the default `--model-key model` (§2.6, §3.1). |
| Missing dependencies (lmdb, decord…) | use the conda env with dependencies installed (§0.1). |
| DCP directory unreadable by `torch.load` | the shard directory is not a single file, first merge with `tools/export_checkpoint.py` (§3.1). |
| `unrecognized arguments: --checkpoint / --prompts / --input-json ...` | copied the old flag style. `infer_mwm.py` now only eats `--config-file` + trailing dotlist (`inference.benchmark=` / `inference.output_dir=` …), see §3.2. |
| `unknown key(s) '...' in Inference` | there's a misspelled key in the `inference` block. That block is a strict schema (`minwm/config/schema.py:Inference`), and a wrong key errors out directly; fix per the legal key names listed in the error. |
| HY bidirectional inference OOM (`Tried to allocate ... GiB`) | `inference.vae_tiling` must be `False` (HY's 3D convolution decode), all five infer configs are already set; don't miss it when writing your own config (§3.2). |
| Inference can't find weights / `build_model` reports `OSError` | the config's `_from_pretrained` points to `$CKPT_ROOT/{stage}/`, export there after training (§3.1). |
| Key overlay reports `ffmpeg not found` | `overlay_from_manifest.py` needs ffmpeg/ffprobe, run it on a machine that has them (§4.1). |

---

</details>

<details>
<summary><big><b>4.5 References</b></big></summary>

- Data encoding scripts and `.pt` key convention: [`tools/data/hy/README.md`](../../../tools/data/hy/README.md).
- Each stage config: `configs/hy/action2v/train/` + `configs/hy/action2v/infer/`.
- Weights / dataset downloads, naming symlinks, Quick Start inference: main [`README.md`](../../../README.md).
- Installation: `INSTALL.md`.
- Wan backbone cross-reference: `configs/wan21/action2v/README.md`.

</details>

