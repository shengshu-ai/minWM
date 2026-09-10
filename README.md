# minWM: The First Full-Stack Open-Source World Model Framework

>  ***A full-stack framework and tutorial for newcomers, rather than a specific model.***

<p align="center">
  <a href="https://arxiv.org/abs/2605.30263"><img src="https://img.shields.io/badge/Technical_Report-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Technical Report"></a>
  <a href="https://huggingface.co/MIN-Lab/minWM"><img src="https://img.shields.io/badge/Hugging_Face-Models-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

**minWM** is our contribution to the world-model community: a **full-stack open-source framework** that walks you end-to-end through turning a bidirectional T2V foundation model into an action-conditioned video world model — with example data, runnable scripts, **Claude skills** capturing our hands-on experience, and **onboarding knowledge** for newcomers. We hope more researchers and developers join us in growing the community together.

https://github.com/user-attachments/assets/99c25915-7fe7-4a20-a2c4-9d291502fccf

## 🔥 News

- **2026-09-05** We release a much better version with an optimized infrastructure — a more elegant, efficient, and systematic system. For the legacy version, see the [migration guide](MIGRATION.md) (old layout preserved at tag `v0.1-legacy`).
- **2026-05-29** We release the [technical report](https://arxiv.org/pdf/2605.30263).
- **2026-05-17** We release **minWM** — the first full-stack open-source world model framework.


## Table of Contents
- [Why minWM?](#-why-minwm)
  - [1. Full-Stack Framework](#1-full-stack-framework)
  - [2. Multi-Backbone Support](#2-multi-backbone-support)
  - [3. Claude Skills — Modify the Framework with an LLM Assistant](#3-claude-skills--modify-the-framework-with-an-llm-assistant)
- [Installation](#installation)
- [Inference](#inference)
  - [1. Download the demo checkpoints](#1-download-the-demo-checkpoints)
  - [2. Run the demos](#2-run-the-demos)
  - [3. Optional: overlay the key indicator](#3-optional-overlay-the-key-indicator)
- [Data & Training & Reproduction](#data--training--reproduction)


## ✨ Why minWM?

### 1. Full-Stack Framework

The complete **data → training → inference** pipeline is open-sourced; every stage exposes input/output checkpoints so you can stop, swap, or fork anywhere.

**1.1 Data.** We walk you through how to construct training-ready datasets paired with camera poses, and the full data processing pipeline that turns them into latents.

**1.2 Training.** Including FSDP + sequence parallelism, single-/multi-node training, and the full distillation pipeline from a bidirectional diffusion model to a 4-step AR student:

```
Phase 1                            Phase 2 — Distillation to Causal Few-Step
─────────────────────              ────────────────────────────────────────────
Bidirectional SFT      ──▶   Stage 1   Teacher Forcing AR Diffusion
                             Stage 2a  Causal ODE  (proposed in [Causal Forcing](https://arxiv.org/abs/2602.02214))
                             Stage 2b  Causal CD   (proposed in [Causal Forcing++](https://arxiv.org/abs/2605.15141))
                             Stage 3   Asymmetric DMD with Self Rollout
                                                ▼
                                         4-step real-time
```

**1.3 Inference**: 4-step DMD inference for HY Action2V / HY TI2V / Wan Action2V, multi-GPU sequence parallelism, camera-trajectory control via pose strings (`"a*4,w*8,s*7"`) or JSON files


### 2. Multi-Backbone Support

> From Scratch: Bidirectional T2V Foundation → Real-Time World Model

The HunyuanVideo 1.5 and Wan 2.1 lines walk through the full 4-stage pipeline — starting from a bidirectional T2V foundation model and ending at a 4-step autoregressive world model.

| Backbone             | Architecture          | Params | Training       | Inference    |
| -------------------- | --------------------- | ------ | -------------- | ------------ |
| **Wan 2.1**          | Cross-attention + DiT | 1.3 B  | all 4 stages | 4-step DMD |
| **HunyuanVideo 1.5** | MMDiT                 | 8 B    | all 4 stages | 4-step DMD |

Both lines share the same trainer / loss / dataset abstractions, so adding a third backbone is structurally a wrapper-and-config exercise.

### 3. Claude Skills — Modify the Framework with an LLM Assistant
We are packaging our project experience across the CF / CF++ pipeline as Claude skills, so that an LLM assistant can help users debug failures and integrate new models without reverse-engineering the whole repo.

- **`debug-world-model`** — collected failure modes from the training pipeline (loss NaN, frame-to-frame jitter, camera drift, memory attenuation, distillation collapse, …). Claude diagnoses likely root causes from your symptoms instead of guessing.
- **`integrate-new-backbone`** — step-by-step recipe for plugging a new video DiT into minWM, grounded in the HunyuanVideo and Wan reference integrations — e.g. *"look at how HY does teacher forcing here, do the same for your model there"*.
- **`onboarding-world-model`** — A third Claude skill aimed at researchers entering the world-model space for the first time. Two parts:
    - **Foundations** — the minimal background to follow the pipeline: Teacher Forcing for AR diffusion training and Causal Forcing & Causal Forcing++ for AR diffusion distillation.
    - **Pitfalls** — the non-obvious mistakes we hit while building minWM, distilled so you don't repeat them.

Intended audience: graduate students, independent researchers, and junior labs that want to enter the world-model space without spending three months reverse-engineering existing repos.

## Installation

```bash
conda create -n minwm python=3.12 -y
conda activate minwm
pip install -r requirements/base.txt
pip install flash-attn --no-build-isolation
pip install -e .          # editable install: makes `import minwm` resolve, no PYTHONPATH
```

> Full requirements, verification, developer setup, and troubleshooting:
> see [`INSTALL.md`](INSTALL.md).
>
> Saving checkpoints to a remote object store (`s3://`, `oss://`, or an
> S3-compatible store like Baidu BOS)? Install the matching fsspec backend —
> see [Remote Checkpoint Storage](INSTALL.md#remote-checkpoint-storage-optional).

<details> <summary> Model Checkpoints (Click to expand) </summary> 

All weights live under `./ckpts/` after download.


| Checkpoint                                                                | Backbone | Stage                               | Use case                               | Download                                              |
| ------------------------------------------------------------------------- | -------- | ----------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `HunyuanVideo-1.5` (base)                                                 | HY 1.5   | —                                   | Required by both HY pipelines          | [HF](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| `HY15/Action2V/bidirectional`                                             | HY 1.5   | Phase 1 SFT                         | Starting point for HY Action2V Phase 2 | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/ar_diffusion_tf`                                           | HY 1.5   | Phase 2 Stage 1                     | Teacher Forcing AR diffusion           | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/causal_ode`                                                | HY 1.5   | Phase 2 Stage 2a (proposed in Causal Forcing)   | DMD initialization               | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/causal_cd`                                                 | HY 1.5   | Phase 2 Stage 2b (proposed in Causal Forcing++) | DMD initialization               | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `HY15/Action2V/dmd`                                                       | HY 1.5   | Phase 2 Stage 3                     | **4-step real-time inference**         | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `Wan21/Action2V/{bidirectional,ar_diffusion_tf,causal_ode,causal_cd,dmd}` | Wan 2.1  | Same 4 stages                       | Wan pipeline                           | [HF](https://huggingface.co/MIN-Lab/minWM)            |
| `Wan21/Action2V/bidirectional/model_v2.pt`                                | Wan 2.1  | Phase 1 SFT (v2, retrained data)    | Improved Wan Action2V bidirectional    | [HF](https://huggingface.co/MIN-Lab/minWM/tree/main/Wan21/Action2V/bidirectional/model_v2.pt) |
| `Wan2.1-T2V-1.3B` (base)                                                  | Wan 2.1  | —                                   | Required by Wan pipeline               | [HF](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)   |

</details>

## Inference

### 1. Download the demo checkpoints

```bash
# Wan base (T2V-1.3B)
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B 

# The ODE data-curation tool loads the base from `wan_models/` relative to the repo
# root, so mirror it there too (inference itself reads ./ckpts/Wan2.1-T2V-1.3B).
mkdir -p wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" wan_models/Wan2.1-T2V-1.3B


# HY base + text/vision encoders (required by HY pipelines)
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*"  "scheduler/*" "transformer/480p_i2v/*"
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm
hf download google/byt5-small           --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir ./ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2
hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip --token <your_hf_token>


# 4-step DMD checkpoints
## Wan Action2V (DMD, 4-step)
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/dmd/*"

## HY Action2V (DMD, 4-step, worldplay teacher) 
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/dmd/*"

# The HF repo publishes stage weights under release names (`dmd`, `causal_cd`, …)
# while the configs load the `stage{N}_*` names. Link them so the configs resolve
# with no extra flags (relative + idempotent; safe to re-run).
ln -sfnT dmd ./ckpts/Wan21/Action2V/stage3_ar_dmd
ln -sfnT dmd ./ckpts/HY15/Action2V/stage3_ar_dmd
```

<details> <summary> Checkpoint naming: release names → config names (Click to expand) </summary>

The two naming schemes differ, so each downloaded stage needs one directory-level
symlink. The links are relative (so `./ckpts/` stays movable) and re-running is a
no-op. Wan configs load `<stage>/model.pt`; HY configs load `<stage>/` as a
diffusers directory — a single link per stage satisfies both.

Keep the `-T`: without it, if the `stage{N}_*` path already exists as a **real**
directory (e.g. you exported your own checkpoint there), `ln` would quietly create
a nested link *inside* it and the config would keep loading the old weights. With
`-T` you get a loud `cannot overwrite directory` instead, and nothing is touched.

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

Dangling links for stages you did not download are harmless — nothing reads them.
Alternatively, skip the links entirely and pass the release path explicitly:
`inference.checkpoint=./ckpts/Wan21/Action2V/dmd/model.pt`.

</details>


### 2. Run the demos

All inference goes through one entrypoint — **`tools/infer_mwm.py`** — and the loop,
sampler, guidance and step count come from the `--config-file`, not from CLI flags. So
switching model line or stage means switching the config; the command shape never changes.

The input is a benchmark JSON named by the config key `inference.benchmark`: a list of
`[{id, caption, trajectory}]` items (extra keys ignored), each output named `{id}.mp4`. An
`image` field makes the item image-to-video (HY); omitting it makes it text-to-video (Wan).

```bash
# 2.1  Wan Action2V (4-step DMD, camera control)
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/wan21/action2v/infer/stage3_ar_dmd.py \
    inference.benchmark=assets/example_t2v.json inference.limit=2 \
    inference.output_dir=./outputs/quickstart_wan_action2v

# 2.2  HY Action2V (4-step DMD, camera control)
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage3_ar_dmd.py \
    inference.benchmark=assets/example.json inference.limit=2 \
    inference.strict=False \
    inference.output_dir=./outputs/quickstart_hy_action2v
```

Each run writes one `.mp4` per sample (Wan Action2V: 832×480, 77 frames @ 16 fps) plus a
**`manifest.json`** recording the config and the per-item `{prompt, trajectory, seed, video}`.

Every config value is overridable inline as a dotlist `key=value`: `inference.checkpoint=` /
`inference.output_dir=` override the config's values, `inference.limit=N` runs only the first
N items, `inference.seed=`, and `inference.sp_size=N` (with a matching `--nproc_per_node=N`)
for sequence-parallel sampling.

> **Camera trajectories.** Format is `key*N` segments joined by commas — e.g. `d*8,i*5,l*6`
> = pan right 8, tilt up 5, pan left 6. `w/s/a/d` translate, `i/k/j/l` rotate. For 20 latent
> frames the segment counts sum to 19. The trajectory is per-sample, read from each benchmark
> item's `"trajectory"` field.

### 3. Optional: overlay the key indicator

Renders the WASD/KIJL key presses onto each clip and concatenates them into one overview.
It reads `manifest.json`, so it works on any output directory produced above:

```bash
python demos/overlay_from_manifest.py \
    --input-dir ./outputs/quickstart_wan_action2v \
    --output final_with_keys.mp4
```

> Needs `ffmpeg` / `ffprobe` on `PATH`. Cluster images often ship without them — run this
> step locally on the (shared-filesystem) output directory instead.

## Data & Training & Reproduction

### 1. Data preparation

Before starting any training stage, prepare the raw videos and camera trajectories. Choose one option; both produce the same `./dataset/` layout.

#### Option A: Download minWM Dataset

The videos are generated with HunyuanVideo (HY-WorldPlay); their use is subject to the upstream model's license terms.

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "preencode_input.json" "videos/**"
```

The resulting layout is:

```text
./dataset/
├── preencode_input.json
└── videos/
    ├── 000000_right8a11/gen.mp4
    ├── 000001_w10d9/gen.mp4
    └── ...
```

For HY Action2V, download the CFG negative prompt embeddings separately:

```bash
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/Action2V/**"
```

#### Option B: Use Your Own Videos and Trajectories

Match Option A's layout by providing your own `preencode_input.json` and `videos/` directory. The JSON file must be a list, and each entry must contain at least `image_path`, `caption`, and `pose_str`:

```json
[
    {
        "image_path": "/abs/path/to/image1.png",
        "caption": "A scenic mountain view",
        "pose_str": "right-8, a-11"
    }
]
```

Each video must be stored at `videos/{i:06d}_{slug(pose_str)}/gen.mp4`, where `i` is the entry's index in the JSON list and `slug` is `pose_str` lowercased with non-alphanumeric characters removed.

### 2. Training and Reproduction Guides

The Wan 2.1 and HunyuanVideo 1.5 pipelines follow the same four-part workflow:
**(1) Setup → (2) Data encoding → (3) Training → (4) Inference**.

The complete guides are split by backbone:

- [`configs/wan21/`](configs/wan21/README.md) — Wan 2.1 backbone
    - [`configs/wan21/action2v/`](configs/wan21/action2v/README.md) — **Wan Action2V**
- [`configs/hy/`](configs/hy/README.md) — HY1.5-8B backbone
    - [`configs/hy/action2v/`](configs/hy/action2v/README.md) — **HY Action2V**

## Citation

If minWM helps your research, please cite:

```bibtex

# ICML 2026
@article{zhu2026causal,
  title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
  author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2602.02214},
  year={2026}
}

# Technical Report
@article{zhao2026causal,
  title={Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation},
  author={Zhao, Min and Zhu, Hongzhou and Zheng, Kaiwen and Zhou, Zihan and Yan, Bokai and Li, Xinyuan and Yang, Xiao and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2605.15141},
  year={2026}
}

# Technical Report
@article{zhao2026minwm,
  title={minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models},
  author={Zhao, Min and Zhu, Hongzhou and Yan, Bokai and Zhou, Zihan and Chen, Yimin and Sun, Wenqiang and Zheng, Kaiwen and He, Guande and Yang, Xiao and Li, Chongxuan and others},
  journal={arXiv preprint arXiv:2605.30263},
  year={2026}
}

```

## License

minWM's own framework code is released under the [Apache License 2.0](LICENSE).
The repository also contains components under other licenses; those terms govern
the corresponding files. See [NOTICE](NOTICE) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full attribution, and
[`licenses/`](licenses/) for the license texts.

| Component | Location | License |
|---|---|---|
| minWM framework (engine, config, data, sampling) | `minwm/` (except modeling backbones below) | Apache-2.0 |
| Wan 2.1 backbone | `minwm/modeling/wan21/` | Apache-2.0 |
| vLLM-derived sequence parallelism | `minwm/distributed/sp/` | Apache-2.0 |
| HunyuanVideo 1.5 backbone | `minwm/modeling/hy15/` | Tencent Hunyuan Community License (THCL) |

> **HunyuanVideo 1.5 components and any derived weights** (fine-tuned, distilled,
> DMD-student) are licensed under the [Tencent Hunyuan Community License](licenses/TENCENT_HUNYUAN_COMMUNITY_LICENSE.txt),
> **not** Apache-2.0. The THCL is not an OSI-approved open-source license: its
> grant is limited to the **Territory** — worldwide **excluding the European
> Union, the United Kingdom, and South Korea** — it treats fine-tuning and
> distillation outputs as "Model Derivatives", and it restricts using Hunyuan
> outputs to improve other AI models. Review the agreement before using these
> parts.

## Contact

For questions, suggestions, or collaboration, please open a GitHub issue or contact: [gracezhao1997@gmail.com](mailto:gracezhao1997@gmail.com).

## Acknowledgements

minWM stands on the shoulders of giants. We thank the authors and maintainers of [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), [Wan 2.1](https://github.com/Wan-AI/Wan), [Causal-Forcing](https://github.com/thu-ml/Causal-Forcing), and [FastVideo](https://github.com/hao-ai-lab/FastVideo) for their open-source contributions, which made this framework possible.
