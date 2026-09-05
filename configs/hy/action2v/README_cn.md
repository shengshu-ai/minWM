# HY Action2V 训练

> English version: [README.md](README.md)

数据 encode → SFT → TF → ODE → CD → DMD，全阶段训练（FSDP2 + 序列并行，默认单机 8 卡）。
本文档假设你已按 §0.2 准备好基础模型、把**原始视频数据**放在默认相对路径下（或自己给路径，
改 §0.1 两行即可）——编码在 §1 里做，产物就是后面各阶段读的训练数据。


**阶段链**（每段的产物是下一段的输入）：

```
encode ─→ SFT ─→ TF ─┬─→ ODE
                     └─→ CD ─→ DMD
```

`ODE` 与 `CD` 都从 TF 权重起步、互不依赖；`DMD` 需要 CD (generator) 和 SFT
(real/fake score) 两份权重。每段训完把 DCP 目录导出成 diffusers 目录（§3.1），它同时是
**本段推理的 checkpoint** 和**下一段训练的 `_from_pretrained`**——所以必须按上图串行推进。

---

## 0. 前置

<details>
<summary><big><b>0.1 ⭐ 配置块（先改这里，后面所有命令都引用它）</b></big></summary>

**本文档后面的每条命令都假设你已经 source 过这一段。** 只有仓库路径和原始数据路径需要按你的
环境改；其余一律用变量，照抄即可。建议存成 `env.sh` 放在仓库外，每开一个新 shell `source` 一次。

```bash
# ---- (1) 仓库与 Python 环境 -------------------------------------------------
cd /path/to/minWM                      # ← 改成你的 checkout 路径
export PROJECT_ROOT="$PWD"
conda activate minwm                   # ← 见 INSTALL.md
pip install -e .                       # editable 安装后 import minwm 生效，无需 PYTHONPATH

# ---- (2) 权重 --------------------------------------------------------------
export HYCKPT="./ckpts/HunyuanVideo-1.5"                     # HY1.5 base + VAE/编码器（编码 + 各阶段都用）
export BASE="$HYCKPT/transformer/480p_i2v"                   # SFT 初始化的 base transformer
export CKPT_ROOT="./ckpts/HY15/Action2V"                     # 各阶段 diffusers 权重根

# ---- (3) 原始视频数据（编码前；自备数据就改这两行）--------------------------
export SRC_JSON="./dataset/preencode_input.json"            # 原始 caption + pose_str（§1）
export SRC_VIDEOS="./dataset/videos"                        # 原始视频根目录（§1）
export NEG_PROMPT="./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"      # CFG 负向 embedding
export NEG_BYT5="./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"
export BENCH="assets/example.json"                           # 评测用 benchmark JSON（§3.2）

# ---- (4) 拓扑与运行时环境 --------------------------------------------------
export NPROC=8                         # GPU 数（默认单机 8 卡）
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True         # HY 显存碎片明显，别省
```

> `$NPROC` 按实际卡数设（`nvidia-smi -L | wc -l`）。**卡数必须能被 `training.sp_size`
> 整除**，否则起不来——见 §0.4。
>
> **编码后 / ODE 数据的路径变量在对应阶段再 export**：`$INDEX`（编码产物）在 §1、
> `$ODE_INDEX`（ODE 预处理产物）在 §2.3，跟着那一步一起给。

</details>

<details>
<summary><big><b>0.2 基础模型</b></big></summary>

HY 训练和推理需要 HunyuanVideo 1.5 的 VAE、scheduler、transformer，以及独立的文本和视觉编码器：

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

上面的 `$HYCKPT` 默认是 `./ckpts/HunyuanVideo-1.5`，与各 config 的路径一致；如果模型存放在
其他位置，直接修改 §0.1 中的 `$HYCKPT` / `$BASE`，或在命令行覆盖对应的 `_from_pretrained`。

</details>

<details>
<summary><big><b>0.3 默认路径约定</b></big></summary>

各 stage config 里写死的默认路径全是**仓库内相对路径**，与本文档变量一一对应。放对位置，
命令里的 `$INDEX` / `$NEG_*` / `$CKPT_ROOT/...` 就直接命中：

| 角色 | 默认相对路径 | 变量 | 何时产生 |
|---|---|---|---|
| HY1.5 base + 编码器 | `./ckpts/HunyuanVideo-1.5/` | `$HYCKPT` / `$BASE` | 下载 |
| 各阶段权重 | `./ckpts/HY15/Action2V/{stage}/`（diffusers 目录） | `$CKPT_ROOT/{stage}` | 各段训完导出（§3.1） |
| 原始视频 + JSON | `./dataset/{videos, preencode_input.json}` | `$SRC_VIDEOS` / `$SRC_JSON` | 你自备 |
| 编码后训练数据 | `./dataset/HY15/Action2V/{latents, train_index.json}` | `$INDEX`（§1） | §1 编码 |
| ODE 数据 | `./dataset/HY15/Action2V_ode/{latents, train_index.json}` | `$ODE_INDEX`（§2.3） | §2.3 预处理 |
| CFG 负向 embedding | `./dataset/others/HY/Action2V/*.pt` | `$NEG_PROMPT` / `$NEG_BYT5` | §1 生成或下载 |

`{stage}` ∈ `stage0_bi_sft` / `stage1_ar_tf` / `stage2_ar_ode` / `stage2_ar_cd` / `stage3_ar_dmd`。

> **权重来自 HF 时名字对不上**：HF 发布名是 `bidirectional` / `ar_diffusion_tf` / … ，
> config 加载的是 `stage{N}_*`。按各训练小节中的命令建一层软链即可，本文档一律用 config 名。

</details>

<details>
<summary><big><b>0.4 拓扑与并行（按你的卡数选 `sp_size`）</b></big></summary>

`$NPROC = sp_size × DP`，`GBS = DP × data.batch_size`。约束只有一条：**`$NPROC` 必须能被
`sp_size` 整除**，否则起不来。

本文档命令统一 `training.sp_size=2`（默认单机 8 卡 → DP=4）。`sp_size` 越小 DP/GBS 越大、
显存压力越大；越大则单样本被切得越细、显存越省但通信越多。各阶段显存压力见 §4.2。

</details>

<details>
<summary><big><b>0.5 监控</b></big></summary>

**默认不用 wandb**：指标只走 STDOUT。日志在 `outputs/<run>/logs/log.txt`（rank0）+
`log.txt.rank{N}`，`log_interval=10` 每 10 步一行（loss + steps/sec + ms/step + 峰值显存）；
DMD 那行是 `generator_forward/backward` + `critic_forward/backward` + `generator_loss` /
`critic_loss` 四组。要 wandb 就自己配，**key 只放在 shell / 仓库外的文件里，别写进任何进
git 的文件**：

```bash
export WANDB_API_KEY='你的 key'
# 训练命令尾部追加（run name 自己起）：
#   'monitor.backends=["wandb"]' monitor.wandb_project=<project> \
#   monitor.wandb_run_name=<run-name>
```

</details>

---

## 1. 数据 encode (原始视频 → latent + train_index)

<details>
<summary><big><b>1.1 数据编码</b></big></summary>

把原始视频 + caption + `pose_str` 编码成各训练阶段直接读的 `.pt` latent 分片
（VAE + SigLIP 视觉 + LLM 文本 + byT5 字形），每 rank 写自己的分片再由 rank 0 合并出
`train_index.json`。用 WorldPlay 蒸馏视频那套（相机位姿由 `pose_str` DSL 合成）：

```bash
export INDEX="./dataset/HY15/Action2V/train_index.json"    # ← 本步产物，后面 SFT/TF/CD/DMD 都读它

torchrun --nproc_per_node="$NPROC" \
    tools/data/hy/preencode_generated_wdplay.py \
    --input_json "$SRC_JSON" \
    --video_root "$SRC_VIDEOS" \
    --output_dir ./dataset/HY15/Action2V \
    --hunyuan_checkpoint_path "$HYCKPT" \
    --skip_existing
```

- 产物：`./dataset/HY15/Action2V/{latents/*.pt, train_index.json}`（= `$INDEX`）。`.pt` 键约定
  见 [`tools/data/hy/README.md`](../../../tools/data/hy/README.md)（`latent (1,32,20,H,W)` +
  `prompt_embeds` + `vision_states` + `byt5_*` + 相机 `intrinsics`/`poses`/`camera_indices`）。
- 有自己现成的相机位姿 `.npy`（不是 `pose_str` DSL）就改用 `preencode_camera_video.py`，
  入参与输出契约同上，详见该 README。

**输入布局**（原始数据，编码前）——`$SRC_JSON` 每条含 `caption` / `pose_str`（首帧由视频
`gen.mp4` 第 0 帧提供，JSON 里无 `image_path`）；视频命名为 `<idx:06d>_<suffix>`，`suffix` 是
`pose_str` 去空白/逗号、删短横线（`right-8, a-11` → `right8a11`）：

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
<summary><big><b>1.2 CFG 负向 embedding</b></big></summary>

**CFG 负向 embedding**（每个 HY checkpoint 生成一次，各阶段共用）：

```bash
# 也可以直接下载官方提供的 embedding
hf download MIN-Lab/minWM-data --repo-type dataset \
    --local-dir ./dataset \
    --include "others/HY/Action2V/**"

# 如果没有下载现成文件，则使用本地 HY 基础模型生成
python tools/data/hy/generate_negative_prompts.py \
    --hunyuan_checkpoint_path "$HYCKPT" \
    --output_dir ./dataset/others/HY/Action2V     # → $NEG_PROMPT / $NEG_BYT5
```

> 已从 HF 下过 `others/HY/Action2V/*.pt`（主 README）就跳过这一步。

**产物布局**（编码后，`.pt` 键约定见 [`tools/data/hy/README.md`](../../../tools/data/hy/README.md)：
`latent (1,32,20,H,W)` + `prompt_embeds` + `vision_states` + `byt5_*` + 相机
`intrinsics`/`poses`/`camera_indices`）：

```
./dataset/
├── HY15/Action2V/                      # 编码产物
│   ├── latents/                        # 每视频一个 .pt
│   └── train_index.json                # = $INDEX，SFT/TF/CD/DMD 都读它
└── others/HY/Action2V/                 # CFG 负向 embedding
    ├── hunyuan_neg_prompt.pt           # = $NEG_PROMPT
    ├── hunyuan_neg_byt5_prompt.pt      # = $NEG_BYT5
    └── negative_prompt.pt
```

</details>

---

## 2. 训练流程

<details>
<summary><big><b>2.1 Phase-1 Bidirectional SFT (`stage0_bi_sft`)</b></big></summary>

双向 + 相机 (PRoPE) 监督微调。`ARHunyuanVideo_1_5_DiffusionTransformer` + `BiSFTRecipe` +
`FlowMatchingLoss`，读 §1 编码出的 `$INDEX`（clean-latent camera 数据，viewmats / Ks）。

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29635 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage0_bi_sft.py \
    --output-dir outputs/hy_action2v_sft \
    training.sp_size=2
```

- 初始化：config 默认 `_from_pretrained=$BASE`（`./ckpts/HunyuanVideo-1.5/transformer/480p_i2v`）
  ——SFT 是链条起点，从 base 起。**无 `checkpoint.pretrained`**。
- 超参（config 默认）：Muon `lr=2e-5`、`weight_decay=1e-4`、`timestep_shift=3.0`、
  `schedule=linear`、`logit_normal` 采时间步、`window_frames=20`、`batch_size=1`、
  `activation_checkpointing=True`。
- 步数（config 默认）：`max_steps=100000` / `ckpt_interval=1000`；按需覆盖，如
  `training.max_steps=... training.ckpt_interval=...`。
- 输出：`outputs/hy_action2v_sft/{ckpts/checkpoint_{step}/, logs/}`，`ckpts/latest.txt` 指向
  最新。**checkpoint 是 FSDP2 DCP 分片目录，不是单 `.pt`**（见 §3.1）。

**首个 step 的正常告警**：一大批 `double_blocks.*.img_attn_prope_proj.* newly initialized` +
`You should probably TRAIN this model...`——PRoPE 相机参数在 base 里不存在、新初始化，
**符合预期**。

训完导出（§3.1）→ `$CKPT_ROOT/stage0_bi_sft/`。它既是 SFT 推理的 checkpoint，
也是 **TF 训练的起点** + **DMD 的 real/fake score 种子**。

---

</details>

<details>
<summary><big><b>2.2 Phase-2 Stage-1 Teacher-Forcing AR (`stage1_ar_tf`)</b></big></summary>

**如果跳过上一阶段，可下载官方提供的上一阶段 checkpoint：**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/bidirectional/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT bidirectional stage0_bi_sft
)
```

下载后 checkpoint 位于 `./ckpts/HY15/Action2V/bidirectional/`，上面的软链接会将它映射为
本文档中的 `$CKPT_ROOT/stage0_bi_sft/` 路径；也可以在命令行显式覆盖
`model._from_pretrained`。

把双向 SFT 模型转成 causal + teacher forcing 的 AR diffusion。同款
`ARHunyuanVideo_1_5_DiffusionTransformer` + `ARTFRecipe` + `FlowMatchingLoss`，
`use_prope=True` 保留 PRoPE 参数。**数据与 SFT 同一份 `$INDEX`**。

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29636 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage1_ar_tf.py \
    --output-dir outputs/hy_action2v_ar_tf \
    training.sp_size=2 \
    model._from_pretrained="$CKPT_ROOT/stage0_bi_sft"
```

- **初始化必给**：`model._from_pretrained=$CKPT_ROOT/stage0_bi_sft`（§2.1 的导出）。config 默认
  指 base，串行训练要从 SFT 权重起，所以这条覆盖必给。
- 与 SFT 唯一的区别：recipe 换成 `ARTFRecipe`，preprocessor 链尾多一个
  `CleanContextNoiseAug(max_timestep=0)`（写出未加噪的 clean context，把模型切到
  teacher-forcing 的 block-causal `flex_tf` mask 路径）。
- 超参（config 默认）：Muon `lr=1e-5`（比 SFT 的 2e-5 低）、`timestep_shift=3.0`、
  `window_frames=20`；`max_steps=200000` / `ckpt_interval=1000`。

训完导出 → `$CKPT_ROOT/stage1_ar_tf/`。它是 **ODE / CD 训练的起点**。

---

</details>

<details>
<summary><big><b>2.3 Stage-2(a) ODE data curation (数据准备，不是训练)</b></big></summary>

**TF teacher checkpoint（2.2 的产物；如果跳过 TF 阶段，可下载官方提供的 checkpoint）：**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/ar_diffusion_tf/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT ar_diffusion_tf stage1_ar_tf
)
```

下载后 checkpoint 位于 `./ckpts/HY15/Action2V/ar_diffusion_tf/`，上面的软链接会将它映射为
下面 ODE curation 命令使用的 `$CKPT_ROOT/stage1_ar_tf/`；如路径不同，也可显式覆盖
`--generator_ckpt`。

用冻结的 TF teacher 给 SFT latent 预解 6 点 ODE 轨迹，再重建 index，供 §2.4 的
`ARODERecipe` 回归拟合 causal student。**数据是 ODE 专用的 `$ODE_INDEX`**。

#### 2.3.1 ODE 数据预处理（先做，训练前置）

用 §2.2 的 TF teacher 对 SFT latent 跑 48 步 CFG 采样（guidance 5.0），预解出 6 点轨迹，
再重建 index。产物落 `./dataset/HY15/Action2V_ode/`，与 SFT latent 平行、不重叠：

```bash
export ODE_INDEX="./dataset/HY15/Action2V_ode/train_index.json"   # ← 本步产物，§2.4 训练读它

# 1) 48 步 CFG 采样（guidance 5.0）——最重的一步
torchrun --nproc_per_node="$NPROC" --master_port=29640 \
    tools/data/hy/get_causal_ode_data_prope.py \
    --generator_ckpt "$CKPT_ROOT/stage1_ar_tf" \
    --rawdata_path   ./dataset/HY15/Action2V/latents \
    --output_folder  ./dataset/HY15/Action2V_ode/latents \
    --neg_prompt     "$NEG_PROMPT" \
    --neg_byt5       "$NEG_BYT5" \
    --guidance_scale 5.0

# 2) 重建绝对路径 index
python tools/data/create_train_index.py \
    ./dataset/HY15/Action2V_ode \
    --recursive \
    -o "$ODE_INDEX"
```

- 采样器是 minwm 原生（`ARHunyuanVideo_1_5_DiffusionTransformer` + `use_prope=True`），要
  **A 系列（sm_80）** 的 flash-attn kernel（A100/A800 一类 ✓，B200 实测缺）。
- 按 rank 切分，产物用 shape 自检：`latent (1,32,20,30,52)` / `ode_trajectory (1,6,32,20,30,52)`
  + prompts/camera。这是全流程最重的一段。
- HF 上有现成的预生成 ODE latent，可跳过采样直接下（`ODE_data/HY15/Action2V/**`），
  下完只跑上面第 2 步重建 index 即可，权重 / 数据集下载见主 [`README.md`](../../../README.md)。

**产物布局**（ODE 数据与 SFT latent 平行、不重叠；负向 embedding 复用 §1 的）：

```
./dataset/
├── HY15/
│   ├── Action2V/                       # SFT latents（§1）
│   │   ├── latents/
│   │   └── train_index.json            # = $INDEX
│   └── Action2V_ode/                   # ODE 数据（本步）
│       ├── latents/                    # 每 clip 一个 .pt，含 ode_trajectory
│       └── train_index.json            # = $ODE_INDEX
└── others/HY/Action2V/                 # CFG 负向 embedding（复用）
```

</details>

<details>
<summary><big><b>2.4 Stage-2(a) Causal ODE Distillation (`stage2_ar_ode`)</b></big></summary>

在 TF teacher 预解好的 6 点轨迹上回归拟合 causal student。`ARODERecipe` +
`ODERegressionLoss`，`ODETrajectorySample` 每步采一个去噪锚点
（`[1000,750,500,250]`，`warp_denoising_step=True`）。

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29638 \
    tools/train_mwm.py \
    --config-file configs/hy/action2v/train/stage2_ar_ode.py \
    --output-dir outputs/hy_action2v_ar_ode \
    training.sp_size=2
```

- 初始化：config 默认 `_from_pretrained=$CKPT_ROOT/stage1_ar_tf`——**已指向 TF 导出，权重
  放对位置就不用覆盖**。
- 数据：config 默认读 `$ODE_INDEX`（`CausalODEDataset`）；给成普通 `$INDEX` 会报缺
  `ode_trajectory` 键。
- 超参（config 默认）：Muon `lr=1e-5`；scheduler 与别人不同——`schedule="shifted"` +
  `timestep_shift=5.0` + `sigma_min=0.0` + `extra_one_step=True`。轨迹是在 shift=5.0 的
  σ 表上解的，只有 `shifted` 会把 shift 烘进 `scheduler.timesteps`，warp 才能还原每个快照
  真正对应的 σ（`linear` 会把 4 个锚点里 3 个标错时间步）。这些在 config 里，不用覆盖。
  `max_steps=10000` / `ckpt_interval=1000`。

训完导出 → `$CKPT_ROOT/stage2_ar_ode/`。它是 ODE few-step 推理的 checkpoint，
也是 DMD 的备选起点。

---

</details>

<details>
<summary><big><b>2.5 Stage-2(b) Causal Consistency Distillation (`stage2_ar_cd`)</b></big></summary>

把 causal TF 模型蒸馏成 few-step 一致性模型：冻结 teacher 走一步 CFG Euler `t→t_next`，
student 预测 `t` 处 `x0`、EMA 网络预测 `t_next` 处 `x0`，loss 是两者 `x0` 的 MSE。
**数据回到 SFT 的 `$INDEX`**（不需要 ODE 预处理）。

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

- **初始化必给（三份）**：**student / teacher / ema 全从同一个 TF 权重 seed**。config 默认
  三份都指 base，串行训练要从 TF 权重起，所以这三条覆盖必给。`ARCDRecipe` 在第一步用
  `copy_params` 把 student 拷进 teacher / ema，三份都从同一目录加载时这次拷贝就是 no-op。
  teacher + ema 在 config 里声明为冻结的 `auxiliary_models`。
- 负向 prompt：`recipe.adapter.neg_*` 已是 config 默认（§1.2），放对路径不用管。
- 超参（config 默认）：Muon `lr=1e-5`、`discrete_cd_n=50`、`timestep_shift=5.0`、
  `guidance_scale=5.0`、`ema_decay=0.999`、`activation_checkpointing=True`（三份模型，显存较紧，
  别关）；`max_steps=200000` / `ckpt_interval=500`。

训完导出 → `$CKPT_ROOT/stage2_ar_cd/`（CD few-step 推理的 checkpoint +
**DMD 的 generator 种子**）。

---

</details>

<details>
<summary><big><b>2.6 Stage-3 Asymmetric DMD with Self Rollout (`stage3_ar_dmd`)</b></big></summary>

**如果跳过前面的初始化阶段，可下载官方提供的 ODE 或 CD checkpoint：**

```bash
# 默认的 ODE 初始化
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/causal_ode/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT causal_ode stage2_ar_ode
)

# 或使用 CD 初始化
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/causal_cd/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT causal_cd stage2_ar_cd
)
```

下载后 checkpoint 分别位于 `./ckpts/HY15/Action2V/causal_ode/` 和
`./ckpts/HY15/Action2V/causal_cd/`，上面的软链接分别映射为 `stage2_ar_ode` 和
`stage2_ar_cd`。DMD 还需要 bidirectional SFT checkpoint 作为
`real_score` / `fake_score` 的种子；若本地没有，可按 2.2 的命令下载，或在命令行显式
覆盖对应的 `model._from_pretrained` 路径。

Distribution Matching Distillation：generator 从纯噪声 self-roll 一段假视频（AR cm rollout，
单步截断 BPTT），把它的分布对齐到冻结 real_score 的 CFG 引导分布，同时可训的 fake_score
critic 学 generator 的分布。**无真实视频监督**——数据只用来取条件。三份模型：

| aux 名 | 角色 | 种子 |
|---|---|---|
| `model` (主模型) | student / generator, trainable, causal | CD |
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

- **初始化必给（三份）**：generator ← CD，real/fake score ← SFT。config 默认三份都指 base，
  串行训练要这么接，所以这三条覆盖必给。也可让 generator 从 ODE 起步
  （把 `model._from_pretrained` 换成 `$CKPT_ROOT/stage2_ar_ode`）。
- 负向 prompt：`recipe.adapter.neg_*` 已是 config 默认（§1.2）。
- 超参（config 默认）：generator Muon `lr=1e-5`、critic `lr=8e-6`，都 `weight_decay=0.01` /
  `betas=(0.0,0.999)`；`dfake_gen_update_ratio=5`（generator 每 5 步更新一次，critic 每步）、
  `guidance_scale=5.0`、`denoising_step_list=[1000,750,500,250]`、`num_frame_per_block=4`、
  `activation_checkpointing=True`（三份模型 + rollout，**关掉必 OOM**）；`max_steps=200000` /
  `ckpt_interval=500`。

训完导出 → `$CKPT_ROOT/stage3_ar_dmd/`，**4-step 实时推理**的权重。
**HY stage3 没有 `generator_ema`**（aux 只有 `real_score` / `fake_score`），所以导出用默认
`--model-key model`，**不要**用 `aux/generator_ema`（那是 Wan 线的做法，§3.1）。

---

</details>

## 3. 推理流程

<details>
<summary><big><b>3.1 导出 checkpoint (DCP 目录 → diffusers 目录)</b></big></summary>

训练存的是 **FSDP2 DCP 分片目录** `ckpts/checkpoint_{step}/`（每卡一个 `.distcp` +
一个 `.metadata`），`torch.load` 读不了。`tools/export_checkpoint.py` 合并分片、剥
FSDP / compile 前缀，写成下一段训练 / 推理能直接吃的 diffusers 目录。

```bash
BEST_STEP=3000    # 由验证选出，须是 ckpt_interval 的倍数
# HY 推理 / 下一段训练用 diffusers 目录（_from_pretrained），导 diffusers 格式，需 --config：
python tools/export_checkpoint.py \
    --checkpoint outputs/hy_action2v_ar_tf/ckpts/checkpoint_${BEST_STEP} \
    --output   "$CKPT_ROOT/stage1_ar_tf" \
    --format   diffusers \
    --config   configs/hy/action2v/train/stage1_ar_tf.py
```

- 单进程、纯 CPU、不用 torchrun。`--checkpoint` 必须指向真实存在的 DCP 目录。
- **DMD 也是 `--model-key model`（默认）**：HY stage3 没有 `generator_ema`（§2.6），主模型
  `model` 就是要导的 generator。别照搬 Wan README 的 `aux/generator_ema`。
- `--format` 还能选 `pt` / `safetensors`；HY 的 infer / train config 走 `_from_pretrained`，
  用 `diffusers`。

导出目标是 diffusers 目录（`config.json` + safetensors）。8.55B 模型 > diffusers 默认
`max_shard_size=10GB`，`save_pretrained` 会**自动分片**成多个 `-0000N-of-M` + 一个
`.index.json`；**分片不影响加载**——`from_pretrained`（即 config 的 `_from_pretrained`）对
单文件和分片一视同仁。全流程各段都导到 `$CKPT_ROOT` 下，布局：

```
./ckpts/HY15/Action2V/                                    # = $CKPT_ROOT
├── stage0_bi_sft/                                        # SFT 导出（§2.1）
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00002.safetensors
│   ├── diffusion_pytorch_model-00002-of-00002.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json    # 分片索引
├── stage1_ar_tf/                                         # TF 导出（§2.2）
├── stage2_ar_ode/                                        # ODE 导出（§2.4）
├── stage2_ar_cd/                                         # CD 导出（§2.5）
└── stage3_ar_dmd/                                        # DMD 导出（§2.6），4-step 实时推理权重
```

#### 3.1.1 单权重的双重用途

`$CKPT_ROOT/{stage}/` 既是**本 stage 推理的 `_from_pretrained`**，又是**下一 stage 训练的
`model._from_pretrained`**——一次导出两处共用。所以每段训完就导到这里，下一段（§2.2/§2.4/§2.5/§2.6
的初始化覆盖，以及 §2.3.1 的 ODE 采样）直接引用。

---

</details>

<details>
<summary><big><b>3.2 推理</b></big></summary>

五个训练阶段各有一个对应的 infer config（`configs/hy/action2v/infer/`），共用同一入口
`tools/infer_mwm.py`。loop / sampler / guidance / 步数**全由 `--config-file` 决定**，
所以换 stage 只需换 config——命令形状不变。config 里的 `inference.checkpoint` 已指向
`$CKPT_ROOT/{stage}/`（§3.1 导出的目录），放对位置直接跑即可。

命令行**只有** `--config-file` 加尾随 dotlist（`inference.*=` / `model.*=`）；值型 flag
（`--checkpoint` / `--output-dir` / `--prompts` / `--input-json` / `--limit` …）已全部移除。

**输入是 benchmark JSON**（`inference.benchmark`）：`[{id, caption, trajectory}]`，多余键忽略，
每条出一个 `{id}.mp4`。**HY 是 i2v，所以每项都要带 `image`**（相对路径按 JSON 所在目录解析）；
没有 `image` 的项会退化成 t2v。

如果只做官方 4-step DMD 推理、跳过训练阶段，可直接下载最终 checkpoint：

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "HY15/Action2V/dmd/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/HY15/Action2V || exit 1
    ln -sfnT dmd stage3_ar_dmd
)
```

下载后 checkpoint 位于 `./ckpts/HY15/Action2V/dmd/`，软链接会将它映射为
`$CKPT_ROOT/stage3_ar_dmd/`。

各 stage 的差异（config 默认，无需手动传）：

| stage | config | `inference.loop` | `sampler.solver` | 步数 | guidance |
|---|---|---|---|---|---|
| SFT | `infer/stage0_bi_sft.py` | `BidirectionalGenerationLoop` | `UniPCSolver` | 50 | 6.0 |
| TF | `infer/stage1_ar_tf.py` | `ARGenerationLoop` | `EulerSolver` | 50 | 6.0 |
| ODE | `infer/stage2_ar_ode.py` | `ARGenerationLoop` | `EulerSolver` | 4 | 1.0（关） |
| CD | `infer/stage2_ar_cd.py` | `ARGenerationLoop` | `EulerSolver` | 4 | 1.0（关） |
| DMD | `infer/stage3_ar_dmd.py` | `ARGenerationLoop` | `EulerSolver` | 4 | 1.0（关） |

`loop` / `solver` 都是**类名字符串**，按名字解析（`minwm/engine/inference/{loop,samplers}.py`），
所以换 loop / 换 sampler 全在 config 里。

公共默认：`num_frames=20`（→ 480×832 像素、20 latent 帧）、`fps=16`、`sp_size=1`、
`seed=42`。**轨迹是逐条的**，来自 benchmark item 的 `trajectory` 字段。
五个 stage 都设了 `vae_tiling=False`（HY 的 3D 卷积全量解码，开 tiling 会 OOM）。

**五个阶段分别推理（单卡，每条只跑 benchmark 前 2 条）**：

**SFT（50 步双向 CFG）**：

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage0_bi_sft.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_sft
```

**AR-TF（50 步 teacher-forcing）**：

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage1_ar_tf.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_ar_tf
```

**AR-ODE（4 步）**：

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage2_ar_ode.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_ar_ode
```

**AR-CD（4 步）**：

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage2_ar_cd.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_ar_cd
```

**DMD（4 步，最终产物）**：

```bash
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/hy/action2v/infer/stage3_ar_dmd.py \
    inference.benchmark=assets/example.json \
    inference.limit=2 \
    inference.strict=False \
    inference.output_dir=outputs/infer_hy_action2v_dmd
```

- 常用覆盖：`inference.output_dir=`、`inference.limit=N`（只跑前 N 条）、`inference.seed=`、
  `inference.dtype=`、`inference.sp_size=N`（配 `--nproc_per_node=N` 开序列并行）。
- 每条输出一个 mp4（480×832 / 20 latent 帧 → 77 像素帧 @ 16fps）+ `manifest.json`
  （记录 config 与每条 `{index, id, prompt, trajectory, seed, video}`）。
- **轨迹语义**：`key*N` 段用逗号拼接，如 `d*8,i*5,l*6` = 右移 8 + 上仰 5 + 左移 6；
  `w/s/a/d` 平移、`i/k/j/l` 转视角，20 latent 帧对应 19 段。

**产物布局**（`inference.output_dir` 下）：

```
outputs/infer_hy_action2v_dmd/       # = inference.output_dir
├── {id}.mp4                          # benchmark 每条一个（480×832 / 77 帧 @16fps）
├── manifest.json                     # config + 每条 {index, id, prompt, trajectory, seed, video}
└── final_with_keys.mp4               # 叠按键拼接总览（跑了 overlay 才有）
```

---

</details>

## 4. 辅助工具与运行维护

<details>
<summary><big><b>4.1 叠按键 + 拼接</b></big></summary>

**叠按键指示器（可选，纯 CPU）**：把 WASD/KIJL 按键叠到每个 clip 上再拼成总览，读
`manifest.json`，对上面任一输出目录都适用：

```bash
python demos/overlay_from_manifest.py \
    --input-dir outputs/infer_hy_action2v_dmd \
    --output final_with_keys.mp4
```

> 需要 `ffmpeg` / `ffprobe` 在 `PATH` 上；集群镜像常没有，这步在有 ffmpeg 的机器上跑
> （读共享盘上的输出目录即可）。

</details>

<details>
<summary><big><b>4.2 训练健康检查、显存与并行</b></big></summary>

训练启动后先确认以下状态：

**① 没有 OOM、loss 正常**

```bash
grep "num_ooms: [^0]" outputs/<run>/logs/log.txt          # 应为空
grep -riE "traceback|out of memory" outputs/<run>/logs/    # 应为空
```

loss 初期在合理量级、随步数下降，`grad_norm` 不发散。DMD 看 `generator_loss` /
`critic_loss` 两条，注意 `dfake_gen_update_ratio=5` 的更新不对称。

**② checkpoint 正常落盘**

```bash
grep "saved checkpoint" outputs/<run>/logs/log.txt         # 每个 ckpt_interval 一行 + 收尾一行
```

checkpoint 落 `outputs/<run>/ckpts/checkpoint_{step}/`，`ckpts/latest.txt` 指向最新。

**③ 续训**：崩了 / 手动 kill 后接着跑，在原命令上加 `checkpoint.resume=True`（`--output-dir`
不变）。resume 从 `<output_dir>/ckpts/latest.txt` 恢复全部训练态（model + optimizer + aux +
step）从断点续，**优先于**初始化的 `_from_pretrained`；找不到 `latest.txt` 会大声报错，
不会静默回退到 fresh init。其余超参保持不变。

各阶段的显存压力（相对，同 `sp_size` 下）：

| 阶段 | 显存压力 | 说明 |
|---|---|---|
| SFT | 最低 | 单模型，backward 占大头 |
| TF / ODE | 中 | 单模型 + teacher-forcing / ODE 回归 |
| CD | 高 | 三份模型（student + teacher + ema） |
| DMD | **最高** | 三份模型 + self rollout，最易 OOM |

- **DMD 最易 OOM**。config 的 `activation_checkpointing=True` 是保命设置，别关；真 OOM 就
  **调大 `training.sp_size`**（把激活摊到更多卡）。
- **约束只有一条：`$NPROC` 必须能被 `sp_size` 整除**，否则起不来。
- `sp_size` 调大时，`dp_size = $NPROC / sp_size` 随之变小、GBS 变小，但单样本被切得更细、
  显存更省；单步因 SP 通信略增。

---

</details>

<details>
<summary><big><b>4.3 全流程串行速查</b></big></summary>

每段训完导出 diffusers 目录（§3.1），下一段引用它。`<BEST_STEP>` 由验证选，须是
`ckpt_interval` 的倍数：

```bash
# ① encode (原始视频 → latent + index)          → $INDEX                        (§1)
# ② SFT (config 默认从 base)                     → 导出 stage0_bi_sft/            (§2.1, §3.1)
# ③ TF  (model._from_pretrained=stage0_bi_sft)   → 导出 stage1_ar_tf/             (§2.2, §3.1)
# ④ ODE 预处理 (TF teacher 48 步预解)            → $ODE_INDEX                     (§2.3)
# ⑤ ODE (config 默认从 stage1_ar_tf)             → 导出 stage2_ar_ode/            (§2.4, §3.1)
# ⑥ CD  (student/teacher/ema=stage1_ar_tf)       → 导出 stage2_ar_cd/             (§2.5, §3.1)
# ⑦ DMD (gen=stage2_ar_cd, score=stage0_bi_sft)  → 导出 stage3_ar_dmd/            (§2.6, §3.1)
# ⑧ 推理 (换 infer config + $CKPT_ROOT/{stage})  → outputs/infer_*/ + 叠按键      (§3.2, §4.1)
```

④⑤ 与 ⑥ 都从 ③ 起步、互不依赖；⑦ 同时要 ⑥（generator 种子）和 ②（real/fake score 种子）。
ODE 预处理（④）是全流程最重的一段，要 sm_80 及以上的 flash-attn kernel（§2.3.1）。
各段 `--master_port` 已错开（29635–29640），依次跑不用改端口。

---

</details>

<details>
<summary><big><b>4.4 排错</b></big></summary>

| 现象 | 原因 / 处理 |
|---|---|
| `world_size not divisible by sp_size` 起不来 | `$NPROC` 不能被 `training.sp_size` 整除（§0.4）。 |
| 编码后 `train_index.json` 为空 / 少条目 | 视频目录布局不对：须 `$SRC_VIDEOS/<idx:06d>_<suffix>/gen.mp4`，`suffix` 见 §1。 |
| ODE 采样 flash-attn kernel 报错 | 需要 sm_80 及以上（A100/A800 ✓，B200 缺）（§2.3.1）。 |
| ODE 训练报缺 `ode_trajectory` 键 | `json_path` 给成了普通 `$INDEX`。ODE 必须用 `$ODE_INDEX`（§2.4）。 |
| TF/CD/DMD 训出来像没微调 / 从 base 起 | 漏了 `model._from_pretrained`（及 CD/DMD 的 aux 覆盖）。config 默认是 base，串行训练必给（§2.2/§2.5/§2.6）。 |
| CD / DMD CFG 相关报错 / uncond 为空 | 负向 embedding 路径不对。config 默认在 `$NEG_PROMPT` / `$NEG_BYT5`，放对或显式覆盖 `recipe.adapter.neg_*`（§1.2）。 |
| DMD OOM | 别关 `gradient_checkpointing`；或调大 `training.sp_size` 分摊激活（§4.2）。 |
| `double_blocks.*.img_attn_prope_proj.* newly initialized` 告警 | 正常。PRoPE 相机参数在 base 里不存在、新初始化（§2.1）。 |
| 导出 DMD 效果不对 / 找不到 `aux/generator_ema` | HY stage3 没有 generator_ema，用默认 `--model-key model`（§2.6、§3.1）。 |
| 依赖缺失（lmdb、decord…） | 用装好依赖的 conda env（§0.1）。 |
| DCP 目录 `torch.load` 读不了 | 分片目录不是单文件，先 `tools/export_checkpoint.py` 合并（§3.1）。 |
| `unrecognized arguments: --checkpoint / --prompts / --input-json ...` | 抄了旧的 flag 写法。`infer_mwm.py` 现在只吃 `--config-file` + 尾随 dotlist（`inference.benchmark=` / `inference.output_dir=` …），见 §3.2。 |
| `unknown key(s) '...' in Inference` | `inference` 节里有拼错的键。该节是强 schema（`minwm/config/schema.py:Inference`），错键直接报错；照报错列出的合法键名改。 |
| HY 双向推理 OOM（`Tried to allocate ... GiB`） | `inference.vae_tiling` 必须是 `False`（HY 的 3D 卷积解码），五个 infer config 都已设好；自己新写 config 时别漏（§3.2）。 |
| 推理找不到权重 / `build_model` 报 `OSError` | config 的 `_from_pretrained` 指向 `$CKPT_ROOT/{stage}/`，训完先导出到那里（§3.1）。 |
| 叠按键报 `ffmpeg not found` | `overlay_from_manifest.py` 要 ffmpeg/ffprobe，在有它的机器上跑（§4.1）。 |

---

</details>

<details>
<summary><big><b>4.5 参考</b></big></summary>

- 数据编码脚本与 `.pt` 键约定：[`tools/data/hy/README.md`](../../../tools/data/hy/README.md)。
- 各 stage config：`configs/hy/action2v/train/` + `configs/hy/action2v/infer/`。
- 权重 / 数据集下载、命名软链、Quick Start 推理：主 [`README.md`](../../../README.md)。
- 安装：`INSTALL.md`。
- Wan backbone 对照：`configs/wan21/action2v/README_cn.md`。

</details>
