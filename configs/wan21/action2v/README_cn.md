# Wan2.1 Action2V

> English version: [README.md](README.md)

数据 encode → SFT → TF → ODE data curation → ODE → CD → DMD → 推理 → 叠按键。


**阶段链**（每段的产物是下一段的输入，必须串行）：

```
encode ─→ SFT ─→ TF ─┬─→ ODE curation ─→ ODE ─┐
                     └─→ CD ───────────────────┴─→ DMD
```

`ODE` 与 `CD` 都从 TF 权重起步、互不依赖；`DMD` 需要 CD (student+ema) 和 SFT
(teacher+critic) 两份权重。每段训完都要把 DCP 目录导出成单文件 `.pt`（§3.1），
它同时是**本段推理的 checkpoint** 和**下一段训练的 `checkpoint.pretrained`**。

---

## 0. 前置

<details>
<summary><big><b>0.1 ⭐ 配置块（先改这里，后面所有命令都引用它）</b></big></summary>

**本文档后面的每条命令都假设你已经 source 过这一段。** 只有这里需要按你的环境改；
其余章节一律用变量，照抄即可。建议存成 `env.sh` 放在仓库外，每开一个新 shell `source` 一次。

```bash
# ---- (1) 仓库与 Python 环境 -------------------------------------------------
cd /path/to/minWM                      # ← 改成你的 checkout 路径
export PROJECT_ROOT="$PWD"
conda activate minwm                   # ← 见 INSTALL.md

# ---- (2) 数据集短名：贯穿所有产物路径的唯一标识 -----------------------------
# 随便起，只要在你这台机器上唯一。所有 dataset/ outputs/ 路径都由它派生，
# 所以换一份数据只改这一行，各 run 自然不撞车。
export DS=my_dataset                   # ← 改成你的数据集名

# ---- (3) 源数据（只有这两个是仓库外的绝对路径）------------------------------
# 视频目录布局必须是 <VIDEO_DIR>/<idx:06d>_<suffix>/gen.mp4，见 §1。
export SRC_JSON=/abs/path/to/preencode_input.json   # ← caption + pose_str
export SRC_VIDEOS=/abs/path/to/videos               # ← 视频根目录

# ---- (4) 派生路径（不用改）--------------------------------------------------
export BASE="$PROJECT_ROOT/ckpts/Wan2.1-T2V-1.3B"   # Wan2.1 base（§0.2 下载）
export DATA_ROOT="dataset/Wan21/Action2V_$DS"       # 编码/curation 产物
export LMDB="$DATA_ROOT/data"                       # SFT/TF/CD/DMD 读的 LMDB
export ODE_LATENTS="$DATA_ROOT/ode_latents"         # ODE per-clip .pt
export ODE_LMDB="$DATA_ROOT/ode_lmdb"               # ODE 训练读的 LMDB
export CKPT_ROOT="ckpts/Wan21/Action2V"             # 导出的单文件权重
export NPROC=8                                      # GPU 数

# ---- (5) 评测用 benchmark JSON（可选，§3.2.5）-------------------------------
# 格式 [{id, caption, trajectory}]；留空就用各 infer config 自带的默认 benchmark。
export BENCH="assets/example_t2v.json"              # ← 评测用 benchmark JSON

# ---- (6) 运行时环境 --------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
```

> `$NPROC` 按实际卡数设（`nvidia-smi -L | wc -l`）。**卡数必须能被 `training.sp_size`
> 整除**，否则起不来——见 §0.3 的对照表。
>
> `$DS` 是整套路径的唯一变量：`DATA_ROOT` / 训练输出 / 推理输出全由它派生，所以同一台机器
> 上跑多份数据只需换 `DS`，产物天然隔离。

</details>

<details>
<summary><big><b>0.2 base 模型</b></big></summary>

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$BASE" \
    --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" \
              "diffusion_pytorch_model.safetensors" "config.json"
```

所有 train / infer config 里写的是 `_BASE = "./ckpts/Wan2.1-T2V-1.3B"`（仓库内相对路径），
所以 `$BASE` 指到别处时要么软链过去，要么在命令里加 `model._from_pretrained=$BASE`。
另外 ODE curation 工具把 base 路径写死成 `wan_models/Wan2.1-T2V-1.3B/`，务必建这个软链：

```bash
mkdir -p wan_models && ln -sfn "$BASE" wan_models/Wan2.1-T2V-1.3B
```

</details>

<details>
<summary><big><b>0.3 拓扑与并行（按你的卡数选 `sp_size`）</b></big></summary>

`总卡数 = sp_size × DP`，`GBS = DP × data.batch_size`。约束只有一条：**`$NPROC` 必须能被 `sp_size` 整除**。

本文档命令统一写 `training.sp_size=2`（1.3B 在 20 latent 帧下的稳妥选择）。`sp_size` 越小 DP/GBS 越大、显存压力越大;
越大则单样本被切得越细、显存越省但通信越多。

</details>

<details>
<summary><big><b>0.4 路径约定</b></big></summary>

全部由 §0.1 的 `$DS` 派生，**换数据集只改 `DS` 一处**，各 run 的产物天然隔离：

| 角色 | 路径 | 变量 |
|---|---|---|
| 源 JSON / 视频 | 仓库外，你自己的数据 | `$SRC_JSON` / `$SRC_VIDEOS` |
| SFT/TF/CD/DMD LMDB | `dataset/Wan21/Action2V_$DS/data` | `$LMDB` |
| ODE per-clip `.pt` | `dataset/Wan21/Action2V_$DS/ode_latents` | `$ODE_LATENTS` |
| ODE LMDB | `dataset/Wan21/Action2V_$DS/ode_lmdb` | `$ODE_LMDB` |
| 训练输出 | `outputs/wan_action2v_{stage}_$DS/` | — |
| 导出单权重 | `ckpts/Wan21/Action2V/{stage}/model.pt` | `$CKPT_ROOT/{stage}/model.pt` |
| 推理输出 | `outputs/infer_{stage}_$DS/{step}/` | — |

`{stage}` ∈ `stage0_bi_sft` / `stage1_ar_tf` / `stage2a_ar_ode` / `stage2_ar_cd` / `stage3_ar_dmd`。

> **权重路径不带 `$DS`**：`$CKPT_ROOT/{stage}/model.pt` 是各 config 的默认值（写死在
> config 里），所以同一台机器上跑多份数据时**这些导出权重会互相覆盖**。要并存就自己加后缀，
> 例如 `--output "$CKPT_ROOT/stage0_bi_sft/model_$DS.pt"`，推理时 `--checkpoint` 指同一个。
>
> config 里数据路径默认指向 `./dataset/Wan21/Action2V/{data,ode_lmdb}`（不带 `$DS`），
> 所以本文档命令一律显式覆盖 `data.dataset.data_path=`；也可以反过来软链：
> `ln -sfn "$PROJECT_ROOT/$LMDB" dataset/Wan21/Action2V/data`。

</details>

<details>
<summary><big><b>0.5 监控</b></big></summary>

**默认不用 wandb**：不传 `monitor.*`，指标只走 STDOUT（`training.log_interval=10`
每 10 步一行：loss + steps/sec + ms/step + 峰值显存）。要 wandb 就自己配，
**key 只放在 shell / 仓库外的文件里，别写进任何进 git 的文件**：

```bash
export WANDB_API_KEY='你的 key'
export WANDB_ENTITY='你的 entity'      # 不设则用你账号的默认 entity
# 训练命令尾部追加（run name 自己起，建议带上 $DS 和 sp_size 便于区分）：
#   monitor.backends="['wandb']" monitor.wandb_project=wan21 \
#   monitor.wandb_run_name="sft-$DS-sp4"
```

</details>

---

## 1. 数据 encode (VAE → LMDB)

<details>
<summary><big><b>1.1 数据编码</b></big></summary>

把 WorldPlayGen 视频 + caption + `pose_str` 编码成 Wan VAE latent 的合并 LMDB。
每 rank 写自己的 `.rank_{r}` 分片 → rank0 流式合并进 `data/` 并删分片（内存有界）。

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29700 \
    tools/data/wan21/build_worldplaygen_lmdb.py \
    --input_json "$SRC_JSON" \
    --video_dir  "$SRC_VIDEOS" \
    --output_dir "$DATA_ROOT" \
    --vae_path   "$BASE/Wan2.1_VAE.pth" \
    --target_h 480 --target_w 832
```

产出 `$LMDB`（= `$DATA_ROOT/data`，LMDB 目录，`data.mdb` + `lock.mdb`）。

**LMDB 键契约**：

```
latents    (N, 20, 16, 60, 104) float16   Wan VAE latent（77 像素帧 → 20 latent 帧）
prompts    (N,)                 str       caption
intrinsics (N, 4)               float32   [fx/W, fy/H, cx/W, cy/H]
poses      (N, 20, 7)           float32   [tx,ty,tz, qx,qy,qz,qw]（w2c）
```

`N` = 有效 clip 数，跑完看 rank0 的合并日志（参考量级：某个 14975 条的数据集 0 丢失）。

**输入格式要求**（换数据集时按这个准备）：

- `$SRC_JSON`：每条含 `caption` + `pose_str`。
- `$SRC_VIDEOS`：布局必须是 `<$SRC_VIDEOS>/<idx:06d>_<suffix>/gen.mp4`，其中
  `suffix` = 该条 `pose_str` 转小写后删掉所有非 `[a-z0-9]` 字符；`idx` 是它在 JSON 里的下标。
- 视频 77 帧、480×832（`--target_h/--target_w` 会 resize，帧数不符会被跳过）。

**注意点**：

- 多机版靠共享盘 + NCCL barrier 合并分片；单机就是 `$NPROC` 个分片本地合并，不需要共享盘。
- 编码内部细节（Wan21VAE z_dim=16 归一化、`pose_str`→相机轨迹合成、键契约）见
  [`tools/data/wan21/README.md`](../../../tools/data/wan21/README.md)。

</details>

<details>
<summary><big><b>1.2 数据体检（可选但推荐）</b></big></summary>

训之前先确认相机轨迹 / caption / 画面对得上。把 latent 解回像素，每 clip 出一个
`[RGB | BEV]` 并排 mp4（俯视 X-Z 轨迹 + yaw 箭头 + pitch 表 + 路径长度 + caption 条），
并跑全库尺度普查（`trans_span` / `path_len` / `rot_span` 直方图 + 离群清单）：

```bash
python tools/data/wan21/check_dataset_bev.py \
    --data_path "$LMDB" \
    --vae "$BASE/Wan2.1_VAE.pth" \
    --num_videos 8 --concat 10 --gpu 0
```

只要尺度普查图、不要视频（不需要 GPU）：加 `--scale_plot`。
逐 flag 说明见 [`tools/data/wan21/README.md`](../../../tools/data/wan21/README.md)。
位姿归一化与训练完全一致，所以看到的就是模型看到的。

---

</details>

## 2. 训练流程

<details>
<summary><big><b>2.1 Phase-1 Bidirectional SFT (`stage0_bi_sft`)</b></big></summary>

双向 + 相机 (PRoPE) 监督微调。`Wan21Model` + `BiSFTRecipe` + `FlowMatchingLoss`，
读 §1 的 clean-latent camera LMDB（viewmats / Ks）。

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

- 初始化：`_from_pretrained=./ckpts/Wan2.1-T2V-1.3B`（非 strict 加载 base 权重）。
  **无 `checkpoint.pretrained`**——SFT 是链条起点。
- 超参（config 默认）：`lr=2e-6`、`betas=(0.0,0.999)`、`weight_decay=0.01`、
  `batch_size=1`、`num_workers=4`、`activation_checkpointing=True`。
- config 默认 `max_steps=10000` / `ckpt_interval=1000`；上面覆盖成集群 run 用的
  `20000` / `2500`。冒烟跑：`training.max_steps=100 training.ckpt_interval=50`。
- 输出：`outputs/wan_action2v_sft_$DS/{ckpts/checkpoint_{step}/, logs/}`，
  `ckpts/latest.txt` 指向最新。**checkpoint 是 FSDP2 DCP 分片目录，不是单 `.pt`**（见 §3.1）。

**首个 step 的正常告警**：一大批
`blocks.*.self_attn.prope_o.* newly initialized` + `You should probably TRAIN this model...`
——PRoPE 相机投影参数在 base 里不存在、新初始化，**符合预期**。

**显存很松时可以试的提速**：`model.gradient_checkpointing=False`（backward 占单步
~70%，部分是重算开销）。不改 GBS，纯换显存换速度。

**验证 override 生效**（不起训练）：

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

训完选最优 step 导出（§3.1）→ `$CKPT_ROOT/stage0_bi_sft/model.pt`。
它既是 SFT 推理的 checkpoint，也是 **TF 训练的起点** + **DMD 的 teacher/critic 种子**。

---

</details>

<details>
<summary><big><b>2.2 Phase-2 Stage-1 Teacher-Forcing AR (`stage1_ar_tf`)</b></big></summary>

**如果跳过上一阶段，可下载官方提供的上一阶段 checkpoint：**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/bidirectional/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT bidirectional stage0_bi_sft
)
```

下载后 checkpoint 位于 `./ckpts/Wan21/Action2V/bidirectional/`，上面的软链接会将它映射为
本文档中的 `$CKPT_ROOT/stage0_bi_sft/model.pt` 路径；也可以在命令行显式覆盖
`checkpoint.pretrained`。

把双向 SFT 模型转成 causal + teacher forcing 的 AR diffusion。`CausalWan21Model`
(`num_frame_per_block=4`, `local_attn_size=20`) + `ARTFRecipe` + `FlowMatchingLoss`，
`causal=True` 的 `Wan21Adapter`，`use_prope=True` 保留 PRoPE 参数。**数据与 SFT 同一个
LMDB**。

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

- **必须先有 SFT 导出**：`$CKPT_ROOT/stage0_bi_sft/model.pt`（§3.1）。
  trainer 的 loader 会自动剥 `generator` / `model.` 前缀。
- 超参同 SFT (`lr=2e-6`, `betas=(0.0,0.999)`)。TF 比 SFT 慢约 2.3×。
- 输出：`outputs/wan_action2v_tf_$DS/`。

训完导出 → `$CKPT_ROOT/stage1_ar_tf/model.pt`。它是 **ODE curation 的 teacher**、
**ODE 训练的起点**、**CD 训练的 student/teacher/ema 三份种子**。

---

</details>

<details>
<summary><big><b>2.3 Stage-2(a) ODE data curation (数据准备，不是训练)</b></big></summary>

**TF teacher checkpoint（2.2 的产物；如果跳过 TF 阶段，可下载官方提供的 checkpoint）：**

```bash
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/ar_diffusion_tf/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT ar_diffusion_tf stage1_ar_tf
)
```

下载后 checkpoint 位于 `./ckpts/Wan21/Action2V/ar_diffusion_tf/`，上面的软链接会将它映射为
下面 ODE curation 命令默认使用的 `$CKPT_ROOT/stage1_ar_tf/model.pt`；如路径不同，也可
显式覆盖 `--generator_ckpt`。

用**冻结的 TF teacher** 给 SFT LMDB 里每个 clip **预解一条 6 点 ODE 轨迹**、冻进一个新
LMDB，供 §2.4 的 `ARODERecipe` 回归拟合。**无 config、纯 CLI 工具**，两步。

#### 2.3.1 采样 (48 步 CFG flow → per-clip `.pt`)

```bash
torchrun --nproc_per_node="$NPROC" --master_port=29703 \
    tools/data/wan21/ode/get_causal_ode_data_prope.py \
    --generator_ckpt $CKPT_ROOT/stage1_ar_tf/model.pt \
    --rawdata_path   "$LMDB" \
    --output_folder  "$ODE_LATENTS" \
    --guidance_scale 6.0
```

- 加载 TF causal 模型（取 `state_dict["generator"]`、剥 FSDP/compile 前缀、`strict=True`）
  + umt5 T5。对每个 clip 以其 `clean_latent` + 相机 `viewmats`/`Ks` 为条件跑 **48 步 CFG
  flow**（`shift=5.0`, `sigma_min=0.0`, `extra_one_step`, guidance 6.0），保留下标
  `[0,12,24,36,-2,-1]` → **6 点** = 4 个去噪锚点 (t≈1000/750/500/250) + 48 步 target + clean。
- 按 `index * world_size + rank` 分片，每 clip 写 `{idx:05d}.pt`：
  `{prompt, latents(1,6,20,16,60,104), viewmats(1,20,4,4), Ks(1,20,3,3)}`。
- **必须从仓库根跑**：T5 / 模型 `config.json` 路径写死在 `wan_models/Wan2.1-T2V-1.3B/`（§0.2 软链）。
- **要求 sm_80 及以上的 flash-attn kernel**（A100/A800 一类 ✓；实测 B200 上缺对应 kernel）。
  拿不准就先跑一小批（`--num_videos` 那种试探不适用，这里直接看头几个 `.pt` 是否落盘）。
- **这是全流程最重的一段**：每 clip 要跑 48 步 × 2 次前向（CFG），单卡摊到
  `ceil(N / $NPROC)` 个 clip（`N` = LMDB 条数）。量级感受：某个 ~15k 条的数据集在
  128 卡上是 ~117 clip/GPU，卡少就按比例放大。**先用小数据集试通再上全量。**
- 进度看已落盘的 `.pt` 数递增，别信 tqdm（`\r` 帧重定向后不落盘）：
  ```bash
  ls "$ODE_LATENTS" | wc -l        # 目标 = LMDB 条数 N
  ```

#### 2.3.2 合并 (per-clip `.pt` → 单 LMDB)

单进程、纯 CPU。torchrun 末尾的 `dist.barrier()` 保证所有 rank 的 `.pt` 都已落盘。

```bash
python tools/data/wan21/ode/wan_utils/build_ode_prope_lmdb.py \
    --input_dir  "$ODE_LATENTS" \
    --output_dir "$ODE_LMDB" \
    --map_size_gb 10000
```

产出 `ode_lmdb/{data.mdb,lock.mdb}`，键为
`latents_{i}_data` / `prompts_{i}_data` / `viewmats_{i}_data` / `Ks_{i}_data`
+ `latents_shape="N 6 20 16 60 104"`。预期约 46 GB（`map_size_gb=10000` 有充足余量）。

**只重跑合并**（`.pt` 已在，跳过 2.3.1）：直接跑 2.3.2 这条命令即可。

#### 2.3.3 省掉 2.3.1 的办法：下载现成 ODE latents

HF 上发布的是**未合并的 `.pt`**（正是 2.3.1 的产物），下载后本地只跑 2.3.2 的合并：

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

**初始化 checkpoint：**使用 `2.3` 中准备好的 TF checkpoint（`$CKPT_ROOT/stage1_ar_tf/model.pt`）。
如果跳过 `2.3`、直接使用预生成的 ODE latents，仍需先按 `2.3` 开头的说明下载或准备该
TF checkpoint。

用 §2.3 预解好的 6 点轨迹 LMDB 回归拟合 causal student。`ODETrajectorySample` 每步采一个
去噪锚点（`denoising_step_list=[1000,750,500,250]`，`warp_denoising_step=True` 映到 shift
后的 schedule 时刻），`ODERegressionLoss` 在 **x0 空间**回归近净 target。模型与 TF 同款
`CausalWan21Model` + PRoPE 相机流，只是训练信号从 flow-matching 换成 ODE 回归。

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

- **数据是 `ode_lmdb`，不是 SFT 的 `data`**（`CameraODERegressionLMDBDataset` 读）。
  config 默认指向 `./dataset/Wan21/Action2V/ode_lmdb`，所以这条 override 必给。
- 初始化：TF promoted 权重，strict load 保留 PRoPE 参数。
- 超参（config 默认）：`lr=2e-6`、**`betas=(0.9,0.999)`**（注意与 SFT/TF 的 `(0.0,0.999)`
  不同）、`weight_decay=0.01`、`timestep_shift=5.0`、`sigma_min=0.0`、`extra_one_step=True`。
- 输出：`outputs/wan_action2v_ode_$DS/`。

**验证 override 生效**：

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

训完导出 → `$CKPT_ROOT/stage2a_ar_ode/model.pt`。ODE 是**单模型 recipe**
（无 `generator_ema`），导出用默认 `--model-key model`。它是 ODE few-step 推理的
checkpoint，也是 CD 的备选起点。

---

</details>

<details>
<summary><big><b>2.5 Stage-2(b) Causal Consistency Distillation (`stage2_ar_cd`)</b></big></summary>

**初始化 checkpoint：**CD 与 ODE 分支都从 TF checkpoint 开始，使用 `2.3` 中准备好的
`$CKPT_ROOT/stage1_ar_tf/model.pt`。如果跳过 `2.2` 和 `2.3`，请按 `2.3` 开头的说明下载
或准备该 checkpoint。

把 causal TF 模型蒸馏成 few-step 一致性模型：冻结 teacher 走一步 CFG Euler `t→t_next`，
student 预测 `t` 处 `x0`、EMA 网络预测 `t_next` 处 `x0`，loss 是两者 `x0` 的 MSE。

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

- **初始化（关键）**：**student / teacher / ema 三份 1.3B 全从同一个 TF 权重 seed**。
  只给一个 `checkpoint.pretrained`，`ARCDRecipe` 在 step 0 用 `copy_params` 把 student
  参数拷到 teacher / ema。teacher + EMA 在 config 里声明为冻结的 `auxiliary_models`。
- **数据回到 SFT 的 clean-latent LMDB**（CD 不需要预解轨迹，直接用干净 latent）。
- 超参（config 默认）：`lr=2e-6`、`guidance_scale=3.0`、`discrete_cd_n=50`、`ema_decay=0.99`、
  `activation_checkpointing=True`（三份 1.3B，显存较紧，别关）。
- 输出：`outputs/wan_action2v_cd_$DS/`。

**验证 override + aux 模型**：

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

训完导出 → `$CKPT_ROOT/stage2_ar_cd/model.pt`（CD few-step 推理的 checkpoint
+ **DMD 的 generator / generator_ema 种子**）。

---

</details>

<details>
<summary><big><b>2.6 Stage-3 Asymmetric DMD with Self Rollout (`stage3_ar_dmd`)</b></big></summary>

**如果跳过前面的初始化阶段，可下载官方提供的 ODE 或 CD checkpoint：**

```bash
# 默认的 ODE 初始化
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/causal_ode/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT causal_ode stage2_ar_ode
)

# 或使用 CD 初始化
hf download MIN-Lab/minWM --local-dir ./ckpts \
    --include "Wan21/Action2V/causal_cd/**"

# 将 HF 发布名映射到 config 使用的 stage 名
(
    cd ./ckpts/Wan21/Action2V || exit 1
    ln -sfnT causal_cd stage2_ar_cd
)
```

下载后 checkpoint 分别位于 `./ckpts/Wan21/Action2V/causal_ode/` 和
`./ckpts/Wan21/Action2V/causal_cd/`，上面的软链接分别映射为 `stage2_ar_ode` 和
`stage2_ar_cd`。DMD 还需要 bidirectional SFT checkpoint 作为
`real_score` / `fake_score` 的种子；若本地没有，可按 2.2 的命令下载，或在命令行显式
覆盖对应的 `checkpoint.*` 路径。

Distribution Matching Distillation 把 causal generator 蒸馏成 4-step 模型。**四份 1.3B**：

| aux 名 | 角色 | 种子 |
|---|---|---|
| `generator` (主模型) | student, trainable, causal | CD `model.pt` |
| `generator_ema` | frozen EMA, causal | CD `model.pt` |
| `real_score` | frozen teacher, **bidirectional** | SFT `model.pt` |
| `fake_score` | trainable critic, **bidirectional** | SFT `model.pt` |

集群版脚本为此做了「rank-0 导出 + 全节点 FS barrier」两步预处理。**本地不需要 barrier**
——先把两个 `.pt` 导好（§3.1），再起单条 torchrun：

```bash
# 前置：两个源权重都已导出为单文件（幂等，已存在就跳过）
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

- 上面的 `checkpoint_6000` / `checkpoint_15000` 是集群 run 选出的最优 step，按自己的
  验证结果换。**两个 `.pt` 已存在时跳过导出**（想换 step 重导要先删 `.pt`）。
- **⚠️ 四条 `checkpoint.*` override 必给（实测）**：`stage3_ar_dmd.py` 的 config 默认把
  generator / generator_ema 种子写成 **ODE** 权重
  （`_ODE = ./$CKPT_ROOT/stage2_ar_ode/model.pt`），想从 ODE 而非 CD 起步是另一种合法
  选择，那就把 `$CD_PT` 换成 `$CKPT_ROOT/stage2a_ar_ode/model.pt`。
- **数据只用来取 latent shape + prompt + 相机做条件**，clean latent 不直接进 loss
  （self rollout，无 real-video 监督）。
- 超参（config 默认）：generator `lr=2e-6`、critic `lr=4e-7`、`dfake_gen_update_ratio=5`
  （generator 每 5 步更新一次）、`guidance_scale=3.0`、
  `denoising_step_list=[1000,750,500,250]`、`generator_ema_decay=0.99`、
  `activation_checkpointing=True`（**四份 1.3B，关掉必 OOM**）。
- 输出：`outputs/wan_action2v_dmd_$DS/`。DCP 目录含 4 份模型，约 46 GB。

训完导出 → `$CKPT_ROOT/stage3_ar_dmd/model.pt`，**必须
`--model-key aux/generator_ema`**（见 §3.1.2）。

---

</details>

## 3. 推理流程

<details>
<summary><big><b>3.1 导出 checkpoint (DCP 目录 → 单文件 `.pt`)</b></big></summary>

训练存的是 **FSDP2 DCP 分片目录** `ckpts/checkpoint_{step}/`（`__*.distcp` + `.metadata`），
`torch.load` 读不了、`cp` 也没法「提升」它。`tools/export_checkpoint.py` 合并分片、剥
FSDP / compile 前缀，写出 `{"<model_key>": state_dict}` 单文件。

```bash
BEST_STEP=15000    # 由验证选出，须是 ckpt_interval 的倍数
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_sft_$DS/ckpts/checkpoint_${BEST_STEP}" \
    --output     $CKPT_ROOT/stage0_bi_sft/model.pt \
    --format     pt
```

单进程、纯 CPU、约 2 min，1.3B 出来 ≈5.96 GB / 885 tensors。`--output` 的父目录会自动创建。
`--format` 可选 `pt` / `safetensors` / `diffusers`（后者需 `--config`）；不给时按 `--output`
后缀推断。也支持 `s3://` / `oss://` 的 DCP 目录。


#### 3.1.1 各 stage 的 `--model-key`

| stage | `--model-key` | 目标路径 |
|---|---|---|
| SFT | `model`（默认） | `$CKPT_ROOT/stage0_bi_sft/model.pt` |
| TF | `model`（默认） | `$CKPT_ROOT/stage1_ar_tf/model.pt` |
| ODE | `model`（默认） | `$CKPT_ROOT/stage2a_ar_ode/model.pt` |
| CD | `model`（默认） | `$CKPT_ROOT/stage2_ar_cd/model.pt` |
| **DMD** | **`aux/generator_ema`** | `$CKPT_ROOT/stage3_ar_dmd/model.pt` |

SFT / TF / ODE 都是单模型 recipe（无 EMA），默认 key 就对。CD 的主模型也是 `model`。

> **⚠️ ODE 的路径不一致（实测）**：`infer/stage2_ar_ode.py` 里的默认
> `inference.checkpoint` 写的是 `./$CKPT_ROOT/stage2_ar_ode/model.pt`，
> 而本文档 / 现有产物用的是 **`stage2a_ar_ode/`**（`stage2_ar_ode/` 这个目录在盘上不存在）。
> 所以 ODE 推理**必须显式给 `--checkpoint`**（§3.2.4 已这样写），否则会去找一个不存在的默认路径。
> 想省掉这个 flag 就做个软链：
> `ln -sfn "$PWD/$CKPT_ROOT/stage2a_ar_ode" $CKPT_ROOT/stage2_ar_ode`。

#### 3.1.2 DMD 必须导 EMA

```bash
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_dmd_$DS/ckpts/checkpoint_1000" \
    --output     $CKPT_ROOT/stage3_ar_dmd/model.pt \
    --model-key  aux/generator_ema
```

DMD checkpoint 里有四份模型，推理用的是 **EMA generator**；用默认 `--model-key model`
会导出裸 student。DMD 推理 config 设了 `prefer_ema=True` 与之匹配。

#### 3.1.3 单权重的双重用途

`$CKPT_ROOT/{stage}/model.pt` 既是**本 stage 推理的 `inference.checkpoint`**，
又是**下一 stage 训练的 `checkpoint.pretrained`**——一次导出两处共用。

#### 3.1.4 边训边导 + 自动推理 (可选)

`tools/auto_dump.py` 轮询 `ckpts/`，等 DCP 的 `.metadata` 出现（DCP **最后**写它，所以它
就是「存完了」的信号，绝不会合并半成品）后自动合并；`tools/auto_sample.py` 接着自动跑推理。
**本地要加 `--local`**（否则它会用 `<launch-tool> submit` 提集群 job；`<launch-tool>` 是
集群提交工具的占位符，通过 `MINWM_LAUNCH_TOOL` 指向你自己的工具，见 `tools/_cluster.py`）：

```bash
# 本地进程内合并（吃 RAM ≥ 模型大小）
python tools/auto_dump.py \
    --output-dir "outputs/wan_action2v_sft_$DS" \
    --ckpt-steps 10000 15000 20000 --format safetensors --local

# DMD 要 EMA 权重
python tools/auto_dump.py \
    --output-dir "outputs/wan_action2v_dmd_$DS" \
    --ckpt-steps 1000 --model-key aux/generator_ema --format pt --local

# 自动推理（prompt 来自 config 的 inference.benchmark；--benchmark 只是覆盖它）
python tools/auto_sample.py \
    --output-dir "outputs/wan_action2v_sft_$DS" \
    --ckpt-steps 10000 15000 \
    --config-file configs/wan21/action2v/infer/stage0_bi_sft.py \
    --benchmark "$BENCH" \
    --local
```

样本落 `<output-dir>/<sample-name>/{step}-{exp_name}/`。这条流水线只自动化了「导出 +
推理」，**叠按键（§4.1）仍要单独跑**。详见 [`docs/auto-pipeline.md`](../../../docs/auto-pipeline.md)。

---

</details>

<details>
<summary><big><b>3.2 推理</b></big></summary>

#### 3.2.1 统一入口（config + benchmark JSON 驱动）

五个 stage 共用 `tools/infer_mwm.py`，loop / sampler / guidance / 步数**全由
`--config-file` 决定**，所以换 stage 只需换 config + checkpoint。命令行**只有**
`--config-file` 加尾随 dotlist（`inference.*=` / `model.*=`）——值型 flag
（`--checkpoint` / `--output-dir` / `--prompts` / `--input-json` / `--limit` …）已全部移除。

输入是 benchmark JSON（`inference.benchmark`，格式 `[{id, caption, trajectory}]`），
每条出一个 `{id}.mp4`；`image` 字段有则 i2v（HY），无则 t2v（Wan）。

```bash
# 先按 stage 设这三个
CFG=configs/wan21/action2v/infer/stage0_bi_sft.py    # 该 stage 的 infer config
PT="$CKPT_ROOT/stage0_bi_sft/model.pt"              # 导出的单文件权重
OUT="outputs/infer_sft_$DS/15000"                   # 输出目录，15000 只是例子

torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file "$CFG" \
    inference.checkpoint="$PT" \
    inference.benchmark="$BENCH" \
    inference.output_dir="$OUT" \
    inference.num_inference_steps=50
```

`inference.benchmark` 已在每个 stage config 里给了默认值（50 条官方 benchmark），
所以只想跑默认评测时那一行可以省掉。

常用覆盖：`inference.limit=N`（只跑前 N 条，冒烟用）、`inference.seed=`、`inference.dtype=`、
`inference.prefer_ema=True`、`inference.sp_size=N`（配 `--nproc_per_node=N` 开序列并行）。

**各 stage 的 config 差异**（都在 `configs/wan21/action2v/infer/`）：

| stage | config | `inference.loop` | `sampler.solver` | guidance | 步数 |
|---|---|---|---|---|---|
| SFT | `stage0_bi_sft.py` | `BidirectionalGenerationLoop` | `UniPCSolver` | 8.0 | 50 |
| TF | `stage1_ar_tf.py` | `ARGenerationLoop` | `UniPCSolver` | 3.0 | 50 |
| ODE | `stage2_ar_ode.py` | `ARGenerationLoop` | `CMSolver` | 1.0 | 4 |
| CD | `stage2_ar_cd.py` | `ARGenerationLoop` | `CMSolver` | 1.0 | 4 |
| DMD | `stage3_ar_dmd.py` | `ARGenerationLoop` | `CMSolver` | 1.0 | 4 |

- `loop` / `solver` 都是**类名字符串**，由代码按名字解析（`ARGenerationLoop` 在
  `minwm/engine/inference/loop.py`，`CMSolver` 在 `.../samplers.py`），所以换 loop / 换
  sampler 全在 config 里，代码没有 `if pipeline == ...` 分支。
- SFT/TF 走 50 步 flow-UniPC。**TF 是 AR loop 但仍然 50 步**——few-step=4 只针对
  ODE/CD/DMD，别混淆。
- ODE/CD/DMD 的 `CMSolver` 用 `denoising_step_list=[1000,750,500,250]`，所以
  `inference.num_inference_steps` 传什么都无害（实际步数由 `denoising_step_list` 决定）。
- 公共默认：`num_frames=20`（→ 77 像素帧）、`latent_shape=(16,60,104)` → 832×480、
  `fps=16`、`seed=0`、`sp_size=1`。**轨迹是逐条的**，来自 benchmark item 的 `trajectory`
  字段（没有全局 `inference.trajectory` 这个 knob 了）。
- checkpoint：`checkpoint_key="auto"` + `prefer_ema=True`，依次找
  `generator_ema` → `generator` → `model`，所以只含 `model` 的 SFT/TF/ODE/CD 导出也能正常加载。

**推理三步固定**：① 导出 DCP → 单 `.pt`（§3.1，幂等）② `infer_mwm.py` 采样
③ `overlay_from_manifest.py` 叠按键 + 拼接（§4.1，纯 CPU）。都是单卡。

#### 3.2.2 SFT 推理 (50 步双向)

```bash
STEP=15000
OUT="outputs/infer_sft_$DS/${STEP}"

# ① 导出（已存在则跳过；换 step 重导要先删 model.pt）
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_sft_$DS/ckpts/checkpoint_${STEP}" \
    --output     $CKPT_ROOT/stage0_bi_sft/model.pt --format pt

# ② 采样
mkdir -p "$OUT"
torchrun --nproc_per_node=1 tools/infer_mwm.py \
    --config-file configs/wan21/action2v/infer/stage0_bi_sft.py \
    inference.checkpoint=$CKPT_ROOT/stage0_bi_sft/model.pt \
    inference.benchmark="$BENCH" \
    inference.output_dir="$OUT" \
    inference.num_inference_steps=50

# ③ 叠按键 + 拼接（见 §4.1）
python demos/overlay_from_manifest.py --input-dir "$OUT" --output final_with_keys_${STEP}.mp4
```

冒烟（1 条）：②里加 `inference.limit=1`。

#### 3.2.3 TF 推理 (causal AR，仍是 50 步)

与 3.2.2 同构，只换 config / DCP 目录 / 输出目录：

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

#### 3.2.4 few-step 推理 (ODE / CD / DMD — 同一条代码路径)

三者**同属一条 `few_step` + `causal_ar` 代码路径**，只差 config 与权重，所以别为它们各写
一套：

```bash
# --- CD ---
python tools/export_checkpoint.py \
    --checkpoint "outputs/wan_action2v_cd_$DS/ckpts/checkpoint_6000" \
    --output     $CKPT_ROOT/stage2_ar_cd/model.pt
CFG=configs/wan21/action2v/infer/stage2_ar_cd.py
PT=$CKPT_ROOT/stage2_ar_cd/model.pt
OUT="outputs/infer_cd_$DS/cd_bench50"

# --- ODE（换这三行即可）---
# CFG=configs/wan21/action2v/infer/stage2_ar_ode.py
# PT=$CKPT_ROOT/stage2_ar_ode/model.pt
# OUT="outputs/infer_ode_$DS/6000_bench50"

# --- DMD（注意导出用 --model-key aux/generator_ema，见 §3.1.2）---
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

- **没有 `$BENCH`**（§0.1 留空）就把 `inference.benchmark=` 那行删掉，用 config 自带的默认
  benchmark（50 条官方 benchmark）。
- **指定用哪张卡**：前面加 `CUDA_VISIBLE_DEVICES=<id>`（推理是单进程，多卡空闲时可以并行
  跑几个 stage，各占一张）。
- 耗时量级：50 条、单卡约 20 min。

#### 3.2.5 benchmark JSON（唯一的输入格式）

轨迹字符串语义：`token*n` 段用逗号拼接，如 `d*8,i*5,l*6` = 右移 8 + 上仰 5 + 左移 6，
**共 19 段动作配 20 latent 帧**。token 是 WASD（平移）+ KIJL（转视角）。

benchmark JSON（`inference.benchmark`，上面的 `$BENCH`）：一个 JSON 数组，每项
`{id, caption, trajectory}`（`image` 可选）。自己攒一个即可：

```json
[
  {"id": "0001", "caption": "a drone shot over a forest lake", "trajectory": "d*8,i*5,l*6"},
  {"id": "0002", "caption": "walking through a narrow alley at dusk", "trajectory": "w*19"}
]
```

- `id` 决定输出文件名 `{id}.mp4`，所以要唯一。
- `trajectory` 同上面的段语义，**段数和建议为 19**（对齐 20 latent 帧；§4.1 的叠按键按这个换算）。
- `image` 字段在 Wan21 推理时被**静默忽略**（Wan 的预处理器不用它），退化为纯 T2V，不报错
  ——所以 HY 那边的 i2v benchmark 文件可以直接复用。
- 想要固定的评测集就自己维护几档规模（1 条冒烟 / 20 条小批 / 50 条标准 / 200 条大批），
  用 `$BENCH` 切换。团队内已有共享 benchmark 的话，问一下放在哪、`chmod o+r` 后填进 `$BENCH`。

**别覆盖旧输出**：输出目录带上用途后缀（如 `_bench50` / `_smoke`），见 §3.2.4 的 `OUT=`。

#### 3.2.6 输出目录契约

`outputs/infer_{stage}_$DS/{step}/` 下：

- 每条一个 mp4：**832×480 / 77 帧 / 16 fps**，文件名 `{id}.mp4`（`id` 来自 benchmark item）。
- `manifest.json` — `{inference: <配置>, items: [{index, id, prompt, trajectory, seed, video}]}`。
- `final_with_keys_{...}.mp4` — 叠 WASD/KIJL 的拼接总览（§4.1）。
- `overlay_config.json` — 每条 `{input, trajectory, sequence}`（§4.1 写）。

---

</details>

## 4. 辅助工具与运行维护

<details>
<summary><big><b>4.1 叠按键 + 拼接 (第③步)</b></big></summary>

读 `manifest.json` 的每条 `trajectory`，换算逐段像素帧数、给每个 mp4 叠 WASD/KIJL 指示器，
再拼成一个总览视频。纯 CPU，不占 GPU。

```bash
python demos/overlay_from_manifest.py \
    --input-dir "outputs/infer_sft_$DS/15000" \
    --output    final_with_keys_15000.mp4
```

产物落在 `--input-dir` 下：`final_with_keys_*.mp4` + `overlay_config.json`。
`--keep-temp` 保留每段叠加后的中间文件。

- **依赖 ffmpeg / ffprobe + PIL + numpy**。本地 minwm 环境有 `/usr/bin/ffmpeg`，所以这步
  在本地能跑；集群镜像**没有** ffmpeg，那边跑必 fail（这也是当初把它拆成独立第③步的原因）。
- **它直接读 manifest 的 `trajectory` 字段，不解析文件名**，所以对 `{id}.mp4` 这种简洁
  命名也适用。（同目录的 `demos/batch_overlay.py` 是靠解析文件名的老版本，对 `{id}.mp4`
  用不了。）
- **帧数换算**：模型出 20 latent 帧，VAE 时间解码成 **77 像素帧**（首帧 1 帧、其余每 latent
  帧 4 帧）。轨迹的 19 段动作按段位置分配像素帧：首段 `1+4*(n-1)`、末段 `4+4*n`、
  中间段 `4*n`、单段 `1+4*(n-1)+4`，各段和恒为 77。
- 轨迹段数和应为 19。bench50 已核实 50 条都是 832×480 / 77 帧 / 16fps、段和均为 19。

---

</details>

<details>
<summary><big><b>4.2 断点续训与重跑</b></big></summary>

#### 4.2.1 RESUME（崩溃 / 手动 kill 后接着跑）

`ckpts/checkpoint_{step}/` 是 **DCP 完整训练态**（model + optimizers + aux + step，
不只是权重），`ckpts/latest.txt` 指向最新。trainer 里 **`checkpoint.resume=True` 优先于
`checkpoint.pretrained`**：resume 恢复全部训练态、从断点 step 续，然后 `return` 跳过
pretrained。把 `checkpoint.pretrained=...` 换成 `checkpoint.resume=True` 即可（**二者别同时给**，
容易误读）：

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

（`tools/train_mwm.py` 也有等价的 `--resume` flag。）

**验证真的续上了**：日志出现 `loaded checkpoint .../checkpoint_{step}`，且首条监控行的 step
是**断点后的下一个 log_interval**（如从 2500 续 → 首条 `step 2510`，**不是 step 10**）。
找不到 `latest.txt` 时会**大声报错**，不会静默回退到 fresh init / pretrained
——避免从错误权重开始训练。

其它超参（`max_steps` / `sp_size` / `ckpt_interval`）续训时保持不变。

#### 4.2.2 从头重训的覆盖陷阱

save 用 `checkpoint_{step}` 命名 + 改写 `latest.txt`。**复用一个已有 run 的 `--output-dir`
从 step 0 重跑，step 涨到 2500 时会无声覆盖旧的 `checkpoint_2500` 并改写 `latest.txt`。**


```bash
# 推荐：换一个新 output-dir，零风险，不碰旧 run（后缀自己起，如 _v2 / _lr2e6）
torchrun --nproc_per_node="$NPROC" ... \
    --output-dir "outputs/wan_action2v_tf_${DS}_v2"

# 或者换一个 DS（连数据产物一起隔离，见 §0.4）
```

要就地覆盖旧 run，先自己确认那份 ckpt 真的可以丢：
`ls "outputs/wan_action2v_tf_$DS/ckpts"` 看清楚再 `rm -rf` 那个 `ckpts` 目录。

> 对 TF/ODE/CD/DMD 而言，「from scratch」= 从上一 stage 的 `pretrained` 的 **step 0** 起
> （真·随机初始化对 causal 模型无意义，必须先从上游权重转）。

---

</details>

<details>
<summary><big><b>4.3 全流程串行速查</b></big></summary>

按顺序跑完即可（每段的 `<BEST_STEP>` 由自己的验证选，须是 `ckpt_interval` 的倍数）：

先 source §0.1 的配置块，然后按顺序推进（每段的 `BEST_STEP` 由自己的验证选，
须是该 stage `ckpt_interval` 的倍数）：

```bash
# 每个新 shell 都先来这一遍（§0.1）
source /path/to/your/env.sh        # 或把 §0.1 那段直接粘进来

# ① encode                                      → $LMDB                        (§1)
# ② SFT 训练                                     → 导出 $CKPT_ROOT/stage0_bi_sft/model.pt   (§2.1, §3.1)
# ③ TF  训练 (pretrained=stage0_bi_sft)          → 导出 $CKPT_ROOT/stage1_ar_tf/model.pt    (§2.2, §3.1)
# ④ ODE curation: 采样 + 合并                    → $ODE_LMDB                    (§2.3)
# ⑤ ODE 训练 (pretrained=stage1, data=$ODE_LMDB) → 导出 $CKPT_ROOT/stage2a_ar_ode/model.pt  (§2.4, §3.1)
# ⑥ CD  训练 (pretrained=stage1, data=$LMDB)     → 导出 $CKPT_ROOT/stage2_ar_cd/model.pt    (§2.5, §3.1)
# ⑦ DMD 训练 (CD 种 generator/ema + SFT 种 real/fake_score)
#                                                → 导出 $CKPT_ROOT/stage3_ar_dmd/model.pt
#                                                  --model-key aux/generator_ema            (§2.6, §3.1.2)
# ⑧ 每段推理 + 叠按键                                                            (§3.2, §4.1)
```

④⑤ 与 ⑥ 只依赖 ③，互不依赖；⑦ 需要 ⑥ 和 ② 两份权重（见开头的阶段链图）。
各段的 `--master_port` 已错开（29700–29706），同时跑多段不撞端口。

---

</details>

<details>
<summary><big><b>4.4 排错</b></big></summary>

| 现象 | 原因 / 处理 |
|---|---|
| `unrecognized arguments: --checkpoint / --prompts / --input-json ...` | 抄了旧的 flag 写法。`infer_mwm.py` 现在**只吃** `--config-file` + 尾随 dotlist（`inference.checkpoint=` / `inference.benchmark=` / `inference.output_dir=`），见 §3.2.1。 |
| `unknown key(s) '...' in Inference` | `inference` 节里有拼错的键。该节是强 schema（`minwm/config/schema.py:Inference`），错键直接报错而不是静默走默认值；照报错里列出的合法键名改。 |
| `blocks.*.prope_o.* newly initialized` + `You should probably TRAIN this model` | **正常**。PRoPE 相机参数在 base 里不存在、新初始化。推理加载导出权重时也会打印同一批告警。 |
| DCP 目录 `torch.load` 读不了 | 分片目录不是单文件，必须先 `tools/export_checkpoint.py` 合并（§3.1）。 |
| DMD 推理效果明显不对 | 导出时忘了 `--model-key aux/generator_ema`，导成了裸 student（§3.1.2）。 |
| ODE curation 加载 base 失败 / 卡住 | 代理没 unset。`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`（§0.1）；且必须从仓库根跑、`wan_models/Wan2.1-T2V-1.3B` 软链存在。 |
| ODE curation flash-attn kernel 报错 | 需要 sm_80 及以上的 kernel；实测 B200 上缺（§2.3.1）。 |
| ODE 训练报数据键找不到 | 数据路径给成 `$LMDB` 了。ODE 必须用 `$ODE_LMDB`（§2.4）。 |
| `world_size not divisible by sp_size` 之类的起不来 | `$NPROC` 不能被 `training.sp_size` 整除。按 §0.3 的表调 `sp_size`。 |
| 变量为空导致路径变成 `dataset/Wan21/Action2V_/data` | 没 source §0.1 的配置块（新 shell / 新终端要重新 source）。`echo "$DS $LMDB"` 自检。 |
| DMD OOM | 别关 `gradient_checkpointing`（四份 1.3B）。也可把 `training.sp_size` 调大分摊激活。 |
| tqdm 进度条停在第一帧 | 重定向到文件时 `\r` 帧不落盘，**不代表卡住**。看 `log_interval` 的打印行，或数落盘文件（`.rank_*` 分片 / `$ODE_LATENTS/*.pt`）。 |
| 训练无声覆盖了旧 checkpoint | 复用了同一个 `--output-dir` 从头跑（§4.2.2）。换新目录或换 `$DS`。 |
| 换了数据集但权重被覆盖 | 导出权重路径不带 `$DS`（§0.4 的提醒）。加后缀或换 `$CKPT_ROOT`。 |
| 叠按键报 ffmpeg not found | 这步要 ffmpeg/ffprobe，装一下（`conda install -c conda-forge ffmpeg` 或系统包管理器）。 |
| 磁盘不足 | 先 `df -h .`。量级参考：ODE LMDB ≈46 GB、DMD DCP ≈46 GB/step、导出单权重 ≈6 GB。 |

---

</details>

<details>
<summary><big><b>4.5 参考</b></big></summary>

- 各 stage config：[`train/`](train/) + [`infer/`](infer/)（`stage0_bi_sft` / `stage1_ar_tf` /
  `stage2_ar_ode` / `stage2_ar_cd` / `stage3_ar_dmd`）。
- 数据工具与键契约：[`tools/data/wan21/README.md`](../../../tools/data/wan21/README.md)。
- 自动导出 / 自动采样：[`docs/auto-pipeline.md`](../../../docs/auto-pipeline.md)。
- 安装与模型线总览：[`configs/wan21/README.md`](../README.md)；
  HunyuanVideo backbone：[`configs/hy/`](../../hy/README.md)。
- Wan 的 CFG negative prompt 在代码里（`DEFAULT_NEGATIVE_PROMPT`,
  `minwm/modeling/wan21/adapter.py`），被需要它的 config 引用，**不需要 `.pt` 预编码**。

</details>
