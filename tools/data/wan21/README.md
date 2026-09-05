# Wan2.1 Data Preprocessing / 数据预处理

**EN** — Offline preprocessor that turns raw WorldPlayGen videos + captions +
`pose_str` motion strings into a single camera-aware LMDB the Wan2.1 Action2V
(PRoPE) training recipes consume. The VAE encode core lives in
`minwm.modeling.wan21.vae._video_vae` (z_dim=16) and the camera trajectory in
`minwm.data.preprocessing.trajectory.generate_camera_trajectory_local`; this
script owns only argument parsing, the local `Wan21VAE` normalization wrapper, and
the distributed shard-then-merge loop. It is a standalone `tools/` entrypoint —
run it with `torchrun`, never `import` it.

**中文** — 离线预处理脚本，把 WorldPlayGen 原始视频 + 文本 + `pose_str` 运动串编码成
Wan2.1 Action2V（PRoPE）训练 recipe 直接读取的单个相机感知 LMDB。VAE 编码核心在
`minwm.modeling.wan21.vae._video_vae`（z_dim=16），相机轨迹在
`minwm.data.preprocessing.trajectory.generate_camera_trajectory_local`；本脚本只负责
参数解析、本地 `Wan21VAE` 归一化封装，以及分布式「分片→合并」循环。它是独立的 `tools/`
入口，用 `torchrun` 运行，不要被 `import`。

This directory also ships `check_dataset_bev.py`, a sanity-check tool that reads
a built LMDB back and renders camera-vs-video overlays — see the last section.

本目录还提供 `check_dataset_bev.py`，把已构建的 LMDB 读回并渲染相机-视频对照图，
用于数据校验，详见最后一节。

---

## `build_worldplaygen_lmdb.py` — Action2V LMDB / Action2V 潜变量库

**EN** — Each 77-frame video is encoded into 20 latent frames (VAE 4x temporal
downsample). Camera poses are not loaded from disk; they are synthesized from a
`pose_str` DSL string per sample (identical DSL to HY
`preencode_generated_wdplay.py`), then converted to `(intrinsics, poses)` in the
`CameraPluckerDataset` format. Videos live at
`<video_dir>/<idx:06d>_<suffix>/gen.mp4`, where `suffix` is `pose_str`
lowercased with every non-`[a-z0-9]` character stripped
(`right-8, A 11` → `right8a11`).

**中文** — 每个 77 帧视频编码成 20 个 latent 帧（VAE 4 倍时间下采样）。相机位姿不从
磁盘读取，而是按每条样本的 `pose_str` DSL 字符串合成（DSL 与 HY
`preencode_generated_wdplay.py` 完全一致），再转成 `CameraPluckerDataset` 格式的
`(intrinsics, poses)`。视频位于 `<video_dir>/<idx:06d>_<suffix>/gen.mp4`，`suffix`
为 `pose_str` 转小写后删除所有非 `[a-z0-9]` 字符（`right-8, A 11` → `right8a11`）。

**Input JSON / 输入 JSON:**

```json
[{"image_path": "...", "caption": "a city street", "pose_str": "w-3, right-8", "pose_json_path": "..."}]
```

`image_path` and `pose_json_path` are carried for provenance but not read by the
encoder — only `caption` and `pose_str` are used. Entries whose computed video
path is missing are silently skipped.

**Run / 运行:**

```bash
torchrun --nproc_per_node=8 tools/data/wan21/build_worldplaygen_lmdb.py \
    --input_json ./dataset/preencode_input.json \
    --video_dir  ./dataset/videos \
    --output_dir ./dataset/Wan21/Action2V \
    --vae_path   wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
    --target_h 480 --target_w 832
```

Wrapper: `run_build_worldplaygen_lmdb.sh` (edit `VAE_PATH`, `INPUT_JSON`,
`VIDEO_DIR`, `OUTPUT_DIR`, `NUM_GPUS_PER_NODE` at the top). The final merged LMDB
is written to `<output_dir>/data/`.

**中文** — `image_path` / `pose_json_path` 仅作溯源记录，编码器不读取，只用 `caption`
和 `pose_str`。计算出的视频路径不存在的条目会被静默跳过。最终合并的 LMDB 写到
`<output_dir>/data/`。

---

## pose_str DSL / 运动串语法

**EN** — Comma-separated `action-count` tokens (e.g. `w-3, right-8, d-4`); each
token repeats one per-frame motion `count` times. The DSL is identical to HY
`preencode_generated_wdplay.py`:

| Token | Motion / 运动 | Token | Motion / 运动 |
| --- | --- | --- | --- |
| `w` | forward `+0.08` | `s` | forward `-0.08` |
| `d` | right `+0.08` | `a` | right `-0.08` |
| `up` | pitch `+deg2rad(3)` | `down` | pitch `-deg2rad(3)` |
| `right` | yaw `+deg2rad(3)` | `left` | yaw `-deg2rad(3)` |

**中文** — 逗号分隔的 `action-count` 词元（如 `w-3, right-8, d-4`），每个词元把一帧
运动重复 `count` 次。DSL 与 HY `preencode_generated_wdplay.py` 完全一致。

---

## Output LMDB contract / 输出 LMDB 键约定

**EN** — The merged LMDB at `<output_dir>/data/` stores `N` samples under
indexed keys (`latents_{i}_data`, `prompts_{i}_data`, `intrinsics_{i}_data`,
`poses_{i}_data`) plus shape metadata. Latents are Wan VAE encodings
(z_dim=16, 4x temporal downsample).

| Key | Shape | Type | Notes |
| --- | --- | --- | --- |
| `latents` | (N, 20, 16, 60, 104) | fp16 | 20 latent frames, 16-channel Wan VAE |
| `prompts` | (N,) | str | caption per sample |
| `intrinsics` | (N, 4) | fp32 | `[fx/W, fy/H, cx/W, cy/H]` normalized |
| `poses` | (N, 20, 7) | fp32 | `[tx,ty,tz, qx,qy,qz,qw]` w2c per latent frame |

Final metadata keys: `latents_shape`, `prompts_shape`, `intrinsics_shape`,
`poses_shape` (each a space-joined dim string, e.g. `"512 20 16 60 104"`).

**中文** — 合并后的 LMDB 在 `<output_dir>/data/`，以索引键
（`latents_{i}_data` 等）存 `N` 条样本及形状元信息。latent 为 Wan VAE 编码
（z_dim=16，4 倍时间下采样）。元信息键 `latents_shape` / `prompts_shape` /
`intrinsics_shape` / `poses_shape` 各为空格拼接的维度串。

---

## Encoding & merge flow / 编码与合并流程

**EN** — Each rank initializes a local `Wan21VAE` (hardcoded 16-channel mean/std
normalization wrapping `_video_vae`), streams its shard to a per-rank LMDB
(`<output_dir>/data.rank_{r}`), then rank 0 streaming-merges all shards into
`<output_dir>/data/` and removes the per-rank dirs. This keeps memory bounded —
no rank holds the full dataset.

**中文** — 每个 rank 初始化本地 `Wan21VAE`（封装 `_video_vae`，硬编码 16 通道
mean/std 归一化），把自己的分片流式写入 per-rank LMDB
（`<output_dir>/data.rank_{r}`），最后 rank 0 把所有分片流式合并到
`<output_dir>/data/` 并删除 per-rank 目录，全程内存有界。

---

## `check_dataset_bev.py` — dataset sanity-check / 数据校验

**EN** — Read a built camera LMDB back and verify that the stored trajectory
matches the video. It instantiates the real
`minwm.data.datasets.lmdb.CameraLatentLMDBDataset` (so pose normalization is
identical to training), decodes each sampled `clean_latent (F,C,H,W)` through
the `Wan21VAE`, and renders a side-by-side `[ RGB frame | animated top-down
BEV ]` mp4 per clip. The BEV panel draws the top-down world X-Z trajectory, a
yaw arrow, a pitch gauge, a path-length readout, and the caption. It handles
single or sharded LMDB layouts and needs one GPU for the VAE decode. It is a
standalone entrypoint — run it with `python`, do not `import` it.

**中文** — 把已构建的相机 LMDB 读回，校验存储的轨迹是否与视频一致。它实例化真实的
`minwm.data.datasets.lmdb.CameraLatentLMDBDataset`（位姿归一化与训练完全一致），
用 `Wan21VAE` 解码每个采样的 `clean_latent (F,C,H,W)`，每条 clip 渲染一个左右并排的
`[ RGB 帧 | 俯视 BEV 动画 ]` mp4。BEV 面板绘制俯视世界 X-Z 轨迹、偏航箭头、俯仰计、
路径长度和文本。支持单库或分片布局，VAE 解码需一张 GPU。它是独立入口，用 `python`
运行，不要被 `import`。

**Run / 运行:**

```bash
python tools/data/wan21/check_dataset_bev.py \
    --data_path ./dataset/Wan21/Action2V/data --num_videos 8 --scale_plot
```

**Key flags / 主要参数:**

| Flag | Default | Meaning / 含义 |
| --- | --- | --- |
| `--data_path` | — | single LMDB dir or sharded parent / 单库目录或分片父目录 |
| `--num_videos` / `--rows` | `8` / — | random sample count / explicit row indices |
| `--concat` | `10` | also stitch the first N clips into one montage (0 disables) |
| `--scale_plot` | off | scale-only mode (no VAE / GPU) / 仅统计模式（无需 VAE/GPU） |
| `--out_dir` | `outputs/data_check/ds_check_<name>` | output dir |

**EN** — Camera convention is OpenCV: `viewmats` are world->cam, `c2w =
inv(viewmats)`, camera center is `c2w[:3,3]` and forward is `c2w[:3,2]`. The Wan
VAE is 4x temporally compressed with a special first frame, so `F` latent frames
decode to `1 + (F-1)*4` pixel frames; each pixel frame maps to the nearest pose
so the BEV arrow tracks the video. By default a full-dataset scale sweep
(`trans_span`/`path_len`/`rot_span` histograms + an outlier list at
`median(trans_span) * --outlier_k`) is written alongside the videos.

**中文** — 相机约定为 OpenCV：`viewmats` 为世界→相机，`c2w = inv(viewmats)`，相机中心
`c2w[:3,3]`，朝向 `c2w[:3,2]`。Wan VAE 时间 4 倍压缩且首帧特殊，故 `F` 个 latent 帧解码
为 `1 + (F-1)*4` 个像素帧；每个像素帧映射到最近位姿，使 BEV 箭头与视频同步。默认还会在
视频旁写出整库尺度扫描（`trans_span`/`path_len`/`rot_span` 直方图，以及
`median(trans_span) * --outlier_k` 之上的异常清单）。

