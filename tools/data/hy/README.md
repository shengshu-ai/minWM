# HunyuanVideo Data Preprocessing / 数据预处理

**EN** — Offline preprocessors that turn raw videos + captions (+ optional camera
data) into the `.pt` latent shards the HunyuanVideo training recipes consume.
The encode core lives in `minwm.data.preprocessing.hy.VideoLatentExtractor`
(VAE + SigLIP vision + LLM text + byT5 glyph); these scripts own only argument
parsing and the distributed shard-then-merge loop. They are standalone
`tools/` entrypoints — run them with `torchrun`, never `import` them.

**中文** — 离线预处理脚本，把原始视频 + 文本（+ 可选相机数据）编码成
HunyuanVideo 训练 recipe 直接读取的 `.pt` latent 分片。编码核心在
`minwm.data.preprocessing.hy.VideoLatentExtractor`（VAE + SigLIP 视觉 + LLM 文本
+ byT5 字形）；这些脚本只负责参数解析和分布式「分片→合并」循环。它们是独立的
`tools/` 入口，用 `torchrun` 运行，不要被 `import`。

---

## Scripts / 脚本一览

| Script | Use case / 用途 | Camera? | Output |
| --- | --- | --- | --- |
| `preencode_video.py` | TI2V — sequential frame sampling | no | `.pt` per video |
| `preencode_camera_video.py` | Action2V — camera-aligned sampling, raw poses from disk | yes (`.npy`) | `.pt` per video |
| `preencode_generated_wdplay.py` | Action2V — WorldPlay distilled videos, poses from `pose_str` | yes (DSL) | `.pt` per video |
| `generate_negative_prompts.py` | CFG negative-prompt embeddings | — | shared `.pt` |

**EN** — All three pre-encoders shard the input list across ranks
(`i % world_size == rank` or contiguous slices), write one `.pt` per video into
`<output_dir>/latents/`, and emit a per-rank `train_index_rank{r}.json` that
rank 0 merges into a single `train_index.json`. Pass `--skip_existing` to resume.

**中文** — 三个预编码脚本都把输入列表按 rank 切分，每个视频写一个 `.pt` 到
`<output_dir>/latents/`，并各自产出 `train_index_rank{r}.json`，最后由 rank 0
合并成 `train_index.json`。加 `--skip_existing` 可断点续跑。

---

## `preencode_video.py` — TI2V latents / TI2V 潜变量

**EN** — Sequential frame sampling (no camera). `image_cond` is the first frame
of the VAE-encoded latents, matching the training pipeline.

**中文** — 顺序采帧（无相机）。`image_cond` 取 VAE 编码 latent 的首帧，与训练管线一致。

**Input JSON / 输入 JSON:**

```json
[{"video_path": "/abs/a.mp4", "caption": "a dog runs"}, ...]
```

**Run / 运行:**

```bash
torchrun --nproc_per_node=8 tools/data/hy/preencode_video.py \
    --input_json ./dataset/videos.json \
    --output_dir ./dataset/HY15/TI2V \
    --hunyuan_checkpoint_path ./ckpts/HunyuanVideo-1.5 \
    --target_height 480 --target_width 832 --skip_existing
```

Wrapper: `run_preencode_video.sh` (env-overridable `HUNYUAN_CHECKPOINT`,
`INPUT_JSON`, `OUTPUT_DIR`, `NUM_GPUS`).

---

## `preencode_camera_video.py` — Action2V latents (raw poses) / 相机对齐潜变量（原始位姿）

**EN** — Camera-aligned sampling: frames are picked to line up with the VAE's
4x temporal downsampling (via the extractor's `camera_indices` path), and raw
`intrinsics`/`poses` `.npy` files are stored next to the latents for
plucker-embedding construction in the dataloader.

**中文** — 相机对齐采帧：按 VAE 的 4 倍时间下采样对齐挑帧（走 extractor 的
`camera_indices` 路径），并把原始 `intrinsics`/`poses` 的 `.npy` 一并存入 latent，
供 dataloader 构建 plücker 嵌入。

**Input JSON / 输入 JSON** (from `prepare_camera_json.py`):

```json
[{"video_path": "/abs/a.mp4", "caption_path": "/abs/a.json",
  "intrinsics_path": "/abs/K.npy", "poses_path": "/abs/poses.npy",
  "camera_indices": [0, 4, 8], "num_frames": 77, "video_id": "000000"}]
```

`caption_path` is read as JSON and the `"SceneSummary"` field is used as the
caption (missing/garbled → empty caption, non-fatal).

**Run / 运行:**

```bash
torchrun --nproc_per_node=8 tools/data/hy/preencode_camera_video.py \
    --input_json ./dataset/train_camera.json \
    --output_dir ./dataset/HY15/Action2V \
    --hunyuan_checkpoint_path ./ckpts/HunyuanVideo-1.5 \
    --target_height 480 --target_width 832 --max_frames 77 --skip_existing
```

Wrapper: `run_preencode_downloaded_camera_video.sh`.

---

## `preencode_generated_wdplay.py` — Action2V latents (pose_str DSL) / WorldPlay 蒸馏视频潜变量

**EN** — For WorldPlay distilled videos. Camera poses are not loaded from disk;
they are synthesized from a `pose_str` DSL string per sample, then converted to
`(intrinsics, poses)` matching the `CameraPluckerDataset` format. Videos live at
`<video_root>/<idx:06d>_<suffix>/gen.mp4`, where `suffix` is `pose_str` with
whitespace/commas stripped and dashes removed (`right-8, a-11` → `right8a11`).

**中文** — 针对 WorldPlay 蒸馏视频。相机位姿不从磁盘读取，而是按每条样本的
`pose_str` DSL 字符串合成，再转成与 `CameraPluckerDataset` 一致的
`(intrinsics, poses)`。视频位于 `<video_root>/<idx:06d>_<suffix>/gen.mp4`，
`suffix` 为去掉空白/逗号、删除短横线后的 `pose_str`（`right-8, a-11` → `right8a11`）。

**pose_str DSL** — comma-separated `action-count` tokens (e.g. `w-3, right-8, d-4`);
each token repeats one per-frame motion `count` times:

| Token | Motion / 运动 | Token | Motion / 运动 |
| --- | --- | --- | --- |
| `w` | forward `+0.08` | `s` | forward `-0.08` |
| `d` | right `+0.08` | `a` | right `-0.08` |
| `up` | pitch `+deg2rad(3)` | `down` | pitch `-deg2rad(3)` |
| `right` | yaw `+deg2rad(3)` | `left` | yaw `-deg2rad(3)` |

**Input JSON / 输入 JSON:**

```json
[{"image_path": "...", "caption": "...", "pose_str": "w-3, right-8", "pose_json_path": "..."}]
```

**Run / 运行:**

```bash
torchrun --nproc_per_node=8 tools/data/hy/preencode_generated_wdplay.py \
    --input_json ./dataset/preencode_input.json \
    --video_root ./dataset/videos \
    --output_dir ./dataset/HY15/Action2V \
    --hunyuan_checkpoint_path ./ckpts/HunyuanVideo-1.5 --skip_existing
```

---

## `generate_negative_prompts.py` — CFG negative embeddings / 分类器引导负提示

**EN** — Single-GPU script that generates the empty-prompt LLM + byT5 embeddings
the training/dataloader pipeline loads as the CFG negative prompt. Run once per
checkpoint; output goes to the same directory as your preencoded latents.

**中文** — 单 GPU 脚本，生成训练和数据管线用作 CFG 负提示的空提示 LLM + byT5 嵌入。
每个 checkpoint 跑一次；输出放到与 latent 同一目录。

**Run / 运行:**

```bash
python tools/data/hy/generate_negative_prompts.py \
    --hunyuan_checkpoint_path ./ckpts/HunyuanVideo-1.5 \
    --output_dir ./dataset/HY15/TI2V
```

**Output / 输出:**

- `hunyuan_neg_prompt.pt` — `negative_prompt_embeds` (1,1000,3584), `negative_prompt_mask` (1,1000)
- `hunyuan_neg_byt5_prompt.pt` — `byt5_text_states` (1,256,1472), `byt5_text_mask` (1,256)
- `negative_prompt.pt` — compat copy (embeds + mask)

---

## Output .pt contract / 输出 `.pt` 键约定

**EN** — All three pre-encoders produce `.pt` files with the following tensors:

| Key | Shape | Type | Source | Notes |
| --- | --- | --- | --- | --- |
| `latent` | (1,32,N_latent,H,W) | fp16 | VAE | N_latent = (max_frames-1)//4 + 1 = 20 for 77 frames |
| `image_cond` | (1,32,1,H,W) | fp16 | VAE | first latent frame (TI2V mode) |
| `prompt_embeds` | (1,1000,3584) | fp16 | LLM | — |
| `prompt_mask` | (1,1000) | int64 | LLM | — |
| `vision_states` | (1,729,1152) | fp16 | SigLIP | first video frame |
| `byt5_text_states` | (1,256,1472) | fp16 | byT5 | glyph encoder |
| `byt5_text_mask` | (1,256) | int64 | byT5 | — |
| `intrinsics` | (4,) | fp32 | camera script | `[fx_norm, fy_norm, cx_norm, cy_norm]` (camera only) |
| `poses` | (N_cam,7) | fp32 | camera script | `[tx,ty,tz, qx,qy,qz,qw]` w2c (camera only) |
| `camera_indices` | (N_cam,) | int64 | camera script | latent-frame indices (camera only) |
| `video_frame_indices` | (max_frames,) | int64 | extractor | sequential 0..76 for 77 frames |
| `num_video_frames` | scalar | int | JSON | total frames in source video (camera only) |
| `video_id` | — | str | JSON | unique identifier (camera only) |

**中文** — 三个预编码脚本生成的 `.pt` 包含以下张量（相机相关键仅出现在
`preencode_camera_video.py` / `preencode_generated_wdplay.py` 输出中）。



