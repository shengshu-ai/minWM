#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"

# ===== Configuration =====
HUNYUAN_CHECKPOINT="${HUNYUAN_CHECKPOINT:-./ckpts/HunyuanVideo-1.5}"

INPUT_JSON="${INPUT_JSON:-./dataset/videos.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./dataset/HY15/TI2V}"
TARGET_HEIGHT=480
TARGET_WIDTH=832
NUM_GPUS=8

# ===== Pre-encode video latents =====
echo "Pre-encoding video latents (${NUM_GPUS} GPUs)..."

torchrun --nproc_per_node=$NUM_GPUS \
    tools/data/hy/preencode_video.py \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --hunyuan_checkpoint_path "$HUNYUAN_CHECKPOINT" \
    --target_height $TARGET_HEIGHT \
    --target_width $TARGET_WIDTH \
    --skip_existing

echo "Done. Latents at: $OUTPUT_DIR"
