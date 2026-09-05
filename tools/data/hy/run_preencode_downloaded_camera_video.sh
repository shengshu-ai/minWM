#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"

# =============================================================================
# User Input (override via env vars if needed)
# =============================================================================
# HunyuanVideo-1.5 model directory
HUNYUAN_CHECKPOINT="${HUNYUAN_CHECKPOINT:-./ckpts/HunyuanVideo-1.5}"
# Directory containing downloaded data (preencode_input.json + videos/)
INPUT_DIR="${INPUT_DIR:-./dataset}"
# Directory to write pre-encoded latents and negative prompts
OUTPUT_DIR="${OUTPUT_DIR:-./dataset/HY15/Action2V}"

# =============================================================================
# Resources
# =============================================================================
NUM_GPUS=8

# ===== Pre-encode videos + camera =====
echo "Pre-encoding videos with camera data..."

torchrun --nproc_per_node=$NUM_GPUS \
    tools/data/hy/preencode_generated_wdplay.py \
    --input_json "${INPUT_DIR}/preencode_input.json" \
    --video_root "${INPUT_DIR}/videos" \
    --output_dir "${OUTPUT_DIR}" \
    --hunyuan_checkpoint_path "$HUNYUAN_CHECKPOINT" \
    --skip_existing

echo "Done. Latents at: ${OUTPUT_DIR}"
