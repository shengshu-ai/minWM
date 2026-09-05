#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# User Input (MUST set before running)
# =============================================================================
# Wan2.1 VAE checkpoint
VAE_PATH="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
# preencode_input.json (same format as HY15 camera: caption + pose_str per sample)
INPUT_JSON="./dataset/preencode_input.json"
# Directory containing generated videos: {idx:06d}_{pose_suffix}/gen.mp4
VIDEO_DIR="./dataset/videos"

# =============================================================================
# Intermediate / Output Paths
# =============================================================================
OUTPUT_DIR="./dataset/Wan21/Action2V"

# =============================================================================
# Resources
# =============================================================================
NUM_GPUS_PER_NODE=8
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29700}

echo "=== Wan Camera LMDB Build ==="
echo "  INPUT_JSON: $INPUT_JSON"
echo "  VIDEO_DIR:  $VIDEO_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  NNODES: $NNODES, NODE_RANK: $NODE_RANK"
echo "  TOTAL_GPUS: $((NUM_GPUS_PER_NODE * NNODES))"
echo "=============================="

torchrun \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    tools/data/wan21/build_worldplaygen_lmdb.py \
    --input_json "$INPUT_JSON" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --vae_path "$VAE_PATH"

echo "Done. Output: $OUTPUT_DIR"
