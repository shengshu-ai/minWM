#!/bin/bash
# torchrun launcher for minWM inference.
#
# Runs tools/infer_mwm.py under torchrun, deriving the distributed topology from
# the cluster environment (WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT) the
# same way the training scripts do, so one script serves both a local run and a
# cluster job submitted with `<launch-tool> submit -- scripts/infer.sh ...`.
#
# All arguments are forwarded verbatim to tools/infer_mwm.py, which takes only
# --config-file plus dotlist overrides:
#
#   bash scripts/infer.sh --config-file <cfg> \
#       inference.checkpoint=<weights> inference.output_dir=<dir> ...
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# Processes per node. minWM inference is single-process unless the config sets
# sequence parallelism (sp_size>1), so default to 1 rather than every visible
# GPU — an 8-GPU node would otherwise launch 8 redundant workers. Override with
# NPROC_PER_NODE (e.g. to match sp_size).
GPUS_PER_NODE=${NPROC_PER_NODE:-1}

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "[infer.sh] python=$PYTHON nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node=$GPUS_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

# Run torchrun via the resolved interpreter (`-m torch.distributed.run`) rather
# than the bare `torchrun` on PATH, which resolves to /usr/bin and its deps-less
# torch on the cluster image.
"$PYTHON" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    tools/infer_mwm.py "$@"
