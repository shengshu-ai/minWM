#!/usr/bin/env bash
set -euo pipefail

# Launch through torchrun --no-python so every rank gets its own report. Pass
# the framework capture window explicitly through dotlist arguments, e.g.:
#
#   torchrun --no-python --nproc_per_node=2 tools/profile_nsys.sh \
#       python tools/train_mwm.py --config-file configs/train.py \
#       profile.enabled=true profile.start_iter=1 profile.end_iter=3 profile.rank=-1

if (($# == 0)); then
    echo "usage: tools/profile_nsys.sh <command> [args ...]" >&2
    exit 2
fi

output_dir=${NSYS_OUTPUT_DIR:-nsys-traces}
run_name=${NSYS_RUN_NAME:-minwm}
rank=${LOCAL_RANK:-${RANK:-0}}
mkdir -p "$output_dir"

exec nsys profile \
    --output="${output_dir}/${run_name}.rank${rank}" \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    "$@"
