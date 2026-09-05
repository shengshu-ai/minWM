#!/bin/bash
# Single-process launcher for a DCP -> single-file checkpoint dump.
#
# Wraps tools/export_checkpoint.py so `auto_dump.py` can submit one consolidation
# per step as a cluster job (`<launch-tool> submit -- scripts/dump.sh ...`). The dump
# is CPU-only and single-process — no torchrun, no GPU — but it rebuilds the whole
# model state dict in RAM, so it wants a node with RAM >= model size.
#
# All arguments are forwarded verbatim to tools/export_checkpoint.py:
#
#   bash scripts/dump.sh --checkpoint <dcp_dir> --output <weight> \
#       --format safetensors --model-key model [--config <cfg>]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

echo "[dump.sh] python=$PYTHON"
"$PYTHON" tools/export_checkpoint.py "$@"
