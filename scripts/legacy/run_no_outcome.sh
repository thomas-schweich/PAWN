#!/usr/bin/env bash
# Launch no-outcome-token ablation pretraining on RunPod.
# Trains small/base/large simultaneously with outcome token stripped.
set -euo pipefail

cd /opt/pawn

uv run python scripts/train_all.py \
    --no-outcome-token \
    --hf-repo thomas-schweich/pawn-no-outcome \
    --total-steps 100000 \
    --batch-size 256 \
    --shm-checkpoints \
    --run-evals
