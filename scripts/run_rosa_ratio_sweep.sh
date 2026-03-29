#!/usr/bin/env bash
# Launch a RoSA bottleneck-to-sparse ratio sweep.
#
# Explores how to allocate a fixed parameter budget between bottleneck adapters
# and gradient-informed sparse masks in retro-bottleneck mode.
#
# Budgets tested: 100K, 250K, 500K total params
# Primary variable: bottleneck_ratio (fraction allocated to bottleneck)
set -euo pipefail

uv run python scripts/sweep.py \
    --adapter rosa-ratio --in-process \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --elo-min 1800 --elo-max 1900 \
    --n-trials 30 --epochs 30 \
    --pruner hyperband \
    --output-dir sweeps/rosa-ratio \
    --study-name rosa-ratio
