#!/bin/bash
# Lichess PGN -> PAWN Parquet extraction entrypoint.
# Downloads monthly database dumps, parses via Rust engine, writes sharded
# Parquet with train/val/test splits, and optionally pushes to HuggingFace.
#
# Required env vars:
#   MONTHS          — space-separated training months (e.g., "2025-01 2025-02 2025-03")
#
# Optional env vars:
#   HF_TOKEN        — HuggingFace token (for pushing dataset)
#   HF_REPO         — HuggingFace dataset repo (e.g., "thomas-schweich/pawn-lichess-full")
#   HOLDOUT_MONTH   — month for val/test (e.g., "2023-12")
#   HOLDOUT_GAMES   — games per split from holdout month (default: 50000)
#   BATCH_SIZE      — games per parsing batch (default: 500000)
#   SHARD_SIZE      — games per output shard (default: 1000000)
#   MAX_GAMES       — stop after this many training games (for testing)
#   OUTPUT_DIR      — output directory (default: /workspace/lichess-parquet)
#   SEED            — random seed for holdout sampling (default: 42)
set -euo pipefail

echo "=== Lichess Parquet Extraction ==="
echo "  Training months: ${MONTHS:?MONTHS env var is required}"
echo "  Holdout month: ${HOLDOUT_MONTH:-none}"
echo "  Holdout games/split: ${HOLDOUT_GAMES:-50000}"
echo "  HF Repo: ${HF_REPO:-none}"
echo "  Batch size: ${BATCH_SIZE:-500000}"
echo "  Shard size: ${SHARD_SIZE:-1000000}"
echo ""

# Persist HF token if provided
if [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
    echo "HF token persisted"
fi

# Install zstandard if not available (needed for streaming decompression)
python3 -c "import zstandard" 2>/dev/null || pip install --no-cache-dir zstandard

# Build the command
CMD="python3 /opt/pawn/scripts/extract_lichess_parquet.py"
CMD="$CMD --months $MONTHS"
CMD="$CMD --output ${OUTPUT_DIR:-/workspace/lichess-parquet}"
CMD="$CMD --batch-size ${BATCH_SIZE:-500000}"
CMD="$CMD --shard-size ${SHARD_SIZE:-1000000}"
CMD="$CMD --seed ${SEED:-42}"

if [ -n "${HOLDOUT_MONTH:-}" ]; then
    CMD="$CMD --holdout-month $HOLDOUT_MONTH"
    CMD="$CMD --holdout-games ${HOLDOUT_GAMES:-50000}"
fi
if [ -n "${HF_REPO:-}" ]; then
    CMD="$CMD --hf-repo $HF_REPO"
fi
if [ -n "${MAX_GAMES:-}" ]; then
    CMD="$CMD --max-games $MAX_GAMES"
fi

echo "Running: $CMD"
echo ""
exec $CMD
