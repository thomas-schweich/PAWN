#!/bin/bash
# Lichess PGN -> PAWN Parquet extraction entrypoint.
# Streams monthly database dumps, parses via the Rust engine, and uploads
# each shard to a HuggingFace dataset repo as it's written.
#
# Required env vars:
#   MONTHS              — space-separated training months (e.g., "2025-01 2025-02")
#   DEST_REPO           — destination HuggingFace dataset repo
#
# Optional env vars:
#   HF_TOKEN            — HuggingFace token (also writes to ~/.cache/huggingface/token)
#   HOLDOUT_MONTH       — month for val/test splits (e.g., "2026-01")
#   VAL_DAYS            — day range for validation split (e.g., "1-3")
#   TEST_DAYS           — day range for test split (e.g., "15-17")
#   REVISION            — destination branch (default: "run/extract")
#   MAX_PLY             — ply cap per game (default: 512)
#   BATCH_SIZE          — games per parsing batch (default: 500000)
#   SHARD_SIZE          — games per output parquet shard (default: 1000000)
#   JOBS                — parallel worker processes (default: 4)
#   THREADS_PER_WORKER  — rayon/polars threads per worker (default: 2)
#   SCRATCH_DIR         — local staging dir (default: /dev/shm/pawn-lichess-extract)
#   MAX_GAMES_PER_MONTH — cap games per month, for smoke tests
#   SOURCE              — raw PGN path template with '{year_month}' placeholder
#                         (hf://buckets/..., hf://datasets/..., or local path).
#                         Defaults to the raw Lichess bucket.
#   FORCE               — set to 1 to ignore existing sentinels and re-extract
set -euo pipefail

echo "=== Lichess Parquet Extraction ==="
echo "  Training months:   ${MONTHS:?MONTHS env var is required}"
echo "  Destination repo:  ${DEST_REPO:?DEST_REPO env var is required}"
echo "  Revision:          ${REVISION:-run/extract}"
echo "  Holdout month:     ${HOLDOUT_MONTH:-none}"
echo "  Jobs:              ${JOBS:-4}"
echo "  Threads/worker:    ${THREADS_PER_WORKER:-2}"
echo ""

# Persist HF token if provided
if [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
    echo "HF token persisted"
fi

# Install zstandard if not available (needed for streaming decompression)
python3 -c "import zstandard" 2>/dev/null || pip install --no-cache-dir zstandard

# Build the command as an array to avoid shell injection
CMD=(python3 /opt/pawn/scripts/extract_lichess_parquet.py
    --months $MONTHS
    --dest-repo "$DEST_REPO"
    --revision "${REVISION:-run/extract}"
    --max-ply "${MAX_PLY:-512}"
    --batch-size "${BATCH_SIZE:-500000}"
    --shard-size "${SHARD_SIZE:-1000000}"
    --jobs "${JOBS:-4}"
    --threads-per-worker "${THREADS_PER_WORKER:-2}"
    --scratch-dir "${SCRATCH_DIR:-/dev/shm/pawn-lichess-extract}"
)

if [ -n "${HOLDOUT_MONTH:-}" ]; then
    CMD+=(--holdout-month "$HOLDOUT_MONTH")
fi
if [ -n "${VAL_DAYS:-}" ]; then
    CMD+=(--val-days "$VAL_DAYS")
fi
if [ -n "${TEST_DAYS:-}" ]; then
    CMD+=(--test-days "$TEST_DAYS")
fi
if [ -n "${MAX_GAMES_PER_MONTH:-}" ]; then
    CMD+=(--max-games-per-month "$MAX_GAMES_PER_MONTH")
fi
if [ -n "${SOURCE:-}" ]; then
    CMD+=(--source "$SOURCE")
fi
if [ -n "${FORCE:-}" ] && [ "${FORCE}" != "0" ]; then
    CMD+=(--force)
fi

echo "Running: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
