#!/bin/bash
# Entrypoint for RoSA sweep container.
# Runs all three ablation sweeps sequentially, uploads results to HF.
#
# Required env vars:
#   HF_TOKEN     — HuggingFace write token
#   PARQUET_REPO — HF dataset repo (e.g. thomas-schweich/lichess-1800-1900)
#   CHECKPOINT   — HF model repo (e.g. thomas-schweich/pawn-base)
#
# Optional env vars:
#   N_TRIALS     — Trials per sweep (default: 30)
#   N_GPUS       — Number of GPUs (default: 2)
#   N_JOBS       — Parallel trials (default: same as N_GPUS)
#   EPOCHS       — Epochs per trial (default: 50)
#   MAX_GAMES    — Max training games (default: 12000)
set -euo pipefail

N_TRIALS="${N_TRIALS:-30}"
N_GPUS="${N_GPUS:-2}"
N_JOBS="${N_JOBS:-$N_GPUS}"
EPOCHS="${EPOCHS:-50}"
MAX_GAMES="${MAX_GAMES:-12000}"
PARQUET_REPO="${PARQUET_REPO:-thomas-schweich/lichess-1800-1900}"
CHECKPOINT="${CHECKPOINT:-thomas-schweich/pawn-base}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required"
    exit 1
fi

# Persist HF token
mkdir -p /root/.cache/huggingface
echo -n "$HF_TOKEN" > /root/.cache/huggingface/token

cd /opt/pawn

echo "=== RoSA Ablation Sweeps ==="
echo "  Checkpoint: $CHECKPOINT"
echo "  Data: $PARQUET_REPO"
echo "  Trials: $N_TRIALS per sweep"
echo "  GPUs: $N_GPUS, Jobs: $N_JOBS"
echo "  Epochs: $EPOCHS, Max games: $MAX_GAMES"
echo ""

# Pull checkpoint from HF
echo "=== Pulling checkpoint ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$CHECKPOINT', local_dir='checkpoints/model')
print('Downloaded to checkpoints/model/')
"

# Download training parquet
echo "=== Downloading training data ==="
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('$PARQUET_REPO',
    'data/train-00000-of-00001.parquet', repo_type='dataset',
    local_dir='/dev/shm/data')
print(f'Downloaded: {path}')
"
PARQUET_PATH="/dev/shm/data/data/train-00000-of-00001.parquet"

# Verify GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

mkdir -p /workspace/sweeps

# Run all three sweeps
for ADAPTER in rosa retro-sparse retro-bottleneck; do
    echo ""
    echo "============================================================"
    echo "=== Sweep: $ADAPTER ($(date)) ==="
    echo "============================================================"

    python3 scripts/sweep.py \
        --adapter "$ADAPTER" \
        --in-process \
        --n-trials "$N_TRIALS" \
        --n-gpus "$N_GPUS" \
        --n-jobs "$N_JOBS" \
        --epochs "$EPOCHS" \
        --checkpoint checkpoints/model \
        --pgn "$PARQUET_PATH" \
        --output-dir /workspace/sweeps \
        || echo "WARNING: $ADAPTER sweep failed with exit code $?"

    echo "=== $ADAPTER sweep complete ($(date)) ==="
done

# Copy sweep DBs to workspace (persisted on volume)
echo ""
echo "=== Results ==="
find /workspace/sweeps -name 'study.db' -exec ls -lh {} \;

echo ""
echo "=== All sweeps done ($(date)) ==="
echo "View results: optuna-dashboard sqlite:///workspace/sweeps/<adapter>/study.db"
