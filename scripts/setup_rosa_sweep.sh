#!/usr/bin/env bash
# Setup and launch RoSA sweep on a GPU pod.
# Run from inside the pod SSH session.
set -euo pipefail

echo "=== PAWN RoSA Sweep Setup ==="

# Install uv + Rust
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Clone repo
cd /workspace
if [ -d pawn/.git ]; then
    echo "Repo exists, pulling..."
    cd pawn && git pull origin main && cd ..
else
    echo "Cloning repo..."
    git clone https://github.com/thomas-schweich/PAWN.git pawn
fi
cd /workspace/pawn

# Build engine
echo "Building chess engine..."
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install deps
echo "Installing Python deps..."
uv sync --extra cu128

# Pull checkpoint
echo "Pulling pawn-base checkpoint..."
git submodule update --init checkpoints/pawn-base

# Download one Lichess Parquet file for training
echo "Downloading Lichess training data..."
mkdir -p /workspace/data
uv run python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('thomas-schweich/lichess-1800-1900',
    'data/train-00000-of-00001.parquet', repo_type='dataset',
    local_dir='/workspace/data')
print(f'Downloaded: {path}')
"

# Create sweep output dir
mkdir -p /workspace/sweeps

# Verify
echo ""
echo "=== Verification ==="
uv run python -c "import chess_engine; print(f'Engine OK: {len(chess_engine.export_move_vocabulary()[\"token_to_move\"])} tokens')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "=== Ready ==="
echo "Launch sweeps with:"
echo ""
echo "  cd /workspace/pawn"
echo ""
echo "  # RoSA (joint LoRA + gradient sparse)"
echo "  nohup uv run python scripts/sweep.py \\"
echo "    --adapter rosa --in-process --n-trials 30 --n-gpus 2 --n-jobs 2 \\"
echo "    --checkpoint checkpoints/pawn-base \\"
echo "    --pgn /workspace/data/data/train-00000-of-00001.parquet \\"
echo "    --output-dir /workspace/sweeps --local-checkpoints \\"
echo "    > /workspace/sweeps/rosa.log 2>&1 &"
echo ""
echo "  # Then retro-sparse, then retro-bottleneck (same command, change --adapter)"
