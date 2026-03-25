#!/usr/bin/env bash
# One-shot setup for rosa-sweep pod. Run from inside the pod SSH session:
#   bash <(curl -sL https://raw.githubusercontent.com/thomas-schweich/PAWN/main/scripts/setup_rosa_pod.sh)
# Or paste into the terminal.
set -euo pipefail

echo "=== PAWN RoSA Sweep Pod Setup ==="

# Install uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Rust (needed to build engine)
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Clone repo to /workspace (persisted on volume)
cd /workspace
if [ -d pawn/.git ]; then
    echo "Repo exists, pulling latest..."
    cd pawn && git pull origin main && cd ..
else
    echo "Cloning repo..."
    git clone https://github.com/thomas-schweich/PAWN.git pawn
fi

cd /workspace/pawn

# Build Rust engine
echo "Building chess engine..."
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python deps
echo "Installing Python dependencies..."
uv sync --extra cu128

# Pull pawn-base checkpoint
echo "Pulling pawn-base checkpoint..."
git submodule update --init checkpoints/pawn-base

# Create sweep output dir on volume
mkdir -p /workspace/sweeps

# Verify
echo ""
echo "=== Verification ==="
uv run python -c "import chess_engine; print(f'Engine OK: {len(chess_engine.export_move_vocabulary())} moves')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "=== Ready ==="
echo "Run sweeps with:"
echo "  cd /workspace/pawn"
echo "  CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/sweep.py \\"
echo "    --adapter rosa --in-process --n-trials 30 --n-gpus 2 --n-jobs 2 \\"
echo "    --checkpoint checkpoints/pawn-base --pgn <PGN_PATH> \\"
echo "    --output-dir /workspace/sweeps --local-checkpoints"
