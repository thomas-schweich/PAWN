#!/usr/bin/env bash
# One-time setup on a fresh Runpod (or similar) GPU VM.
# Run from the project root: cd /workspace/pawn && bash deploy/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== PAWN Setup ==="
echo "Root: $ROOT"

# --- System packages ---
NEED_INSTALL=()
for pkg in rsync zstd; do
    command -v "$pkg" &>/dev/null || NEED_INSTALL+=("$pkg")
done
if [ ${#NEED_INSTALL[@]} -gt 0 ]; then
    apt-get update -qq && apt-get install -y -qq "${NEED_INSTALL[@]}"
fi

# --- Rust (needed for chess engine) ---
if ! command -v cargo &>/dev/null; then
    echo ""
    echo "--- Installing Rust ---"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env" 2>/dev/null || true

# --- uv ---
echo ""
echo "--- Installing/upgrading uv ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# --- Build chess engine ---
echo ""
echo "--- Building chess engine ---"
cd "$ROOT/engine"
uv run --with maturin maturin develop --release
cd "$ROOT"

# --- Sync workspace with CUDA torch ---
echo ""
echo "--- Syncing workspace (cu128) ---"
uv sync --extra cu128 --no-dev

# --- Verify ---
echo ""
echo "--- Verifying CUDA ---"
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "--- Verifying chess engine ---"
uv run python -c "import chess_engine; print(f'chess_engine OK: {len(chess_engine.export_move_vocabulary()[\"token_to_move\"])} tokens')"

# --- Decompress data if needed ---
if ls "$ROOT/data"/*.zst &>/dev/null; then
    echo ""
    echo "--- Decompressing training data ---"
    cd "$ROOT/data"
    for zst in *.zst; do
        [ -f "$zst" ] || continue
        pgn="${zst%.zst}"
        if [ -f "$pgn" ]; then
            echo "  $pgn already exists, skipping"
        else
            echo "  Decompressing $zst..."
            zstd -d "$zst" -o "$pgn"
        fi
    done
    cd "$ROOT"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To pretrain:"
echo "  uv run python scripts/train.py --variant base"
echo ""
echo "To train an adapter:"
echo "  uv run python scripts/train.py --run-type adapter --strategy bottleneck \\"
echo "      --checkpoint thomas-schweich/pawn-base \\"
echo "      --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \\"
echo "      --bottleneck-dim 32 --lr 1e-4"
