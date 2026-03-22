#!/usr/bin/env bash
# Build the deploy directory for rsync to Runpod.
# Usage: bash deploy/build.sh [--checkpoint PATH] [--data-dir PATH]
#
# This creates deploy/pawn-deploy/ with everything needed to run
# experiments on a fresh GPU VM. rsync the result:
#   rsync -avz --progress deploy/pawn-deploy/ root@<pod-ip>:/workspace/pawn/
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$REPO/deploy/pawn-deploy"

# Parse args
CHECKPOINT=""
DATA_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Building deploy package ==="
echo "Source: $REPO"
echo "Target: $DEPLOY"

# Clean slate
rm -rf "$DEPLOY"
mkdir -p "$DEPLOY"/{data,checkpoints,logs}

# Code — rsync with excludes
echo ""
echo "--- Syncing code ---"
for dir in engine pawn; do
    rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
          --exclude='target' --exclude='*.so' --exclude='*.egg-info' \
          --exclude='.mypy_cache' --exclude='logs' --exclude='data' \
          --exclude='uv.lock' \
          "$REPO/$dir/" "$DEPLOY/$dir/"
done

# Scripts
rsync -a --exclude='__pycache__' "$REPO/scripts/" "$DEPLOY/scripts/"

# Tests
rsync -a --exclude='__pycache__' "$REPO/tests/" "$DEPLOY/tests/"

# Root config
cp "$REPO/pyproject.toml" "$DEPLOY/pyproject.toml"

# Deploy scripts
mkdir -p "$DEPLOY/deploy"
cp "$REPO/deploy/setup.sh" "$DEPLOY/deploy/setup.sh"
chmod +x "$DEPLOY/deploy/setup.sh"

# Checkpoint (if provided)
if [ -n "$CHECKPOINT" ]; then
    echo ""
    echo "--- Copying checkpoint ---"
    cp "$CHECKPOINT" "$DEPLOY/checkpoints/$(basename "$CHECKPOINT")"
fi

# Data — compress PGN files with zstd (if provided)
if [ -n "$DATA_DIR" ]; then
    echo ""
    echo "--- Compressing training data ---"
    for pgn in "$DATA_DIR"/*.pgn; do
        [ -f "$pgn" ] || continue
        name="$(basename "$pgn")"
        zst="$DEPLOY/data/${name}.zst"
        if [ -f "$zst" ]; then
            echo "  $name.zst already exists, skipping"
        else
            echo "  Compressing $name..."
            zstd -T0 -3 "$pgn" -o "$zst"
        fi
    done
fi

# Summary
echo ""
echo "=== Deploy package ready ==="
du -sh "$DEPLOY"
echo ""
echo "Contents:"
du -sh "$DEPLOY"/*
echo ""
echo "To deploy:"
echo "  rsync -avz --progress $DEPLOY/ root@<pod-ip>:/workspace/pawn/"
echo "  ssh root@<pod-ip> 'cd /workspace/pawn && bash deploy/setup.sh'"
