#!/usr/bin/env bash
# Build the patched Stockfish binary used for tier-0 (evallegal) data gen.
#
# Clones the `stockfish-ml-extensions` fork (Stockfish 18 + the `evallegal`
# UCI command) and builds it. Output binary is placed in the repo root as
# `stockfish-patched` (gitignored).
#
# Usage:
#   bash scripts/build_patched_stockfish.sh           # default: x86-64-avx2
#   bash scripts/build_patched_stockfish.sh x86-64-avx512
#
# Requirements: g++, make, git, an internet connection (one-time clone).

set -euo pipefail

ARCH="${1:-x86-64-avx2}"
# Pin to the v18.evallegal.0 tag of our fork — Stockfish 18 release
# (cb3d4ee9, tag sf_18 upstream) plus the additive `evallegal` UCI command.
# Bundles nn-c288c895ea92.nnue (big) + nn-37f18f62d772.nnue (small), the same
# NNUE weights as vanilla SF18. Patch is purely additive: every command other
# than `evallegal` is bit-identical to vanilla SF18.
SF_REPO="https://github.com/thomas-schweich/stockfish-ml-extensions.git"
SF_TAG="v18.evallegal.0"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SF_DIR="$ROOT_DIR/upstream-stockfish"
OUT_BIN="$ROOT_DIR/stockfish-patched"

if [ ! -d "$SF_DIR" ]; then
    echo "Cloning $SF_REPO into $SF_DIR..."
    git clone "$SF_REPO" "$SF_DIR"
fi

cd "$SF_DIR"
git fetch --quiet --tags origin
git checkout --quiet "$SF_TAG"
git reset --hard --quiet "$SF_TAG"

cd src
echo "Building Stockfish ($ARCH)..."
make -j build ARCH="$ARCH" >/dev/null

cp stockfish "$OUT_BIN"
echo "Built: $OUT_BIN"
"$OUT_BIN" bench 2>&1 | tail -1
