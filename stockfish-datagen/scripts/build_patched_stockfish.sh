#!/usr/bin/env bash
# Build the patched Stockfish binary used for tier-0 (evallegal) data gen.
#
# Clones upstream Stockfish, applies our patches, builds. Output binary is
# placed in the repo root as `stockfish-patched` (gitignored).
#
# Usage:
#   bash scripts/build_patched_stockfish.sh           # default: x86-64-avx2
#   bash scripts/build_patched_stockfish.sh x86-64-avx512
#
# Requirements: g++, make, git, an internet connection (one-time clone).

set -euo pipefail

ARCH="${1:-x86-64-avx2}"
# Pin to SF18 release. Bundles nn-c288c895ea92.nnue (big) + nn-37f18f62d772.nnue
# (small) — the same NNUE weights as `~/bin/stockfish` SF18 release. The patch
# is purely additive (new `evallegal` UCI command), so every other command in
# the patched binary is bit-identical to vanilla SF18.
SF_TAG="sf_18"
SF_COMMIT="cb3d4ee9b47d0c5aae855b12379378ea1439675c"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SF_DIR="$ROOT_DIR/upstream-stockfish"
PATCH_DIR="$ROOT_DIR/patches"
OUT_BIN="$ROOT_DIR/stockfish-patched"

if [ ! -d "$SF_DIR" ]; then
    echo "Cloning Stockfish into $SF_DIR..."
    git clone https://github.com/official-stockfish/Stockfish.git "$SF_DIR"
fi

cd "$SF_DIR"
git fetch --quiet --tags origin
git checkout --quiet "$SF_COMMIT"
git reset --hard --quiet "$SF_COMMIT"

echo "Applying patches from $PATCH_DIR..."
for p in "$PATCH_DIR"/*.patch; do
    git apply "$p"
done

cd src
echo "Building Stockfish ($ARCH)..."
make -j build ARCH="$ARCH" >/dev/null

cp stockfish "$OUT_BIN"
echo "Built: $OUT_BIN"
"$OUT_BIN" bench 2>&1 | tail -1
