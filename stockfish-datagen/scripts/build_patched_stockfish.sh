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
# Pin to the sf18-v0.2.0 tag of our fork — based on Stockfish 18 release
# (cb3d4ee9, tag sf_18 upstream) plus two additive extensions:
#   - `evallegal` UCI command (per-legal-move NNUE eval, single line)
#   - `NetSelection` UCI option (auto|small|large, default auto)
# The `sf18-` prefix makes the upstream base explicit in the tag itself, so
# future rebases onto a new Stockfish release land at e.g. `sf19-v0.1.0`
# without semver-only ambiguity. Bundles nn-c288c895ea92.nnue (big) +
# nn-37f18f62d772.nnue (small), the same NNUE weights as vanilla SF18. With
# `NetSelection=auto` (the default), every other command is bit-identical
# to vanilla SF18.
SF_REPO="https://github.com/thomas-schweich/stockfish-ml-extensions.git"
# Pinning to commit SHA, not tag. Lightweight tags can be force-moved on
# the remote and a `git fetch` against a stale local clone will silently
# leave the local tag pointing at the old commit (fetch never moves an
# existing local tag without --force). Pinning to SHA matches the
# Dockerfile's pin and guarantees byte-identical builds across machines.
# This SHA is the commit annotated tag `sf18-v0.2.0` currently points at;
# bump together if/when the tag advances.
SF_COMMIT="777b88071fa00decaed2d1d6b1d4b2031bacb428"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SF_DIR="$ROOT_DIR/upstream-stockfish"
OUT_BIN="$ROOT_DIR/stockfish-patched"

if [ ! -d "$SF_DIR" ]; then
    echo "Cloning $SF_REPO into $SF_DIR..."
    git clone "$SF_REPO" "$SF_DIR"
fi

cd "$SF_DIR"
git fetch --quiet origin
git checkout --quiet "$SF_COMMIT"
git reset --hard --quiet "$SF_COMMIT"

cd src
echo "Building Stockfish ($ARCH)..."
make -j build ARCH="$ARCH" >/dev/null

cp stockfish "$OUT_BIN"
echo "Built: $OUT_BIN"
"$OUT_BIN" bench 2>&1 | tail -1
