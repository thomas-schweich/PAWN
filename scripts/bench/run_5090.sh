#!/usr/bin/env bash
# Comprehensive 5090 perf measurement for jax_migration HEAD (384cd9a+).
#
# Run sequence:
#   1. cost_curve.py at SUPERNET LARGE B=64 1v + 3v-stoch, T ∈ {128,256,384,512}
#   2. run.py full matrix (BASE/LARGE × B={64,128,256} × {1v, 3v-det, 3v-stoch})
#   3. end-to-end train_jax.py 300 steps bucketed vs --no-bucketing
#
# Writes results into bench/results/. Pull with rsync after run.
set -euo pipefail

cd "$(dirname "$0")/../.."

LABEL="${1:-5090-$(git rev-parse --short HEAD)}"

echo "===================================================="
echo "5090 bench harness — label: $LABEL"
echo "Git SHA: $(git rev-parse --short HEAD)"
echo "JAX device: $(uv run --extra cu128 python -c 'import jax; print(jax.devices()[0])' 2>&1 | tail -1)"
echo "===================================================="

# Step 1: cost(T) curve at production B=64
echo
echo "==> Step 1a: cost(T) curve SUPERNET LARGE B=64 1v"
uv run --extra cu128 python scripts/bench/cost_curve.py \
    --supernet production --batch-size 64 --k 50 \
    --warmup-outers 2 --timed-outers 10 \
    --seq-lens 128 256 384 512 \
    --n-variants-mode 1v \
    --label "${LABEL}-1v"

echo
echo "==> Step 1b: cost(T) curve SUPERNET LARGE B=64 3v-stoch"
uv run --extra cu128 python scripts/bench/cost_curve.py \
    --supernet production --batch-size 64 --k 50 \
    --warmup-outers 2 --timed-outers 10 \
    --seq-lens 128 256 384 512 \
    --n-variants-mode 3v-stoch \
    --label "${LABEL}-3v-stoch"

# Step 2: full bench matrix
echo
echo "==> Step 2: full bench matrix"
uv run --extra cu128 python scripts/bench/run.py \
    --k 50 --warmup-outers 2 --timed-outers 30 \
    --label "${LABEL}-matrix"

# Step 3: end-to-end bucketed vs no-bucketing — all features on
echo
echo "==> Step 3a: end-to-end bucketed (default, AdamW, no grad-norm emit)"
rm -rf logs/5090-bucketed 2>/dev/null
uv run --extra cu128 python scripts/train_jax.py \
    --supernet production --total-steps 500 --batch-size 64 --seq-len 512 --k 50 \
    --local-checkpoints --logs-dir logs/5090-bucketed

echo
echo "==> Step 3b: end-to-end no-bucketing"
rm -rf logs/5090-nobucket 2>/dev/null
uv run --extra cu128 python scripts/train_jax.py \
    --supernet production --total-steps 500 --batch-size 64 --seq-len 512 --k 50 \
    --no-bucketing --local-checkpoints --logs-dir logs/5090-nobucket

echo
echo "==> Step 3c: end-to-end with all opt-ins (Lion + emit_grad_norms)"
rm -rf logs/5090-allopt 2>/dev/null
uv run --extra cu128 python scripts/train_jax.py \
    --supernet production --total-steps 500 --batch-size 64 --seq-len 512 --k 50 \
    --optimizer lion --lr 1e-4 \
    --emit-grad-norms \
    --local-checkpoints --logs-dir logs/5090-allopt

echo
echo "===================================================="
echo "5090 bench complete. Files in bench/results/ + logs/"
echo "===================================================="
ls -la bench/results/*.json | tail
