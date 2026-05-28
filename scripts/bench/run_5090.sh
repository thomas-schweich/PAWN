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

# Python launcher. On a dev checkout `uv run --extra cu128 python` resolves
# the GPU extra; on the prebuilt runtime image (thomasschweich/pawn:jax) the
# venv is already active on PATH and `uv run` would instead try to rebuild
# the chess-engine workspace member from source (no Rust toolchain present)
# and fail — so there, launch with `PYRUN=python`.
PYRUN="${PYRUN:-uv run --extra cu128 python}"

# Which stages to run: "all" (default), "step12" (cost_curve + matrix only),
# or "step3" (end-to-end training only). Lets us re-run just the end-to-end
# stage after an interrupt without repeating the ~30-min matrix.
BENCH_ONLY="${BENCH_ONLY:-all}"

# Step-3 end-to-end training batch size. The *production supernet joint
# 3-variant* training (forward+backward for small+base+large on one batch,
# plus optimizer state and the K-step lax.scan) does NOT fit a 32 GB card at
# B=64 T=512 with default materialised attention — it OOMs. 32 fits with
# headroom; override with STEP3_BSZ for larger cards.
STEP3_BSZ="${STEP3_BSZ:-32}"

# Git SHA is unavailable in the runtime image (.git is excluded from the
# build context); fall back to the baked-in $PAWN_GIT_HASH, then "unknown".
_sha="$(git rev-parse --short HEAD 2>/dev/null || echo "${PAWN_GIT_HASH:-unknown}")"
LABEL="${1:-5090-${_sha}}"

echo "===================================================="
echo "5090 bench harness — label: $LABEL"
echo "Git SHA: ${_sha}"
echo "Launcher: $PYRUN"
echo "Stages: $BENCH_ONLY   Step-3 batch: $STEP3_BSZ"
echo "JAX device: $($PYRUN -c 'import jax; print(jax.devices()[0])' 2>&1 | tail -1)"
echo "===================================================="

if [ "$BENCH_ONLY" = "all" ] || [ "$BENCH_ONLY" = "step12" ]; then
    # Step 1: cost(T) curve at production B=64
    echo
    echo "==> Step 1a: cost(T) curve SUPERNET LARGE B=64 1v"
    $PYRUN scripts/bench/cost_curve.py \
        --supernet production --batch-size 64 --k 50 \
        --warmup-outers 2 --timed-outers 10 \
        --seq-lens 128 256 384 512 \
        --n-variants-mode 1v \
        --label "${LABEL}-1v"

    echo
    echo "==> Step 1b: cost(T) curve SUPERNET LARGE B=64 3v-stoch"
    $PYRUN scripts/bench/cost_curve.py \
        --supernet production --batch-size 64 --k 50 \
        --warmup-outers 2 --timed-outers 10 \
        --seq-lens 128 256 384 512 \
        --n-variants-mode 3v-stoch \
        --label "${LABEL}-3v-stoch"

    # Step 2: full bench matrix
    echo
    echo "==> Step 2: full bench matrix"
    $PYRUN scripts/bench/run.py \
        --k 50 --warmup-outers 2 --timed-outers 30 \
        --label "${LABEL}-matrix"
fi

if [ "$BENCH_ONLY" = "all" ] || [ "$BENCH_ONLY" = "step3" ]; then
    # Step 3: end-to-end bucketed vs no-bucketing vs all-opt-in.
    # Each run is non-fatal (|| true): an OOM or crash in one variant must
    # not abort the others (under `set -e` it otherwise kills the script,
    # which is how the B=64 attempt previously lost 3b/3c). A failed run
    # leaves no val record; the analysis step treats a missing log as N/A.
    echo
    echo "==> Step 3a: end-to-end bucketed (default, AdamW, no grad-norm emit) B=$STEP3_BSZ"
    rm -rf logs/5090-bucketed 2>/dev/null
    $PYRUN scripts/train_jax.py \
        --supernet production --total-steps 500 --batch-size "$STEP3_BSZ" --seq-len 512 --k 50 \
        --local-checkpoints --logs-dir logs/5090-bucketed \
        || echo "  (Step 3a FAILED — see traceback above)"

    echo
    echo "==> Step 3b: end-to-end no-bucketing B=$STEP3_BSZ"
    rm -rf logs/5090-nobucket 2>/dev/null
    $PYRUN scripts/train_jax.py \
        --supernet production --total-steps 500 --batch-size "$STEP3_BSZ" --seq-len 512 --k 50 \
        --no-bucketing --local-checkpoints --logs-dir logs/5090-nobucket \
        || echo "  (Step 3b FAILED — see traceback above)"

    echo
    echo "==> Step 3c: end-to-end with all opt-ins (Lion + emit_grad_norms) B=$STEP3_BSZ"
    rm -rf logs/5090-allopt 2>/dev/null
    $PYRUN scripts/train_jax.py \
        --supernet production --total-steps 500 --batch-size "$STEP3_BSZ" --seq-len 512 --k 50 \
        --optimizer lion --lr 1e-4 \
        --emit-grad-norms \
        --local-checkpoints --logs-dir logs/5090-allopt \
        || echo "  (Step 3c FAILED — see traceback above)"
fi

echo
echo "===================================================="
echo "5090 bench complete. Files in bench/results/ + logs/"
echo "===================================================="
ls -la bench/results/*.json | tail
