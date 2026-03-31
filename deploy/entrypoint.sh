#!/usr/bin/env bash
set -euo pipefail

# ── Workspace symlinks (persistent storage) ────────────────────────
if mkdir -p /workspace/logs /workspace/sweep_results /workspace/plots \
            /workspace/optuna-storage /opt/pawn/local 2>/dev/null; then
    ln -sfn /workspace/sweep_results /opt/pawn/local/optuna_results
    ln -sfn /workspace/logs /opt/pawn/logs
    echo "Workspace symlinks ready"
else
    echo "WARNING: /workspace not available — skipping symlinks"
fi

# ── CUDA MPS (multi-process service for GPU sharing) ───────────────
if command -v nvidia-cuda-mps-control &>/dev/null; then
    nvidia-cuda-mps-control -d 2>/dev/null && echo "CUDA MPS daemon started" \
        || echo "CUDA MPS already running or unavailable"
fi

# Hand off to RunPod entrypoint (SSH + Jupyter)
exec /start.sh
