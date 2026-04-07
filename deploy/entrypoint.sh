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

# ── HF token persistence ─────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
fi

# ── SSH key injection (RunPod sets PUBLIC_KEY env var) ────────────────
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
fi

# ── CUDA MPS (multi-process service for GPU sharing) ───────────────
if command -v nvidia-cuda-mps-control &>/dev/null; then
    nvidia-cuda-mps-control -d 2>/dev/null && echo "CUDA MPS daemon started" \
        || echo "CUDA MPS already running or unavailable"
fi

# ── Configure and start SSH as the foreground process ────────────────
# tini (PID 1) reaps zombies and forwards signals.
if [ -x "$(command -v sshd)" ]; then
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
    echo "PAWN container ready — starting sshd"
    exec /usr/sbin/sshd -D
fi

# Fallback if sshd is somehow missing
echo "PAWN container ready (no sshd found)"
exec sleep infinity
