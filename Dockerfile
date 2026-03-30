# PAWN — single image for all RunPod workloads
#
# Built automatically by CI on merge to main and pushed to Docker Hub.
# Uses the RunPod base image (CUDA + SSH + Jupyter) with uv for
# reproducible Python dependency management.
#
# Usage: create a RunPod template pointing at thomasschweich/pawn:latest,
# SSH in, and run experiments. Code lives at /opt/pawn.
#
# IMPORTANT: Always attach a network volume. Set HF_TOKEN as a pod env var.

# ── Builder: compile Rust engine wheel ───────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /build
COPY engine/ engine/
COPY pyproject.toml uv.lock ./
COPY pawn/ pawn/
COPY scripts/ scripts/

RUN cd engine && uv run --no-project --with maturin maturin build --release

# ── Runtime ──────────────────────────────────────────────────────────
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404 AS runtime

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Project files + lockfile
WORKDIR /opt/pawn
COPY pyproject.toml uv.lock ./
COPY pawn/ pawn/
COPY scripts/ scripts/
COPY tests/ tests/
COPY deploy/ deploy/
COPY docs/ docs/
COPY cards/ cards/

# Install engine wheel, then sync Python deps from lockfile
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/
RUN uv venv --system-site-packages && \
    uv sync --extra cu128 --no-dev --frozen --no-install-workspace && \
    uv pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Bake git version info and set PATH for all contexts (docker exec, SSH, cron)
ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/opt/pawn/.venv/bin:/root/.local/bin:${PATH}"

# Inherits /start.sh entrypoint from RunPod base image (SSH + Jupyter)

# ── Dev container: non-root user + Claude Code + tools ──────────────
# Build:  docker build --target dev -t thomasschweich/pawn:dev .
# Extends the runtime image with a non-root user, CLI tools for
# interactive work, and a startup script that symlinks ephemeral dirs
# to /workspace. Use `su - pawn` after SSH to run Claude Code.
FROM runtime AS dev

# CLI tools for interactive/agent work
RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required for claude --dangerously-skip-permissions)
RUN useradd -m -s /bin/bash pawn && \
    chown -R pawn:pawn /opt/pawn

# Install Claude Code as pawn user
USER pawn
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/pawn/.local/bin:/opt/pawn/.venv/bin:${PATH}"

# Startup script: workspace setup, data prefetch, and CUDA MPS
# Run as root first: bash /home/pawn/setup-workspace.sh
COPY --chown=pawn:pawn <<'SETUP' /home/pawn/setup-workspace.sh
#!/usr/bin/env bash
set -euo pipefail

# ── Workspace symlinks (persistent storage) ────────────────────────
mkdir -p /workspace/logs /workspace/sweep_results /workspace/plots \
         /workspace/optuna-storage /opt/pawn/local
ln -sfn /workspace/sweep_results /opt/pawn/local/optuna_results
ln -sfn /workspace/logs /opt/pawn/logs
echo "Workspace symlinks ready"

# ── CUDA MPS (multi-process service for GPU sharing) ───────────────
# Allows concurrent trials to share a GPU efficiently. Needs root.
if command -v nvidia-cuda-mps-control &>/dev/null; then
    if [ "$(id -u)" = "0" ]; then
        nvidia-cuda-mps-control -d 2>/dev/null && echo "CUDA MPS daemon started" \
            || echo "CUDA MPS already running or unavailable"
    else
        echo "Skipping MPS (run setup-workspace.sh as root to enable)"
    fi
fi

# ── Polars file cache on /workspace ────────────────────────────────
# Polars caches whole parquet files (~140MB each, ~40GB for all 289
# shards). Cache on /workspace so it persists across pod restarts.
# /dev/shm would be faster but the full dataset won't fit in RAM.
POLARS_CACHE="/workspace/polars-cache"
mkdir -p "$POLARS_CACHE" 2>/dev/null || true
chmod 777 "$POLARS_CACHE" 2>/dev/null || true
grep -q POLARS_FILE_CACHE_DIR /home/pawn/.bashrc 2>/dev/null || \
    echo 'export POLARS_FILE_CACHE_DIR=/workspace/polars-cache' >> /home/pawn/.bashrc

echo "Setup complete"
SETUP
RUN chmod +x /home/pawn/setup-workspace.sh

USER root
# Inherits /start.sh entrypoint from RunPod base image
