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

# ── Prefetch dataset to /dev/shm ───────────────────────────────────
# Downloads all parquet shards (~4GB for 1800-1900 Elo band once
# filtered). Polars file cache lives in /dev/shm for zero-latency
# reads across all concurrent trials.
POLARS_CACHE="/dev/shm/polars-cache"
mkdir -p "$POLARS_CACHE"
# Make writable by pawn user
chmod 777 "$POLARS_CACHE"

# Set polars cache env for pawn user
grep -q POLARS_FILE_CACHE_DIR /home/pawn/.bashrc 2>/dev/null || \
    echo 'export POLARS_FILE_CACHE_DIR=/dev/shm/polars-cache' >> /home/pawn/.bashrc

# Warm the cache by scanning each shard individually (avoids glob rate limits)
echo "Warming Polars cache in /dev/shm (first run downloads from HF)..."
POLARS_FILE_CACHE_DIR="$POLARS_CACHE" uv run python -c "
import polars as pl, os, sys
sys.path.insert(0, '/opt/pawn')
from pawn.shard_loader import _list_shards, _hf_storage_options

repo = 'thomas-schweich/pawn-lichess-full'
shards = _list_shards(repo)
opts = _hf_storage_options()
total = 0
for i, shard in enumerate(shards):
    url = f'hf://datasets/{repo}/{shard}'
    try:
        n = pl.scan_parquet(url, storage_options=opts or None).select(pl.len()).collect().item()
        total += n
        print(f'  [{i+1}/{len(shards)}] {shard}: {n:,} rows', flush=True)
    except Exception as e:
        print(f'  [{i+1}/{len(shards)}] {shard}: FAILED ({e})', flush=True)
print(f'Cache warmed: {total:,} total rows across {len(shards)} shards')
"

echo "Setup complete"
SETUP
RUN chmod +x /home/pawn/setup-workspace.sh

USER root
# Inherits /start.sh entrypoint from RunPod base image
