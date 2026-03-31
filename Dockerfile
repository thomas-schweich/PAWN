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

RUN pip install --no-cache-dir maturin

# Cache Cargo dependency downloads — only re-fetched when Cargo.toml/lock change
WORKDIR /build/engine
COPY engine/Cargo.toml engine/Cargo.lock ./

# Stub out the expected source layout so Cargo can resolve the crate,
# then fetch dependencies into a cached layer. The real source files
# are copied in the next step — only Cargo.toml/lock changes trigger
# a re-download.
RUN mkdir -p src python/chess_engine && \
    touch src/lib.rs python/chess_engine/__init__.py && \
    cargo fetch

# Now copy actual source and build the wheel
COPY engine/pyproject.toml ./
COPY engine/src/ src/
COPY engine/python/ python/
RUN maturin build --release

# ── Runtime ──────────────────────────────────────────────────────────
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404 AS runtime

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# Copy `uv` from their official distroless image.
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

# Lockfile first — Python deps only re-installed when dependencies change
WORKDIR /opt/pawn
COPY pyproject.toml uv.lock ./
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/
RUN uv venv --system-site-packages && \
    uv sync --extra cu128 --no-dev --frozen --no-install-workspace && \
    uv pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Source code (changes here don't invalidate the dependency layer)
COPY . .

# Bake git version info and set PATH for all contexts (docker exec, SSH, cron)
ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/opt/pawn/.venv/bin:${PATH}"

# Entrypoint wrapper: workspace setup + CUDA MPS, then hand off to RunPod
COPY <<'ENTRYPOINT_WRAPPER' /opt/pawn/entrypoint.sh
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
ENTRYPOINT_WRAPPER
RUN chmod +x /opt/pawn/entrypoint.sh
ENTRYPOINT ["/opt/pawn/entrypoint.sh"]

# ── Dev container: non-root user + Claude Code + tools ──────────────
# Build:  docker build --target dev -t thomasschweich/pawn:dev .
# Built independently (not FROM runtime) so all /opt/pawn files are
# owned by pawn from the start, avoiding a slow chown -R layer.
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404 AS dev

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# CLI tools for interactive/agent work
RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

# Developer-friendly tmux defaults
RUN cat <<'TMUX' > /etc/tmux.conf
set -g mouse on
set -g history-limit 50000
set -g default-terminal "tmux-256color"
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
set -g set-clipboard on
TMUX

# Create non-root user before any COPY --chown
RUN useradd -m -s /bin/bash pawn

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

# Install deps as pawn user — everything under /opt/pawn is owned correctly
WORKDIR /opt/pawn
COPY --chown=pawn:pawn pyproject.toml uv.lock ./
COPY --from=builder --chown=pawn:pawn /build/engine/target/wheels/*.whl /tmp/
USER pawn
RUN uv venv --system-site-packages && \
    uv sync --extra cu128 --no-dev --frozen --no-install-workspace && \
    uv pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Source code
COPY --chown=pawn:pawn . .

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/home/pawn/.local/bin:/opt/pawn/.venv/bin:${PATH}"

# Convenience script: drop into pawn user with claude in a tmux session.
# Starts bash first, then sends the claude command as keystrokes so that
# CTRL+Z safely suspends to a shell prompt (fg to resume) instead of
# leaving the pane in an irrecoverable state.
USER root
COPY <<'CLAUDE_DEV' /usr/local/bin/claude-dev
#!/usr/bin/env bash
set -euo pipefail
SESSION="claude"
exec su - pawn -c "
    if tmux has-session -t $SESSION 2>/dev/null; then
        exec tmux attach -t $SESSION
    fi
    tmux new-session -d -s $SESSION -c /opt/pawn
    tmux send-keys -t $SESSION 'cd /opt/pawn && claude --dangerously-skip-permissions' Enter
    exec tmux attach -t $SESSION
"
CLAUDE_DEV
RUN chmod +x /usr/local/bin/claude-dev

# Entrypoint wrapper: workspace setup + CUDA MPS, then hand off to RunPod
COPY <<'ENTRYPOINT_WRAPPER' /opt/pawn/entrypoint.sh
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
ENTRYPOINT_WRAPPER
RUN chmod +x /opt/pawn/entrypoint.sh
ENTRYPOINT ["/opt/pawn/entrypoint.sh"]
