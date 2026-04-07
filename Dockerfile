# PAWN — multi-GPU Docker image for RunPod and bare-metal workloads
#
# Built automatically by CI on merge to main and pushed to Docker Hub.
# All targets use python:3.12-slim as the base. PyTorch cu128 wheels
# bundle their own CUDA runtime; PyTorch ROCm wheels bundle their own
# ROCm/HIP libraries. No nvidia/cuda or rocm base image needed — the
# only host requirements are the GPU kernel drivers.
#
# Targets:
#   runtime       — CUDA production image (default)
#   runtime-rocm  — ROCm production image
#   dev           — CUDA dev image (non-root, Claude Code, tmux)
#   dev-rocm      — ROCm dev image
#
# Usage:
#   docker build --target runtime      -t pawn:latest .
#   docker build --target runtime-rocm -t pawn:rocm   .
#   docker build --target dev          -t pawn:dev     .
#   docker build --target dev-rocm     -t pawn:dev-rocm .
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


# ═══════════════════════════════════════════════════════════════════════
# CUDA stages (--extra cu128)
# ═══════════════════════════════════════════════════════════════════════

# ── Deps (CUDA) ──────────────────────────────────────────────────────
FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

WORKDIR /opt/pawn
COPY pyproject.toml uv.lock ./
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/
RUN uv venv && \
    uv sync --extra cu128 --no-dev --frozen --no-install-workspace && \
    uv pip install /tmp/*.whl && rm -rf /tmp/*.whl ${UV_CACHE_DIR}

# ── Runtime (CUDA) ───────────────────────────────────────────────────
FROM deps AS runtime

COPY . .

ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/opt/pawn/.venv/bin:${PATH}"

COPY deploy/entrypoint.sh /opt/pawn/entrypoint.sh
RUN chmod +x /opt/pawn/entrypoint.sh
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/entrypoint.sh"]

# ── Dev (CUDA) ───────────────────────────────────────────────────────
# Built independently (not FROM runtime) so all /opt/pawn files are
# owned by pawn from the start, avoiding a slow chown -R layer.
FROM python:3.12-slim AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tini tmux ripgrep jq curl git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

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

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

# Create non-root user, then copy installed deps with correct ownership
RUN useradd -m -s /bin/bash pawn && \
    mkdir -p /opt/pawn && chown pawn:pawn /opt/pawn
COPY --from=deps --chown=pawn:pawn /opt/pawn /opt/pawn

# Source code + entrypoint
USER pawn
WORKDIR /opt/pawn
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

COPY deploy/entrypoint.sh /opt/pawn/entrypoint.sh
RUN chmod +x /opt/pawn/entrypoint.sh
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/entrypoint.sh"]


# ═══════════════════════════════════════════════════════════════════════
# ROCm stages (--extra rocm)
# Same python:3.12-slim base — the ROCm torch wheel (~2.8 GB) bundles
# HIP, rocBLAS, MIOpen, etc. inside the wheel itself.
# ═══════════════════════════════════════════════════════════════════════

# ── Deps (ROCm) ──────────────────────────────────────────────────────
FROM python:3.12-slim AS deps-rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

WORKDIR /opt/pawn
COPY pyproject.toml uv.lock ./
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/
RUN uv venv && \
    uv sync --extra rocm --no-dev --frozen --no-install-workspace && \
    uv pip install /tmp/*.whl && rm -rf /tmp/*.whl ${UV_CACHE_DIR}

# ── Runtime (ROCm) ───────────────────────────────────────────────────
FROM deps-rocm AS runtime-rocm

COPY . .

ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/opt/pawn/.venv/bin:${PATH}"

COPY deploy/entrypoint.sh /opt/pawn/entrypoint.sh
RUN chmod +x /opt/pawn/entrypoint.sh
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/entrypoint.sh"]

# ── Dev (ROCm) ───────────────────────────────────────────────────────
FROM python:3.12-slim AS dev-rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tini tmux ripgrep jq curl git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

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

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

# Create non-root user, then copy installed deps with correct ownership
RUN useradd -m -s /bin/bash pawn && \
    mkdir -p /opt/pawn && chown pawn:pawn /opt/pawn
COPY --from=deps-rocm --chown=pawn:pawn /opt/pawn /opt/pawn

# Source code + entrypoint
USER pawn
WORKDIR /opt/pawn
COPY --chown=pawn:pawn . .

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/home/pawn/.local/bin:/opt/pawn/.venv/bin:${PATH}"

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

COPY deploy/entrypoint.sh /opt/pawn/entrypoint.sh
RUN chmod +x /opt/pawn/entrypoint.sh
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/entrypoint.sh"]
