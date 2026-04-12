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

# ── Caddy: single static binary for reverse-proxying the dashboard ──
FROM python:3.12-slim AS caddy
ARG CADDY_VERSION=2.11.2
ARG CADDY_SHA256=6d07b9bda92ac46e3b874e90dabc33192eca7e64c4b36ea661f4fd7dd27a5129
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "https://caddyserver.com/api/download?os=linux&arch=amd64&version=v${CADDY_VERSION}" \
       -o /usr/local/bin/caddy \
    && echo "${CADDY_SHA256}  /usr/local/bin/caddy" | sha256sum -c \
    && chmod +x /usr/local/bin/caddy \
    && rm -rf /var/lib/apt/lists/*

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
# Shared deps base — everything before the GPU-specific uv sync
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS deps-common

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /opt/pawn
COPY pyproject.toml uv.lock ./
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/

# External binaries last — they don't depend on our layers, so placing
# them here avoids invalidating the layers above on a caddy/uv release.
COPY --from=caddy /usr/local/bin/caddy /usr/local/bin/caddy
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/


# ═══════════════════════════════════════════════════════════════════════
# CUDA stages (--extra cu128)
# ═══════════════════════════════════════════════════════════════════════

# ── Deps (CUDA) ──────────────────────────────────────────────────────
FROM deps-common AS deps
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

RUN chmod +x deploy/entrypoint.sh
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/deploy/entrypoint.sh"]


# ═══════════════════════════════════════════════════════════════════════
# ROCm stages (--extra rocm)
# Same python:3.12-slim base — the ROCm torch wheel (~2.8 GB) bundles
# HIP, rocBLAS, MIOpen, etc. inside the wheel itself.
# ═══════════════════════════════════════════════════════════════════════

# ── Deps (ROCm) ──────────────────────────────────────────────────────
FROM deps-common AS deps-rocm
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

RUN chmod +x deploy/entrypoint.sh
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/deploy/entrypoint.sh"]


# ═══════════════════════════════════════════════════════════════════════
# Shared dev base — dev tools, non-root user, Claude Code, tmux
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS dev-common

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
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

# Create non-root user
RUN useradd -m -s /bin/bash pawn && \
    mkdir -p /opt/pawn && chown pawn:pawn /opt/pawn

# Install Claude Code and Rust toolchain (for building the chess engine).
# BuildKit auto-updates $HOME based on the current USER's passwd entry at
# each RUN, so USER pawn gives HOME=/home/pawn without an explicit ENV.
# Do NOT set `ENV HOME=...` here: an explicit value becomes sticky and
# would propagate into child stages (dev, dev-rocm), breaking their
# USER pawn / uv sync step with "cannot create /root/.rustup: permission
# denied".
USER pawn
WORKDIR /home/pawn
RUN curl -fsSL https://claude.ai/install.sh | bash && \
    { test -x /home/pawn/.local/bin/claude \
      || test -x /home/pawn/.claude/local/claude \
      || { echo "claude install failed — binary not found" >&2; \
           find /home/pawn -name claude 2>/dev/null; exit 1; }; }
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Expose claude system-wide so both root and pawn find it on PATH without
# relying on .bashrc aliases (which don't fire in non-interactive shells).
USER root
RUN set -e; \
    for p in /home/pawn/.local/bin/claude /home/pawn/.claude/local/claude; do \
        if [ -x "$p" ]; then ln -sf "$p" /usr/local/bin/claude; break; fi; \
    done; \
    test -x /usr/local/bin/claude

# Convenience script: drop into pawn user with claude in a tmux session
COPY --chmod=755 <<'CLAUDE_DEV' /usr/local/bin/claude-dev
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

# External binaries last (same rationale as deps-common)
COPY --from=caddy /usr/local/bin/caddy /usr/local/bin/caddy
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/


# ═══════════════════════════════════════════════════════════════════════
# Dev images — GPU deps + source code on top of dev-common
# Built independently from runtime/deps so every file in /opt/pawn
# enters via COPY --chown=pawn:pawn, avoiding a slow chown -R layer
# that would duplicate the multi-GB venv.
# ═══════════════════════════════════════════════════════════════════════

# ── Dev (CUDA) ───────────────────────────────────────────────────────
FROM dev-common AS dev
COPY --from=deps --chown=pawn:pawn /opt/pawn /opt/pawn

USER pawn
WORKDIR /opt/pawn
COPY --chown=pawn:pawn . .

# Build the engine so uv run doesn't trigger a rebuild on first use
RUN PATH="/home/pawn/.cargo/bin:${PATH}" \
    uv sync --extra cu128 --frozen

ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/home/pawn/.cargo/bin:/home/pawn/.local/bin:/opt/pawn/.venv/bin:${PATH}"

USER root
RUN chmod +x /opt/pawn/deploy/entrypoint-dev.sh /opt/pawn/deploy/entrypoint.sh
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/deploy/entrypoint-dev.sh"]

# ── Dev (ROCm) ───────────────────────────────────────────────────────
FROM dev-common AS dev-rocm
COPY --from=deps-rocm --chown=pawn:pawn /opt/pawn /opt/pawn

USER pawn
WORKDIR /opt/pawn
COPY --chown=pawn:pawn . .

# Build the engine so uv run doesn't trigger a rebuild on first use
RUN PATH="/home/pawn/.cargo/bin:${PATH}" \
    uv sync --extra rocm --frozen

ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn \
    PATH="/home/pawn/.cargo/bin:/home/pawn/.local/bin:/opt/pawn/.venv/bin:${PATH}"

USER root
RUN chmod +x /opt/pawn/deploy/entrypoint-dev.sh /opt/pawn/deploy/entrypoint.sh
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["/opt/pawn/deploy/entrypoint-dev.sh"]
