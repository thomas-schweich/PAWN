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
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

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
COPY data/ data/

# Install engine wheel, then sync Python deps from lockfile
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/
RUN uv venv --system-site-packages && \
    uv sync --extra cu128 --no-dev --frozen --no-install-workspace && \
    uv pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Bake git version info
ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG} \
    PYTHONPATH=/opt/pawn

# Persist env vars for SSH sessions
RUN echo "export PYTHONPATH=/opt/pawn" >> /etc/environment && \
    echo "export PAWN_GIT_HASH=${GIT_HASH}" >> /etc/environment && \
    echo "export PAWN_GIT_TAG=${GIT_TAG}" >> /etc/environment && \
    echo 'export PATH="/opt/pawn/.venv/bin:$PATH"' >> /etc/environment && \
    cat /etc/environment >> /root/.bashrc

# Inherits /start.sh entrypoint from RunPod base image (SSH + Jupyter)
