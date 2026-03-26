# PAWN Container for Runpod
#
# Uses runpod/base (CUDA + SSH + Jupyter, no PyTorch) with uv for
# reproducible Python dependency management from the lockfile.
#
# Build targets:
#   interactive (default) — SSH + Jupyter, stays alive
#   runner               — runs a command then exits (pod auto-stops)
#   rosa-sweep           — runs RoSA ablation sweeps then exits
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg GIT_HASH=$(git rev-parse HEAD) \
#     --build-arg GIT_TAG=$(git tag --points-at HEAD) \
#     [--target runner] \
#     -t pawn:<tag> .
#
# IMPORTANT: Always attach a Runpod network volume. Checkpoints use
# atomic directory writes (tmp + rename) that require persistent disk.
# Set HF_TOKEN as a pod env var for HuggingFace checkpoint push.

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

# ── Runtime base (shared by all targets) ─────────────────────────────
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404 AS runtime-base

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

# Install engine wheel first
COPY --from=builder /build/engine/target/wheels/*.whl /tmp/

# Create venv with system packages (picks up pre-installed torch + CUDA)
# and install remaining deps from lockfile
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

# ── Runner — executes command then exits (pod auto-stops) ────────────
FROM runtime-base AS runner
COPY deploy/entrypoint-run.sh /entrypoint-run.sh
RUN chmod +x /entrypoint-run.sh
ENTRYPOINT ["/entrypoint-run.sh"]

# ── RoSA sweep — runs all three ablation sweeps then exits ───────────
FROM runtime-base AS rosa-sweep
COPY deploy/entrypoint-rosa-sweep.sh /entrypoint-rosa-sweep.sh
RUN chmod +x /entrypoint-rosa-sweep.sh
ENTRYPOINT ["/entrypoint-rosa-sweep.sh"]

# ── Interactive (default) — SSH + Jupyter, stays alive ───────────────
FROM runtime-base AS interactive
# Inherits /start.sh entrypoint from Runpod base image
