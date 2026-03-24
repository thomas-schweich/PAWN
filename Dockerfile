# PAWN Training Container for Runpod BYOC
#
# Extends the official Runpod PyTorch template — SSH and JupyterLab
# start automatically via the base image's /start.sh entrypoint.
#
# Build targets:
#   interactive (default) — SSH + Jupyter, stays alive
#   runner               — runs a command then exits (pod auto-stops)
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg GIT_HASH=$(git rev-parse HEAD) \
#     --build-arg GIT_TAG=$(git tag --points-at HEAD) \
#     [--target runner] \
#     -t pawn:<tag> .
#
# Run (interactive):
#   docker run --gpus all pawn:<tag>
#
# Run (auto-stop):
#   docker run --gpus all -e PAWN_MODEL=thomas-schweich/pawn-base \
#     pawn:<tag>-runner python scripts/train.py --variant base

# ── Builder ──────────────────────────────────────────────────────────
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 AS builder

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/pawn
COPY pyproject.toml uv.lock ./
COPY engine/ engine/
COPY pawn/ pawn/
COPY scripts/ scripts/
COPY tests/ tests/

# Build engine wheel for runtime install
RUN cd engine && \
    uv run --no-project --with maturin maturin build --release && \
    cd ..

# ── Runtime base (shared by all targets) ─────────────────────────────
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 AS runtime-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/pawn

# Direct deps only (torch + numpy already in base image)
RUN pip install --no-cache-dir psutil safetensors tqdm wandb huggingface-hub

COPY --from=builder /workspace/pawn/engine/target/wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Project source
WORKDIR /opt/pawn
COPY pawn/ pawn/
COPY scripts/ scripts/
COPY tests/ tests/

# Bake git version info for trainer config.json
ARG GIT_HASH=""
ARG GIT_TAG=""
ENV PAWN_GIT_HASH=${GIT_HASH} \
    PAWN_GIT_TAG=${GIT_TAG}

# Persist env vars for SSH sessions (Docker ENV doesn't propagate)
RUN echo "export PYTHONPATH=/opt/pawn" >> /etc/environment && \
    echo "export PAWN_GIT_HASH=${GIT_HASH}" >> /etc/environment && \
    echo "export PAWN_GIT_TAG=${GIT_TAG}" >> /etc/environment && \
    cat /etc/environment >> /root/.bashrc

# ── Interactive (default) — SSH + Jupyter, stays alive ───────────────
FROM runtime-base AS interactive
# Inherits /start.sh entrypoint from Runpod base image

# ── Runner — executes command then exits (pod auto-stops) ────────────
FROM runtime-base AS runner
COPY deploy/entrypoint-run.sh /entrypoint-run.sh
RUN chmod +x /entrypoint-run.sh
ENTRYPOINT ["/entrypoint-run.sh"]
