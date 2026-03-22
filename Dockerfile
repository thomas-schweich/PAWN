# PAWN Training Container for Runpod BYOC
#
# Extends the official Runpod PyTorch template — SSH and JupyterLab
# start automatically via the base image's /start.sh entrypoint.
#
# Build:
#   docker build --platform linux/amd64 -t pawn:v1.0 .
#
# Runpod BYOC:
#   Push to a registry, then set as the container image in a Pod template.
#   Configure HTTP port 8888 (Jupyter) and TCP port 22 (SSH).
#   Mount a network volume at /workspace for data, checkpoints, and logs.
#   Code lives at /opt/pawn (outside the volume mount).

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

# ── Runtime ──────────────────────────────────────────────────────────
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/pawn

# Direct deps only (torch + numpy already in base image)
RUN pip install --no-cache-dir psutil tqdm wandb

COPY --from=builder /workspace/pawn/engine/target/wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Project source
WORKDIR /opt/pawn
COPY pawn/ pawn/
COPY scripts/ scripts/
COPY tests/ tests/

# Persist PYTHONPATH for SSH sessions (Docker ENV doesn't propagate)
RUN echo 'export PYTHONPATH=/opt/pawn' >> /etc/environment && \
    echo 'export PYTHONPATH=/opt/pawn' >> /root/.bashrc
