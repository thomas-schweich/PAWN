# Training Guide

## Prerequisites

- **Rust** (stable) -- required to build the chess engine native extension
- **uv** -- Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **GPU** with ROCm (AMD) or CUDA (NVIDIA). CPU works only for `--variant toy`

## Installation

```bash
# Build the chess engine (one-time, or after engine/ changes)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python dependencies
uv sync --extra rocm    # AMD GPUs (ROCm)
uv sync --extra cu128   # NVIDIA GPUs (CUDA 12.8)
```

Verify the install:

```bash
uv run python -c "import chess_engine; print('engine OK')"
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Pretraining from Scratch

PAWN pretrains on random chess games generated on-the-fly by the Rust engine. No external datasets are needed.

```bash
uv run python scripts/train.py --variant base
```

### Model variants

| Variant | Params | d_model | Layers | Heads | d_ff |
|---------|--------|---------|--------|-------|------|
| `small` | ~9.5M  | 256     | 8      | 4     | 1024 |
| `base`  | ~36M   | 512     | 8      | 8     | 2048 |
| `large` | ~68M   | 640     | 10     | 8     | 2560 |
| `toy`   | tiny   | 64      | 2      | 4     | 256  |

### Default training configuration

- **Total steps**: 100,000 (the published v1.0.0 backbones were cotrained for 200,000 steps)
- **Batch size**: 256
- **Optimizer**: [AdamW](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017) (lr=3e-4, weight_decay=0.01)
- **LR schedule**: [cosine decay](https://arxiv.org/abs/1608.03983) (Loshchilov & Hutter, 2016) with 1,000-step warmup (10,000 for the published backbones)
- **Mixed precision**: fp16 [AMP](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2017) (auto-detected)
- **Checkpoints**: saved every 5,000 steps to `checkpoints/`
- **Eval**: every 500 steps on 512 held-out random games (1,000 / 2,048 for the published backbones)
- **Outcome conditioning**: off by default. The training config exposes a `prepend_outcome` flag that prefixes each example with one of 11 game-outcome tokens; the published v1.0.0 checkpoints use the no-prefix mode (pure moves, no outcome leakage), and so should any new backbones intended to be comparable to standard chess models

### Common overrides

```bash
# Resume from a checkpoint
uv run python scripts/train.py --variant base --resume checkpoints/step_00050000

# Custom batch size and step count
uv run python scripts/train.py --variant base --batch-size 128 --total-steps 200000

# Gradient accumulation (effective batch = batch_size * accumulation_steps)
uv run python scripts/train.py --variant base --accumulation-steps 4

# Enable W&B logging
uv run python scripts/train.py --variant base --wandb
```

## Adapter Training (Behavioral Cloning)

Adapter training freezes the pretrained PAWN backbone and trains lightweight adapter modules on Lichess games to predict human moves.

### Requirements

1. A pretrained PAWN checkpoint (from pretraining above)
2. A Lichess PGN file filtered to an Elo band

Download standard rated game archives from the [Lichess open database](https://database.lichess.org/) ([Lichess](https://lichess.org/)), filtered to your target Elo band. The scripts expect a single `.pgn` file.

### Available adapters

All adapter strategies dispatch through the unified `scripts/train.py` via `--run-type adapter --strategy STRATEGY`:

| Strategy          | `--strategy` value | Key flag             |
|-------------------|--------------------|----------------------|
| Bottleneck        | `bottleneck`       | `--bottleneck-dim 8` |
| FiLM              | `film`             | `--no-output-film`   |
| LoRA              | `lora`             | `--lora-rank 4`      |
| Sparse            | `sparse`           | `--density 0.01`     |
| Hybrid (LoRA+FiLM)| `hybrid`           | `--lora-rank 4`      |
| RoSA              | `rosa`             | `--rosa-mode rosa`   |
| From-scratch      | `specialized_clm`  | `--d-model 84`       |
| Unfreeze top-N    | `unfreeze`         | `--unfreeze-layers 6,7` |

### Example: bottleneck adapter

```bash
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \
    --bottleneck-dim 32 \
    --lr 1e-4 --local-checkpoints
```

### Adapter training defaults

- **Epochs**: 50 (early stopping is opt-in via `--patience N`; default is no early stopping)
- **Batch size**: 64
- **Optimizer**: AdamW (lr=3e-4)
- **LR schedule**: cosine with 5% warmup
- **Min ply**: 10 (games shorter than 10 plies are skipped)
- **Max games**: 12,000 train + 2,000 validation
- **Legal masking**: move legality enforced via the Rust engine at every position

### Resuming adapter training

```bash
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \
    --resume logs/bottleneck_20260315_120000/checkpoints/best --local-checkpoints
```

### Selective layer placement

Adapters can target specific layers or sublayer positions:

```bash
# Only FFN adapters on layers 4-7
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \
    --no-adapt-attn --adapter-layers 4,5,6,7 --local-checkpoints
```

## Cloud Deployment (Runpod)

The `deploy/` directory provides scripts for managing GPU pods.

### Pod lifecycle with `pod.sh`

```bash
bash deploy/pod.sh create myexp --gpu a5000        # Create a pod
bash deploy/pod.sh deploy myexp                     # Build + transfer + setup
bash deploy/pod.sh launch myexp scripts/train.py --variant base  # Run training
bash deploy/pod.sh ssh myexp                        # SSH in
bash deploy/pod.sh stop myexp                       # Stop (preserves volume)
```

GPU shortcuts: `a5000`, `a40`, `a6000`, `4090`, `5090`, `l40s`, `h100`.

### Manual deployment

If you prefer to deploy manually:

```bash
# 1. Build deploy package locally
bash deploy/build.sh --checkpoint thomas-schweich/pawn-base --data-dir data/

# 2. Transfer to pod
rsync -avz --progress deploy/pawn-deploy/ root@<pod-ip>:/workspace/pawn/

# 3. Run setup on the pod (installs Rust, uv, builds engine, syncs deps)
ssh root@<pod-ip> 'cd /workspace/pawn && bash deploy/setup.sh'
```

`setup.sh` handles: Rust installation, uv installation, building the chess engine, `uv sync --extra cu128`, and decompressing any zstd-compressed PGN data.

## GPU Auto-Detection

The `pawn.gpu` module auto-detects your GPU and configures:

- **torch.compile**: enabled on CUDA, uses inductor backend
- **AMP**: fp16 automatic mixed precision on CUDA
- **SDPA backend**: flash attention on NVIDIA; MATH backend on AMD (ROCm's flash attention backward has stride mismatches with torch.compile)

No manual flags are needed in most cases. Override with `--no-compile`, `--no-amp`, or `--sdpa-math` if needed.

## Monitoring

All training scripts log metrics to JSONL files in `logs/`. Each run creates a timestamped directory (e.g., `logs/bottleneck_20260315_120000/metrics.jsonl`).

Every log record includes:

- Training metrics (loss, accuracy, learning rate)
- System resource stats (RAM, GPU VRAM peak/current)
- Timestamps and elapsed time

The JSONL format is one JSON object per line, readable with standard tools:

```bash
# Watch live training progress
tail -f logs/*/metrics.jsonl | python -m json.tool
```
