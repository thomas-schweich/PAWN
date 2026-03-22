# Adapter Methods

PAWN is designed as a testbed for parameter-efficient fine-tuning. The frozen 30M-parameter backbone provides learned chess representations from pretraining on random games; adapters specialize those representations for downstream tasks like predicting human moves at a given Elo level.

All adapter implementations live in `pawn/adapters/`. Each wraps a frozen `PAWNCLM` backbone and exposes a uniform interface: `forward_hidden()`, `project_head()`, `forward()`, and `forward_generate()` (with KV-cache).

## Bottleneck (Houlsby-style)

**Module:** `pawn.adapters.bottleneck.BottleneckCLM`

Inserts small residual MLP bottlenecks after the attention sublayer and/or FFN sublayer within each transformer block:

```
x = x + up(gelu(down(x)))
```

The up-projection is zero-initialized, so the model starts identical to the frozen backbone. `bottleneck_dim` controls the parameter budget.

**Key parameters:**
- `bottleneck_dim` -- hidden dimension of the bottleneck (default: 8)
- `adapt_attn` / `adapt_ffn` -- which sublayers to adapt (default: both)
- `layers` -- which transformer layers to adapt (default: all)
- `attn_layers` / `ffn_layers` -- per-sublayer layer selection overrides

**Param count:** `2 * n_positions * n_layers * 2 * d_model * bottleneck_dim` (e.g. 131K at dim=8, both positions, all 8 layers).

Best performer at low parameter budgets. The GELU nonlinearity and full-rank projections provide the most expressive per-parameter adaptation.

## LoRA (Low-Rank Adaptation)

**Module:** `pawn.adapters.lora.LoRACLM`

Injects rank-r adapters into attention projections (and optionally FFN) in every transformer layer (Hu et al., 2021):

```
output = frozen_linear(x) + (x @ A^T) @ B^T * (alpha / rank)
```

B is zero-initialized for identity start. A is Kaiming-initialized. LoRA modifies the linear projections in-place (replacing `nn.Linear` with `LoRALinear`), so the backbone's own forward pass automatically includes the LoRA contribution.

**Key parameters:**
- `rank` -- rank of the low-rank matrices (default: 4)
- `alpha` -- scaling factor (default: same as rank)
- `attn_targets` -- which projections: `"qkvo"`, `"qv"`, or `"qkv"` (default: `"qkvo"`)
- `adapt_ffn` -- also adapt FFN projections (w_gate, w_up, w_down)
- `layers` -- which transformer layers to adapt (default: all)

**Param count:** `n_layers * n_targets * 2 * d_model * rank` (e.g. 131K at rank=4, qkvo, all 8 layers).

## FiLM (Feature-wise Linear Modulation)

**Module:** `pawn.adapters.film.FiLMCLM`

Applies learned per-channel affine transforms after each transformer block and optionally on the output logits:

```
h_adapted = gamma * h + beta       (hidden layers, dim = d_model)
logits_adapted = gamma * logits + beta  (output, dim = vocab_size)
```

Identity-initialized: gamma=1, beta=0.

**Key parameters:**
- `use_output_film` -- apply FiLM to output logits as well (default: True)

**Param count:** `n_layers * 2 * d_model + 2 * vocab_size` = ~17K. The lightest adapter by far -- only diagonal (per-channel) modulation with no cross-channel mixing.

## Sparse

**Module:** `pawn.adapters.sparse.SparseCLM`

Perturbs a random subset of frozen weight elements. A fixed binary mask selects which weight positions get a trainable additive delta (zero-initialized):

```
W_effective = W_frozen + delta * mask
```

The mask is generated once at initialization from a fixed seed and never changes. Only the masked delta values are learned, but the full delta tensor is stored (unmasked entries remain zero and contribute no gradient due to the mask).

**Key parameters:**
- `density` -- fraction of weight elements to unmask (default: 0.01)
- `attn_targets` -- which attention projections (default: qkvo)
- `adapt_ffn` -- also adapt FFN projections
- `layers` -- which layers (default: all)
- `seed` -- RNG seed for reproducible mask generation (default: 42)

**Param count:** `density * total_weight_elements` in targeted projections. E.g. density=0.031 on qkvo gives ~65K active params; density=0.081 on qkvo+FFN gives ~2.7M.

Excels at high parameter budgets where many small perturbations to existing weights can outperform structured adapters.

## Hybrid (LoRA + FiLM)

**Module:** `pawn.adapters.hybrid.HybridCLM`

Combines LoRA and FiLM on a single frozen backbone. LoRA modifies attention projections within layers (cross-channel mixing); FiLM rescales the residual stream between layers (diagonal modulation). Both are identity-initialized.

**Key parameters:** Union of LoRA and FiLM parameters, plus:
- `lora_layers` / `film_layers` -- independent layer selection for each method
- `use_output_film` -- FiLM on logits (default: False)

Supports separate learning rates for LoRA and FiLM parameters via the training script (`--lora-lr`, `--film-lr`).

---

## Design Patterns

### Identity initialization

All adapters initialize to the identity function. Bottleneck and LoRA zero-initialize their output projections (up-project B-matrix). FiLM initializes gamma=1, beta=0. Sparse initializes all deltas to zero. This guarantees the adapted model produces identical outputs to the frozen backbone before any training.

### Sparse logit projection

All adapter wrappers expose `forward_hidden()` (returns `(B, T, d_model)`) and `project_head()` (applies `lm_head`) as separate methods. Training scripts use this split to avoid materializing the full `(B, T, V)` logit tensor: only positions included in the loss mask are projected through `lm_head`. This reduces peak memory significantly since `V=4278` and most positions in a batch are padding.

### Legal masking via Rust engine

Evaluation uses `LegalMaskBuilder` to replay games through the Rust chess engine and produce per-position legal move masks. These are scattered into a pre-allocated GPU buffer as sparse indices, avoiding dense `(B, T, V)` boolean masks.

### DataLoader worker safety

The chess engine uses rayon for internal parallelism. Python's default `fork` multiprocessing can deadlock if forked after rayon initializes its thread pool. All DataLoader usage must specify `multiprocessing_context='spawn'`. The `LegalMaskCollate` callable moves Rust replay work into spawned workers, and `LichessDataset.share_memory()` avoids per-worker copies of the dataset.

### Parameter management

Each adapter exposes three helpers:
- `adapter_state_dict()` / `lora_state_dict()` / etc. -- extract only trainable weights for saving
- `load_adapter_state_dict()` -- restore adapter weights into a freshly wrapped backbone
- `adapter_weight_report()` -- per-layer weight norms for monitoring training dynamics

---

## Results Summary

All results on the PAWN-Base backbone (~36M params, 8 layers, d_model=512), trained via behavioral cloning on Lichess games with legal-move-masked cross-entropy loss.

### Method comparison (~65K params, 1000-1100 Elo, 100K games)

| Method | Params | Val top-1 |
|--------|--------|-----------|
| Bottleneck (dim=8) | 65K | 39.3% |
| Sparse (density=0.031) | 65K | 35.2% |
| Hybrid (LoRA+FiLM) | ~65K | 34.1% |
| FiLM | ~33K | 30.3% |

Bottleneck dominates at matched parameter budgets on low-Elo data.

### Backbone leverage

| Model | Params | Val top-1 |
|-------|--------|-----------|
| Standalone tiny transformer | 529K | 30.9% |
| Bottleneck on frozen PAWN | 524K | 42.2% |

The frozen backbone provides ~11 percentage points of "free" accuracy (36% relative improvement). Adapters specialize existing representations rather than learning from scratch.

### Capacity scaling (1800-1900 Elo, 1M games)

| Method | Params | Val top-1 |
|--------|--------|-----------|
| Sparse (density=0.081, qkvo+FFN) | 2.7M | 44.7% |
| Bottleneck (dim=64, all layers) | 1.0M | 43.5% |
| Bottleneck (dim=32, all layers) | 524K | 41.7% |
| Sparse (density=0.015, qkvo+FFN) | 503K | 40.2% |

Below ~1M params, bottleneck is more parameter-efficient. Above ~1M, sparse catches up and overtakes -- likely because direct weight perturbation can modify more individual weight entries than a structured bottleneck at the same total parameter count.

### Data scaling (1000-1100 Elo, bottleneck dim=32)

10x more data (100K to 1M games) yields +2.9pp (39.3% to 42.2%) and reduces train/val gap from ~1pp to ~0.3pp.

---

## Quick Start

All commands assume you are in the `pawn/` directory. `--checkpoint` points to a pretrained PAWN backbone and `--pgn` to a Lichess PGN file.

```bash
# Bottleneck (recommended default)
uv run python scripts/train_bottleneck.py \
    --checkpoint checkpoints/pawn.pt --pgn data/lichess_1800.pgn \
    --bottleneck-dim 32 --max-games 100000 --lr 3e-4

# LoRA
uv run python scripts/train_lora.py \
    --checkpoint checkpoints/pawn.pt --pgn data/lichess_1800.pgn \
    --lora-rank 8 --lora-targets qkvo --lr 3e-4

# FiLM
uv run python scripts/train_film.py \
    --checkpoint checkpoints/pawn.pt --pgn data/lichess_1800.pgn \
    --lr 1e-3

# Sparse
uv run python scripts/train_sparse.py \
    --checkpoint checkpoints/pawn.pt --pgn data/lichess_1800.pgn \
    --density 0.015 --sparse-ffn --lr 3e-4

# Hybrid (LoRA + FiLM)
uv run python scripts/train_hybrid.py \
    --checkpoint checkpoints/pawn.pt --pgn data/lichess_1800.pgn \
    --lora-rank 4 --lora-lr 3e-4 --film-lr 1e-3
```

All scripts share common flags: `--epochs`, `--batch-size`, `--patience` (early stopping), `--no-compile` (required on ROCm), `--device`, `--num-workers`, `--resume` (checkpoint path for resuming).

Run `--help` on any script for the full argument list.
