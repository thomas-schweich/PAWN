# Adapter Methods

PAWN is designed as a testbed for parameter-efficient fine-tuning. The frozen ~36M-parameter backbone provides learned chess representations from pretraining on random games; adapters specialize those representations for downstream tasks like predicting human moves at a given Elo level.

All adapter implementations live in `pawn/adapters/`. Each wraps a frozen `PAWNCLM` backbone and exposes a uniform interface: `forward_hidden()`, `project_head()`, `forward()`, and `forward_generate()` (with KV-cache).

## Bottleneck ([Houlsby et al., 2019](https://arxiv.org/abs/1902.00751))

**Module:** `pawn.adapters.bottleneck.BottleneckCLM`

Inserts small residual MLP bottlenecks after the attention sublayer and/or FFN sublayer within each transformer block, following "Parameter-Efficient Transfer Learning for NLP" (ICML 2019):

```
x = x + up(gelu(down(x)))
```

The up-projection is zero-initialized, so the model starts identical to the frozen backbone. `bottleneck_dim` controls the parameter budget.

**Key parameters:**
- `bottleneck_dim` -- hidden dimension of the bottleneck (default: 8)
- `adapt_attn` / `adapt_ffn` -- which sublayers to adapt (default: both)
- `layers` -- which transformer layers to adapt (default: all)
- `attn_layers` / `ffn_layers` -- per-sublayer layer selection overrides

**Param count:** `n_positions * n_layers * 2 * d_model * bottleneck_dim` where n_positions is 2 (attn + ffn) by default (e.g. `2 * 8 * 2 * 512 * 8 = 131K` at dim=8, both positions, all 8 layers).

Best performer at low parameter budgets. The GELU nonlinearity and full-rank projections provide the most expressive per-parameter adaptation.

## LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685))

**Module:** `pawn.adapters.lora.LoRACLM`

Injects rank-r adapters into attention projections (and optionally FFN) in every transformer layer, following "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022):

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

## FiLM ([Perez et al., 2017](https://arxiv.org/abs/1709.07871))

**Module:** `pawn.adapters.film.FiLMCLM`

Applies learned per-channel affine transforms after each transformer block and optionally on the output logits, following "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018):

```
h_adapted = gamma * h + beta       (hidden layers, dim = d_model)
logits_adapted = gamma * logits + beta  (output, dim = vocab_size)
```

Identity-initialized: gamma=1, beta=0.

**Key parameters:**
- `use_output_film` -- apply FiLM to output logits as well (default: False)

**Param count:** `n_layers * 2 * d_model + 2 * vocab_size` = ~17K. The lightest adapter by far -- only diagonal (per-channel) modulation with no cross-channel mixing.

## RoSA ([Nikdan et al., 2024](https://arxiv.org/abs/2401.04679))

**Module:** `pawn.adapters.rosa.RoSACLM`

Implements Robust Sparse Adaptation from "RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation". Combines a low-rank adapter (LoRA) with a gradient-informed sparse adapter on each frozen projection matrix:

```
output = frozen(x) + (x @ A^T) @ B^T * scaling + F.linear(x, delta * mask)
```

Unlike the random masks used by the Sparse adapter, RoSA selects its sparse mask positions based on gradient information from a LoRA warm-up phase. Training proceeds in three phases:

1. **LoRA warm-up** -- train LoRA-only for `warmup_steps` steps to build gradient signal
2. **Mask generation** -- accumulate squared gradient magnitudes over a small data subset, select top-k positions per weight matrix at the target density (Algorithm 1 from the paper)
3. **Joint training** -- train both LoRA and sparse adapters simultaneously

The `rosa` strategy (via `scripts/train.py --run-type adapter --strategy rosa`) also supports two **retrospective ablation** modes via `--rosa-mode`:

- **`retro-sparse`** -- use LoRA purely as a probe for mask selection, then discard it and train sparse-only on a fresh backbone with the found masks
- **`retro-bottleneck`** -- same as retro-sparse, but adds bottleneck adapters (`RetroBottleneckCLM`) after each sublayer for nonlinearity that sparse-only cannot express

In retrospective modes, warm-up LoRA weights are saved as a checkpoint for analysis before the backbone is reloaded.

**Key parameters:**
- `rank` -- LoRA rank during warm-up and joint training (default: 4)
- `density` -- target sparse mask density (default: 0.01)
- `attn_targets` -- which attention projections: `"qkvo"`, `"qv"`, or `"qkv"` (default: `"qkvo"`)
- `adapt_ffn` -- also adapt FFN projections
- `warmup_steps` -- LoRA-only steps before mask generation (default: 128)
- `mask_samples` -- batches for gradient accumulation (default: 32)
- `grad_alpha` -- gradient exponent: 1=mean magnitude, 2=Fisher diagonal (default: 2)
- `bottleneck_dim` -- bottleneck dimension for retro-bottleneck mode (default: 8)

**Param count:** Depends on mode and configuration. In `rosa` mode: `n_lora_params + density * total_weight_elements`. In `retro-sparse`: `density * total_weight_elements`. In `retro-bottleneck`: sparse params + `2 * n_positions * n_layers * 2 * d_model * bottleneck_dim`.

## Sparse

**Module:** `pawn.adapters.sparse.SparseCLM`

Perturbs a random subset of frozen weight elements (related to sparse fine-tuning ideas from the [lottery ticket hypothesis](https://arxiv.org/abs/1803.03635); Frankle & Carbin, 2018). A fixed binary mask selects which weight positions get a trainable additive delta (zero-initialized):

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

## Hybrid ([LoRA](https://arxiv.org/abs/2106.09685) + [FiLM](https://arxiv.org/abs/1709.07871))

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

All adapter wrappers expose `forward_hidden()` (returns `(B, T, d_model)`) and `project_head()` (applies `lm_head`) as separate methods. Training scripts use this split to avoid materializing the full `(B, T, V)` logit tensor: only positions included in the loss mask are projected through `lm_head`. This reduces peak memory significantly since `V=1980` and most positions in a batch are padding.

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

All results on the v1.0.0 `pawn-base` backbone (8 layers, d_model=512, 1,980-token vocab, 512-token context, no outcome prefix) trained via behavioral cloning on Lichess games with legal-move-masked cross-entropy loss. Every run streams one pass through 2M games filtered to Elo 1800-1900 with `min_ply=10`, bs=128 (bs=96 for dim=512 to fit activation memory on a 21 GB card), `lr=3e-4`, bf16 AMP, flash SDPA + `torch.compile`. FiLM and Hybrid remain in the codebase but were not part of this sweep.

### Phase 2 v2 sweep (2M games, 1800-1900 Elo, one pass)

| Method | Params | Val top-1 | Val top-5 | Val loss | Wall |
|--------|-------:|----------:|----------:|---------:|------|
| Sparse (density=0.015, qkvo, random mask) | 126K | 29.18% | 61.38% | 2.495 | 1h27m |
| RoSA retro-sparse (density=0.01, grad mask) | 84K | **30.45%** | 63.15% | 2.438 | 1h28m |
| LoRA (rank=16, qkvo) | 524K | 35.62% | 70.33% | 2.206 | 1h36m |
| Bottleneck (dim=32, attn+ffn, all layers) | 524K | **39.82%** | 76.07% | 2.013 | 1h30m |
| Bottleneck (dim=64) | 1.05M | **41.56%** | 78.07% | 1.942 | 1h28m |
| Bottleneck (dim=512, bs=96) | 8.4M | **46.14%** | 82.88% | 1.751 | 1h44m |

Pareto-optimal points are in bold.

### Takeaways

- **Bottleneck dominates at matched parameter budgets.** At 524K, bottleneck dim=32 (39.82%) beats LoRA rank=16 qkvo (35.62%) by 4.2pp — the gap is structural and does not close under more data (consistent with the earlier 12K-game phase-1 sweep, where the same ordering held).
- **Gradient-informed masks beat random masks.** RoSA retro-sparse (84K) beats random-mask sparse (126K) by 1.27pp at a third fewer trainable parameters. The 3-phase training overhead (LoRA warmup → mask generation → sparse-only training) pays for itself cleanly.
- **Bottleneck hasn't saturated.** 16× params (dim=32 → dim=512) gave +6.32pp, and the last 8× step (dim=64 → dim=512) contributed +4.58pp on its own. The sweep didn't reach the point of diminishing returns.
- **The legacy v0.x adapter sweep (which used `prepend_outcome=True` plus a coarser 4,278-token vocab and 256-token context) is preserved in git history at tag `pre-vocab-transition`.** Direct comparison with those numbers is apples-to-oranges because the backbone architecture, vocabulary, and training schedule all changed; use the table above as the canonical reference for v1.0.0.

### Backbone leverage (legacy reference)

For historical context, the legacy v0.x sweep measured a standalone tiny transformer (no backbone, trained from scratch) at 524K params and 30.9% top-1 vs. a 524K bottleneck on the frozen legacy backbone at 42.2%. The v1.0.0 bottleneck dim=32 (524K trainable) lands at 39.82% on 2M games; we haven't yet re-run the from-scratch baseline on the v1.0.0 data pipeline (1,980-vocab, 512-ctx, no outcome prefix) to directly quantify the frozen-backbone lift under current conditions, so the ~9pp implied by subtracting the two eras is indicative rather than a like-for-like measurement. A matched from-scratch 524K standalone on the current data is on the to-do list.

---

## Quick Start

All commands assume you are in the repo root. `--checkpoint` points to a pretrained PAWN backbone and `--pgn` to a Lichess PGN file. Every strategy dispatches through `scripts/train.py --run-type adapter --strategy <name>`.

```bash
# Bottleneck (recommended default)
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --bottleneck-dim 32 --max-games 100000 --lr 3e-4 --local-checkpoints

# LoRA
uv run python scripts/train.py --run-type adapter --strategy lora \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --lora-rank 8 --lora-targets qkvo --lr 3e-4 --local-checkpoints

# FiLM
uv run python scripts/train.py --run-type adapter --strategy film \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --lr 1e-3 --local-checkpoints

# Sparse
uv run python scripts/train.py --run-type adapter --strategy sparse \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --density 0.015 --sparse-ffn --lr 3e-4 --local-checkpoints

# Hybrid (LoRA + FiLM)
uv run python scripts/train.py --run-type adapter --strategy hybrid \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --lora-rank 4 --lr 3e-4 --local-checkpoints

# RoSA (standard: joint LoRA + gradient-informed sparse)
uv run python scripts/train.py --run-type adapter --strategy rosa \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --rosa-mode rosa --density 0.01 --lora-rank 4 --rosa-warmup-steps 128 \
    --lr 3e-4 --local-checkpoints

# RoSA (retrospective sparse + bottleneck)
uv run python scripts/train.py --run-type adapter --strategy rosa \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --rosa-mode retro-bottleneck --density 0.01 --bottleneck-dim 8 --lr 3e-4 \
    --local-checkpoints
```

Common flags across all strategies: `--epochs`, `--batch-size`, `--patience` (early stopping), `--no-compile` (required on ROCm), `--device`, `--num-workers`, `--resume` (checkpoint path for resuming).
