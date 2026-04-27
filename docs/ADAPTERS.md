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
- `n_hidden` -- extra `Linear(bn,bn)+GELU` stages between `down` and `up` (default: 0 reproduces the standard two-layer Houlsby block)
- `adapt_attn` / `adapt_ffn` -- which sublayers to adapt (default: both)
- `layers` -- which transformer layers to adapt (default: all)
- `attn_layers` / `ffn_layers` -- per-sublayer layer selection overrides

**Param count:** `n_positions * n_layers * (2 * d_model * bottleneck_dim + n_hidden * bottleneck_dim^2)` where n_positions is 2 (attn + ffn) by default (e.g. `2 * 8 * (2 * 512 * 8 + 0) = 131K` at dim=8, n_hidden=0, both positions, all 8 layers; n_hidden=2 adds `2 * 8 * 2 * 8^2 = 2,048` → 133K).

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

Training and evaluation use `LegalMaskBuilder` to replay games through the Rust chess engine and produce per-position legal move masks. These are scattered into a pre-allocated GPU buffer as sparse indices, avoiding dense `(B, T, V)` boolean masks.

By default the loss is computed over legal moves only (illegal logits masked to `-inf` before cross-entropy). Two CLI flags tune this:

- `--disable-legal-mask` — train without the hard mask, so the CE runs over the full 1,980-token vocabulary the same way pretraining does. Useful when you want to measure whether the adapter preserves the backbone's rule-tracking or is leaning on the mask as a crutch.
- `--illegal-penalty λ` — when masking is disabled, add `λ · E[P_illegal]` (the expected softmax mass assigned to illegal moves, averaged over scored positions) to the loss. Illegal moves are then strictly worse than legal-but-wrong moves without being forbidden outright.

Two new per-eval metrics fall out: `illegal_pred_rate` (fraction of argmax picks that are illegal) and `illegal_prob_mass` (softmax mass on illegal tokens). Both are analytically zero under the hard mask and are skipped entirely in that path to avoid a numerically unstable softmax over all-`-inf` rows.

### DataLoader worker safety

The chess engine uses rayon for internal parallelism. Python's default `fork` multiprocessing can deadlock if forked after rayon initializes its thread pool. All DataLoader usage must specify `multiprocessing_context='spawn'`. The `LegalMaskCollate` callable moves Rust replay work into spawned workers, and `LichessDataset.share_memory()` avoids per-worker copies of the dataset.

### Parameter management

Each adapter exposes three helpers:
- `adapter_state_dict()` / `lora_state_dict()` / etc. -- extract only trainable weights for saving
- `load_adapter_state_dict()` -- restore adapter weights into a freshly wrapped backbone
- `adapter_weight_report()` -- per-layer weight norms for monitoring training dynamics

---

## Results Summary

All results below use Lichess games filtered to Elo 1800-1900 with both players in band and `min_ply=10` (≈16.2 M games per epoch). Default settings: bs=128, bf16 AMP, flash SDPA + `torch.compile`, infinite LR schedule (`warmup_frac=0.05`, `cooldown_frac=0.4`, `stable_lr_ratio=0.75`, `decay_frac=0.1`, `wsd_decay_shape=cosine`), peak LR 4e-4 for bottleneck families and 3e-4 for retro-bottleneck and specialized_clm. "1 ep" rows train on one full pass; "2 ep" rows do an honest two passes with the LR decay completing in the tail of epoch 2.

### Pareto frontier across param budgets

| Method | Backbone | Params | Epochs | Val top-1 | Notes |
|--------|----------|-------:|:------:|----------:|-------|
| Sparse density=0.015 qkvo random mask | pawn-base | 126K | 1 | 29.18% | random binary mask baseline |
| RoSA retro-sparse density=0.01 qkvo | pawn-base | 84K | 1 | **30.45%** | gradient-informed sparse mask, smaller and better |
| Retro-sparse density=0.05 | pawn-base | 419K | 1 | 36.99% | pure-sparse scaling |
| LoRA rank=16 qkvo | pawn-base | 524K | 1 | 35.62% | low-rank attention |
| Bottleneck dim=32 all layers | pawn-base | 524K | 1 | **39.82%** | beats LoRA at matched params |
| Bottleneck dim=64 | pawn-base | 1.05M | 1 | 41.56% | |
| Retro-bottleneck (composition-matched) | pawn-large | 2.35M | 1 | **46.14%** | bdim=82, density=0.0153 — clean H2H against pawn-base |
| Retro-bottleneck dim=160 d=0.02 | pawn-large | 4.43M | 1 | 47.47% | |
| Bottleneck dim=512 dist | pawn-base | 8.4M | 1 | 48.51% | |
| Bottleneck dim=1024 top-4 | pawn-base | 8.4M | 1 | **49.23%** | top-4 placement +0.7 pp over distributed at matched params |
| Bottleneck dim=512 dist | pawn-large | 13.1M | 1 | 49.93% | |
| Bottleneck dim=1280 top-4 | pawn-large | 13.1M | 1 | 50.12% | top-4 generalizes to large |
| Combo: retro-bn dim=1280 top-4 + d=0.05 sparse | pawn-large | 13.93M | 1 | **50.60%** | combo +0.5 pp over pure bottleneck at matched params |
| Bottleneck dim=640 dist | pawn-large | 16.4M | 1 | 49.88% | |
| Bottleneck dim=1024 top-4 | pawn-base | 8.4M | 2 | **51.03%** | extension closes the backbone-size gap to 1-ep large |
| Combo at 13.93M, max_games=14M | pawn-large | 13.93M | 1.4 | **51.16%** | matches 26.4M combo at half the params |
| Combo: retro-bn dim=2500 top-4 + d=0.05 sparse | pawn-large | 26.4M | 1 | 51.18% | |
| Bottleneck dim=1024 dist | pawn-large | 25.6M | 1 | 50.39% | |
| Bottleneck dim=2500 top-4 | pawn-large | 25.6M | 1 | 50.74% | top-4 +0.35 pp over distributed at ceiling scale |
| Bottleneck dim=2580 top-4 (decomp test) | pawn-large | 26.4M | 1 | 50.76% | within noise of T29 — extra bottleneck params add ~nothing, isolating the combo's sparse contribution |
| Bottleneck dim=2500 top-4 | pawn-large | 25.6M | 2 | **52.37%** 🥇 | current ceiling on this dataset |

Pareto-optimal points (best val top-1 at their param budget × epoch tier) are in bold.

### Backbone size at matched adapter capacity

Holding params equal across `pawn-base` (8 layers, d=512) and `pawn-large` (10 layers, d=640):

| Family | Params | base | large | Δ |
|--------|-------:|-----:|------:|--:|
| Bottleneck distributed | 16.4M | 49.52% | 49.88% | **+0.36** |
| Retro-bottleneck (composition-matched, see note) | 2.35M | 45.91% | 46.14% | **+0.23** |
| Retro-bottleneck | 4.4M | 47.10% | 47.47% | **+0.37** |

`pawn-large` wins consistently by 0.2–0.4 pp once parameters are matched. **Composition matters for multi-component adapters**: an earlier 2.35M H2H landed at a tie because density was matched (0.03) but composition wasn't — at fixed density, the larger backbone gets more sparse and less bottleneck. Matching the bottleneck and sparse param counts separately recovers the small backbone-size advantage at every scale tested.

### Layer placement (matched 8.4M, base, distributed budget)

| Placement | Layers | Val top-1 | Δ from top-4 |
|-----------|--------|----------:|-------------:|
| top-4 | 4,5,6,7 | **49.23%** | — |
| top-2 | 6,7 | 48.63% | −0.60 |
| distributed (all 8) | 0–7 | 48.51% | −0.72 |
| mid-4 | 2,3,4,5 | 47.46% | −1.77 |
| bottom-4 | 0,1,2,3 | 45.85% | −3.38 |
| top-1 | 7 | 44.11% | −5.12 |

**Top-4 (≈40% of depth) is the sweet spot.** Concentrating the same parameter budget on the top half of the backbone gains ~0.7 pp over uniform distribution; over-concentrating onto a single layer drops 5 pp. The same pattern holds on `pawn-large` (top-4 of 10 = layers 6–9), where it adds +0.19 pp over distributed at 13.1 M and +0.35 pp at 25.6 M.

### Adapter vs. from-scratch transformer (matched budget)

`specialized_clm` is a standalone decoder-only transformer trained on the same Lichess data without a frozen backbone, sized to match adapter param counts. The matched-2-epoch comparison is the apples-to-apples one — most of the apparent adapter advantage at the 1-epoch frontier was the dedicated side being undertrained.

| Params | Dedicated 1ep | Adapter 1ep | Δ | Dedicated 2ep | Adapter 2ep | Δ (matched) |
|-------:|--------------:|------------:|--:|--------------:|------------:|------------:|
| 420K | 33.66% | 36.99% (retro-sparse) | +3.3 | — | — | — |
| 4.55M | 40.09% | 47.47% (retro-bn large) | +7.4 | 43.24% | — | — |
| 7.84M | 42.10% | 49.23% (bottleneck top-4 base) | +7.1 | — | — | — |
| 16.75M | 43.94% | 49.88% (bottleneck large dist) | +5.9 | — | — | — |
| 25.85M | 44.81% | 50.74% (bottleneck large top-4) | +5.9 | 48.97% | 52.37% | **+3.4** |

At ~25 M with both sides trained for 2 full epochs, the adapter wins by **~3 pp**, not the 6 pp the 1-epoch ladder suggested. The frozen backbone genuinely helps, but more of the early-epoch gap was undertraining than was generally appreciated.

### Takeaways

- **Bottleneck dominates at matched parameter budgets.** At 524K, bottleneck dim=32 beats LoRA rank=16 qkvo by 4.2 pp — the gap is structural across the whole budget range.
- **Gradient-informed masks beat random masks.** RoSA retro-sparse (84K) beats random-mask sparse (126K) by 1.27 pp at fewer trainable parameters. The 3-phase training overhead (LoRA warmup → mask generation → sparse-only training) pays for itself.
- **Combo (top-4 bottleneck + qkvo sparse) adds orthogonal capacity.** Decomposition test at 26.4 M: pure bottleneck dim=2580 lands at 50.76% (within 0.02 pp of the dim=2500 baseline), so the +0.42 pp T43-over-T51 gain is the genuine contribution of the gradient-informed sparse component, not a param-count effect.
- **Top-4 layer placement scales.** Sweet spot at ~40% of depth on every backbone tested.
- **2-epoch extension closes the backbone-size gap.** `pawn-base` top-4 at 8.4 M with two epochs (51.03%) beats `pawn-large` distributed at 13.1 M with one epoch (49.93%). For tight param budgets, training longer on the smaller backbone is competitive.

### Vs. the legacy v0.x sweep

The legacy v0.x adapter sweep (preserved in git at tag `pre-vocab-transition`) used a coarser ~60 K-token vocabulary, `prepend_outcome=True`, and a 256-token context. Direct comparison with the table above is apples-to-oranges because the backbone architecture, vocabulary, and training schedule all changed during the v1.0.0 transition. Use the current results as the canonical reference.

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
