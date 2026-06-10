# Factored-embedding (v1-architecture) LR-confound experiment

## Question

v2-large@400k underperforms v1-large on compound legality — teacher-forced
game-completion 97.71% vs 99.90% (the converted v1-large re-measured under
our eval; its published figure is 99.76%), and autoregressive
game-completion **34.6% vs 65.9%** (the AR numbers measured by
`scripts/eval_ar_legality.py` / `scripts/eval_v1_ar_legality.py`). Two
confounded explanations:

1. **Architecture** — v1's factored `src+dst+promo` move embeddings (square-
   aware, plausibly more sample-efficient for legality) + larger dims
   (66.91M params, head_dim 80, d_ff 2560) vs v2's uniform `embed_tokens`
   (53M, head_dim 64, d_ff 1792).
2. **Training recipe** — v1 trained 200k steps with its own schedule
   (cosine-ish, wd 0.01); v2 trained 400k with the `infinite` (WSD-style)
   schedule at wd 0.0. The LR schedule could explain more of the gap than
   assumed.

This experiment **trains the exact v1 architecture under the exact v2
recipe**. If the factored model under the v2 recipe recovers v1-class
compound legality, the architecture is the culprit and the recipe is
exonerated; if it lands at v2-class numbers, the recipe (or data volume /
schedule) is implicated and the architecture story weakens.

## What trains

`pawn.factored_model.FactoredPAWNModel` at `pawn.config.FACTORED_V1_LARGE` —
v1-large's published architecture verbatim (from
`thomas-schweich/pawn-large` `config.json`):

| field | value | v2-large (for contrast) |
|---|---|---|
| d_model | 640 | 640 |
| n_layers | 10 | 10 |
| n_heads × head_dim | 8 × 80 | 10 × 64 |
| d_ff (SwiGLU) | 2560 | 1792 |
| vocab | 1980 (no BOS/NULL/reserved) | 2000 |
| embeddings | factored src+dst+promo (+pad, +outcome) | uniform `embed_tokens` |
| lm_head | untied (always) | untied (since 18a0047) |
| params | 66.91M | ~53M |

**The transformer trunk is byte-shared with v2** (`pawn.model._run_layers_impl`
+ fp32 RoPE + fp32 head matmul + the fixed flash custom-VJP): the ported
factored class originally carried the pre-`cda3f8a` numerical recipe, which
would have re-introduced the very instabilities v2 fixed and confounded the
comparison. Only `_embed`, the always-present `lm_head`, and the config dims
differ from `PAWNModel`.

## Sequence contract

v1's native bare-moves layout, via `pawn.corpus.to_v1_contract`: the v2
pipeline's `[BOS][m_1…]` corpus has slot 0 rewritten to a masked,
unsupervised PAD (BOS=1980 is out-of-vocab for this model; v1 never
supervised the first move). The transform is validated — the converted
v1-large reproduces its published eval numbers through it
(`scripts/eval_v1_legality.py`). Supervision is otherwise identical to the
v2 run (first move excepted: 1 of ~512 positions).

## Recipe (replicated from the v2 teacher run)

Source of truth: the `hardy-finch` config record
(`logs/teacher_large/pretrain_20260604_220846_178138_hardy-finch/metrics.jsonl`)
— the from-scratch lr=3e-4 run whose lineage became v2-large@400k.
Committed as `configs/factored_v1_lr_confound.json`:

400k steps · effective batch 64 (32×2 accumulation) · seq 512 · K=50 ·
AdamW(0.9, 0.95) wd 0.0 · clip 1.0 · `infinite` schedule (warmup 5%,
stable-ratio 0.5, cooldown 20%, decay 10%, linear) · peak LR 3e-4 ·
bf16 AMP · conditioning [] · val every 10k on 2048 games.

Launch:

```bash
uv run --extra rocm python scripts/train_jax.py \
    --config configs/factored_v1_lr_confound.json --local-checkpoints
```

### Known asymmetries vs the v2 run (deliberate, documented)

- **Attention kernel**: this run uses the plain materialised-QK^T path from
  step 0. The v2 run's history mixed kernels (stock Pallas flash — with the
  then-undiagnosed bf16-backward defect — for its first ~255k steps, plain
  thereafter). Replicating that churn is neither possible nor desirable;
  the plain path is the bit-stable baseline. If anything this *favours* the
  factored run — worth remembering if it wins by a hair (it is not expected
  to matter at the effect sizes of interest, ~2pt teacher-forced / ~31pt AR).
- **Batch micro-structure**: 32×2 accumulation vs hardy-finch's 64×1
  (identical effective batch; the v2 lineage itself switched to 32×2
  mid-run when memory pressure rose). The factored model's d_ff is 43%
  larger, so 32×2 is the known-safe local shape.
- **First-move supervision**: the v1 contract leaves m_1 unsupervised
  (~0.2% of supervised positions). Inherent to the architecture's vocab.

## Eval plan (after the run)

Same evals as the v2-large@400k report, all already factored-aware:

1. `compute_val_metrics` (per-move legal, top-1/5, ppl) — in-run val.
2. `compute_compound_legality` (teacher-forced game-completion) via the
   checkpoint loader (`pawn.checkpoint.load_model` dispatches on
   `factored_embeddings`).
3. AR game-completion via `scripts/eval_v1_ar_legality.py` (the no-BOS
   seeded-m_1 loop; pass the local checkpoint dir).
4. Optional: `scripts/eval_probes_jax.py` (probes accept the factored model).

Comparison targets: v2-large@400k (97.71% TF / 34.6% AR), converted v1-large
(99.90% TF / 65.9% AR, 200k steps under v1's own recipe).
