# Phase A — Format Core: Implementation Spec

Authoritative, file-by-file contract for **Phase A** of `docs/v2_redesign_plan.md`
(uniform vocab + BOS + un-factor/tie + conditioning prefix + loss-mask/position
contract). The plan has the *why*; this has the *what* and *exact names*. Where
they conflict, the plan's intent wins — raise it, don't silently diverge.

## Process & constraints (READ FIRST)
- Branch: `feat/v2-phase-a-format` (already created off `jax_migration`). **Commit
  per chunk; never push.**
- **The local benchmark is finished** — running tests, `pyright`, and a small
  smoke run is allowed and expected.
- **No parallel JAX/training processes** (`feedback_no_parallel_training_local`:
  RAM can't hold multiple model copies). Test/train execution is **serialized** —
  only one JAX process at a time. Read-only review runs in parallel; it must not
  execute pytest or import-run model code.
- Verify with `pyright` (use the project extras, e.g.
  `uv run --extra rocm --extra dashboard --extra lab --extra wandb pyright <paths>`)
  and `uv run --extra rocm pytest <targeted>`. **Never suppress a type error** —
  type it properly (user global rule). **Never gut a test to pass** — fix the
  implementation or replace a genuinely-obsolete test with an equivalent guard.
- §2 (vocab) and §3 (un-factor/tie) **MUST co-land** — no intermediate commit may
  feed `BOS`/`NULL` into the still-factored `_embed` (it would alias them onto the
  last outcome embedding via the `clip()` override at `model.py:644-648`). Land
  config + model + checkpoint + adapter-compat together (Chunks 1–3 + 6) before
  anything imports a model with the new vocab.

## Global invariants (acceptance gates for the whole phase)
- `pyright` clean on `pawn/`, `scripts/`, `tests/` (modulo pre-existing unrelated
  errors — note them, don't fix out of scope).
- Full `pytest` green (rewritten tests included).
- Input vocab == output vocab == `V`. No `N_INPUT_TOKENS`/`VOCAB_SIZE` split.
- A model with `tie_embeddings=True` has no separate `lm_head` array; logits =
  `x @ embed_tokens.T`.
- Reserved/NULL/control columns (≥1981) are masked to `-inf` before softmax-CE
  (and will be before distillation-KL in Phase B). They must receive **zero
  gradient** after a training step (add a test).
- Move positions are at a **fixed, run-constant offset `C`** in every code path
  that builds a sequence (train, eval, probes, generation). One assembler owns it.
- A tiny end-to-end pretrain + checkpoint round-trip + tiny adapter run succeed.

---

## Chunk 1 — Vocab + config foundation  (`pawn/config.py`, `engine/src/vocab.rs`)
- `NUM_ACTIONS=1968`, `PAD_TOKEN=1968`, `OUTCOME_TOKEN_BASE=1969`,
  `N_TOTAL_OUTCOMES=11` stay. **Add** `BOS_TOKEN=1980`, `NULL_TOKEN=1981`,
  `N_CONTROL_RESERVED=18` (IDs 1982–1999).
- Replace the `N_INPUT_TOKENS`(1980)/`VOCAB_SIZE`(1969) split with a single
  **`VOCAB_SIZE: Final[int] = 2000`** (`V`). Delete the A.2 "lm_head covers 1969"
  comment block. `ModelConfig.vocab_size` default → `VOCAB_SIZE`.
- **Add `ModelConfig.tie_embeddings: bool = True`.**
- `validate_nested`: keep the `vocab_size` equality check; **add** a
  `tie_embeddings` equality check (variant must match supernet). Keep the existing
  `head_dim`/`d_model`/`n_layers` checks. (Optional defense-in-depth: `d_V %
  head_dim == 0`.)
- `engine/src/vocab.rs`: only touch if a Rust consumer references total vocab
  size. `BOS`/`NULL`/reserved are **Python-side** tokens (the engine emits moves +
  PAD + outcomes only). Keep `export_move_vocabulary`'s factored decomposition —
  it's the action data table (plan §10-Q5). If you edit `vocab.rs`, **do not run
  `maturin` mid-phase**; defer the rebuild to the verify stage and note it.
- Tests: rewrite (don't delete) `tests/test_jax_config.py:64-65` and
  `tests/test_jax_model.py:371` from the A.2-trim contract to the `V=2000`
  contract. Add a test pinning `BOS_TOKEN`/`NULL_TOKEN`/reserved IDs and
  `tie_embeddings` validation.

## Chunk 2 — Model  (`pawn/model.py`)  [co-lands with 1,3,6]
- Field set: **remove** `embed_src[64,d]`, `embed_dst[64,d]`, `embed_promo[5,d]`,
  `embed_pad[d]`, `embed_outcome[n_out,d]`. **Add** `embed_tokens: Float[V, d]`.
  `lm_head: Float[d, V] | None` — `None` when `tie_embeddings`.
- `_embed(ids)` → single gather `embed_tokens[ids]`. Delete the factored sum, the
  `clip(...)` index guards, and both `jnp.where` overrides (`:629-648`).
- `__call__`/forward: when tied, `logits = jnp.einsum("btd,dv->btv", x,
  embed_tokens.T)` (respect `compute_dtype`); else use `lm_head`. Keep the
  argmax/sampling restriction to `[0, NUM_ACTIONS)` (`:550`) — output columns
  existing does **not** loosen sampling.
- `sliced()` (`:1146`): slice `embed_tokens[:, :d_V]`; tied head slices via the
  transpose; untied `lm_head[:d_V, :]`. Remove the factored-field slicing.
- KV cache (`forward_with_cache`, `:852-947`): no structural change, but verify it
  works with the prefix — the first-move query is at `pos_start = C-1` (Chunk 5
  exercises it).
- **Delete the 64/64/5 literals** at `:1133-1135` and `:459-460`.
- Update the three RoPE-precision docstrings while here (review §8.4-low) only if
  trivial; otherwise leave for Phase D.
- Tests: rewrite `tests/test_jax_model.py` model-construction / embedding /
  lm_head-width / slicing tests to the new fields. Add a tied↔untied logits-shape
  test.

## Chunk 3 — Checkpoint  (`pawn/checkpoint.py`)  [co-lands with 1,2,6]
- `SAVED_FIELDS` (the per-field tensor list) computed **per `tie_embeddings`**:
  untied includes `lm_head`, tied omits it. Keep deterministic declaration order.
- **Hard-error on tied↔untied cross-load** (a checkpoint saved tied can't load
  into an untied config and vice-versa) with a clear `CheckpointIntegrityError`.
- **Delete `_expected_shapes`' hardcoded 64/64/5** (`:210`); derive expected
  shapes from the config (`V`, `d`).
- `config.json` must persist: `tie_embeddings`, `vocab_size` (`V`), the run's
  `conditioning`/`C`, and a `format_version`/`mask_version` tag (Chunk 4 defines
  the value). `load_model` must **return the persisted run block** (`:318`) so eval
  can read the checkpoint's own conditioning (plan §8.1 promotion).
- Tests: bf16-µ optimizer-state round-trip test (review §8.4, pulled forward);
  tied & untied save/load round-trip; cross-load rejection test.

## Chunk 6 — Adapter reconstruction compatibility  (`pawn/adapters/*`, `pawn/adapter_trainer.py`)  [co-lands with 1–3]
- Un-factoring changes `PAWNModel`'s fields, so **every positional
  `PAWNModel(...)` rebuild or field reference to `embed_src/dst/promo/pad/outcome`
  breaks.** Update all adapter model reconstruction to use **`eqx.tree_at`** on the
  specific leaves each strategy modifies (lora/film/bottleneck/sparse/rosa/hybrid/
  unfreeze/specialized_clm). This decouples adapters from the field set.
- Scope: **compatibility only** — adapters must construct, forward, and keep the
  frozen-backbone invariant against the new fields. Behavioral fixes (H2 real FiLM,
  H3 resume) are Phase C; FiLM here must merely not crash on the new fields and
  preserve its current identity-at-init behavior.
- Tests: adapter construct + forward + `backbone-frozen` smoke for each strategy
  (the existing partition/grad-flow contract tests, updated to new fields).

## Chunk 4 — Conditioning prefix + data  (`pawn/run_config.py`, `pawn/corpus.py`, `pawn/lichess_data.py`)
- **Replace `BaseRunConfig.prepend_outcome: bool`** (`run_config.py:122`) with
  **`conditioning: list[str] = []`** (ordered control-token kinds), validated
  against a registry `CONDITIONING_KINDS = {"outcome"}` (extensible). Derived
  `C = 1 + len(conditioning)` (BOS always present). JSON-config migration:
  `prepend_outcome=True` ⇒ `conditioning=["outcome"]` (accept the legacy key with a
  deprecation, or hard-error pointing to the new key — pick one and document).
- **Single prefix assembler + loss-mask helper**, used by *both* `corpus.py` and
  `lichess_data.py` (kills duplicated off-by-one):
  - `build_prefix(conditioning, outcome_tokens, n) -> prefix_ids[n, C]` — slot 0 =
    `BOS`; slots `1..C-1` = the resolved control token per kind, or `NULL` when a
    game lacks that value.
  - `build_loss_mask(C, game_lengths, seq_len) -> bool[n, seq_len]` — **`True`
    exactly on positions `[C-1 .. C-1 + game_length - 1]`** (predicts `ply_1 …
    ply_N`; **first-move BOS→ply_1 IS supervised**; the predict-PAD position is
    **NOT** supervised — a deliberate change from the current
    `corpus.py:257-266`, which supervised the predict-PAD slot and never supervised
    the first move). Document the change; it shifts which positions count toward
    per-move accuracy vs v1.
- Sequence layout everywhere: `[BOS][cond…][ply…][PAD…]`, strictly right-padded
  (NULL-fill introduces **no interior PAD** — preserves the Pallas-flash right-pad
  invariant, `model.py:296-311`). `targets[t]=tokens[t+1]`.
- **Versioning:** bake `C` + `mask_version` into the lichess cache key
  (`lichess_data.py`) **and** `config.json`; add a **load-time assert** that the
  builder's `C` matches the checkpoint's (closes silent absolute-RoPE drift).
- Update the `loss_mask.sum() → game_length` recovery heuristic
  (`lichess_data.py:332-336`) for the new mask definition.
- Python now assembles the prefix itself; **stop calling the engine's
  `generate_clm_batch` prepend path.** Leave the dead Rust path in place for now
  (deleting it + `maturin` rebuild is deferred to verify/Phase-cleanup to avoid
  shipping unbuilt Rust mid-phase).
- `seq_len ≤ max_seq_len` validator is **net-new** (today only a runtime
  `T>max_seq_len` check in `model.py`); add it and ensure it budgets `C` even when
  all conditioning is NULL.
- Tests: `build_loss_mask` off-by-one (C=1 and C=2, incl. first-move-supervised
  and predict-PAD-excluded); prefix assembly + NULL-fill; `conditioning` config
  validation; cache-key includes `C`; load-time `C`-mismatch assert.

## Chunk 5 — Trainer + eval + generation  (`pawn/trainer.py`, `pawn/eval.py`, `pawn/eval_suite/diagnostics.py`, `pawn/generation.py`)
- `cross_entropy_loss` (`trainer.py:149`): consume the new `loss_mask`; **mask
  reserved/NULL/control columns (IDs ≥1981) to `-inf`** before the softmax so they
  don't dilute move mass or accrue gradient. `supernet_joint_loss` structure
  unchanged (still slices one model — there is no cotrain path; plan §7).
- `eval.py` per-phase accuracy (`:128-172`): **add a `+C` offset** —
  `PhaseBoundaries` bins by `ply = t - C`, not raw `t` (plan §8.1 promotion).
  Fold the 8 per-chunk `int()` host-syncs into the jitted body while here
  (review §8.4) if low-risk.
- `eval.py` / `eval_vs_stockfish.py`: build the corpus using the **checkpoint's own
  `conditioning`** (read from the persisted run block, Chunk 3), not a hardcoded
  default — removes the layout/off-by-one mismatch (plan §8.1).
- `generation.autoregressive_generate` (`:239,557`): lay down the full `C`-wide
  prefix and decode from `pos_start = C-1`; remove the hardcoded outcome-at-0.
  Generation diagnostics: gating simplifies to "what's in the conditioning slots";
  keep the 5 diagnostics functional.
- Tests: trainer loss with reserved-column masking + reserved-rows-zero-grad;
  per-phase `+C` offset; eval-reads-checkpoint-conditioning; generation prefix +
  `pos_start=C-1` parity (KV-cache vs full-forward).

---

## Verify stage (after all chunks)
1. **Integration review** (multi-lane, read-only) over the full
   `feat/v2-phase-a-format` diff vs base: spec-conformance, correctness/bug, JAX/
   Equinox footguns, performance, test-adequacy. Plus a best-effort `codex review`
   lane (`feedback_run_codex_in_reviews`). Adversarially verify high/critical
   findings; fix-loop until none remain.
2. **Full suite:** `pyright` clean + `pytest` green. If `vocab.rs` changed, rebuild
   the engine (`cd engine && uv run --with maturin maturin develop --release`)
   once, then re-run.
3. **End-to-end smoke** (single process; prefer GPU, else `PAWN_ALLOW_CPU=1`):
   - Tiny pretrain: `scripts/train_jax.py --supernet tiny --total-steps 30
     --batch-size 8 --seq-len 64 --k 10 --conditioning outcome --local-checkpoints`
     → loss finite & decreasing-ish, 0 NaN, checkpoint written.
   - Checkpoint round-trip: load the written checkpoint (tie_embeddings honored),
     re-instantiate, verify logits match pre-save on a fixed batch.
   - Tiny adapter: `scripts/train_jax_adapter.py --strategy lora --supernet tiny
     --variant base --total-steps 20 --local-checkpoints` → backbone bit-identical,
     val_loss finite.
   - Confirm a `conditioning=["outcome"]` checkpoint's generation diagnostics no
     longer report `_skipped`.

## Done = every Global Invariant gate is green + the smoke stage passes.
