# PAWN v2 Redesign & Parity Plan

> **Status:** proposal, pre-implementation, **panel-reviewed** (changes folded
> in; see the "Review outcomes" callout per section). Source of truth is this
> doc + the commit log, **not** the stale `docs/jax_migration_plan.md` /
> `DEFERRALS.md`. Companion artifacts at repo root:
> `JAX_MIGRATION_REVIEW.md` / `JAX_MIGRATION_REVIEW_findings.json`.

## 0. Framing

PAWN is a **testbed for finetuning/augmentation methods at small scales**, used
primarily by one researcher with no significant downstream consumers. The
principle driving every decision:

> **Optimize the scarce resource, not the abundant one.** Compute and parameter
> count are abundant (tiny models, single ROCm GPU). Scarce are (a) *clean,
> attributable measurement* of method effects and (b) *external validity* —
> results must transfer to real LLMs. v2 choices that traded a scarce resource
> for an abundant one (factored embeddings, the A.2 output-vocab trim, the
> supernet-as-substrate) are reversed.

v2 stays **JAX/Equinox/Optax** (settled). v2 is held to **feature parity with v1
(= current `main`)**: any validated-but-unconsumed knob or silently-dropped v1
capability is a **bug**.

## 1. Decisions (the pivot)

1. **Uniform vocabulary.** Input == output. Revert the A.2 outcome-column trim.
   Add **BOS** + **NULL** + a reserved control block. (§2)
2. **Un-factor the move embeddings** into a monolithic token table, **weight
   tying on by default** (`tie_embeddings=True`). (§3)
3. **Conditioning prefix** ("system-prompt" region): fixed-width `[BOS][cond…]`
   that generalizes `prepend_outcome` and makes discrete control-token
   conditioning a first-class axis. (§4)
4. **Loss-mask, never vocab-surgery.** Decide predictability by masking targets,
   not deleting output columns. (§5)
5. **Scale ladder:** **distillation** (or sequential independent runs) is the
   *canonical published* ladder; the **supernet is retained as a contrast arm**
   behind its existing path and gated by a quality-parity harness. **There is no
   cotrain to re-enable** — see the §7 correction. (§7)
6. **Fresh retrain** of the ladder under one v2 pipeline (the format changes
   force it). The shared 1968-action move vocab keeps per-move accuracy
   *scale-comparable* to v1's published numbers (magnitude, not bit-exact). (§7)
7. **Fix all parity bugs** except those *obviated by redesign*. (§8)

## 2. Vocabulary & token layout

Current (`config.py:79-92`): input `N_INPUT_TOKENS=1980` but **output
`VOCAB_SIZE=1969`** (A.2 trimmed the 11 outcome columns). Asymmetric.

**Target — one uniform vocab `V`:**

| IDs        | Meaning                                  |
|------------|------------------------------------------|
| 0–1967     | searchless_chess move actions (unchanged) |
| 1968       | `PAD`                                     |
| 1969–1979  | outcome tokens (11, unchanged IDs)        |
| 1980       | `BOS`                                     |
| 1981       | `NULL` (unconditioned slot filler)        |
| 1982–1999  | reserved control block (inert until assigned) |

`V = 2000`. Existing IDs 0–1979 preserved → action-space metrics stay
v1-comparable. Both input embedding and lm_head are width `V`.

> **Review outcomes (§2):**
> - **Mask un-assigned columns (NULL + reserved 1981–1999) to `-inf`** before
>   softmax-CE *and* distillation-KL. Reserved rows are inert as *targets* but as
>   distractor columns they dilute move mass / spread teacher mass — masking
>   preserves the V=2000 headroom *and* keeps perplexity/KL genuinely
>   v1-comparable. (Resolves §10-Q1: keep V=2000 + mask, not tight-V=1982.)
> - **Lockstep checklist** with `engine/src/vocab.rs` must also include the
>   theoretical-ceiling stride (`scripts/compute_theoretical_ceiling.py`,
>   `flat_pos = sparse // vocab_size`) — a silent-corruption coupling on
>   `vocab_size`.
> - **Test churn:** `tests/test_jax_config.py:64-65` and `test_jax_model.py:371`
>   pin the A.2 trim — rewrite (don't delete) to the V=2000 contract.

Move *sampling* still restricts argmax to `[0, NUM_ACTIONS)` (`model.py:550`);
output columns existing doesn't change that — it stops foreclosing
outcome/conditioning prediction.

## 3. Embeddings — un-factor + tie

Current (`model.py:459-466, 616-648`): factored
`embed_src[64,d]+embed_dst[64,d]+embed_promo[5,d]` + `embed_pad`/`embed_outcome`
overrides + a separate `lm_head[d,V]`.

**Target:**
- Single `embed_tokens: Float[V, d]`; `_embed(ids) = embed_tokens[ids]`. Delete
  the factored decomposition, clamps, and `jnp.where` overrides.
- `tie_embeddings: bool = True` (new `ModelConfig` field). Tied:
  `logits = x @ embed_tokens.T`, no `lm_head` array. Untied: keep `lm_head[d,V]`
  (a studyable knob).
- **Supernet slicing:** `embed_tokens[:, :d_V]`; tied head slices via transpose.
  `validate_nested` gains a `tie_embeddings` equality check (compares `V`).

> **Review outcomes (§3):**
> - **Param framing corrected:** the input table grows ~15× *in isolation*, but
>   net embedding budget (input+head) barely moves because tying removes the head
>   that already existed (≈ −0.1% total params tied; +2–5% untied). Conclusion
>   ("irrelevant on this hardware") stands; the "15× bigger model" reading does
>   not.
> - **`lm_head: Array | None`** driven by static `tie_embeddings`; compute
>   `SAVED_FIELDS` per-mode and **hard-error on tied↔untied checkpoint
>   cross-loads** (deterministic PyTree structure → JIT/partition-safe).
> - **Adapters reconstruct via `eqx.tree_at`** on the specific leaves they touch,
>   not positional `PAWNModel(...)` rebuilds — decouples lora/bottleneck/sparse/
>   rosa/hybrid from the field set so this change (and future ones) touches only
>   `model.py`/`checkpoint.py`.
> - **Delete all 64/64/5 literals:** `model.py:1133-1135` and `:459-460`, not
>   just the `checkpoint.py:210` copy.
> - §10-Q2 resolved: **tied is canonical** (representative of modern small LLMs;
>   net params ≈unchanged). No current adapter touches embeddings, so the
>   confound is forward-looking — *document* that embedding/head-adapter studies
>   must set `tie_embeddings=False` to attribute cleanly.

**Rationale:** factored embeddings are a chess-specific prior that makes
embedding-touching adapter results non-representative of real LLMs. Monolithic
`V×d` is the canonical LLM surface, enables tying, and kills the dual-vocab-size
off-by-one footgun.

## 4. Conditioning prefix ("system prompt" region)

Generalizes the (inert — H1) `prepend_outcome: bool`.

```
[BOS] [cond_1] … [cond_{C-1}] [ply_1] … [ply_N] [PAD] … [PAD]
└────── prefix (width C) ──────┘└──── moves ────┘└── right-pad ──┘
```
- `BOS` always at position 0 (`C ≥ 1`).
- `conditioning: list[str] = []` (new `BaseRunConfig` field, ordered) →
  `C = 1 + len(conditioning)`, validated against a registry (`{"outcome", …}`).
  Migration: `prepend_outcome=True` → `conditioning=["outcome"]`.
- **Fixed width + `NULL`-fill:** games lacking a conditioning value get `NULL` in
  that slot so `C` is constant across the dataset → stable move positions.
- **Prefix assembly is Python-side** (`corpus.py`/`lichess_data.py`); the engine
  already returns `move_ids` + a separate `outcome_tokens` column
  (`lib.rs:993-995`).

> **Review outcomes (§4):**
> - **Claim scoped:** this natively enables *discrete control-token* conditioning
>   (control tokens, and the substrate for prefix-style methods). **Continuous /
>   soft prompts need a later trainable-vector injection hook** this design does
>   not provide — the fixed-`C` contract is the right substrate either way, but
>   don't overstate "soft-prompt natively."
> - **Delete the in-engine prepend path** (`lib.rs:175-200,1001-1017,1056-1061` +
>   `batch.rs`) rather than leaving it dual-pathed — Python becomes the single
>   source of truth *in fact*. Nothing live calls it; low risk.
> - **`autoregressive_generate` must be rewritten** to lay down the full `C`-wide
>   prefix and decode from `pos_start = C-1` (`generation.py:239,557` hardcode
>   outcome-at-0). This is explicit work, not "verify."
> - **Consider now** (only if soft-prompt studies are near-term): type slots as
>   `(name, kind ∈ {token, soft})` to avoid a second format migration.

## 5. Loss-masking contract

Keep the existing masked-CE machinery (`trainer.py:217-220`, targets =
left-shifted next-token, `trainer.py:82`). Mask definition:

- `targets[t] = input_ids[t+1]`.
- `loss_mask[t] = True` on `[C-1 .. C-1+N-1]`: position `C-1` (last prefix slot)
  predicts `ply_1` → the **first-move distribution** is learned there; through
  the position predicting `ply_N`. `False` on interior prefix `[0..C-2]` and the
  PAD tail.
- **Principle:** change predictability via the mask, never by trimming columns.

> **Review outcomes (§5):**
> - **Deliberate semantic change, documented:** the current builder
>   (`corpus.py:262-266`) supervises one *extra* position whose target is the
>   first PAD; the new contract **drops it** (denominator −1 per game). This
>   slightly perturbs the v1 per-move-accuracy comparison — expected, not a bug.
>   If an explicit end-of-game signal is wanted, use a reserved control token,
>   **not** the implicit predict-PAD slot.
> - **Single `build_loss_mask(C, game_lengths, seq_len)` helper** consumed by
>   both `_pack_clm` and the lichess path — kills duplicated off-by-one drift.
> - **The new mask breaks the `loss_mask.sum() → game_length` recovery
>   heuristic** (`lichess_data.py:332-336`) — update it.
> - §10-Q3 resolved: **supervise `C-1` by default**, but treat as load-bearing —
>   it's the only position the prefix gets direct next-token gradient; gating it
>   off yields a materially different "pure continuation" mode (prefix trained
>   only via attention).

## 6. Position / RoPE contract

RoPE uses absolute positions (`model.py:235, 942`). With fixed `C`: prefix
`0..C-1`, moves `C..C+N-1` — stable **iff `C` is identical everywhere**.

> **Review outcomes (§6):**
> - **Version the mask contract + `C` into the lichess cache key
>   (`lichess_data.py`) *and* `config.json`, with a load-time assert** that the
>   builder's `C` matches the checkpoint's. Closes the two highest-severity
>   interactions: stale `$HF_HOME/pawn-lichess-cache` entries trained under the
>   old mask, and **silent absolute-RoPE position drift** across checkpoints with
>   different `C` (degraded accuracy, no shape error).
> - The `seq_len ≤ max_seq_len` validator is **net-new** (today only a runtime
>   `T > max_seq_len` check exists in `model.py`) and must budget `C` even when
>   all conditioning is NULL.
> - **One source of truth** for prefix assembly + `C`, consumed by trainer / eval
>   / probes / generation. **This must be mandatory, not opt-in** (see Residual
>   Risks) — any consumer defaulting to `C=1` against a `C>1` checkpoint silently
>   reincarnates the H1 mismatch.
> - KV-cache decode (`forward_with_cache`, `model.py:852-947`) takes `pos_start`;
>   verify the first-move query (`pos_start = C-1`) is exercised.

## 7. Scale-ladder production

> **CORRECTION (load-bearing).** The earlier draft assumed "cotrain is already a
> flag that disables the supernet." **It is not.** `supernet_joint_loss`
> (`trainer.py:225`) takes a *single* `PAWNModel` and `sliced()`s it per variant
> — all variants share one weight tensor. `stochastic_variants` only switches
> between "sum all slices" and "supernet + one sampled slice", both *inside* the
> shared-weight supernet. `run_config.py:15` ("v1 `CotrainConfig` GONE BY
> DESIGN") + `trainer.py:7` ("the only pretrain path") confirm: **no
> independent-weight pretrain path exists.** "Cotrain canonical" would mean
> building a new 3-model joint trainer (3 models, 3 optimizer states, 3-model
> checkpoint/resume/publish) — and `feedback_no_parallel_training_local` flags
> that local RAM can't hold multiple model copies at once. So cotrain is **not**
> the cheap default it was framed as.

**Revised ladder:**
- **Canonical published ladder — distillation (default) or sequential
  independent runs.** Distill `base`/`small` from a frozen `large` teacher,
  reusing the existing `specialized_clm` shapes + the adapter trainer's
  frozen-tier partition machinery (`eqx.partition` + `jax.grad` over trainable),
  swapping CE for `KL(student ‖ softmax(teacher_logits/T)) + α·CE`. **One student
  trained at a time → sidesteps the multi-model RAM problem.** Logit distillation
  only (students aren't slices → no hidden-state matching). If function-sharing
  is *not* wanted as a studied axis, **sequential independent from-scratch runs**
  (one model at a time) are the lowest-machinery alternative — also RAM-safe.
  (Resolves §10-Q4.)
- **Supernet — contrast arm.** Keep the sliced shared-weight path behind its
  existing toggle. Gate with a **quality-parity harness** that goes *beyond*
  per-phase move accuracy:
  - linear-probe decodability (reuse `probes.py` / H6), and
  - a reference-LoRA val-loss delta (supernet-`small` vs the canonical `small`).

    Move accuracy can match while the properties the testbed actually measures
    (probe-decodability, adapter response) diverge.
- **Distillation-KL must be masked to the supervised vocab support** (vocab-axis
  restriction, §2) so teacher mass doesn't spread onto dead reserved/PAD/NULL
  columns — in addition to the §5 time-axis mask.

Net effect: "how the ladder relates to itself" (function-shared distillation vs
weight-shared supernet vs fully-independent) stays a deliberate research axis,
without standing up a from-scratch multi-model trainer.

### 7.1 Distillation generality — design for adapted/conditioned teachers

The distillation trainer must be built so that distilling an **adapted,
augmented, or conditioned** model later is a *wiring* change, not a new pipeline.
This is structurally free: both teacher and student are already
"frozen-or-trainable PyTree → logits over `V`", and the adapter trainer is
already "freeze part / train part / `grad` over trainable" (`eqx.partition` +
`eqx.combine`). Distillation is that same machine with **CE swapped for
KL-against-a-frozen-teacher's-logits**. An adapter-adapted model is
forward-identical to a plain `PAWNModel` after `eqx.combine`; a conditioned model
is the same weights run with a different prefix. So one trainer covers: plain →
plain (§7 canonical); **adapted teacher → fresh student** (compress a specialist);
**plain teacher → adapter student** ("distill-LoRA", student is an adapter on a
frozen backbone, loss = KL); **adapted → adapted** (transfer an adaptation across
scale); **conditioned → unconditioned** (bake a behavior into weights).

**Two non-negotiable design choices (Phase B), both just "don't hardcode":**
1. **Teacher is an injectable `logits_fn`**, not the literal `large` checkpoint —
   decouples "what produces the soft targets" from "it's the pretrained large",
   so any frozen model (plain / `eqx.combine`d-adapter / conditioned) drops in.
2. **Pluggable objective** (`CE` / `KL` / `α·CE + (1-α)·KL`) layered on the
   **existing adapter-trainer partition** — i.e. **unify the distillation trainer
   with the adapter trainer**; "distillation" is adapter/full training with a
   teacher-KL term, so the *student* can be a full model or an adapter on a frozen
   backbone with no new trainer.

**Why it just works** (no extra machinery): the uniform `V=2000` +
`-inf`-masked support (§2) makes KL well-defined across any teacher/student pair;
the fixed-`C` layout (§4/§6) aligns teacher/student move-positions *even when
their conditioning differs* (the enabler for conditioned→unconditioned distill);
logit-only distillation (no hidden-state matching) captures adapter effects
functionally, so cross-scale/cross-arch adapted teachers need nothing extra.

**Usage notes (not code gaps):** distill a *narrow* specialist teacher on its
*target distribution* (the teacher's soft targets are only meaningful there — the
Lichess Elo-band path already supports this); teacher-logit caching
(`(position, conditioning) → logits`) is an equal-footing perf option for
expensive adapted teachers.

**Research payoff:** this turns *adaptation transfer / specialist compression*
into a first-class studyable axis (e.g. "does distilling a LoRA specialist into
base weights preserve the adaptation? is distillation a better
adaptation-transfer mechanism than re-adapting the small model directly?") — on
mission for the testbed, for the cost of the two "don't hardcode" decisions.

## 8. Parity & bug fixes

### 8.1 Obviated by redesign (folded into §2–§6)
- **H1** `prepend_outcome` inert → conditioning *is* the format (§4).
- Eval/train **distribution** mismatch from `prepend_outcome`
  (`eval_jax.py:31-36`, `eval_vs_stockfish.py:42-49`) → single-source layout (§6).
- Generation-diagnostic gating complexity → "what's in the conditioning slots".
- `OUTCOME_TOKEN_BASE` dead import (`generation.py:55`).
- `_expected_shapes` 64/64/5 (`checkpoint.py:210`) → un-factor (§3).
- FiLM output-axis-at-`d_model` sibling → output FiLM now targets uniform `V`
  logits; folds into H2.
- `pgn_val_split=""` sentinel coercion → handled in H4.

> **Review correction — NOT obviated, promoted to kept work:**
> - **Per-phase binning offset.** `compute_per_phase_accuracy` (`eval.py:128-132`)
>   bins by *raw* sequence index `t`; a stable `C` does **not** fix this — once
>   `C>1` every per-phase number is off by `C` plies, corrupting the v1
>   regression check §6 relies on. → `PhaseBoundaries` takes a `+C` offset (bin by
>   `ply = t - C`). **Kept under H5 / Phase D.**
> - **Eval-reads-conditioning** is the linchpin that makes the obviation cluster
>   *structurally* true. → promoted to a **Phase-A deliverable** (same commit as
>   the §6 assembler); `load_model` must return the persisted run block
>   (`checkpoint.py:318`).

### 8.2 High — kept
- **H2** Real FiLM via post-residual `attn_hook`/`ffn_hook` (`γ⊙h+β`; output FiLM
  at logit space, now `V`-wide). Hybrid inherits. Add the numeric test vs an
  explicit `γ⊙h+β` reference **and fix the stale `(d_model,)` "v1 parity"
  assertion in `tests/test_jax_adapters.py`** (don't just add a new test).
- **H3** Adapter `--resume`: persist+reload the adapter PyTree for all strategies
  (sidecar) **or** cold-start `opt_state` when the adapter is cold.
- **H4** Implement the carve-from-train val split (deterministic hash-of-game-id;
  fraction+seed in the cache key) or hard-error; distinguish `None` from `""`.
- **H5** Edge-case off-by-one: pair `correct[:,:-1]` with `bits[:,1:]` against the
  prefix-shifted layout; **+ the per-phase `+C` offset above**. Unit test on a
  known in-check ply **and a terminal label (checkmate/stalemate)** — the latter
  has a second off-by-one (scored against a PAD target).
- **H6** Probes on real frozen hidden states (label via `extract_board_states`),
  per-layer, with a train/val split. Don't emit `results['probes']` until wired.
- **H7** Re-implement `schedule_health.json` at both exit paths; lab runner reads
  + flags `actual≠planned ∧ completed`.
- **H8** Route `AdapterObjective` params through temp-JSON `--config` (not
  kebab-flags); argv-acceptance test per suggester. **Migrate `suggest_*` off
  `prepend_outcome` to `conditioning` and test against the post-Phase-A schema**
  (else `extra="forbid"` re-breaks sweeps).
- **H9** `rosa-ratio`: map onto `rosa` via a consumed flag/config, or drop it.
- **H10** Thread `cfg.max_grad_norm` into `make_optimizer`; `did_clip` then
  matches the actual clip.
- **H11** Drive the adapter loop through `make_adapter_scan_step` (K-step scan).

### 8.3 Medium / inert-knob parity bugs — kept (wire-or-reject)
Grad accumulation `(K,N,B,T)` prefetcher wiring (`trainer.py:760-858`,
`train_jax.py:259`); `mate_boost`, `epochs`/`steps_per_epoch`/`data_seed`/
`val_every` (wire or reject); `--wandb` (wire or reject + fix docstring);
persist scheduler+RNG in `training_state.json`; adapter `step_time` resume
divisor (`/(step-start)`); `improbable_task` unreachable `DRAW_BY_AGREEMENT`.

### 8.4 Low — tracked checklist (detail in the review artifact)
Pallas-flash mask/segment_ids doc+guard; **Pallas-flash backward
gradient-parity test** (the one untested load-bearing correctness dep);
**bf16-μ checkpoint round-trip test — pulled forward into Phase A** (Phase A
rewrites the tensor set and the fresh retrain is the first long run to cross a
bf16-μ checkpoint boundary); per-phase host-sync folding; `setup_jax_caching`
in eval scripts; three RoPE-precision docstrings; dead imports; hoist duplicated
device helpers; `validate_nested` `d_V % head_dim` guard; tautological PAD-init
test → drive a real export path; drop dead inner `train_step` `filter_jit`;
reconcile `eqx.partition` docstring; pre-existing engine `game_length` index
panic. **Plus an audit of the perf rounds for other A.2-style
"trade-optionality-for-FLOPs" micro-opts.**

## 9. Sequencing

Each phase: implement → subagent review → fix → next. No heavy training/bench
runs while a local benchmark is active.

- **Phase A — Format core (retrain-forcing; §2 and §3 MUST co-land):** uniform
  vocab + BOS/NULL + un-factor + tie; conditioning prefix + loss-mask +
  position contract; one prefix/`C` assembler; `C`+mask versioned into cache key
  + `config.json` with load-time assert; **eval-reads-conditioning**; bf16-μ
  round-trip test. Absorbs §8.1.
- **Phase B — Pretraining substrate:** distillation trainer (or sequential runs)
  as canonical ladder, **built with an injectable teacher `logits_fn` + pluggable
  loss and unified with the adapter trainer (§7.1)**; supernet contrast + parity
  harness (§7); trainer fixes (H10, grad-accum, inert knobs).
- **Phase C — Adapters:** H2, H3, H11, adapter inert knobs + step_time;
  `eqx.tree_at` reconstruction (§3).
- **Phase D — Eval & observability:** H5 (+ per-phase `+C` offset), H6, H7,
  probes-real, wandb, scheduler/RNG persistence, diagnostics cleanup.
- **Phase E — Sweeps:** H8 (incl. `conditioning` migration), H9.
- **Then:** one fresh pretraining run → publish `pawn-{small,base,large}-v2`;
  sanity-check per-move accuracy against v1 (scale-comparable, §10/Residual).

## 10. Open questions — resolved by the panel
- **Q1 (V=2000 vs 1982):** V=2000 + **mask un-assigned columns to `-inf`** in
  CE+KL. Headroom *and* clean perplexity/KL parity. (§2)
- **Q2 (tie default):** **tied canonical**; document that embedding/head-adapter
  studies set `tie_embeddings=False`. (§3)
- **Q3 (supervise C-1):** **yes by default**, load-bearing (first-move dist +
  only direct prefix gradient). (§5)
- **Q4 (cotrain vs distillation canonical):** **distillation (or sequential
  independent runs)** — cotrain is unbuilt + RAM-risky; distillation trains one
  student at a time and reuses existing machinery. (§7)
- **Q5 (engine factored decomposition):** **keep** — it's the action data table;
  only the *model* un-factors. No consumer breaks. (settled)

## 11. Residual risks to watch
- **Single assembler must be mandatory.** Any consumer calling
  `generate_corpus`/`load_lichess_corpus` with default `C=1` against a `C>1`
  checkpoint silently reincarnates the H1 mismatch — absolute-RoPE makes it
  silent (degraded accuracy, no error).
- **§2/§3 co-land.** An intermediate commit feeding `BOS@1980`/`NULL@1981` into
  the still-factored `_embed` aliases them onto the last outcome embedding via
  `clip(...)` (`model.py:644-648`).
- **Reserved/NULL rows stay at init** — add a test that they receive zero
  gradient (via the tied head) after training.
- **Multi-model RAM** — only relevant if independent-cotrain is ever built;
  verify the `large` (d=640) envelope fits the single ROCm GPU first.
- **Pallas-flash right-pad invariant** (`model.py:296-311`) — NULL-fill adds no
  interior PAD, so it's preserved; re-verify after the corpus-builder rewrite
  (it's the untested load-bearing correctness dep, §8.4).
- **v1 regression check is scale-comparable, not bit-comparable** — fresh retrain
  changes init (tied), param count, mask denominator (§5), and phase binning;
  validate metric *magnitude*, don't over-read small deltas.
