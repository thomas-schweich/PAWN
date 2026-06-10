# Deferrals

> Per `docs/jax_migration_plan.md` §9.4, a deferral is legitimate only
> when actually doing the work is *nonsensical, impossible, or actively
> detrimental*. Things on the §3 acceptance-criteria list cannot be
> deferred under any reason.

This file replaces the earlier "No deferrals" claim, which was
inaccurate per `docs/JAX_PARITY_SHORTFALLS.md` §16. The parity sweep
that document drove (`feat/jax-migration/*` section branches and
commits `0b3` through `93dd9` on `jax_migration`) addressed every §3
acceptance-criterion gap and every behavioral parity gap that had a
sensible JAX implementation. The items below are the residual list:
either gone-by-design or explicitly carried as follow-ups.

## Gone-by-design

These v1 surfaces are intentionally absent from v2 and will not return:

- **Legacy v1→JAX checkpoint converter** (`pawn/legacy.py`,
  `pawn.legacy.convert_legacy_checkpoint`,
  `scripts/convert_published_checkpoints.py`,
  `pawn/_torch_legacy_fixture.py`, `tests/test_jax_legacy.py`) — and with
  it the plan's **acceptance criteria 4, 5, and 20**.
  **Removed in:** the H.2 housekeeping commit.
  **Reason:** actively detrimental + (post-redesign) nonsensical. The
  converter *was* built and verified working (it hit the criterion-4 logit
  tolerance on the real published `pawn-small`: mean Δ ≈ 5.9e-6, argmax
  100%). But the **Phase-A format redesign that landed afterward**
  changed the model's vocabulary and embedding structure — `VOCAB_SIZE`
  went 1980→2000 and the factored `src+dst+promo` move embeddings were
  replaced by a single un-factored, output-tied `embed_tokens[2000, d]`
  table (`pawn/model.py`, `pawn/config.py`). A v1 PyTorch checkpoint's
  weights have **no shape-compatible image** in the v2 architecture, so a
  weight-transposing converter can no longer produce a loadable v2 model;
  keeping a converter that cannot round-trip would be a maintenance trap
  and a correctness footgun. Criteria 4/5/20 were written (plan §3, §6:143,
  §7:177) before that redesign was contemplated and are **superseded**: v1
  artifacts stay reachable via the `v1.0.0` git tag (CLAUDE.md), and v2
  trains + publishes fresh `pawn-{small,base,large}-v2` checkpoints rather
  than converting the v1 weights. This is the one place where v2 is **not**
  a superset of the *original plan*; it is a deliberate, documented design
  decision taken after the plan was written, not an unfinished migration.

- **Cotrain config + entry point** (`pawn.run_config.CotrainConfig`,
  `scripts/train_cotrain.py`, `tests/training/test_cotrain_*`). The
  supernet's joint loss (`pawn.trainer.supernet_joint_loss`)
  replaces multi-variant cotraining; pretrain the supernet, then
  slice the three variants at publish time. Plan §6 documents the
  removal.

  ### Cotrain per-model (per-variant) early stopping
  **Section:** distill (T2-core-training workstream; uncommitted)
  **Reason:** nonsensical — v1 cotrain ran N independent `ModelSlot`s,
  each with its own optimizer + val loop + patience counter, so a
  variant whose val loss plateaued could be frozen while its siblings
  kept training (`git show main:pawn/cotrain.py` per-slot
  `patience_counter` / `evaluate`). v2 has no `ModelSlot`: the supernet
  is **one** weight tensor and `pawn.trainer.supernet_joint_loss` sums
  every selected variant's cross-entropy into a single scalar that one
  optimizer minimises (gradients accumulate into the shared tensor).
  There is no per-variant parameter set to freeze independently — the
  small/base/large slices are nested `[:d_V, :d_V]` views of the same
  array, so "stop training base but keep training large" is not
  expressible (freezing the inner slice would freeze the corresponding
  region of large too). Per-variant patience is therefore not a feature
  that has a v2 home; it died with the slot abstraction it was built on.
  **What was supposed to happen:** v1's cotrain loop early-stopped each
  model slot independently on its own held-out val loss.
  **What I did instead:** the v2 pretrain loop early-stops the **whole**
  supernet on a single compound patience signal (best val loss + best
  late-game legality across the trained variants), which is the only
  early-stop semantics the shared-tensor design admits
  (`scripts/train_jax.py::_run_validation`, pinned by
  `tests/scripts/test_train_jax_smoke.py::test_train_jax_patience_early_stops`).
  Distillation (the net-new v2 trainer this workstream touches) trains a
  single standalone student, so per-variant patience is doubly
  inapplicable there.
  **To revisit when:** never under the supernet design. If a future v2
  direction reintroduces genuinely-independent multi-model training (not
  nested slices of one tensor), per-model patience would be reconsidered
  alongside that — but that would be a new feature, not a parity port.
- **Torch-specific GPU init** (`tests/core/test_gpu.py`). JAX
  manages its own device detection via `jax.default_backend()`;
  `pawn.logging._query_jax_memory_stats` (parity #7) is the v2
  analogue for process-side memory reporting.
- **Torch DataLoader / shard worker** (`tests/model/test_shard_loader.py`,
  `tests/eval/test_worker.py`). The v2 data path is `jax.lax.scan`
  over a corpus indexed in NumPy/Polars — no multi-process loader
  ceremony.
- **`PAWNCLM` class name + torch parameter inspection**. The single
  v2 model class is `pawn.model.PAWNModel`; `pawn.__getattr__`
  (parity #3) routes the old name through to a clear ImportError so
  stale imports fail loudly rather than picking up a near-namesake.

  ### Multi-GPU sweep trial pinning (`n_gpus` / `CUDA_VISIBLE_DEVICES`)
  **Section:** sweeps (workstream revision; uncommitted)
  **Reason:** nonsensical under the v2 execution model. v1's
  `AdapterObjective` took an `n_gpus` constructor param and, when
  `n_gpus > 1`, pinned each trial's training subprocess to a distinct
  device by setting `env["CUDA_VISIBLE_DEVICES"] = str(trial.number %
  n_gpus)` before `subprocess.run` (`git show
  main:pawn/sweep.py:392-396`). That round-robin presupposes (a) a
  multi-GPU host and (b) NVIDIA/CUDA device selection via
  `CUDA_VISIBLE_DEVICES`. v2 is a single-device JAX/Equinox stack on a
  ROCm (AMD) box: there is exactly one `RocmDevice(id=0)` to assign, so a
  `trial.number % n_gpus` partition collapses to a single device and the
  env var is the wrong knob anyway (ROCm honours
  `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`, not
  `CUDA_VISIBLE_DEVICES`). Pinning N trials to N devices is a
  parallelism feature, not a behavioral-parity one — and there is no
  second device to parallelise across. The `InProcessRoSAObjective`
  path is doubly inapplicable: it runs every trial in **one** process
  sharing a single backbone on a single device, so per-trial device
  affinity is meaningless there.
  **What was supposed to happen:** v1 spread subprocess trials across a
  multi-GPU host by env-var device pinning.
  **What I did instead:** `AdapterObjective` runs trials sequentially on
  the single resolved JAX device; it carries no `n_gpus` field and sets
  no `*_VISIBLE_DEVICES` env var. Trials still isolate cleanly via
  separate processes + per-trial log dirs; they just don't fan out
  across devices that don't exist.
  **To revisit when:** v2 is deployed on a genuine multi-accelerator
  host AND a parallel-trial sweep is wanted. The reinstated knob would
  set `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` (or
  `jax.distributed` device assignment), not `CUDA_VISIBLE_DEVICES` — a
  new device-fan-out feature, not a verbatim v1 port.

## Explicit follow-ups (tracked, time-bounded)

These are intentional follow-ups — outside the parity scope as
defined but not gone-by-design:

- **Live HF push verification.** `tests/test_jax_lifecycle.py`
  covers every load-bearing branch of `HFPushTracker` and
  `_DaemonThreadPoolExecutor` with a mocked `HfApi` (parity #4).
  Actually pushing bytes to an HF repo requires a scratch repo +
  `HF_TOKEN`; deferred to release-time live-verify rather than
  included in the CI test sweep.

## Pretrain CLI flags that have no honouring code in the v2 loop
**Section:** config-cli (workstream revision; uncommitted)
**Reason:** detrimental — promoting these to direct argparse flags would
advertise knobs that silently no-op on the pretrain path, which is worse
than an honest argparse error. They split into two groups:

1. **Validation / patience / pause** — `--patience`, `--eval-interval`,
   `--val-games`, `--pause-after-steps`. The v2 backbone-pretrain loop
   (`scripts/train_jax.py` main loop + `pawn.trainer`) has NO held-out
   validation pass, NO early-stop/patience break, and NO pause primitive.
   `docs/V2_PARITY_AUDIT.md` §2 lists "No backbone-pretrain validation loop
   … no early-stopping/patience" as an open **major** gap, separate from the
   config/CLI workstream. v1's `run_pretrain` wired all three
   (`git show main:scripts/train.py:232,267,274`). A user passing
   `--patience 5 --eval-interval 200` would get no val records and no early
   stop.

2. **Lichess-path / unconsumed fields** — `--min-ply`, `--cache-dir`,
   `--max-corpus-gb`. `min_ply` and `cache_dir` were Lichess-path knobs in
   v1: they fed `prepare_lichess_cached`
   (`git show main:scripts/train.py:396-412`), NOT the random-game pretrain
   corpus. v2's pretrain corpus comes from `pawn.corpus.generate_corpus`
   (random self-play via the Rust engine), whose signature has no
   `min_ply`/`cache_dir` parameter, so `--min-ply 20` would silently no-op
   (unfiltered games). `max_corpus_gb` is a v2-only soft resident-memory cap
   that currently has NO consumer in either the pretrain or the Lichess path
   (no reader in `pawn/trainer.py` or `pawn/corpus.py`), so `--max-corpus-gb 4`
   would not cap anything (possible OOM with no error).

**What was supposed to happen:** v1 exposed every `BaseRunConfig` field as
a `--flag value` CLI arg, including all of these pretrain knobs.
**What I did instead:** none of these seven knobs are exposed as direct
pretrain CLI flags. They remain valid `PretrainConfig` fields settable via
`--config` JSON (so a verbatim v1 config still loads — pinned by
`tests/scripts/test_train_jax_smoke.py::test_train_jax_inert_v1_pretrain_knobs_still_settable_via_config_json`),
and argparse rejects them as flags (pinned by
`test_train_jax_inert_v1_pretrain_knobs_not_exposed_as_cli_flags`). The
pretrain knobs that the loop DOES honour ARE exposed as flags — `--log-interval`,
`--checkpoint-interval`, the LR-schedule knobs, and `--mate-boost` /
`--discard-ply-limit` (both fed straight into `generate_corpus`); the
`--mate-boost` wiring is pinned *behaviourally* (not just "lands on config")
by `test_train_jax_mate_boost_cli_flag_is_honoured_end_to_end`, which drives
`main()` and asserts the CLI value reaches the `generate_corpus` call site.
**To revisit when:** group 1 — the audited "backbone-pretrain validation
loop + patience/early-stop" gap is implemented in the pretrain path; promote
those four flags in the same change that adds `log_val` + the patience break
+ the pause primitive. Group 2 — `max_corpus_gb` gets a real consumer (a
corpus-size cap in the prefetch path); `min_ply`/`cache_dir` are inapplicable
to random-game pretrain and stay JSON-only for v1-config load compatibility.

## Resolved follow-ups

The original three items here landed in the parity sweep's follow-up
push (commits on `jax_migration` after `3a0f8e2`):

- **KV-cached generation decoder.** :class:`pawn.model.KVCache` +
  :meth:`PAWNModel.forward_with_cache` decode one position at a time
  by writing fresh K/V slices into a pre-allocated cache and attending
  over the populated window. :func:`pawn.generation.autoregressive_generate`
  auto-detects the cached path (`use_kv_cache=None` → enabled
  whenever the model exposes ``forward_with_cache``; bare
  :class:`PAWNModel` and :class:`BottleneckEffective` both qualify),
  and the parity test
  `tests/test_jax_eval.py::test_autoregressive_generate_kv_cache_matches_full_forward`
  pins bit-identical sequences across the cached and non-cached
  paths. Falls back to the plain forward when ``use_kv_cache=False``
  (for parity testing) or when the model lacks the method.
- **Bottleneck adapter resume across re-init.**
  :func:`pawn.adapters.bottleneck.save_bottleneck_adapter` /
  :func:`load_bottleneck_adapter` round-trip the Houlsby weights
  alongside the backbone in the same checkpoint dir.
  `scripts/train_jax_adapter.py --resume <ckpt_dir>` auto-detects the
  ``adapter.safetensors`` sidecar and re-composes the
  :class:`BottleneckEffective` wrapper, splicing the saved step + Adam
  moments. RoSA + standard adapter resume works the same way (no
  sidecar needed for weight-folded adapters).
- **SDPA fast-path on ROCm.**
  :func:`jax.nn.dot_product_attention` is wired through
  :meth:`PAWNModel.__call__` (parity #43), threaded into the trainer
  via :data:`BaseRunConfig.use_sdpa`, and exposed on both pretrain
  + adapter CLIs as ``--use-sdpa``. Off by default because the gain
  is hardware-dependent: on RDNA 3 the fused kernel OOMs at training
  shapes (B≥2 T=512 wants 128 KB shared memory; the GPU has 64 KB
  per CU). On data-center GPUs and inference shapes the same flag
  produces real speedups without parity loss — the correctness guard
  in `tests/test_jax_model.py::test_use_sdpa_matches_plain_attention_within_fp32_noise`
  pins the bit-identical (within fp32 noise) invariant.

## Acceptance-criteria status

The §3 acceptance criteria were re-audited in `docs/V2_PARITY_AUDIT.md`
(the gated multi-agent audit that superseded the earlier, now-deleted
`docs/JAX_PARITY_SHORTFALLS.md`). Every parity/correctness gap that audit
confirmed has since been implemented and verified (the `fix(v2-parity/*)`
commit series); see that document for the per-item evidence.

Two criteria are **not** automated green checks:

- **§4 / §5 / §20 (legacy-checkpoint conversion)** — superseded and
  gone-by-design; see the "Legacy v1→JAX checkpoint converter" entry under
  *Gone-by-design* above. These are the only §3 criteria with no v2
  implementation, by deliberate post-redesign decision.
- **§18 (live HF push)** — operator-discretion rather than automated: unit
  tests cover the wiring and the per-run `run/{slug}` branch logic; the live
  push itself needs a scratch repo + token.
