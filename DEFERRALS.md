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

- **Cotrain config + entry point** (`pawn.run_config.CotrainConfig`,
  `scripts/train_cotrain.py`, `tests/training/test_cotrain_*`). The
  supernet's joint loss (`pawn.trainer.supernet_joint_loss`)
  replaces multi-variant cotraining; pretrain the supernet, then
  slice the three variants at publish time. Plan §6 documents the
  removal.
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

## Explicit follow-ups (tracked, time-bounded)

These are intentional follow-ups — outside the parity scope as
defined but not gone-by-design:

- **Live HF push verification.** `tests/test_jax_lifecycle.py`
  covers every load-bearing branch of `HFPushTracker` and
  `_DaemonThreadPoolExecutor` with a mocked `HfApi` (parity #4).
  Actually pushing bytes to an HF repo requires a scratch repo +
  `HF_TOKEN`; deferred to release-time live-verify rather than
  included in the CI test sweep.

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

Every §3 acceptance-criterion gap from
`docs/JAX_PARITY_SHORTFALLS.md` §2 has been re-verified with real
runs in `final_smoke.md` (parity #4). The one criterion that's
operator-discretion rather than automated is §18 (live HF push) —
unit tests cover the wiring; the live push itself needs a scratch
repo + token. See `final_smoke.md` §18 for the explicit honest
write-up.
