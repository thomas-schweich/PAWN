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

- **KV-cached generation decoder.** `pawn.generation.autoregressive_generate`
  runs the model's full forward pass per decode step (parity #6).
  The diagnostics work end-to-end at small `n_per_outcome` (default
  16-32); production runs at v1's `n_per_outcome=1000` would benefit
  from a `forward_generate` path on `PAWNModel`. Tracked because the
  v2 model intentionally uses plain attention (no fused SDPA) and
  the KV-cache work is most naturally bundled with the SDPA
  exploration (parity #43).
- **Bottleneck adapter resume across re-init.** Bottleneck-style
  adapters save the trained Houlsby weights as
  `adapter.safetensors` alongside the backbone checkpoint (parity
  #5). The trainer's `--resume` path currently restores the
  PAWNModel + optimizer; re-composing the `BottleneckEffective`
  wrapper on resume is wired for `pawn.trainer` but not yet for
  `scripts/train_jax_adapter.py --resume`. RoSA + standard adapter
  resume works without this.
- **Live HF push verification.** `tests/test_jax_lifecycle.py`
  covers every load-bearing branch of `HFPushTracker` and
  `_DaemonThreadPoolExecutor` with a mocked `HfApi` (parity #4).
  Actually pushing bytes to an HF repo requires a scratch repo +
  `HF_TOKEN`; deferred to release-time live-verify rather than
  included in the CI test sweep.
- **SDPA fast-path on ROCm.** `jax.nn.dot_product_attention` is
  reportedly working on recent ROCm + jaxlib. Tracked as parity #43
  in this document and `docs/JAX_PARITY_SHORTFALLS.md`. The
  migration plan §5 marked it "out of scope" for the framework
  swap; the bf16 perf win shipped in parity #2 (2.85× v1 baseline)
  reached the throughput bar without it, so it's a future
  enhancement rather than a parity gap.

## Acceptance-criteria status

Every §3 acceptance-criterion gap from
`docs/JAX_PARITY_SHORTFALLS.md` §2 has been re-verified with real
runs in `final_smoke.md` (parity #4). The one criterion that's
operator-discretion rather than automated is §18 (live HF push) —
unit tests cover the wiring; the live push itself needs a scratch
repo + token. See `final_smoke.md` §18 for the explicit honest
write-up.
