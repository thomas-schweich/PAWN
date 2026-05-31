#!/usr/bin/env python3
"""Supernet pretraining entry point — JAX/Optax + lax.scan K-step loop.

Acceptance criterion 6 verification:
    uv run --extra rocm python scripts/train_jax.py \
        --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 \
        --k 50 --local-checkpoints

Reads a JSON run config (validated through `PretrainConfig`), drives
the trainer in `pawn.trainer`, and writes checkpoints via
`pawn.checkpoint`. Wires the SIGTERM / HF-push / `--resume` lifecycle
from `pawn.lifecycle`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from pawn.checkpoint import save_model
from pawn.config import (
    CONDITIONING_KINDS,
    PRETRAIN_BUCKETS,
    SUPERNET,
    TINY_SUPERNET,
    VARIANTS,
    TINY_VARIANTS,
)
from pawn.corpus import Corpus, generate_corpus
from pawn.eval import compute_val_metrics
from pawn.jax_setup import require_accelerator, resolve_device, setup_jax_caching
from pawn.lifecycle import (
    HFPushTracker,
    build_training_state,
    drain_push_queue,
    install_sigterm_handler,
    load_resume_state,
    push_checkpoint_async,
    read_resume_data_anchor,
    write_schedule_health,
)
from pawn.logging import MetricsLogger, get_git_info
from pawn.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    require_wandb_available,
)
from pawn.model import PAWNModel, init_model, sliced
from pawn.run_config import PretrainConfig
from pawn.trainer import (
    Batch,
    TrainState,
    VariantSpec,
    flatten_opt_state,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
    top1_accuracy,
)


# H7 / D2 — base seed for the pretrain data stream. Outer-chunk seeds are
# derived deterministically from `(BASE_DATA_SEED, chunk_index)` so the seed
# for a given chunk is independent of when the prefetcher submitted it. Fixed
# (not configurable) to match the prior seed-0 contract; the per-chunk
# randomness comes from the chunk index, not a runtime seed field.
BASE_DATA_SEED: int = 0


def _derive_chunk_seed(base_seed: int, chunk_index: int) -> int:
    """Deterministic engine seed for outer-chunk ``chunk_index``.

    A pure function of ``(base_seed, chunk_index)`` — the data-stream RNG is
    NOT a stateful generator with prefetcher look-ahead (which silently
    skipped the in-flight chunk on resume; H7 / D2 review finding). Using a
    SHA-256 digest of the two integers decorrelates adjacent chunk seeds (so
    the engine's per-chunk games look independent) while keeping the seed at
    a given index reproducible regardless of prefetch timing. The result is
    masked to the engine's accepted ``[0, 2**31 - 1)`` seed range.
    """
    import hashlib

    payload = f"{int(base_seed)}:{int(chunk_index)}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train_jax")
    ap.add_argument("--config", type=Path, default=None, help="JSON run config")
    ap.add_argument("--supernet", choices=("tiny", "production"), default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--accumulation-steps", type=int, default=None,
                    help="(B2) micro-batches accumulated per optimizer step. "
                         ">1 emits (K, N, B, T) batches so the trainer's "
                         "accumulation scan sums N micro-grads before each "
                         "update — effective batch N×B at B's per-step memory "
                         "cost. Default 1 (no accumulation).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--conditioning", nargs="*", default=None,
                    choices=sorted(CONDITIONING_KINDS),
                    help="ordered control-token kinds to prepend after BOS "
                         "(e.g. `outcome`). The sequence layout is "
                         "[BOS][cond...][ply...][PAD...]; prefix width is "
                         "C = 1 + len(conditioning). Default is BOS-only "
                         "(C=1). Persisted to config.json so eval rebuilds "
                         "the corpus at the same C.")
    # v1 CLI-compat: `--prepend-outcome` is the legacy boolean that
    # `BaseRunConfig._migrate_prepend_outcome` folds into
    # `conditioning=["outcome"]`. v1 users typed it on the CLI (not just
    # in JSON), so expose it here. Mutually exclusive with `--conditioning`
    # (the pydantic before-validator rejects setting both).
    ap.add_argument("--prepend-outcome", dest="prepend_outcome",
                    action="store_true", default=None,
                    help="(v1-compat) shorthand for `--conditioning outcome`; "
                         "prepends the outcome control token after BOS. "
                         "Mutually exclusive with --conditioning.")
    ap.add_argument("--amp-dtype",
                    choices=("bfloat16", "float16", "float32", "none"),
                    default=None,
                    help="mixed-precision forward compute dtype (master "
                         "weights + Adam moments stay fp32). Default "
                         "bfloat16. `none` is the v1 spelling of `float32` "
                         "(no mixed precision) and is migrated for you.")
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="(v1-compat) alias for --seq-len; the v1 run-config "
                         "field name. Migrated to seq_len with a deprecation "
                         "warning.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", default=None)
    ap.add_argument("--warmup-frac", type=float, default=None)
    # v1 CLI-compat: the remaining LR-schedule knobs + core training
    # hyperparameters were reachable only via --config JSON in v2 (v1's
    # generic `--flag value` parser exposed every BaseRunConfig field).
    # Expose the load-bearing ones as direct flags.
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="explicit warmup length in steps (overrides "
                         "--warmup-frac when set).")
    ap.add_argument("--decay-frac", type=float, default=None,
                    help="fraction of total_steps in the final decay phase "
                         "(WSD / infinite schedules).")
    ap.add_argument("--cooldown-frac", type=float, default=None,
                    help="cooldown-phase fraction for the `infinite` "
                         "schedule.")
    ap.add_argument("--stable-lr-ratio", type=float, default=None,
                    help="stable-plateau LR as a fraction of peak LR "
                         "(`infinite` schedule only).")
    ap.add_argument("--wsd-decay-shape", choices=("linear", "cosine"),
                    default=None,
                    help="decay-phase curve for WSD / infinite schedules.")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="global-norm gradient clip threshold (default 1.0).")
    # Held-out validation loop + early-stop/pause primitives (v1 parity —
    # `git show main:pawn/trainer.py` validation/early-stop block). The
    # pretrain loop now runs a periodic held-out eval (fresh random games)
    # and emits `type=val` records, so these knobs are honoured.
    ap.add_argument("--val-every", type=int, default=None,
                    help="run the held-out validation pass + emit type=val "
                         "records every N steps. Omit to disable held-out "
                         "eval (cheap loss-curve-only mode). v1 spelled this "
                         "--eval-interval in pretrain; that name still loads "
                         "via --config JSON.")
    ap.add_argument("--val-games", type=int, default=None,
                    help="number of freshly-generated random games in the "
                         "held-out validation corpus (default 512).")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop after N consecutive validation passes "
                         "with no improvement in best val loss / late-game "
                         "legality. Requires --val-every. Records "
                         "reason_for_stop=patience in schedule_health.json.")
    ap.add_argument("--pause-after-steps", type=int, default=None,
                    help="checkpoint and pause training at this step boundary "
                         "(reason_for_stop=paused). Resume with --resume.")
    # NOTE: no `--min-ply` / `--max-corpus-gb` / `--cache-dir` flags either.
    # `min_ply` and `cache_dir` are Lichess-path fields — in v1 they fed
    # `prepare_lichess_cached` (`git show main:scripts/train.py:396-412`),
    # NOT the random-game pretrain corpus. v2's pretrain corpus comes from
    # `generate_corpus()` (random self-play via the Rust engine), whose
    # signature has no `min_ply`/`cache_dir` parameter, so promoting these
    # to pretrain flags would let `--min-ply 20` silently no-op. `max_corpus_gb`
    # is a v2-only soft resident-memory cap that currently has NO consumer in
    # either the pretrain or the Lichess path (no reader in pawn/trainer.py or
    # pawn/corpus.py), so it is likewise not advertised as a flag. All three
    # remain settable via `--config` JSON. By contrast `--mate-boost` and
    # `--discard-ply-limit` below ARE honoured (passed straight into
    # `generate_corpus`), so they stay as flags.
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--mate-boost", type=float, default=None,
                    help="bias the random-game engine toward mating lines "
                         "(0 = uniform).")
    ap.add_argument("--discard-ply-limit", action="store_true", default=None,
                    help="discard games that hit the ply limit instead of "
                         "truncating them.")
    ap.add_argument("--checkpoint-interval", type=int, default=None)
    # IO
    ap.add_argument("--local-checkpoints", action="store_true")
    ap.add_argument("--hf-repo", default=None)
    # NOTE: no `--hf-bucket` flag. v1's bucket-autosave target is not
    # wired in v2's JAX trainer (no bucket-push primitive), so exposing
    # it would advertise a destination that silently drops every
    # checkpoint. `BaseRunConfig._check_checkpoint_mode` rejects any
    # config that names `hf_bucket`. Tracking: docs/V2_PARITY_AUDIT.md.
    # (No `--cache-dir` flag — Lichess-path-only, unread by pretrain; see
    # the not-honoured-knobs NOTE above.)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--use-sdpa", action="store_true",
                    help="opt the attention block into "
                         "jax.nn.dot_product_attention (parity #43). "
                         "Off by default — superseded by --use-flash on GPU.")
    ap.add_argument("--no-flash", action="store_true",
                    help="disable the Pallas flash-attention kernel "
                         "(force the plain materialised QK^T path). "
                         "Flash is the default on GPU; CPU runs "
                         "auto-fall-back regardless of this flag.")
    # Stochastic supernet sampling is **on by default** (H.1 housekeeping).
    # Use --no-stochastic-variants to force the exhaustive 3-variant sum
    # (for parity tests / debug). --stochastic-variants is accepted as a
    # no-op so existing JSON configs and shell scripts that set it
    # explicitly still parse.
    ap.add_argument("--stochastic-variants", dest="stochastic_variants",
                    action="store_true", default=None,
                    help="(default) sandwich-sample one non-supernet variant "
                         "per step; supernet always runs. ~10%% throughput "
                         "improvement at production 3-variant loss.")
    ap.add_argument("--no-stochastic-variants", dest="stochastic_variants",
                    action="store_false",
                    help="force the deterministic exhaustive sum over all "
                         "variants (parity / debug only).")
    ap.add_argument("--no-bucketing", action="store_true",
                    help="disable A.1 length bucketing — pretrain at a "
                         "single seq_len bucket. Slower but useful for "
                         "parity / debug.")
    ap.add_argument("--bucket-outer-factor", type=int, default=3,
                    help="how many K-batches' worth of games to generate "
                         "per outer prefetch call (default 3 — covers the "
                         "natural bucket distribution).")
    ap.add_argument("--emit-grad-norms", action="store_true",
                    help="(no-op, retained for back-compat) per-step pre-clip "
                         "grad_norm is now logged unconditionally (v1 parity), "
                         "so this flag has no effect.")
    ap.add_argument("--optimizer", choices=("adamw", "lion"), default=None,
                    help="(C.1) optimizer to use. Lion halves opt-state "
                         "memory + ~3-5%% step time but needs LR ~1/3 of "
                         "AdamW's. Default is adamw.")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of {small,base,large} to train jointly "
                         "(default: all three = supernet joint loss). e.g. "
                         "`--variants large` trains only the full model — a "
                         "standalone-large teacher pretrain for the "
                         "distillation-canonical ladder (plan §7).")
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace) -> PretrainConfig:
    """Build a PretrainConfig from --config + CLI overrides."""
    base: dict = {}
    if args.config is not None:
        base = json.loads(args.config.read_text())
        base.setdefault("run_type", "pretrain")
    # CLI flags override config-file values.
    for flag, val in (
        ("supernet", args.supernet),
        ("total_steps", args.total_steps),
        ("accumulation_steps", args.accumulation_steps),
        ("batch_size", args.batch_size),
        ("seq_len", args.seq_len),
        ("k", args.k),
        ("lr", args.lr),
        ("lr_schedule", args.lr_schedule),
        ("warmup_frac", args.warmup_frac),
        ("warmup_steps", args.warmup_steps),
        ("decay_frac", args.decay_frac),
        ("cooldown_frac", args.cooldown_frac),
        ("stable_lr_ratio", args.stable_lr_ratio),
        ("wsd_decay_shape", args.wsd_decay_shape),
        ("weight_decay", args.weight_decay),
        ("max_grad_norm", args.max_grad_norm),
        # Validation loop + early-stop/pause knobs — now honoured by the
        # pretrain loop (held-out eval + patience break + pause boundary).
        # `min_ply` / `max_corpus_gb` / `cache_dir` remain --config-JSON-only
        # (no consumer in the random-game pretrain path).
        ("val_every", args.val_every),
        ("val_games", args.val_games),
        ("patience", args.patience),
        ("pause_after_steps", args.pause_after_steps),
        ("log_interval", args.log_interval),
        ("mate_boost", args.mate_boost),
        ("checkpoint_interval", args.checkpoint_interval),
        ("hf_repo", args.hf_repo),
        ("wandb_project", args.wandb_project),
        ("resume", str(args.resume) if args.resume else None),
        # `--conditioning` with nargs="*" yields a list when passed
        # (possibly empty for `--conditioning` with no args) and None
        # when omitted, so the `is not None` guard below distinguishes
        # "explicitly set to []" from "not passed".
        ("conditioning", args.conditioning),
        # `--variants` (nargs="*") -> list when passed, None when omitted;
        # convert to tuple. A bare `--variants` (empty list) -> () which the
        # PretrainConfig validator rejects with a clear message.
        ("variants", tuple(args.variants) if args.variants is not None else None),
    ):
        if val is not None:
            base[flag] = val
    # `store_true` flags: only merge when actually set so a CLI omit
    # doesn't override a JSON config that has the flag True (round-1
    # codex P2: `--use-sdpa` and `--wandb` defaulted to False, were
    # `not None`, and silently overrode the config). Mirrors the
    # `local_checkpoints` handling below.
    if args.use_sdpa:
        base["use_sdpa"] = True
    if args.optimizer is not None:
        base["optimizer"] = args.optimizer
    # `--amp-dtype none` is the v1 spelling of float32; the
    # `_migrate_amp_dtype_none` before-validator rewrites it, so pass the
    # raw value straight through.
    if args.amp_dtype is not None:
        base["amp_dtype"] = args.amp_dtype
    # `--prepend-outcome` and `--max-seq-len` are v1-compat aliases; merge
    # them as the legacy keys so `BaseRunConfig`'s before-validators run
    # the migration (and reject conflicting `--conditioning` / `--seq-len`
    # with a clear message).
    if args.prepend_outcome:
        base["prepend_outcome"] = True
    if args.max_seq_len is not None:
        base["max_seq_len"] = args.max_seq_len
    if args.discard_ply_limit:
        base["discard_ply_limit"] = True
    # `stochastic_variants` default in argparse is None — only override
    # the config value when the user explicitly passed --stochastic-variants
    # or --no-stochastic-variants on the CLI.
    if args.stochastic_variants is not None:
        base["stochastic_variants"] = args.stochastic_variants
    if args.no_flash:
        base["use_flash"] = False
    if args.wandb:
        base["wandb"] = True
    if args.local_checkpoints:
        base["local_checkpoints"] = True
    base.setdefault("run_type", "pretrain")
    return PretrainConfig(**base)


def build_variants(cfg: PretrainConfig) -> tuple[VariantSpec, ...]:
    """The variants to train jointly for this run.

    ``cfg.variants`` selects a subset of the supernet's nested variants;
    ``None`` (the default) trains all three (small/base/large) as the
    supernet joint loss. ``("large",)`` trains only the full model — a
    standalone-large teacher pretrain for the distillation-canonical
    ladder (plan §7). ``is_supernet`` is True only for ``"large"`` so the
    full model goes through the forward unsliced; any other selected
    variant is a width-slice of the supernet.
    """
    variants_dict = TINY_VARIANTS if cfg.supernet == "tiny" else VARIANTS
    selected = cfg.variants if cfg.variants is not None else ("small", "base", "large")
    return tuple(
        VariantSpec(name, variants_dict[name], is_supernet=(name == "large"))
        for name in selected
    )


def widest_trained_variant(variants: tuple[VariantSpec, ...]) -> VariantSpec:
    """The widest variant actually trained this run (max ``d_model``).

    The supernet joint loss only updates the inner ``[:d_V, :d_V]`` slice of
    each selected variant (``--variants``, caf6a53), so the widest TRAINED
    width is the largest ``d_model`` among ``variants`` — *not* necessarily
    the full supernet. ``train/accuracy`` is measured on this variant so the
    metric never reads untrained outer dims (see :func:`accuracy_model`).
    """
    return max(variants, key=lambda s: s.cfg.d_model)


def accuracy_model(model: PAWNModel, widest: VariantSpec) -> PAWNModel:
    """The model the ``train/accuracy`` forward runs on.

    When ``widest`` is the supernet itself (``is_supernet``) the full model
    *is* the variant, so the slice would be a no-op and we return ``model``
    unchanged (default small+base+large path — identical to before). For a
    ``--variants`` subset that excludes ``large`` the outer
    ``[d_widest:d_supernet]`` dims stay at init, so we ``sliced`` down to the
    widest trained width; the forward then never mixes trained inner dims
    with untrained outer dims (which would report near-random accuracy for a
    healthy run).
    """
    if widest.is_supernet:
        return model
    return sliced(model, widest.cfg)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    # `_build_config` raises a pydantic ValueError if --total-steps is
    # missing — no per-field runtime check needed here.
    cfg = _build_config(args)
    require_accelerator()
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

    # Pick supernet shape.
    supernet_cfg = TINY_SUPERNET if cfg.supernet == "tiny" else SUPERNET
    variants = build_variants(cfg)

    # `PretrainConfig._check_pretrain` validates that `total_steps` is
    # not None — assert it for pyright (the model_validator constraint
    # doesn't narrow the optional type at the field level).
    assert cfg.total_steps is not None
    total_steps: int = cfg.total_steps

    # Optimiser + state.
    schedule = make_lr_schedule(cfg, total_steps)
    optimizer = make_optimizer(cfg, schedule)
    if cfg.resume:
        state = load_resume_state(
            Path(cfg.resume), optimizer, jax.random.key(0),
            conditioning=cfg.conditioning,
        )
    else:
        import equinox as eqx
        # PretrainConfig doesn't carry a runtime seed field — the supernet
        # init key is fixed at 0 for reproducibility. The training stream's
        # randomness comes from the Rust engine's per-batch seed, not from
        # the model-init key.
        model = init_model(supernet_cfg, key=0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        state = TrainState(
            model=model, opt_state=opt_state, step=jnp.int32(0),
            key=jax.random.key(0),
        )

    # Resolve cfg.amp_dtype → jnp dtype. None ⇒ fp32 forward (back-
    # compat with existing tests that synthesise opt_state in fp32 and
    # parity-test the legacy converter bit-exact). The default is
    # bf16 per plan §5 + v1 parity.
    _DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": None,
    }
    compute_dtype = _DTYPE_MAP[cfg.amp_dtype]
    # Pallas flash requires the GPU backend; CPU smoke runs
    # (`PAWN_ALLOW_CPU=1`) auto-fall-back to the plain path regardless
    # of `cfg.use_flash`.
    use_flash = cfg.use_flash and jax.default_backend() == "gpu"
    # B2 / plan §8.3: gradient accumulation is now wired through the data
    # loop. With ``accumulation_steps == N > 1`` the bucketed prefetcher
    # emits batches with an extra leading microbatch axis — each scan
    # element is ``(N, B, T)`` so the full per-chunk tensor is
    # ``(K, N, B, T)`` — and ``make_train_step`` scans the N axis,
    # summing micro-grads before a single optimizer update. ``N == 1``
    # keeps the legacy ``(K, B, T)`` shape (no extra axis).
    accumulation_steps = cfg.accumulation_steps
    train_step = make_train_step(
        optimizer, variants,
        compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        stochastic_variants=cfg.stochastic_variants,
        accumulation_steps=cfg.accumulation_steps,
    )
    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix="pretrain", device=resolve_device()
    )
    logger.log_config(run_type="pretrain", model=cfg.model_dump())

    # W&B mirror — gated on `--wandb` (cfg.wandb) *and* the `wandb` extra.
    # `--wandb` without the extra is a hard error (no silent metric drop).
    wandb_run = None
    if cfg.wandb:
        require_wandb_available()
        wandb_run = init_wandb(
            project=cfg.wandb_project, slug=logger.slug,
            run_config=cfg.model_dump(),
            git_hash=get_git_info().get("git_hash"),
            job_type="pretrain",
            run_dir_name=logger.run_dir.name,
        )

    # HF push tracker (optional).
    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None

    # SIGTERM handler — flips a flag the loop polls between chunks.
    should_shutdown = install_sigterm_handler()

    # K-step `lax.scan` body — plan §10 S6 + §4: "Whole training loop as
    # one compiled program — eliminate per-step launch and dispatch
    # overhead." The scan body never returns to the host inside a chunk;
    # per-chunk metrics flush between chunks.
    # Per-step top-1 accuracy on the widest TRAINED variant — the v2 parity
    # of v1's pretrain `train/accuracy` metric
    # (`git show main:pawn/trainer.py:1030,1145`). The supernet joint loss
    # only updates the inner `[:d_V, :d_V]` slice of whatever variants are
    # selected (`--variants`, caf6a53), so the *full* unsliced model is a
    # meaningful accuracy forward only when the largest variant (`large` =
    # the supernet itself) is in the selection. For any subset that excludes
    # it (e.g. `--variants small base`), the outer `[d_base:d_large]` weights
    # stay at init while the inner slices train fine; a full-model forward
    # would then mix trained inner dims with untrained outer dims and report
    # a misleadingly low accuracy for a healthy run. So we measure on the
    # widest selected variant: if it is the supernet, use `model` directly
    # (slice is a no-op); otherwise `sliced(model, spec.cfg)` first.
    #
    # It is stacked alongside the per-step losses inside the `lax.scan`, so
    # the chart-feeding metric costs one extra per-chunk D→H copy and no
    # extra host round trips. Restricting the argmax to the move-token
    # support (`mask_reserved_columns`) matches the eval contract's
    # `[0, NUM_ACTIONS)` restriction. With accumulation the scan element
    # carries a leading microbatch axis (`(N, B, T)`); we measure accuracy
    # on the first micro so the metric is well-defined without paying N
    # forwards.
    widest_variant = widest_trained_variant(variants)

    def _first_micro(batch: Batch) -> Batch:
        # In accumulation mode each scan element is `(N, B, T)` per leaf;
        # take micro 0 so the accuracy forward runs on a single `(B, T)`
        # batch. Equinox `Batch` is a PyTree, so a leaf-wise index works.
        return jax.tree_util.tree_map(lambda x: x[0], batch)

    def _supernet_accuracy(model: PAWNModel, batch: Batch) -> Float[Array, ""]:
        b = batch if accumulation_steps == 1 else _first_micro(batch)
        # `is_supernet` ⇒ full model is the variant (slice is a no-op);
        # otherwise slice down to the widest trained width so the forward
        # never reads untrained outer dims.
        return top1_accuracy(
            accuracy_model(model, widest_variant), b,
            compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        )

    # grad_norm is logged unconditionally — v1 emitted the per-step
    # grad_norm on every log record (`git show main:pawn/trainer.py`
    # `log_train(..., grad_norm=grad_norm, ...)`), not behind a flag. The
    # v2 `--emit-grad-norms` gate that previously suppressed it by default
    # diverged from that contract, so the scan now always returns the
    # pre-clip grad norm (one extra (K,) scalar per chunk, no extra host
    # round trips). `--emit-grad-norms` is kept as an accepted no-op flag
    # for shell-script / config back-compat.
    scan_step = make_scan_step(
        train_step,
        emit_grad_norms=True,
        accuracy_fn=_supernet_accuracy,
    )

    # H7 / D2 — bit-reproducible data stream. The outer-chunk seeds are a
    # pure function of `(BASE_DATA_SEED, chunk_index)` (see
    # `_derive_chunk_seed`), *not* a stateful numpy Generator with one-chunk
    # look-ahead. That severs the seed sequence from the prefetcher's eager
    # pre-submit so the seed at a given chunk index is identical regardless
    # of construction timing — the resume anchor is just the *consume*
    # position (chunk index + intra-chunk batch offset), restored below.
    resume_chunk = 0
    resume_skip = 0
    if cfg.resume:
        resume_chunk, resume_skip = read_resume_data_anchor(Path(cfg.resume))
    indices = np.arange(cfg.batch_size, dtype=np.int64)

    # A.1 — Length bucketing. Read the schedule from `PRETRAIN_BUCKETS`
    # (the single source of truth). The top edge must equal cfg.seq_len
    # (so every game fits in some bucket). `--no-bucketing` ships the
    # legacy single-bucket path for parity / debug.
    if args.no_bucketing:
        bucket_edges: tuple[int, ...] = (cfg.seq_len,)
    else:
        bucket_edges = tuple(e for e in PRETRAIN_BUCKETS if e <= cfg.seq_len)
        if not bucket_edges or bucket_edges[-1] != cfg.seq_len:
            # Add the top edge if not already present
            bucket_edges = (*bucket_edges, cfg.seq_len) if bucket_edges else (cfg.seq_len,)

    # A.4 — Async corpus prefetch + bucketed batch production. The
    # background thread runs the Rust `generate_corpus`, bucketises into
    # per-T sub-corpora, slices each into (K, B, T_bucket) Batches, and
    # uploads via `jnp.asarray` (which is async). The GPU runs scan_step
    # over the resulting queue while the next outer corpus is being
    # generated.
    def _build_batch_from_corpus_slice(
        corp: Corpus, k_inner: int, batch_size: int, seq_len: int,
        start_idx: int, accum: int,
    ) -> Batch:
        # With ``accum > 1`` the scan element gains a leading microbatch
        # axis, so each leaf is reshaped to ``(K, N, B, T)``; ``accum == 1``
        # keeps the legacy ``(K, B, T)`` shape (no extra axis, so the
        # accumulation-free scan body is unchanged).
        group = k_inner * accum * batch_size
        sel = slice(start_idx, start_idx + group)
        if accum == 1:
            shape: tuple[int, ...] = (k_inner, batch_size, seq_len)
        else:
            shape = (k_inner, accum, batch_size, seq_len)
        tokens = corp.tokens[sel, :seq_len].reshape(shape)
        targets = corp.targets[sel, :seq_len].reshape(shape)
        attn = corp.attn_mask[sel, :seq_len].reshape(shape)
        lmask = corp.loss_mask[sel, :seq_len].reshape(shape)
        return Batch(
            tokens=jnp.asarray(tokens), targets=jnp.asarray(targets),
            attn_mask=jnp.asarray(attn), loss_mask=jnp.asarray(lmask),
        )

    def _prepare_outer_chunk(
        n_games: int, seed: int, edges: tuple[int, ...],
        batch_size: int, inner_k: int, accum: int,
    ) -> list[tuple[int, Batch]]:
        """Generate `n_games` random games, bucketise, slice into
        K-batches per bucket, return as `[(edge, Batch), ...]` in
        ascending-edge order. Games that don't fill a complete K-batch
        in any bucket are dropped (acceptable when n_games >> B*K).

        With ``accum > 1`` each K-batch consumes ``inner_k * accum *
        batch_size`` games and is shaped ``(K, accum, B, T)`` so the
        trainer's accumulation scan can sum ``accum`` micro-grads per
        optimizer step."""
        corpus = generate_corpus(
            n_games=n_games, max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=seed,
            conditioning=cfg.conditioning,
            mate_boost=cfg.mate_boost,
            discard_ply_limit=cfg.discard_ply_limit,
        )
        if edges == (cfg.seq_len,):
            # Unbucketed path: one bucket at full seq_len.
            buckets = {cfg.seq_len: corpus}
        else:
            buckets = corpus.by_bucket(edges)
        out: list[tuple[int, Batch]] = []
        games_per_K = batch_size * inner_k * accum
        for edge in sorted(buckets):
            sub = buckets[edge]
            n_full_K = sub.n_games // games_per_K
            for j in range(n_full_K):
                start = j * games_per_K
                out.append((
                    edge,
                    _build_batch_from_corpus_slice(
                        sub, inner_k, batch_size, edge, start, accum,
                    ),
                ))
        return out

    class BucketedPrefetcher:
        """Yields `(edge, Batch)` tuples with one outer-chunk lookahead.

        Each `next()` returns the next ready batch. When the queue is
        empty we wait on the pending future and immediately submit the
        next outer-chunk for the executor to start producing. The host
        cost (Rust gen + numpy reshape + H2D enqueue) is overlapped with
        the GPU work consuming earlier batches.

        **Resumable, bit-reproducible data stream (H7 / D2).** The outer-
        chunk seeds are a *pure function* of ``(base_seed, chunk_index)``
        via :func:`_derive_chunk_seed`, **not** a stateful ``rng`` with
        look-ahead. This severs the data-stream RNG from the prefetcher's
        eager one-chunk look-ahead: the seed produced for outer-chunk ``i``
        is identical regardless of when (or whether) the prefetcher
        pre-submitted chunk ``i+1``. The resume anchor is therefore just
        the *consume* position — the index of the outer-chunk that owns the
        next batch to be popped, plus how many of that chunk's batches have
        already been consumed (a checkpoint can land mid-outer-chunk because
        one chunk yields several K-batches). On resume the prefetcher
        re-derives from that chunk index and skips the already-consumed
        leading batches, so the resumed batch sequence is bit-identical to
        the uninterrupted run's tail. The buggy predecessor persisted a
        look-ahead-advanced ``rng`` whose next draw skipped the in-flight
        (submitted-but-untrained) chunk on resume.
        """

        def __init__(self, resume_chunk: int = 0, resume_skip: int = 0) -> None:
            self._queue: deque[tuple[int, Batch]] = deque()
            self._pending: Future[list[tuple[int, Batch]]] | None = None
            self._closed = False
            self.outer_factor = max(1, args.bucket_outer_factor)
            # `_submit_index` is the index of the *next* outer-chunk to
            # submit; its seed is `_derive_chunk_seed(base_seed, idx)`.
            self._submit_index = resume_chunk
            # `_skip_remaining` drops the leading batches of the first
            # produced chunk that the interrupted run already consumed
            # before the checkpoint (mid-outer-chunk resume).
            self._skip_remaining = resume_skip
            # Consume cursor: the chunk index + intra-chunk batch offset of
            # the *next* batch `next()` will return. Persisted at checkpoint
            # time as the bit-reproducible resume anchor.
            self._consume_chunk = resume_chunk
            self._consume_offset = resume_skip
            self._submit_next()

        def _submit_next(self) -> None:
            if self._closed:
                return
            idx = self._submit_index
            seed = _derive_chunk_seed(BASE_DATA_SEED, idx)
            self._submit_index += 1
            # Each scan step consumes ``B * accumulation_steps`` games (the
            # micro-batches summed into one optimizer update), so scale the
            # outer-chunk size by ``accumulation_steps`` to keep the same
            # ~``outer_factor`` K-batches of lookahead.
            n_games = (
                cfg.batch_size * cfg.k * accumulation_steps * self.outer_factor
            )
            self._pending = executor.submit(
                _prepare_outer_chunk, n_games, seed, bucket_edges,
                cfg.batch_size, cfg.k, accumulation_steps,
            )

        def next(self) -> tuple[int, Batch] | None:
            while not self._queue:
                if self._pending is None:
                    return None
                outer = self._pending.result()
                # Immediately queue the next outer-chunk so its host work
                # overlaps with the GPU work consuming this one.
                self._submit_next()
                # Drop the leading batches already consumed pre-checkpoint
                # (mid-outer-chunk resume). `_skip_remaining` is non-zero
                # only for the first produced chunk after a resume.
                if self._skip_remaining:
                    drop = min(self._skip_remaining, len(outer))
                    outer = outer[drop:]
                    self._skip_remaining -= drop
                self._queue.extend(outer)
                if not self._queue:
                    # Pathological: a generated corpus has fewer than B*K
                    # games in ANY bucket. Try again rather than spinning.
                    if self._pending is None:
                        return None
            batch = self._queue.popleft()
            # Advance the consume cursor. A boundary between outer-chunks is
            # crossed when the queue empties *and* a fresh chunk is fetched;
            # we detect that by tracking how many batches remain queued from
            # the current chunk. Simpler: re-derive the cursor from the
            # number of batches consumed so far is brittle under bucket-
            # dependent chunk sizes, so track per-batch via `_advance_cursor`.
            self._advance_cursor()
            return batch

        def _advance_cursor(self) -> None:
            """Move the consume cursor forward by one batch.

            The cursor names the *next* batch to return: ``(chunk_index,
            offset_within_chunk)``. When a chunk is exhausted the offset
            wraps to 0 and the chunk index advances. ``_queue`` plus
            ``_skip_remaining`` is empty exactly when the current chunk has
            no more batches queued, so the next pop belongs to the next
            outer-chunk.
            """
            self._consume_offset += 1
            if not self._queue:
                # Just emptied the current chunk's queued batches → the next
                # batch starts the following outer-chunk.
                self._consume_chunk += 1
                self._consume_offset = 0

        def resume_anchor(self) -> tuple[int, int]:
            """Return ``(chunk_index, batches_consumed_in_chunk)`` for the
            *next* batch to be consumed — the bit-reproducible resume anchor
            persisted in ``training_state.json``."""
            return self._consume_chunk, self._consume_offset

        def close(self) -> None:
            self._closed = True
            if self._pending is not None:
                self._pending.cancel()
            self._pending = None
            self._queue.clear()

    def _data_anchor_block(prefetcher: "BucketedPrefetcher") -> dict[str, int]:
        """Serialise the prefetcher's consume anchor for the checkpoint.

        ``{"base_seed", "chunk_index", "batch_offset"}`` is everything a
        resumed run needs to re-derive the data stream from the exact batch
        the interrupted run was about to consume (H7 / D2)."""
        chunk_index, batch_offset = prefetcher.resume_anchor()
        return {
            "base_seed": BASE_DATA_SEED,
            "chunk_index": chunk_index,
            "batch_offset": batch_offset,
        }

    def _save_checkpoint(step_int: int) -> None:
        out = logger.run_dir / f"step_{step_int:08d}"
        if out.exists():
            return
        # Flatten the Optax PyTree to a name → ndarray dict so
        # save_model can drop it into `optimizer.safetensors`. The
        # resume path uses the matching `unflatten_opt_state` to rebuild
        # the PyTree against a freshly-initialised template — that's
        # what keeps Adam's first/second moment estimates + the clip
        # counter across the resume boundary.
        opt_tensors = flatten_opt_state(state.opt_state)
        save_model(
            state.model, out,
            run_config=cfg.model_dump(),
            optimizer_state=opt_tensors,
            training_state=build_training_state(
                step=int(state.step),
                schedule=cfg.lr_schedule,
                lr_peak=cfg.lr,
                rng_key=state.key,
                # Persist the data-stream *consume anchor* — the outer-chunk
                # index + intra-chunk batch offset of the next batch the
                # prefetcher will hand out. Outer-chunk seeds derive purely
                # from `(BASE_DATA_SEED, chunk_index)`, so re-deriving from
                # this anchor on resume reproduces the exact tail of the
                # uninterrupted run's batch sequence (bit-reproducible
                # resume; H7 / D2). This replaces the old look-ahead-polluted
                # `numpy_rngs={"data": rng}`, which skipped the in-flight
                # (submitted-but-untrained) chunk on resume.
                extra={"data_anchor": _data_anchor_block(producer)},
            ),
        )
        if push_tracker:
            push_checkpoint_async(out, push_tracker)

    # --- Held-out validation loop (v1 CLMTrainer.evaluate parity) --------
    # The pretrain corpus is freshly-generated random self-play, so the
    # held-out val set is a fixed corpus of `cfg.val_games` games generated
    # off a dedicated seed (disjoint from the training stream's per-chunk
    # seeds, which derive from BASE_DATA_SEED=0). Built once up front so the
    # periodic eval pays only the forward cost, not regeneration. `None`
    # cfg.val_every disables the held-out pass entirely.
    VAL_DATA_SEED: int = 2**31 - 7  # disjoint from BASE_DATA_SEED's chunk seeds
    val_every = cfg.val_every
    val_corpus: Corpus | None = None
    if val_every is not None:
        val_corpus = generate_corpus(
            n_games=cfg.val_games, max_ply=cfg.seq_len, seq_len=cfg.seq_len,
            seed=VAL_DATA_SEED, conditioning=cfg.conditioning,
            mate_boost=cfg.mate_boost, discard_ply_limit=cfg.discard_ply_limit,
        )
    # legality late-ply threshold: explicit override, else seq_len // 2 (v1
    # `legality_late_ply` default).
    legality_late_ply = (
        cfg.legality_late_ply if cfg.legality_late_ply is not None
        else cfg.seq_len // 2
    )
    val_acc_model = accuracy_model
    # Compound early-stop state (v1: best val loss + best late-game
    # legality drive the patience counter; an improvement in *either*
    # resets it).
    best_val_loss = float("inf")
    best_late_legality = 0.0
    patience_counter = 0
    last_val_step = -1  # de-dupe: never eval the same step twice

    start = int(state.step)
    t0 = time.time()
    next_step = start

    def _run_validation(at_step: int) -> bool:
        """Run the held-out validation pass at ``at_step``, log a
        ``type=val`` record, update the patience state, and return True
        when the patience budget has been exhausted (caller should stop).

        No-op (returns False) when the held-out eval is disabled or the
        step was already evaluated.
        """
        nonlocal best_val_loss, best_late_legality, patience_counter
        nonlocal last_val_step
        if val_corpus is None or at_step == last_val_step:
            return False
        last_val_step = at_step
        vm = compute_val_metrics(
            val_acc_model(state.model, widest_variant), val_corpus,
            batch_size=cfg.batch_size, late_ply=legality_late_ply,
        )
        extra_log: dict[str, float | int] = {}
        stop = False
        if cfg.patience is not None:
            improved = False
            if vm.val_loss < best_val_loss:
                best_val_loss = vm.val_loss
                improved = True
            if vm.late_legal_move_rate > best_late_legality:
                best_late_legality = vm.late_legal_move_rate
                improved = True
            patience_counter = 0 if improved else patience_counter + 1
            extra_log = {
                "patience_counter": patience_counter,
                "best_val_loss": best_val_loss,
                "best_late_legality": best_late_legality,
            }
            stop = patience_counter >= cfg.patience
        val_kwargs = vm.as_log_kwargs()
        logger.log_val(step=at_step, **val_kwargs, **extra_log)
        log_metrics(
            wandb_run,
            {f"val/{k}": v for k, v in val_kwargs.items()},
            step=at_step,
        )
        print(
            f"  val @ {at_step}: loss {vm.val_loss:.4f} | "
            f"top1 {vm.top1:.3f} | top5 {vm.top5:.3f} | "
            f"legal {vm.legal_move_rate:.3f} | "
            f"late_legal {vm.late_legal_move_rate:.3f}"
            + (
                f" | pat {patience_counter}/{cfg.patience}"
                if cfg.patience is not None else ""
            ),
            flush=True,
        )
        return stop

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pawn-prefetch")
    producer = BucketedPrefetcher(
        resume_chunk=resume_chunk, resume_skip=resume_skip
    )
    # Per-bucket throughput counters (visible in the JSONL stream).
    bucket_steps: dict[int, int] = {edge: 0 for edge in bucket_edges}
    # Stop-reason tracking for schedule_health.json (H7). `step_limit` is
    # the normal full-budget completion; the SIGTERM / loader-exhausted
    # paths overwrite it below. `exception` is set in the except clause.
    reason_for_stop = "completed"
    try:
        while next_step < total_steps:
            nxt = producer.next()
            if nxt is None:
                # Should be unreachable in practice — generator always
                # produces at least one batch for a non-empty bucket.
                reason_for_stop = "loader_exhausted"
                break
            edge, chunk_batches = nxt
            this_chunk_k = int(chunk_batches.tokens.shape[0])

            # grad_norm is always emitted now (v1 parity — see the
            # `make_scan_step(emit_grad_norms=True, ...)` call site), so the
            # scan returns the 4-tuple unconditionally.
            state, chunk_losses, chunk_gnorms, chunk_accs = scan_step(
                state, chunk_batches
            )
            chunk_gnorms_np = np.asarray(chunk_gnorms)
            next_step += this_chunk_k
            bucket_steps[edge] = bucket_steps.get(edge, 0) + this_chunk_k

            # One D→H per chunk, not per step.
            chunk_losses_np = np.asarray(chunk_losses)
            chunk_accs_np = np.asarray(chunk_accs)

            # Log every step that crossed a log_interval boundary inside
            # the chunk — replays the within-chunk loss curve without
            # per-step syncs.
            chunk_start = next_step - this_chunk_k
            for i in range(this_chunk_k):
                step = chunk_start + i + 1
                if step % cfg.log_interval == 0:
                    extra = {}
                    if chunk_gnorms_np is not None:
                        extra["grad_norm"] = float(chunk_gnorms_np[i])
                        extra["did_clip"] = bool(chunk_gnorms_np[i] > cfg.max_grad_norm)
                    lr_now = np.asarray(schedule(step)).item()
                    step_loss = float(chunk_losses_np[i])
                    step_acc = float(chunk_accs_np[i])
                    # Dashboard `pawn` run_type keys its loss chart + KPIs on
                    # `train/loss` and its accuracy chart on `train/accuracy`
                    # (pawn/dashboard/charts.py:344,373; sol.py:552-563). Emit
                    # the namespaced keys as canonical and keep the bare `loss`
                    # / `accuracy` aliases so older log readers (sweep / lab
                    # monitors that fall back to the bare names) still resolve
                    # them. `train/accuracy` is the widest TRAINED variant's
                    # top-1 (see `_supernet_accuracy`) — v1 parity
                    # (`git show main:pawn/trainer.py:1145`).
                    train_metrics = dict(
                        {
                            "train/loss": step_loss, "loss": step_loss,
                            "train/accuracy": step_acc, "accuracy": step_acc,
                        },
                        lr=lr_now,
                        step_time=(time.time() - t0) / max(1, step - start),
                        bucket=edge,
                        **extra,
                    )
                    logger.log_train(step=step, **train_metrics)
                    log_metrics(wandb_run, train_metrics, step=step)

            # Checkpoint when we cross a checkpoint boundary. Use
            # division-based crossing so chunk_k doesn't have to divide
            # checkpoint_interval. (Pre-A.1 chunks were exactly cfg.k
            # steps; bucketing keeps that, but the test is more robust.)
            crossed_checkpoint = (
                next_step // cfg.checkpoint_interval
                != (next_step - this_chunk_k) // cfg.checkpoint_interval
            )
            if crossed_checkpoint or next_step >= total_steps:
                if cfg.local_checkpoints or cfg.hf_repo:
                    _save_checkpoint(next_step)

            # Held-out validation pass at every val_every boundary crossed
            # inside this chunk. Division-based crossing (like the
            # checkpoint test) so cfg.k need not divide val_every; the eval
            # runs on the post-chunk model at the crossing step. A patience
            # exhaustion breaks the loop with reason=patience.
            if val_every is not None:
                crossed_val = (
                    next_step // val_every
                    != (next_step - this_chunk_k) // val_every
                )
                if crossed_val or next_step >= total_steps:
                    if _run_validation(next_step):
                        _save_checkpoint(next_step)
                        reason_for_stop = "patience"
                        print(
                            f"\nEarly stopping at step {next_step} "
                            f"(no val improvement for {cfg.patience} evals)",
                            flush=True,
                        )
                        break

            # Checkpoint-and-pause at a step boundary (v1 pause_after_steps).
            if (
                cfg.pause_after_steps is not None
                and next_step >= cfg.pause_after_steps
            ):
                _save_checkpoint(next_step)
                reason_for_stop = "paused"
                print(
                    f"\nPaused at step {next_step} "
                    f"(pause_after_steps={cfg.pause_after_steps}); "
                    f"resume with --resume",
                    flush=True,
                )
                break

            if should_shutdown():
                _save_checkpoint(next_step)
                reason_for_stop = "sigterm"
                break
        else:
            # `while` exited via its condition (next_step >= total_steps):
            # the full planned budget ran. `completed` already set above.
            reason_for_stop = "completed"
    except BaseException:
        reason_for_stop = "exception"
        raise
    finally:
        producer.close()
        executor.shutdown(wait=False, cancel_futures=True)
        # H7: write schedule_health.json at *every* exit path (normal,
        # SIGTERM, exception) so a post-hoc reader can tell whether the LR
        # schedule ran to completion. `reason_for_stop == "completed"`
        # with `actual == planned` is the healthy full-run case. The final
        # LR is the schedule value at the last step actually applied
        # (`next_step - 1`, clamped to the start so a 0-step run is sane).
        last_lr = float(
            np.asarray(schedule(max(start, next_step - 1))).item()
        )
        # OBS-1: on a `completed` stop, clamp `actual_total_steps` to the
        # planned budget. The pretrain loop is `while next_step < total_steps`
        # and advances by whole K-batches (`next_step += this_chunk_k`), so the
        # final chunk OVERSHOOTS whenever `cfg.k ∤ (total_steps - start)` —
        # `next_step` lands exactly on `total_steps` only when the chunk size
        # divides the remaining budget, which is the exception, not the rule.
        # That overshoot is a chunk-granularity artifact, not a real schedule
        # shortfall: a `completed` run never *under*-runs the budget. Reporting
        # the raw `next_step` would trip `write_schedule_health`'s red banner
        # AND the lab runner's `structural_mismatch` on healthy runs, gutting
        # the H7 tripwire's signal. Genuine early exits (sigterm /
        # loader_exhausted / exception) report the real `next_step` so a true
        # schedule shortfall still surfaces.
        actual_total_steps = (
            min(next_step, total_steps)
            if reason_for_stop == "completed"
            else next_step
        )
        write_schedule_health(
            logger.run_dir,
            schedule=cfg.lr_schedule,
            planned_total_steps=total_steps,
            actual_total_steps=actual_total_steps,
            lr_peak=cfg.lr,
            actual_final_lr=last_lr,
            reason_for_stop=reason_for_stop,
        )
        # A clean completion / SIGTERM / resume-no-op exits 0; an in-loop
        # exception is the only failed-run signal worth flagging in W&B.
        finish_wandb(wandb_run, exit_code=1 if reason_for_stop == "exception" else 0)

    # Per-bucket step distribution at end of training.
    print(f"Per-bucket step counts: {bucket_steps}", flush=True)

    if push_tracker:
        # `drain_push_queue` reports (timeouts, errors). Only `timeouts`
        # implies a worker is still running — that's the abandon-thread
        # case (`drain_succeeded=False`) so the daemon worker is killed
        # at interpreter exit. `errors > 0` means uploads raised but
        # the worker exited normally, which is safe to `wait=True` on.
        # Conflating the two (round-1 bug-detector finding) would
        # cause a transient network error to spuriously skip the
        # `wait=True` path that's still semantically correct.
        timeouts, _errors = drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown(drain_succeeded=(timeouts == 0))
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
