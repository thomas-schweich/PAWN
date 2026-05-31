#!/usr/bin/env python3
"""Distillation entry point — distil a frozen teacher into a from-scratch student.

The canonical-ladder mechanism (plan §7/§7.1): load a frozen teacher
checkpoint, build a from-scratch student (specialized_clm shapes), and train
the student to match the teacher's soft targets via a temperature-scaled KL
(optionally mixed with ground-truth CE). Logit-only — no hidden-state
matching.

Validates the run config via pydantic (`DistillConfig`); reads Lichess data
via `pawn.lichess_data.load_lichess_corpus` (or random games with `--no-pgn`).
The teacher's conditioning / `C` is inherited from its own checkpoint and the
student corpus is built under it (load-time C-assert, Phase-A Chunk 4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from pawn.checkpoint import load_model, resolve_checkpoint_source, save_model
from pawn.config import SUPERNET, TINY_SUPERNET, ModelConfig
from pawn.corpus import (
    conditioning_from_run_block,
    conditioning_to_C,
    generate_corpus,
)
from pawn.distill import (
    DistillTrainState,
    make_distill_scan_step,
    make_distill_train_step,
)
from pawn.jax_setup import require_accelerator, resolve_device, setup_jax_caching
from pawn.lichess_data import load_lichess_corpus
from pawn.lifecycle import (
    HFPushTracker,
    build_training_state,
    drain_push_queue,
    install_sigterm_handler,
    push_checkpoint_async,
    read_resume_rng_blocks,
    write_schedule_health,
)
from pawn.logging import MetricsLogger, get_git_info
from pawn.model import PAWNModel, init_model
from pawn.run_config import DistillConfig
from pawn.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    require_wandb_available,
)
from pawn.trainer import (
    Batch,
    flatten_opt_state,
    make_lr_schedule,
    make_optimizer,
    slice_batch,
)


def _student_config(cfg: DistillConfig) -> ModelConfig:
    """Resolve the student's :class:`ModelConfig` from the run config.

    Either the ``student_supernet`` preset (tiny / production SUPERNET dims)
    or the four explicit dims — :class:`DistillConfig` already validated
    that exactly one is set. The student reuses the canonical
    vocab / seq / RoPE defaults (it is a standalone model, not a slice).
    """
    if cfg.student_supernet is not None:
        preset = TINY_SUPERNET if cfg.student_supernet == "tiny" else SUPERNET
        return ModelConfig(
            d_model=preset.d_model,
            n_layers=preset.n_layers,
            n_heads=preset.n_heads,
            d_ff=preset.d_ff,
            head_dim=preset.head_dim,
        )
    # Explicit dims (validated non-None + divisible by DistillConfig).
    assert (
        cfg.d_model is not None and cfg.n_layers is not None
        and cfg.n_heads is not None and cfg.d_ff is not None
    )
    return ModelConfig(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        head_dim=cfg.d_model // cfg.n_heads,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train_jax_distill")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--distill-from", default=None,
                    help="frozen teacher checkpoint: local v2 dir or HF repo ID")
    ap.add_argument("--objective", choices=("kl", "ce", "mix"), default=None)
    ap.add_argument("--distill-temp", type=float, default=None,
                    dest="temperature", help="KL softmax temperature")
    ap.add_argument("--distill-alpha", type=float, default=None,
                    dest="alpha", help="mix weight: alpha*ce + (1-alpha)*kl")
    ap.add_argument("--student-supernet", choices=("tiny", "production"),
                    default=None, dest="student_supernet")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--pgn-val-split", default=None)
    ap.add_argument("--elo-min", type=int, default=None)
    ap.add_argument("--elo-max", type=int, default=None)
    ap.add_argument("--min-ply", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--k", type=int, default=None,
                    help="inner-scan length: K steps per lax.scan body")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--accumulation-steps", type=int, default=None,
                    help="(B2) micro-batches accumulated per optimizer step. "
                         ">1 emits (K, N, B, T) chunks so the distill "
                         "trainer's accumulation scan sums N micro-grads "
                         "before each update — effective batch N×B at B's "
                         "per-step memory cost. Default 1 (no accumulation).")
    ap.add_argument("--no-pgn", action="store_true",
                    help="use random-game corpus instead of Lichess parquet")
    ap.add_argument("--local-checkpoints", action="store_true")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from a distill_step_<N> checkpoint dir. "
                         "Restores the trained student + Adam moments + step "
                         "+ RNG so the interrupted run continues rather than "
                         "discarding its progress.")
    ap.add_argument("--use-sdpa", action="store_true",
                    help="opt the attention block into "
                         "jax.nn.dot_product_attention (parity #43)")
    ap.add_argument("--no-flash", action="store_true",
                    help="disable the Pallas flash-attention kernel")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--wandb", action="store_true",
                    help="enable the Weights & Biases metric mirror "
                         "(requires --extra wandb). Hard-errors if the "
                         "extra isn't installed.")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None,
                    dest="checkpoint_interval",
                    help="steps between student checkpoints (boundary-crossing "
                         "cadence; mirrors scripts/train_jax.py)")
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace) -> DistillConfig:
    base: dict = {"run_type": "distill"}
    if args.config is not None:
        base.update(json.loads(args.config.read_text()))
        base.setdefault("run_type", "distill")
    # `--wandb` joins the opt-in store_true set so an absent CLI flag never
    # overrides a JSON-set `"wandb": true` (mirrors train_jax / adapter).
    _CLI_STORE_TRUE_FLAGS = ("use_sdpa", "wandb")
    for flag, val in vars(args).items():
        if flag in (
            "config", "no_pgn", "local_checkpoints", "logs_dir",
            "no_flash", "resume", *_CLI_STORE_TRUE_FLAGS,
        ):
            # `resume` is operated on directly from `args` (Path, not a
            # pydantic-validated string) in main(), mirroring the adapter
            # entry point; the store_true flags need explicit opt-in so an
            # absent CLI flag doesn't clobber a JSON True.
            continue
        if val is None:
            continue
        base[flag] = val
    for flag in _CLI_STORE_TRUE_FLAGS:
        if getattr(args, flag, False):
            base[flag] = True
    if args.no_flash:
        base["use_flash"] = False
    if args.local_checkpoints:
        base["local_checkpoints"] = True
    return DistillConfig(**base)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = _build_config(args)
    if cfg.total_steps is None:
        print("error: --total-steps is required", file=sys.stderr)
        return 2
    # `DistillConfig._check_distill` already guarantees non-None; bind a
    # local so pyright narrows the optional across the rest of main().
    total_steps: int = cfg.total_steps
    require_accelerator()
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

    # Load the frozen teacher. Its conditioning / C is inherited and the
    # student corpus is built under it; a C-mismatch between the run config's
    # conditioning and the teacher's persisted conditioning would place every
    # move at a different absolute RoPE offset (silent drift), so cross-check
    # and fail loudly (Phase-A load-time C-assert).
    ckpt_dir = resolve_checkpoint_source(cfg.distill_from)
    teacher, teacher_run_block = load_model(ckpt_dir)
    teacher_conditioning = conditioning_from_run_block(teacher_run_block)
    teacher_C = conditioning_to_C(teacher_conditioning)
    run_C = conditioning_to_C(cfg.conditioning)
    if run_C != teacher_C:
        raise SystemExit(
            f"[train_jax_distill] conditioning mismatch: run config "
            f"conditioning {cfg.conditioning!r} implies C={run_C}, but the "
            f"teacher checkpoint was trained with C={teacher_C} "
            f"(conditioning {teacher_conditioning!r}). The student corpus is "
            f"built under the teacher's conditioning to avoid RoPE drift; "
            f"rerun with --conditioning matching the teacher."
        )
    # The student corpus is built under the teacher's own conditioning.
    conditioning = teacher_conditioning

    # Build the from-scratch student.
    student_cfg = _student_config(cfg)
    student = init_model(student_cfg, key=jax.random.key(0))

    # Optimizer over the student only.
    schedule = make_lr_schedule(cfg, total_steps)
    optimizer = make_optimizer(cfg, schedule)
    import equinox as eqx

    # --resume: splice (student, opt_state, step, key) from a checkpoint so an
    # interrupted run continues rather than discarding all progress. The
    # student is a standalone PAWNModel saved by `_save` via the plain
    # weight-folded path, so the restore mirrors `pawn.lifecycle.
    # load_resume_state`: load the model, restore the Adam moments + clip
    # counter from `optimizer.safetensors`, and read the step counter from the
    # `training_state.json` sidecar. The teacher is always the freshly-loaded
    # frozen one (it is never re-saved into the student checkpoint).
    resume_step = 0
    resume_key = jax.random.key(0)
    if args.resume is not None:
        from safetensors.numpy import load_file as st_load

        from pawn.checkpoint import OPTIMIZER_FILE
        from pawn.trainer import unflatten_opt_state

        resume_dir = Path(args.resume)
        ts_path = resume_dir / "training_state.json"
        if not ts_path.is_file():
            raise SystemExit(
                f"[train_jax_distill] --resume requires "
                f"{ts_path.name} in the checkpoint dir; got {resume_dir} "
                "with no such file. The training-state sidecar carries the "
                "saved step counter and is load-bearing for the resume "
                "contract."
            )
        # The student checkpoint's own model.safetensors is the trained
        # student; load it and assert its conditioning matches the run's C
        # (same load-time guard as the teacher load above).
        student, student_run_block = load_model(resume_dir)
        student_C = conditioning_to_C(
            conditioning_from_run_block(student_run_block)
        )
        if student_C != run_C:
            raise SystemExit(
                f"[train_jax_distill] --resume conditioning mismatch: the "
                f"checkpointed student was trained at C={student_C} but the "
                f"run config implies C={run_C}. Rerun with --conditioning "
                "matching the checkpoint."
            )
        ts_data = json.loads(ts_path.read_text(encoding="utf-8"))
        resume_step = int(ts_data.get("step", 0))
        template = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
        opt_path = resume_dir / OPTIMIZER_FILE
        if opt_path.is_file():
            flat = st_load(str(opt_path))
            opt_state = unflatten_opt_state(template, flat)
        else:
            print(
                f"[train_jax_distill] WARNING: no {OPTIMIZER_FILE} in "
                f"{resume_dir}; resuming with a fresh opt_state (Adam moments "
                "will cold-start).",
                file=sys.stderr,
            )
            opt_state = template
        # Restore the persisted JAX key if present (older checkpoints fall
        # back to the seed-0 key).
        resume_jax_key, _ = read_resume_rng_blocks(resume_dir)
        if resume_jax_key is not None:
            resume_key = resume_jax_key
    else:
        opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))

    state = DistillTrainState(
        student=student, teacher=teacher, opt_state=opt_state,
        step=jnp.int32(resume_step), key=resume_key,
    )

    _DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": None,
    }
    compute_dtype = _DTYPE_MAP[cfg.amp_dtype]
    use_flash = cfg.use_flash and jax.default_backend() == "gpu"
    # B2: with ``accumulation_steps == N > 1`` the scan element gains a
    # leading microbatch axis (``(N, B, T)``) so each ``train_step`` sums N
    # micro-grads before a single optimizer update — effective batch N×B at
    # B's per-step memory cost. ``N == 1`` keeps the legacy ``(B, T)`` shape.
    accumulation_steps = cfg.accumulation_steps
    train_step = make_distill_train_step(
        optimizer,
        objective=cfg.objective, temperature=cfg.temperature, alpha=cfg.alpha,
        compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        accumulation_steps=accumulation_steps,
    )
    # Drive the hot loop through the K-step `lax.scan` (mirrors
    # `scripts/train_jax.py`'s `make_scan_step`): the per-step body never
    # returns to the host inside a chunk, the donated buffers are amortised
    # across K steps, and metrics flush between chunks — one D→H per chunk,
    # not per step. `cfg.k` is clamped so the final chunk doesn't overrun
    # `total_steps`.
    scan_step = make_distill_scan_step(train_step)

    # Data: Lichess corpus or random games, under the teacher's conditioning.
    if args.no_pgn:
        corpus = generate_corpus(
            n_games=max(cfg.batch_size * 10, 1000),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=0,
            conditioning=conditioning,
        )
        val_corpus = generate_corpus(
            n_games=max(cfg.batch_size * 4, 100),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=1,
            conditioning=conditioning,
        )
    else:
        corpus = load_lichess_corpus(
            cfg.pgn, split="train",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max, min_ply=cfg.min_ply,
            seq_len=cfg.seq_len, max_games=cfg.max_games,
            conditioning=conditioning,
        )
        val_corpus = load_lichess_corpus(
            cfg.pgn, split=cfg.pgn_val_split or "validation",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max, min_ply=cfg.min_ply,
            seq_len=cfg.seq_len, max_games=cfg.val_games,
            conditioning=conditioning,
        )

    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix=f"distill-{cfg.objective}",
        device=resolve_device(),
    )
    logger.log_config(run_type="distill", config=cfg.model_dump())

    # W&B mirror — gated on `--wandb` (cfg.wandb) *and* the `wandb` extra.
    # `--wandb` without the extra is a hard error (no silent metric drop;
    # parity with train_jax / train_jax_adapter).
    wandb_run = None
    if cfg.wandb:
        require_wandb_available()
        wandb_run = init_wandb(
            project=cfg.wandb_project, slug=logger.slug,
            run_config=cfg.model_dump(),
            git_hash=get_git_info().get("git_hash"),
            job_type="distill",
            run_dir_name=logger.run_dir.name,
        )

    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None
    should_shutdown = install_sigterm_handler()

    # Steps already persisted this run. An in-loop SIGTERM save, a
    # boundary-crossing checkpoint, and the run-end final-save can all target
    # the same `distill_step_<final>` path when the interrupt or run-end lands
    # at a step that is not a multiple of `checkpoint_interval` (the common
    # case with k>1). `save_model` refuses to overwrite an existing
    # checkpoint, so `_save` is idempotent per step: a second request for an
    # already-written step is a no-op rather than a `FileExistsError`
    # (mirrors the adapter trainer's `saved_steps` guard).
    saved_steps: set[int] = set()

    def _save(step_int: int) -> None:
        """Save the trained student as an ordinary v2 checkpoint.

        The student is a standalone :class:`pawn.model.PAWNModel`, so the
        save path is the plain weight-folded one. The run config (carrying
        the teacher's inherited conditioning) + optimizer state + step land
        alongside so a downstream consumer treats it as any published
        checkpoint and an eval reads the right sequence layout.

        Idempotent per step: a `crossed_ckpt` save and an in-loop SIGTERM /
        run-end final-save can both name the same `distill_step_<step>` path,
        and `save_model` raises `FileExistsError` on an existing target, so a
        repeat request for an already-written step early-returns.
        """
        if step_int in saved_steps:
            return
        run_cfg = cfg.model_dump()
        # Persist the conditioning the student was actually trained under
        # (inherited from the teacher) so eval rebuilds the right layout.
        run_cfg["conditioning"] = list(conditioning)
        out = logger.run_dir / f"distill_step_{step_int:08d}"
        # Persist the scheduler identity + the JAX key + both numpy
        # data-stream RNGs (train + val) so a `--resume` continues the exact
        # same batch-index sequence rather than replaying from the seed
        # (mirrors train_jax / train_jax_adapter).
        training_state = build_training_state(
            step=int(state.step),
            schedule=cfg.lr_schedule,
            lr_peak=cfg.lr,
            rng_key=state.key,
            numpy_rngs={"train": rng, "val": val_rng},
        )
        save_model(
            state.student, out,
            run_config=run_cfg,
            optimizer_state=flatten_opt_state(state.opt_state),
            training_state=training_state,
        )
        saved_steps.add(step_int)
        if push_tracker:
            push_checkpoint_async(out, push_tracker)

    rng = np.random.default_rng(0)
    val_rng = np.random.default_rng(1)
    # On resume, restore the persisted data-stream RNG state so the resumed
    # run continues the *same* batch-index sequence rather than replaying from
    # the seed (which would re-train on already-seen batches).
    if args.resume is not None:
        _, _resume_np = read_resume_rng_blocks(Path(args.resume))
        if "train" in _resume_np:
            rng = _resume_np["train"]
        if "val" in _resume_np:
            val_rng = _resume_np["val"]
    eval_interval = cfg.eval_interval or cfg.log_interval
    t0 = time.time()
    final_step = resume_step

    def _chunk_batch(k_inner: int) -> Batch:
        """Sample the games for one `scan_step` chunk and stack them into the
        leading-K layout the scan consumes.

        With ``accumulation_steps == 1`` each scan element is a ``(B, T)``
        batch, so a chunk is ``(K, B, T)`` (mirrors `make_scan_step` in
        `scripts/train_jax.py`). With ``N > 1`` each scan element gains a
        leading microbatch axis so the chunk is ``(K, N, B, T)`` and the
        accumulation train step sums N micro-grads per optimizer step."""
        group = k_inner * accumulation_steps * cfg.batch_size
        idx = rng.integers(0, corpus.n_games, size=group)
        flat = slice_batch(corpus, idx)
        if accumulation_steps == 1:
            shape: tuple[int, ...] = (k_inner, cfg.batch_size, -1)
        else:
            shape = (k_inner, accumulation_steps, cfg.batch_size, -1)
        return Batch(
            tokens=flat.tokens.reshape(shape),
            targets=flat.targets.reshape(shape),
            attn_mask=flat.attn_mask.reshape(shape),
            loss_mask=flat.loss_mask.reshape(shape),
        )

    @eqx.filter_jit
    def val_step(
        student_m: PAWNModel, teacher_m: PAWNModel, batch: Batch
    ) -> Float[Array, ""]:
        from pawn.distill import distill_loss, frozen_teacher
        teacher_fn = frozen_teacher(
            teacher_m, compute_dtype=compute_dtype,
            use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        )
        return distill_loss(
            student_m, teacher_fn, batch,
            objective=cfg.objective, temperature=cfg.temperature,
            alpha=cfg.alpha, compute_dtype=compute_dtype,
            use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        )

    # ``run_start`` is the absolute step the run began from — ``resume_step``
    # on a resume, 0 otherwise. The ``step_time`` divisor must be
    # ``(step - run_start)`` so a resume reports the per-step wall time of the
    # *current* run rather than folding in steps from the prior run.
    run_start = resume_step
    completed = resume_step
    # H7: write schedule_health.json at *every* exit path (normal, SIGTERM,
    # exception). `reason_for_stop` defaults to `completed`; SIGTERM and
    # exceptions overwrite it. A `completed` stop whose `actual != planned`
    # on a decay-to-zero schedule is the structural-bug signal the lab
    # runner flags. A resume that runs zero new steps reports `resume_no_op`.
    reason_for_stop = "completed"
    try:
        while completed < total_steps:
            # Clamp the final chunk so the K-step scan never overruns
            # `total_steps`.
            this_k = min(cfg.k, total_steps - completed)
            chunk = _chunk_batch(this_k)
            state, chunk_losses = scan_step(state, chunk)
            # One D→H per chunk, not per step.
            chunk_losses_np = np.asarray(chunk_losses)
            chunk_start = completed
            completed += this_k
            final_step = completed

            # Replay the within-chunk loss curve at every log boundary the
            # chunk crossed, without per-step host syncs.
            for i in range(this_k):
                step = chunk_start + i + 1
                if step % cfg.log_interval == 0:
                    train_metrics = dict(
                        loss=float(chunk_losses_np[i]),
                        lr=np.asarray(schedule(step)).item(),
                        step_time=(time.time() - t0) / max(1, step - run_start),
                    )
                    logger.log_train(step=step, **train_metrics)
                    log_metrics(wandb_run, train_metrics, step=step)

            # Validate / checkpoint on division-based boundary crossing so
            # the chunk size doesn't have to divide the intervals.
            crossed_eval = (
                final_step // eval_interval != chunk_start // eval_interval
            )
            if crossed_eval:
                val_idx = val_rng.integers(
                    0, val_corpus.n_games, size=cfg.batch_size
                )
                val_batch = slice_batch(val_corpus, val_idx)
                val_loss = float(
                    val_step(state.student, state.teacher, val_batch)
                )
                val_record = dict(
                    val_loss=val_loss,
                    val_source=cfg.pgn_val_split if not args.no_pgn else "random",
                )
                logger.log_val(step=final_step, **val_record)
                log_metrics(wandb_run, val_record, step=final_step)

            crossed_ckpt = (
                final_step // cfg.checkpoint_interval
                != chunk_start // cfg.checkpoint_interval
            )
            if crossed_ckpt:
                _save(final_step)
            if should_shutdown():
                _save(final_step)
                reason_for_stop = "sigterm"
                break
        else:
            reason_for_stop = "completed"

        if reason_for_stop != "sigterm" and final_step == resume_step \
                and resume_step > 0:
            # Resumed at/past the budget — no new steps ran. The actual step
            # count is the saved checkpoint's, not 0; skip the final-save (no
            # new student state to persist).
            reason_for_stop = "resume_no_op"
        elif final_step > 0 and final_step % cfg.checkpoint_interval != 0:
            _save(final_step)
    except BaseException:
        reason_for_stop = "exception"
        raise
    finally:
        # `actual` is what ran this session, or the resumed step on a no-op.
        actual_total = final_step if final_step > 0 else resume_step
        actual_final_lr = float(
            np.asarray(schedule(max(0, actual_total - 1))).item()
        )
        write_schedule_health(
            logger.run_dir,
            schedule=cfg.lr_schedule,
            planned_total_steps=total_steps,
            actual_total_steps=actual_total,
            lr_peak=cfg.lr,
            actual_final_lr=actual_final_lr,
            reason_for_stop=reason_for_stop,
        )
        # Only an in-loop exception is a failed run; completed / sigterm /
        # resume_no_op are clean exits.
        finish_wandb(
            wandb_run, exit_code=1 if reason_for_stop == "exception" else 0
        )

    if push_tracker:
        timeouts, _errors = drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown(drain_succeeded=(timeouts == 0))
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
