#!/usr/bin/env python3
"""Adapter fine-tuning entry point — one of 8 strategies on a Lichess Elo band.

Dispatches through `pawn.adapter_trainer.STRATEGIES`. Validates the
run config via pydantic; reads Lichess data via
`pawn.lichess_data.load_lichess_corpus`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.adapter_trainer import (
    STRATEGIES,
    AdapterTrainState,
    dispatch_filter,
    dispatch_init,
    make_adapter_train_step,
)
from pawn.adapters import (
    BottleneckConfig,
    FiLMConfig,
    HybridConfig,
    LoRAConfig,
    RoSAConfig,
    SparseConfig,
    SpecializedCLMConfig as AdapterSpecializedCLMConfig,
    UnfreezeConfig,
)
from pawn.checkpoint import save_model
from pawn.config import SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS
from pawn.corpus import generate_corpus
from pawn.legacy import convert_legacy_checkpoint
from pawn.lichess_data import load_lichess_corpus
from pawn.lifecycle import (
    HFPushTracker,
    drain_push_queue,
    install_sigterm_handler,
    push_checkpoint_async,
)
from pawn.logging import MetricsLogger
from pawn.model import init_model, sliced
from pawn.run_config import AdapterConfig
from pawn.trainer import (
    cross_entropy_loss, make_lr_schedule, make_optimizer, slice_batch,
)


def _strategy_config_from_run(cfg: AdapterConfig) -> object:
    """Build the adapter's strategy Config from the AdapterConfig fields."""
    s = cfg.strategy
    if s == "lora":
        return LoRAConfig(
            rank=cfg.lora_rank or 4,
            targets=cfg.lora_targets or "qkvo",
            ffn=cfg.lora_ffn,
        )
    if s == "film":
        return FiLMConfig(use_output_film=cfg.use_output_film)
    if s == "bottleneck":
        return BottleneckConfig(
            dim=cfg.bottleneck_dim or 8,
            n_hidden=cfg.bottleneck_n_hidden,
            no_adapt_attn=cfg.no_adapt_attn,
            no_adapt_ffn=cfg.no_adapt_ffn,
        )
    if s == "hybrid":
        return HybridConfig(
            lora=LoRAConfig(rank=cfg.lora_rank or 4),
            film=FiLMConfig(use_output_film=cfg.use_output_film),
        )
    if s == "sparse":
        return SparseConfig(
            density=cfg.density or 0.01,
            targets=cfg.sparse_targets or "qkvo",
        )
    if s in ("rosa",):
        return RoSAConfig(
            mode=cfg.rosa_mode or "rosa",
            lora_rank=cfg.lora_rank or 4,
            density=cfg.density or 0.01,
            rosa_warmup_steps=cfg.rosa_warmup_steps,
            mask_samples=cfg.mask_samples,
            grad_alpha=cfg.grad_alpha,
        )
    if s == "unfreeze":
        return UnfreezeConfig(layers=cfg.unfreeze_layers or "5,6,7")
    if s == "specialized_clm":
        return AdapterSpecializedCLMConfig(
            d_model=cfg.d_model or 64,
            n_layers=cfg.n_layers or 2,
            n_heads=cfg.n_heads or 2,
            d_ff=cfg.d_ff or 256,
        )
    raise ValueError(f"unknown strategy {s!r}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train_jax_adapter")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--strategy", required=False)
    ap.add_argument("--supernet", choices=("tiny", "production"), default=None)
    ap.add_argument("--variant", choices=("small", "base", "large"), default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--pgn-val-split", default=None)
    ap.add_argument("--elo-min", type=int, default=None)
    ap.add_argument("--elo-max", type=int, default=None)
    ap.add_argument("--min-ply", type=int, default=None)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-targets", default=None)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--bottleneck-dim", type=int, default=None)
    ap.add_argument("--use-output-film", action="store_true")
    ap.add_argument("--no-adapt-attn", action="store_true")
    ap.add_argument("--no-adapt-ffn", action="store_true")
    ap.add_argument("--rosa-mode", default=None)
    ap.add_argument("--unfreeze-layers", default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-pgn", action="store_true",
                    help="use random-game corpus instead of Lichess parquet")
    ap.add_argument("--local-checkpoints", action="store_true")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--log-interval", type=int, default=None,
                    help="Steps between metrics rows; defaults to the "
                         "BaseRunConfig log_interval (100)")
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace) -> AdapterConfig:
    base: dict = {"run_type": "adapter"}
    if args.config is not None:
        base.update(json.loads(args.config.read_text()))
        base.setdefault("run_type", "adapter")
    for flag, val in vars(args).items():
        if flag in ("config", "no_pgn", "local_checkpoints", "logs_dir"):
            continue
        if val is None or val is False:
            continue
        base[flag] = val
    if args.local_checkpoints:
        base["local_checkpoints"] = True
    return AdapterConfig(**base)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = _build_config(args)
    if cfg.total_steps is None:
        print("error: --total-steps is required", file=sys.stderr)
        return 2
    if cfg.strategy not in STRATEGIES:
        print(f"error: unknown strategy {cfg.strategy!r}", file=sys.stderr)
        return 2

    # Build / load backbone.
    if cfg.strategy == "specialized_clm":
        # Standalone — no backbone needed.
        backbone = init_model(TINY_SUPERNET, key=0)  # placeholder
    elif cfg.checkpoint.startswith("thomas-schweich/") or "/" in cfg.checkpoint:
        # HF repo → convert via legacy.
        from pawn.checkpoint import load_model
        converted = convert_legacy_checkpoint(cfg.checkpoint)
        backbone = load_model(converted)
        # v1 published checkpoints are standalone (each variant has its own
        # depth — e.g. pawn-base is 8 layers, pawn-small is 8 layers), so
        # they don't fit the v2 supernet's "all variants share n_layers"
        # constraint and can't be sliced. Only slice when the loaded model
        # *is* a v2 supernet shape — i.e. it has the same n_layers as the
        # SUPERNET config. Otherwise treat the loaded model as standalone.
        target_supernet = TINY_SUPERNET if cfg.supernet == "tiny" else SUPERNET
        looks_like_supernet = backbone.cfg.n_layers == target_supernet.n_layers
        if cfg.variant != "large" and looks_like_supernet:
            variant_cfg = (
                TINY_VARIANTS[cfg.variant]
                if cfg.supernet == "tiny"
                else VARIANTS[cfg.variant]
            )
            backbone = sliced(backbone, variant_cfg)
    else:
        # Local checkpoint dir.
        from pawn.checkpoint import load_model
        backbone = load_model(Path(cfg.checkpoint))

    # Build adapter.
    strategy_cfg = _strategy_config_from_run(cfg)
    init = dispatch_init(cfg.strategy)
    adapter = init(backbone, strategy_cfg, key=jax.random.key(0))

    # Optimizer over the adapter only.
    schedule = make_lr_schedule(cfg, cfg.total_steps)
    optimizer = make_optimizer(cfg, schedule)
    flt = dispatch_filter(cfg.strategy)(adapter)
    opt_state = optimizer.init(eqx.filter(adapter, flt))
    state = AdapterTrainState(
        backbone=backbone, adapter=adapter, opt_state=opt_state,
        step=jnp.int32(0), key=jax.random.key(0),
    )
    train_step = make_adapter_train_step(cfg.strategy, optimizer)

    apply_fn = STRATEGIES[cfg.strategy].apply

    @eqx.filter_jit
    def val_step(backbone, adapter, batch):
        effective = apply_fn(backbone, adapter)
        return cross_entropy_loss(effective, batch)

    # Data: Lichess corpus or random games. The val_corpus is the held-out
    # `validation` split for PGN (cfg.pgn_val_split, default "validation")
    # or a fresh-seed random corpus when --no-pgn is set.
    if args.no_pgn:
        corpus = generate_corpus(
            n_games=max(cfg.batch_size * 10, 1000),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=0,
        )
        val_corpus = generate_corpus(
            n_games=max(cfg.batch_size * 4, 100),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=1,
        )
    else:
        corpus = load_lichess_corpus(
            cfg.pgn,
            split="train",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max,
            min_ply=cfg.min_ply,
            seq_len=cfg.seq_len,
            max_games=getattr(cfg, "max_games", None),
        )
        val_corpus = load_lichess_corpus(
            cfg.pgn,
            split=cfg.pgn_val_split or "validation",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max,
            min_ply=cfg.min_ply,
            seq_len=cfg.seq_len,
            max_games=getattr(cfg, "val_games", None),
        )

    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix=f"adapter-{cfg.strategy}",
        device="cuda", suffix=cfg.variant,
    )
    logger.log_config(run_type="adapter", config=cfg.model_dump())

    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None
    should_shutdown = install_sigterm_handler()

    rng = np.random.default_rng(0)
    val_rng = np.random.default_rng(1)
    # eval_interval defaults to log_interval when unset so every
    # train-row gets a paired val-row (val-loss is what the sweep
    # objective and §3 criterion 7's "val loss decreases" key on).
    eval_interval = cfg.eval_interval or cfg.log_interval
    t0 = time.time()
    for step in range(cfg.total_steps):
        idx = rng.integers(0, corpus.n_games, size=cfg.batch_size)
        batch = slice_batch(corpus, idx)
        state, loss = train_step(state, batch)
        if (step + 1) % cfg.log_interval == 0:
            logger.log_train(
                step=step + 1, loss=float(loss),
                lr=np.asarray(schedule(int(state.step))).item(),
                step_time=(time.time() - t0) / (step + 1),
            )
        if (step + 1) % eval_interval == 0:
            val_idx = val_rng.integers(
                0, val_corpus.n_games, size=cfg.batch_size
            )
            val_batch = slice_batch(val_corpus, val_idx)
            val_loss = float(val_step(state.backbone, state.adapter, val_batch))
            logger.log_val(
                step=step + 1, val_loss=val_loss,
                val_source=cfg.pgn_val_split if not args.no_pgn else "random",
            )
        if (step + 1) % cfg.checkpoint_interval == 0:
            out = args.logs_dir / f"adapter_step_{step + 1:08d}"
            save_model(
                state.backbone, out,
                run_config=cfg.model_dump(),
                training_state={"step": int(state.step)},
            )
            if push_tracker:
                push_checkpoint_async(out, push_tracker)
        if should_shutdown():
            break
    if push_tracker:
        drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown()
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
