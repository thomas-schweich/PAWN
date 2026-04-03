#!/usr/bin/env python3
"""Unified PAWN training entry point.

Supports both pretraining (random games) and adapter finetuning (Lichess)
via a single Pydantic config model (``pawn.run_config.RunConfig``).

Usage:
    # From a JSON config file
    python3 scripts/train.py --config run.json

    # JSON config with CLI overrides
    python3 scripts/train.py --config run.json --lr 1e-4 --batch-size 128

    # Pure CLI — pretraining
    python3 scripts/train.py --run-type pretrain --variant base --local-checkpoints

    # Pure CLI — adapter
    python3 scripts/train.py --run-type adapter --strategy lora \\
        --checkpoint thomas-schweich/pawn-base \\
        --pgn thomas-schweich/pawn-lichess-full --local-checkpoints
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch.multiprocessing as mp
from pydantic import TypeAdapter

from pawn.run_config import AdapterConfig, PretrainConfig


# -----------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------


def _parse_cli(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI args into a flat dict suitable for Pydantic construction.

    Handles:
      --config PATH      load JSON base config
      --flag value       set field (underscore or hyphen)
      --bool-flag        set to True  (when no value follows)
      --no-bool-flag     set field to False
    """
    if argv is None:
        argv = sys.argv[1:]

    config_path: str | None = None
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            i += 1
            if i >= len(argv):
                _die("--config requires a path argument")
            config_path = argv[i]
            i += 1
            continue

        if not arg.startswith("--"):
            _die(f"Unexpected positional argument: {arg}")

        key = arg[2:]

        # --no-X -> X = False
        if key.startswith("no-"):
            canon = key[3:].replace("-", "_")
            overrides[canon] = False
            i += 1
            continue

        canon = key.replace("-", "_")

        # Peek at next token: if it is another flag or end-of-args, treat
        # the current flag as bool=True.
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            overrides[canon] = True
            i += 1
            continue

        # Otherwise consume the value.
        raw = argv[i + 1]
        overrides[canon] = _coerce(raw)
        i += 2

    base: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if not path.exists():
            _die(f"Config file not found: {config_path}")
        base = json.loads(path.read_text())

    base.update(overrides)
    return base


def _coerce(raw: str) -> Any:
    """Best-effort coercion of a CLI string to a Python value."""
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------
# Pretraining
# -----------------------------------------------------------------------


def run_pretrain(config: PretrainConfig) -> None:
    """Bridge ``PretrainConfig`` into ``CLMConfig`` + ``TrainingConfig``."""
    import torch

    from pawn.config import CLMConfig, TrainingConfig
    from pawn.trainer import CLMTrainer

    variant_factory = {
        "small": CLMConfig.small,
        "base": CLMConfig.base,
        "large": CLMConfig.large,
        "toy": CLMConfig.toy,
    }
    model_cfg = variant_factory[config.variant]()
    train_cfg = (
        TrainingConfig.toy() if config.variant == "toy" else TrainingConfig()
    )

    # Map RunConfig fields onto internal configs
    train_cfg.device = config.device
    if config.total_steps is not None:
        train_cfg.total_steps = config.total_steps
    train_cfg.batch_size = config.batch_size
    train_cfg.num_workers = config.num_workers
    train_cfg.accumulation_steps = config.accumulation_steps
    train_cfg.lr = config.lr
    train_cfg.weight_decay = config.weight_decay
    if config.warmup_steps is not None:
        train_cfg.warmup_steps = config.warmup_steps
    elif config.total_steps is not None:
        train_cfg.warmup_steps = int(config.warmup_frac * config.total_steps)
    else:
        train_cfg.warmup_steps = int(config.warmup_frac * train_cfg.total_steps)
    train_cfg.max_grad_norm = config.max_grad_norm
    train_cfg.discard_ply_limit = config.discard_ply_limit
    train_cfg.no_outcome_token = config.no_outcome_token
    train_cfg.mate_boost = config.mate_boost
    train_cfg.log_interval = config.log_interval
    if config.eval_interval is not None:
        train_cfg.eval_interval = config.eval_interval
    train_cfg.checkpoint_interval = config.checkpoint_interval
    train_cfg.pause_after_steps = config.pause_after_steps
    if config.log_dir:
        train_cfg.log_dir = config.log_dir
    train_cfg.use_wandb = config.wandb
    train_cfg.use_amp = config.amp_dtype != "none"

    if not torch.cuda.is_available() and config.device == "cuda":
        if config.variant == "toy":
            train_cfg.device = "cpu"
            print("CUDA not available, falling back to CPU (toy mode)")
        else:
            _die(
                "CUDA is required for full model training. "
                "Use --variant toy for CPU-based testing."
            )

    # Architecture overrides
    if config.d_model is not None:
        model_cfg.d_model = config.d_model
    if config.n_layers is not None:
        model_cfg.n_layers = config.n_layers
    if config.n_heads is not None:
        model_cfg.n_heads = config.n_heads
    if config.d_ff is not None:
        model_cfg.d_ff = config.d_ff

    print(f"Model config: {model_cfg}")
    print(f"Training config: {train_cfg}")

    trainer = CLMTrainer(train_cfg, model_cfg, hf_repo=config.hf_repo)

    if config.resume:
        trainer.load_checkpoint(config.resume)

    trainer.train()


# -----------------------------------------------------------------------
# Adapter finetuning
# -----------------------------------------------------------------------


def run_adapter(config: AdapterConfig) -> tuple[float, dict[str, Any]]:
    """Run adapter training. Returns ``(best_val_loss, final_val_metrics)``.

    Delegates to building-block functions in ``pawn.adapter_training``.
    """
    import argparse
    import json as json_mod

    import torch
    from torch.utils.data import DataLoader

    from pawn.adapter_training import (
        build_config_json,
        build_model,
        precompute_val_masks,
        rosa_build_phase3,
        rosa_mask_generation,
        rosa_warmup,
        train,
    )
    from pawn.config import CLMConfig
    from pawn.gpu import configure_gpu
    from pawn.lichess_data import (
        LegalMaskBuilder,
        LegalMaskCollate,
        LichessDataset,
        prepare_lichess_dataset,
    )
    from pawn.logging import MetricsLogger

    # Build an argparse.Namespace so existing functions work unchanged.
    args = argparse.Namespace(**config.model_dump())

    device = args.device
    amp_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "none": None,
    }
    amp_dtype = amp_dtype_map[args.amp_dtype]

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else Path(__file__).resolve().parent.parent / "logs"
    )
    logger = MetricsLogger(
        str(log_dir), run_prefix=args.strategy, device=device
    )
    out_dir = logger.run_dir

    print(f"Strategy: {args.strategy}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # GPU config
    gpu_cfg = configure_gpu(
        device,
        no_compile=args.no_compile,
        no_amp=(amp_dtype is None),
        sdpa_math=args.sdpa_math,
    )
    gpu_cfg["use_amp"] = amp_dtype is not None
    gpu_cfg["amp_dtype"] = amp_dtype

    # Build model
    vocab_size: int = CLMConfig().vocab_size
    if args.strategy == "rosa":
        model, _, _, _, _ = build_model(args, device)
        vocab_size = getattr(
            getattr(model, "cfg", None), "vocab_size", vocab_size
        )
    else:
        (
            model,
            trainable_params,
            param_count,
            state_dict_fn,
            weight_report_fn,
        ) = build_model(args, device)
        cfg_obj = getattr(model, "cfg", None) or getattr(
            getattr(model, "bb", None), "cfg", None
        )
        if cfg_obj is not None:
            vocab_size = cfg_obj.vocab_size
        print(f"Trainable params: {param_count:,}")

    # Prepare data
    max_ply = 255
    streaming = args.elo_min is not None or args.elo_max is not None

    from pawn.shard_loader import ShardedLichessDataset, load_val_shards

    if streaming:
        print(
            f"\nShard-parallel loading: {args.pgn} "
            f"[{args.elo_min}, {args.elo_max})"
        )
        val_data = load_val_shards(
            args.pgn,
            elo_min=args.elo_min,
            elo_max=args.elo_max,
            min_ply=args.min_ply,
            max_games=args.val_games,
            cache_dir=args.cache_dir,
        )
        val_ds = LichessDataset(val_data, start=0, end=val_data["n_games"])
        train_ds = ShardedLichessDataset(
            args.pgn,
            elo_min=args.elo_min,
            elo_max=args.elo_max,
            min_ply=args.min_ply,
            max_games=args.max_games,
            cache_dir=args.cache_dir,
        )
        print(
            f"  Val: {len(val_ds):,} games, "
            f"Train: {len(train_ds.shard_files)} shards"
        )
    else:
        print(f"\nPreparing data: {args.pgn}")
        data = prepare_lichess_dataset(
            args.pgn,
            max_ply=255,
            max_games=args.max_games or 1_000_000,
            min_ply=args.min_ply,
        )
        n_total = data["n_games"]
        n_val = min(args.val_games, n_total // 5)
        n_train = n_total - n_val
        print(f"  Train: {n_train} games, Val: {n_val} games")
        train_ds = LichessDataset(data, start=0, end=n_train).share_memory()
        val_ds = LichessDataset(data, start=n_train, end=n_total)

    collate = LegalMaskCollate(seq_len=max_ply + 1, vocab_size=vocab_size)
    n_workers = args.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not streaming,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
        collate_fn=collate,
        multiprocessing_context="spawn" if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    mask_builder = LegalMaskBuilder(
        args.batch_size, max_ply=255, vocab_size=vocab_size, device=device
    )

    # Precompute val legal indices
    val_legal_indices = precompute_val_masks(
        val_loader, mask_builder, vocab_size
    )
    print(
        f"  Precomputed legal masks for {len(val_legal_indices)} val batches"
    )

    # Log config & handle RoSA multi-phase
    if args.strategy == "rosa":
        logger.log_config(
            run_type=args.strategy,
            **{
                k: v
                for k, v in build_config_json(args, 0).items()
                if k not in ("strategy", "param_count")
            },
        )
        rosa_warmup(
            model,
            train_loader,
            mask_builder,
            args,
            device,
            amp_dtype,
            logger,
        )
        masks = rosa_mask_generation(
            model,
            train_loader,
            mask_builder,
            args,
            device,
            amp_dtype,
            logger,
        )
        print(f"\n=== RoSA Phase 3: {args.rosa_mode} training ===")
        (
            model,
            trainable_params,
            param_count,
            state_dict_fn,
            weight_report_fn,
        ) = rosa_build_phase3(model, masks, args, device)
        print(f"Trainable params: {param_count:,}")
    else:
        logger.log_config(
            run_type=args.strategy,
            **{
                k: v
                for k, v in build_config_json(args, param_count).items()
                if k not in ("strategy",)
            },
        )

    # Write config JSON
    config_dict = build_config_json(args, param_count)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json_mod.dump(config_dict, f, indent=2)
    print(f"Config written to {config_path}")

    # Train
    best_val_loss, final_metrics = train(
        model,
        trainable_params,
        train_loader,
        val_loader,
        mask_builder,
        val_legal_indices,
        logger,
        args,
        device,
        amp_dtype,
        gpu_cfg,
        state_dict_fn,
        weight_report_fn,
        param_count,
    )

    logger.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")

    return best_val_loss, final_metrics


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------


def main() -> None:
    raw = _parse_cli()

    run_type = raw.get("run_type")
    if run_type not in ("pretrain", "adapter"):
        _die(
            f"run_type must be 'pretrain' or 'adapter', got {run_type!r}. "
            "Specify via --run-type or in the JSON config."
        )

    ta = TypeAdapter(
        PretrainConfig if run_type == "pretrain" else AdapterConfig
    )
    config = ta.validate_python(raw)

    if isinstance(config, PretrainConfig):
        run_pretrain(config)
    elif isinstance(config, AdapterConfig):
        run_adapter(config)
    else:
        _die(f"Unknown run_type: {run_type}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
