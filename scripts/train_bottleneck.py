#!/usr/bin/env python3
"""Bottleneck adapter behavioral cloning on Lichess games.

Trains residual bottleneck adapters on frozen PAWN to predict human
moves from Lichess games in a given Elo band.

Usage:
    uv run python scripts/train_bottleneck.py \
        --checkpoint /path/to/checkpoint.pt \
        --pgn /path/to/lichess_1200_1400.pgn \
        --bottleneck-dim 8
"""

from __future__ import annotations

import argparse
import gc
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.config import CLMConfig, PAD_TOKEN
from pawn.model import PAWNCLM
from pawn.adapters.bottleneck import BottleneckCLM
from pawn.logging import MetricsLogger
from pawn.gpu import configure_gpu, apply_gpu_config

from pawn.lichess_data import (
    compute_legal_indices,
    prepare_lichess_dataset,
    LegalMaskBuilder,
    LegalMaskCollate,
    LichessDataset,
)


def parse_args():
    p = argparse.ArgumentParser(description="Bottleneck adapter BC on Lichess games")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to PAWN checkpoint")
    p.add_argument("--pgn", type=str, required=True,
                    help="Path to Lichess PGN file (pre-filtered by Elo)")
    p.add_argument("--log-dir", type=str, default=None,
                    help="Parent log directory (default: <project>/logs)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Explicit output directory (overrides --log-dir)")

    # Bottleneck config
    p.add_argument("--bottleneck-dim", type=int, default=8,
                    help="Bottleneck hidden dimension (default: 8)")
    p.add_argument("--no-adapt-attn", action="store_true",
                    help="Skip adapter after attention sublayer")
    p.add_argument("--no-adapt-ffn", action="store_true",
                    help="Skip adapter after FFN sublayer")
    p.add_argument("--adapter-layers", type=str, default=None,
                    help="Comma-separated layer indices (default: all)")
    p.add_argument("--attn-layers", type=str, default=None,
                    help="Comma-separated layers for attn adapters (overrides --no-adapt-attn)")
    p.add_argument("--ffn-layers", type=str, default=None,
                    help="Comma-separated layers for FFN adapters (overrides --no-adapt-ffn)")

    # Data
    p.add_argument("--max-games", type=int, default=12_000)
    p.add_argument("--val-games", type=int, default=2_000)
    p.add_argument("--min-ply", type=int, default=10)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-every", type=int, default=1)

    # Device / precision
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--sdpa-math", action="store_true",
                    help="Use MATH SDPA backend (workaround for ROCm flash attn + compile)")
    p.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers for legal mask prefetch (default: 8)")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from (best.pt or final.pt)")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="Push checkpoints to this HuggingFace repo (requires HF_TOKEN)")
    ckpt_group.add_argument("--local-checkpoints", action="store_true",
                            help="Save checkpoints locally only")

    return p.parse_args()


def load_backbone(checkpoint_path: str, device: str) -> PAWNCLM:
    from pawn.checkpoint import load_backbone_weights
    state_dict, model_config = load_backbone_weights(checkpoint_path, device)
    cfg = CLMConfig(**model_config) if model_config else CLMConfig()
    model = PAWNCLM(cfg).to(device)
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    model.eval()
    return model


def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sparse_forward(model, ids, msk, legal_mask, use_amp, device):
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        hidden = model.forward_hidden(ids)
        valid_hidden = hidden[msk]
        valid_logits = model.project_head(valid_hidden)

    valid_legal = legal_mask[msk]
    valid_logits = valid_logits.float()
    valid_logits.masked_fill_(~valid_legal, float("-inf"))
    return valid_logits


@torch.no_grad()
def evaluate(model, dataloader, mask_builder, device, use_amp: bool = False,
             precomputed_indices: list[torch.Tensor] | None = None):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_positions = 0

    for i, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["targets"].to(device, non_blocking=True)
        msk = batch["loss_mask"].to(device, non_blocking=True)
        if precomputed_indices is not None:
            legal_mask = mask_builder.scatter(precomputed_indices[i], ids.shape[0])
        elif "legal_indices" in batch:
            legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])
        else:
            legal_mask = mask_builder(batch)

        valid_logits = sparse_forward(model, ids, msk, legal_mask, use_amp, device)
        valid_targets = tgt[msk]

        n_pos = valid_targets.shape[0]
        if n_pos == 0:
            continue

        loss = F.cross_entropy(valid_logits, valid_targets)
        preds = valid_logits.argmax(dim=-1)
        top1 = (preds == valid_targets).float().mean().item()
        top5 = valid_logits.topk(5, dim=-1).indices
        top5_acc = (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

        total_loss += loss.item() * n_pos
        total_top1 += top1 * n_pos
        total_top5 += top5_acc * n_pos
        total_positions += n_pos

    if total_positions == 0:
        return {"loss": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}

    return {
        "loss": total_loss / total_positions,
        "top1_accuracy": total_top1 / total_positions,
        "top5_accuracy": total_top5 / total_positions,
    }


def main():
    args = parse_args()

    # Resolve output directory
    device = args.device
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parent.parent.parent / "logs"

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricsLogger.__new__(MetricsLogger)
        logger.run_dir = out_dir
        logger.metrics_path = out_dir / "metrics.jsonl"
        logger._file = open(logger.metrics_path, "a")
        logger._proc = __import__('psutil').Process()
        logger._device = device
        logger._start_time = time.time()
    else:
        logger = MetricsLogger(str(log_dir), run_prefix="bottleneck", device=device)
        out_dir = logger.run_dir

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    hf_branch = None
    if args.hf_repo:
        hf_branch = f"run/{out_dir.name}"

    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    adapter_layers = tuple(int(x) for x in args.adapter_layers.split(",")) if args.adapter_layers else None
    attn_layers = tuple(int(x) for x in args.attn_layers.split(",")) if args.attn_layers else None
    ffn_layers = tuple(int(x) for x in args.ffn_layers.split(",")) if args.ffn_layers else None

    # Write config record
    logger.log_config({
        "run_type": "bottleneck",
        "checkpoint": str(args.checkpoint),
        "pgn": str(args.pgn),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "warmup_frac": args.warmup_frac,
        "max_grad_norm": args.max_grad_norm,
        "bottleneck_dim": args.bottleneck_dim,
        "adapt_attn": not args.no_adapt_attn,
        "adapt_ffn": not args.no_adapt_ffn,
        "adapter_layers": args.adapter_layers,
        "attn_layers": args.attn_layers,
        "ffn_layers": args.ffn_layers,
    })

    # Load backbone
    print(f"Loading backbone: {args.checkpoint}")
    backbone = load_backbone(args.checkpoint, device)
    model = BottleneckCLM(
        backbone,
        bottleneck_dim=args.bottleneck_dim,
        adapt_attn=not args.no_adapt_attn,
        adapt_ffn=not args.no_adapt_ffn,
        layers=adapter_layers,
        attn_layers=attn_layers,
        ffn_layers=ffn_layers,
    ).to(device)

    adapter_params = model.adapter_parameters()
    n_adapt = sum(p.numel() for p in adapter_params)
    n_total = sum(p.numel() for p in model.parameters())
    positions = []
    if attn_layers is not None:
        positions.append(f"attn@{args.attn_layers}")
    elif not args.no_adapt_attn:
        positions.append("attn")
    if ffn_layers is not None:
        positions.append(f"ffn@{args.ffn_layers}")
    elif not args.no_adapt_ffn:
        positions.append("ffn")
    print(f"Adapter params: {n_adapt:,} / {n_total:,} total ({100*n_adapt/n_total:.3f}%)")
    print(f"  bottleneck_dim={args.bottleneck_dim}, positions={'+'.join(positions)}")

    # GPU auto-detection: compile, AMP, SDPA backend
    from pawn import model as model_module
    gpu_cfg = configure_gpu(
        device, no_compile=args.no_compile, no_amp=args.no_amp,
        sdpa_math=args.sdpa_math,
    )
    model.forward_hidden = apply_gpu_config(gpu_cfg, model_module, model.forward_hidden)

    # Prepare data
    print(f"\nPreparing Lichess data: {args.pgn}")
    data = prepare_lichess_dataset(
        args.pgn, max_ply=255, max_games=args.max_games, min_ply=args.min_ply,
    )
    n_total_games = data["n_games"]
    n_val = min(args.val_games, n_total_games // 5)
    n_train = n_total_games - n_val
    print(f"  Train: {n_train} games, Val: {n_val} games")

    vocab_size = backbone.cfg.vocab_size

    train_ds = LichessDataset(data, start=0, end=n_train).share_memory()
    val_ds = LichessDataset(data, start=n_train, end=n_total_games)

    max_ply = 255
    collate = LegalMaskCollate(seq_len=max_ply + 1, vocab_size=vocab_size)
    n_workers = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=n_workers > 0, collate_fn=collate,
        multiprocessing_context='spawn' if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    mask_builder = LegalMaskBuilder(args.batch_size, max_ply=255, vocab_size=vocab_size,
                                    device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    # Mixed precision
    use_amp = gpu_cfg["use_amp"]
    scaler = torch.amp.GradScaler() if use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        from pawn.checkpoint import load_adapter_checkpoint
        ckpt = load_adapter_checkpoint(args.resume, device=device)
        model.load_adapter_state_dict(ckpt["adapter_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_metrics", {}).get("loss", float("inf")))
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, "
              f"best_val_loss={best_val_loss:.4f}")
        del ckpt

    # Precompute val legal indices (fixed dataset — avoids Rust replay each epoch)
    val_legal_indices = []
    for batch in val_loader:
        move_ids = batch["move_ids"]
        if isinstance(move_ids, torch.Tensor):
            move_ids = move_ids.numpy()
        game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
        indices = compute_legal_indices(
            move_ids, game_lengths, mask_builder.T, vocab_size,
        )
        val_legal_indices.append(torch.from_numpy(indices).pin_memory())
    print(f"  Precomputed legal masks for {len(val_legal_indices)} val batches")

    if not args.resume:
        # Baseline
        print("\nBaseline (zero adapters):")
        baseline = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp,
                            precomputed_indices=val_legal_indices)
        print(f"  loss={baseline['loss']:.4f}, top1={baseline['top1_accuracy']:.4%}, "
              f"top5={baseline['top5_accuracy']:.4%}")

        logger.log({
            "train_loss": baseline["loss"], "train_top1": baseline["top1_accuracy"],
            "val_loss": baseline["loss"], "val_top1": baseline["top1_accuracy"],
            "val_top5": baseline["top5_accuracy"],
        }, step=0, epoch=-1)

    val_metrics = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp,
                           precomputed_indices=val_legal_indices) if args.resume else baseline

    _shutdown_requested = False
    def _graceful_exit(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    print(f"\nTraining for up to {args.epochs} epochs ({total_steps} steps)")
    print(f"  Warmup: {warmup_steps} steps, LR: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        epoch_positions = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])

            valid_logits = sparse_forward(model, ids, msk, legal_mask, use_amp, device)
            valid_targets = tgt[msk]

            loss = F.cross_entropy(valid_logits, valid_targets)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params, args.max_grad_norm)
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = valid_logits.argmax(dim=-1)
                top1 = (preds == valid_targets).float().mean().item()

            n_pos = valid_targets.shape[0]
            epoch_loss += loss.item() * n_pos
            epoch_top1 += top1 * n_pos
            epoch_positions += n_pos
            global_step += 1

        dt = time.time() - t0
        train_loss = epoch_loss / max(epoch_positions, 1)
        train_top1 = epoch_top1 / max(epoch_positions, 1)

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            val_metrics = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp,
                                   precomputed_indices=val_legal_indices)

        weight_report = model.adapter_weight_report()

        logger.log({
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "epoch_time_s": dt,
            **weight_report,
        }, step=global_step, epoch=epoch)

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
              f"val_top5={val_metrics['top5_accuracy']:.4%} | "
              f"{dt:.1f}s")

        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                from pawn.checkpoint import save_adapter_checkpoint
                save_adapter_checkpoint(
                    ckpt_dir / "best",
                    model.adapter_state_dict(),
                    config=vars(args),
                    epoch=epoch,
                    step=global_step,
                    val_metrics=val_metrics,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    extra={"best_val_loss": best_val_loss, "patience_counter": patience_counter},
                )
                if args.hf_repo and hf_branch:
                    from pawn.checkpoint import push_checkpoint_to_hf
                    try:
                        push_checkpoint_to_hf(ckpt_dir / "best", args.hf_repo, hf_branch, step=global_step)
                        print(f"Pushed to HF: {args.hf_repo}@{hf_branch}")
                    except Exception as e:
                        print(f"WARNING: HF push failed: {e}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                    break

        if _shutdown_requested:
            print("Shutdown requested, saving checkpoint...")
            break

    from pawn.checkpoint import save_adapter_checkpoint
    save_adapter_checkpoint(
        ckpt_dir / "final",
        model.adapter_state_dict(),
        config=vars(args),
        epoch=epoch,
        step=global_step,
        val_metrics=val_metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        extra={"best_val_loss": best_val_loss, "patience_counter": patience_counter},
    )
    if args.hf_repo and hf_branch:
        from pawn.checkpoint import push_checkpoint_to_hf
        try:
            push_checkpoint_to_hf(ckpt_dir / "final", args.hf_repo, hf_branch, step=global_step)
            print(f"Pushed to HF: {args.hf_repo}@{hf_branch}")
        except Exception as e:
            print(f"WARNING: HF push failed: {e}")

    logger.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
