#!/usr/bin/env python3
"""LoRA behavioral cloning on Lichess games.

Trains LoRA adapters on frozen PAWN to predict human moves from
Lichess games in a given Elo band.

Usage:
    uv run python scripts/train_lora.py \
        --checkpoint /path/to/checkpoint.pt \
        --pgn /path/to/lichess_1200_1400.pgn \
        --output-dir lora_runs/elo_1200_1400
"""

from __future__ import annotations

import argparse
import gc
import json
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
from pawn.adapters.lora import LoRACLM
from pawn.gpu import configure_gpu, apply_gpu_config

from pawn.lichess_data import (
    prepare_lichess_dataset,
    LegalMaskBuilder,
    LichessDataset,
)


def parse_args():
    p = argparse.ArgumentParser(description="LoRA BC on Lichess games")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to PAWN checkpoint")
    p.add_argument("--pgn", type=str, required=True,
                    help="Path to Lichess PGN file (pre-filtered by Elo)")
    p.add_argument("--log-dir", type=str, default=None,
                    help="Parent log directory (default: <project>/logs)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Explicit output directory (overrides --log-dir)")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=4,
                    help="LoRA rank (default: 4)")
    p.add_argument("--lora-alpha", type=float, default=None,
                    help="LoRA alpha scaling (default: same as rank)")
    p.add_argument("--lora-targets", type=str, default="qkvo",
                    choices=["qkvo", "qv", "qkv"],
                    help="Which attention projections to adapt (default: qkvo)")
    p.add_argument("--lora-layers", type=str, default=None,
                    help="Comma-separated layer indices to adapt (default: all). "
                         "E.g. '0,1,6,7' for first/last two of 8 layers")
    p.add_argument("--lora-ffn", action="store_true",
                    help="Also apply LoRA to FFN projections (w_gate, w_up, w_down)")

    # Data
    p.add_argument("--max-games", type=int, default=12_000,
                    help="Max games to parse (train + val)")
    p.add_argument("--val-games", type=int, default=2_000,
                    help="Games to reserve for validation")
    p.add_argument("--min-ply", type=int, default=10,
                    help="Minimum game length in ply")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05,
                    help="Fraction of total steps for LR warmup")
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (epochs)")
    p.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs")

    # Device / precision
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true",
                    help="Disable automatic mixed precision")
    p.add_argument("--no-compile", action="store_true",
                    help="Disable torch.compile")
    p.add_argument("--sdpa-math", action="store_true",
                    help="Use MATH SDPA backend (workaround for ROCm flash attn + compile)")

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
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sparse_forward(model, ids, msk, legal_mask, use_amp, device):
    """Sparse forward: project only loss-masked positions through lm_head.

    Avoids materializing full (B, T, V) logits -- projects only N_valid
    positions, then applies legal mask at those positions only.
    """
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        hidden = model.forward_hidden(ids)          # (B, T, d_model)
        valid_hidden = hidden[msk]                   # (N_valid, d_model)
        valid_logits = model.project_head(valid_hidden)  # (N_valid, V)

    valid_legal = legal_mask[msk]                    # (N_valid, V)
    valid_logits = valid_logits.float()              # ensure float32 for loss
    valid_logits.masked_fill_(~valid_legal, float("-inf"))
    return valid_logits


@torch.no_grad()
def evaluate(model, dataloader, mask_builder, device, use_amp: bool = False):
    """Run validation and return average metrics."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_positions = 0

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        tgt = batch["targets"].to(device)
        msk = batch["loss_mask"].to(device)
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
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parent.parent.parent / "logs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = log_dir / f"lora_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    hf_branch = None
    if args.hf_repo:
        hf_branch = f"run/{out_dir.name}"

    device = args.device
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # Write config record to metrics.jsonl (dashboard reads this)
    metrics_path = out_dir / "metrics.jsonl"
    config_record = {
        "type": "config",
        "run_type": "lora",
        "checkpoint": str(args.checkpoint),
        "pgn": str(args.pgn),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "warmup_frac": args.warmup_frac,
        "max_grad_norm": args.max_grad_norm,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
        "lora_targets": args.lora_targets,
        "lora_layers": args.lora_layers,
        "lora_ffn": args.lora_ffn,
    }
    with open(metrics_path, "w") as f:
        f.write(json.dumps(config_record) + "\n")

    # Load backbone
    print(f"Loading backbone: {args.checkpoint}")
    backbone = load_backbone(args.checkpoint, device)
    lora_layers = tuple(int(x) for x in args.lora_layers.split(",")) if args.lora_layers else None
    model = LoRACLM(
        backbone,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        attn_targets=args.lora_targets,
        adapt_ffn=args.lora_ffn,
        layers=lora_layers,
    ).to(device)

    # Cache LoRA parameter list
    lora_params = model.lora_parameters()

    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {n_lora:,} / {n_total:,} total ({100*n_lora/n_total:.3f}%)")
    layers_str = args.lora_layers or "all"
    print(f"  rank={args.lora_rank}, alpha={args.lora_alpha or args.lora_rank}, "
          f"targets={args.lora_targets}, layers={layers_str}, ffn={args.lora_ffn}")

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

    train_ds = LichessDataset(data, start=0, end=n_train)
    val_ds = LichessDataset(data, start=n_train, end=n_total_games)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Pre-allocated legal mask builder
    mask_builder = LegalMaskBuilder(args.batch_size, max_ply=255, vocab_size=vocab_size,
                                    device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        lora_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    # Mixed precision
    use_amp = gpu_cfg["use_amp"]
    scaler = torch.amp.GradScaler() if use_amp else None

    # Baseline: evaluate frozen model before any training
    print("\nBaseline (zero LoRA):")
    baseline = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp)
    print(f"  loss={baseline['loss']:.4f}, top1={baseline['top1_accuracy']:.4%}, "
          f"top5={baseline['top5_accuracy']:.4%}")

    # Write baseline record
    with open(metrics_path, "a") as f:
        f.write(json.dumps({
            "type": "train",
            "epoch": -1, "step": 0,
            "train_loss": baseline["loss"], "train_top1": baseline["top1_accuracy"],
            "val_loss": baseline["loss"], "val_top1": baseline["top1_accuracy"],
            "val_top5": baseline["top5_accuracy"],
        }) + "\n")

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    val_metrics = baseline

    _shutdown_requested = False
    def _graceful_exit(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    print(f"\nTraining for up to {args.epochs} epochs ({total_steps} steps)")
    print(f"  Warmup: {warmup_steps} steps, LR: {args.lr}")
    if args.val_every > 1:
        print(f"  Validation every {args.val_every} epochs")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        epoch_positions = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["targets"].to(device)
            msk = batch["loss_mask"].to(device)
            legal_mask = mask_builder(batch)

            valid_logits = sparse_forward(model, ids, msk, legal_mask, use_amp, device)
            valid_targets = tgt[msk]

            loss = F.cross_entropy(valid_logits, valid_targets)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
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

        # Validation
        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            val_metrics = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp)

        # LoRA weight report (aggregate B norms -- the learned signal)
        lora_report = model.lora_weight_report()

        # Log
        record = {
            "type": "train",
            "epoch": epoch,
            "step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "epoch_time_s": dt,
            **{f"lora/{k}": v for k, v in lora_report.items()},
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
              f"val_top5={val_metrics['top5_accuracy']:.4%} | "
              f"{dt:.1f}s")

        # Early stopping
        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                from pawn.checkpoint import save_adapter_checkpoint
                save_adapter_checkpoint(
                    ckpt_dir / "best",
                    model.lora_state_dict(),
                    config=vars(args),
                    epoch=epoch,
                    step=global_step,
                    val_metrics=val_metrics,
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

    # Save final checkpoint
    from pawn.checkpoint import save_adapter_checkpoint
    save_adapter_checkpoint(
        ckpt_dir / "final",
        model.lora_state_dict(),
        config=vars(args),
        epoch=epoch,
        step=global_step,
        val_metrics=val_metrics,
    )
    if args.hf_repo and hf_branch:
        from pawn.checkpoint import push_checkpoint_to_hf
        try:
            push_checkpoint_to_hf(ckpt_dir / "final", args.hf_repo, hf_branch, step=global_step)
            print(f"Pushed to HF: {args.hf_repo}@{hf_branch}")
        except Exception as e:
            print(f"WARNING: HF push failed: {e}")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
