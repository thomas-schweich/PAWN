#!/usr/bin/env python3
"""RoSA (Robust Sparse Adaptation) behavioral cloning on Lichess games.

Three training modes:
  rosa             -- Standard RoSA: LoRA warm-up -> gradient masks -> joint LoRA+sparse
  retro-sparse     -- Retrospective: LoRA warm-up -> masks -> restart sparse-only
  retro-bottleneck -- Retrospective: LoRA warm-up -> masks -> restart sparse+bottleneck

Usage:
    uv run python scripts/train_rosa.py \
        --checkpoint /path/to/checkpoint \
        --pgn /path/to/lichess.pgn \
        --mode rosa \
        --density 0.01 \
        --local-checkpoints
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
from pawn.adapters.rosa import RoSACLM, RetroBottleneckCLM, generate_gradient_masks
from pawn.adapters.sparse import SparseCLM, SparseLinear
from pawn.adapters.lora import ATTN_PRESETS, _FFN_TARGETS
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
    p = argparse.ArgumentParser(description="RoSA BC on Lichess games")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to PAWN checkpoint")
    p.add_argument("--pgn", type=str, required=True,
                    help="Path to Lichess PGN file (pre-filtered by Elo)")
    p.add_argument("--log-dir", type=str, default=None,
                    help="Parent log directory (default: <project>/logs)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Explicit output directory (overrides --log-dir)")

    # Mode
    p.add_argument("--mode", type=str, required=True,
                    choices=["rosa", "retro-sparse", "retro-bottleneck"],
                    help="Training mode")

    # LoRA config (used during warm-up in all modes)
    p.add_argument("--lora-rank", type=int, default=4,
                    help="LoRA rank (default: 4)")
    p.add_argument("--lora-alpha", type=float, default=None,
                    help="LoRA alpha scaling (default: same as rank)")
    p.add_argument("--lora-targets", type=str, default="qkvo",
                    choices=["qkvo", "qv", "qkv"],
                    help="Which attention projections to adapt (default: qkvo)")
    p.add_argument("--lora-ffn", action="store_true",
                    help="Also apply adapters to FFN projections")

    # Sparse config
    p.add_argument("--density", type=float, default=0.01,
                    help="Sparse mask density (default: 0.01)")

    # Mask generation
    p.add_argument("--warmup-steps", type=int, default=128,
                    help="LoRA-only warm-up steps before mask generation (default: 128)")
    p.add_argument("--warmup-lr", type=float, default=None,
                    help="Learning rate for warm-up phase (default: same as --lr)")
    p.add_argument("--mask-samples", type=int, default=32,
                    help="Batches for gradient accumulation during mask generation (default: 32)")
    p.add_argument("--grad-alpha", type=int, default=2, choices=[1, 2],
                    help="Gradient accumulation exponent: 1=mean, 2=Fisher (default: 2)")

    # RoSA-specific
    p.add_argument("--restart-lora", action="store_true", default=True,
                    help="Re-initialize LoRA after mask generation (default: True)")
    p.add_argument("--no-restart-lora", action="store_false", dest="restart_lora",
                    help="Keep warm-up LoRA weights for joint training")

    # Bottleneck (retro-bottleneck mode only)
    p.add_argument("--bottleneck-dim", type=int, default=8,
                    help="Bottleneck adapter dimension (retro-bottleneck only, default: 8)")

    # Data
    p.add_argument("--max-games", type=int, default=12_000)
    p.add_argument("--val-games", type=int, default=2_000)
    p.add_argument("--min-ply", type=int, default=10)

    # Training (Phase 3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05,
                    help="Fraction of Phase 3 steps for LR warmup")
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (epochs)")
    p.add_argument("--val-every", type=int, default=1)

    # Device / precision
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--sdpa-math", action="store_true",
                    help="Use MATH SDPA backend (workaround for ROCm flash attn + compile)")
    p.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers for legal mask prefetch (default: 8)")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="Push checkpoints to this HuggingFace repo")
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
    """Sparse forward: project only loss-masked positions through lm_head."""
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        hidden = model.forward_hidden(ids, msk)
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


# ---------------------------------------------------------------------------
# Phase 1: LoRA warm-up
# ---------------------------------------------------------------------------

def run_warmup(model, train_loader, mask_builder, args, device, use_amp):
    """Train LoRA-only for warmup_steps steps. Returns step count."""
    lr = args.warmup_lr if args.warmup_lr is not None else args.lr
    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler() if use_amp else None

    model.train()
    step = 0
    total_loss = 0.0
    t0 = time.time()

    print(f"\n=== Phase 1: LoRA warm-up ({args.warmup_steps} steps, lr={lr}) ===")

    while step < args.warmup_steps:
        for batch in train_loader:
            if step >= args.warmup_steps:
                break

            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            if "legal_indices" in batch:
                legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])
            else:
                legal_mask = mask_builder(batch)

            valid_logits = sparse_forward(model, ids, msk, legal_mask, use_amp, device)
            valid_targets = tgt[msk]
            if valid_targets.shape[0] == 0:
                continue

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

            total_loss += loss.item()
            step += 1

            if step % 32 == 0 or step == args.warmup_steps:
                avg = total_loss / step
                print(f"  Warmup step {step}/{args.warmup_steps} | loss={avg:.4f}")

    dt = time.time() - t0
    print(f"  Warm-up complete in {dt:.1f}s (avg loss={total_loss / max(step, 1):.4f})")
    return step


# ---------------------------------------------------------------------------
# Phase 2: Mask generation
# ---------------------------------------------------------------------------

def run_mask_generation(model, train_loader, mask_builder, args, device, use_amp):
    """Generate gradient-based sparse masks. Returns mask dict."""
    print(f"\n=== Phase 2: Mask generation (density={args.density}, "
          f"alpha={args.grad_alpha}, samples={args.mask_samples}) ===")

    masks = generate_gradient_masks(
        model, train_loader, mask_builder,
        density=args.density, alpha=args.grad_alpha,
        device=device, use_amp=use_amp, max_batches=args.mask_samples,
    )

    # Log mask statistics
    total_active = 0
    total_elements = 0
    for key, mask in masks.items():
        n_active = mask.sum().item()
        n_total = mask.numel()
        total_active += n_active
        total_elements += n_total
        print(f"  {key}: {n_active:,} / {n_total:,} ({100*n_active/n_total:.2f}%)")

    print(f"  Total: {total_active:,} / {total_elements:,} "
          f"({100*total_active/total_elements:.2f}%)")

    return masks


# ---------------------------------------------------------------------------
# Phase 3: Main training loop
# ---------------------------------------------------------------------------

def train_loop(model, adapter_params, train_loader, val_loader, mask_builder,
               val_legal_indices, logger, args, device, use_amp, gpu_cfg,
               weight_report_fn):
    """Standard epoch-based training loop for Phase 3."""
    from pawn import model as model_module
    from pawn.checkpoint import save_adapter_checkpoint, push_checkpoint_to_hf

    # Compile forward_hidden for Phase 3
    model.forward_hidden = apply_gpu_config(gpu_cfg, model_module, model.forward_hidden)

    optimizer = torch.optim.AdamW(
        adapter_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler() if use_amp else None

    # Baseline
    print("\nBaseline (zero/identity adapters):")
    baseline = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp,
                        precomputed_indices=val_legal_indices)
    print(f"  loss={baseline['loss']:.4f}, top1={baseline['top1_accuracy']:.4%}, "
          f"top5={baseline['top5_accuracy']:.4%}")

    logger.log_train(step=0, epoch=-1,
        train_loss=baseline["loss"], train_top1=baseline["top1_accuracy"],
        val_loss=baseline["loss"], val_top1=baseline["top1_accuracy"],
        val_top5=baseline["top5_accuracy"],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    val_metrics = baseline

    ckpt_dir = logger.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    hf_branch = None
    if args.hf_repo:
        hf_branch = f"run/{logger.run_dir.name}"

    _shutdown_requested = False
    def _graceful_exit(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    print(f"\n=== Phase 3: Main training ({args.epochs} epochs, {total_steps} steps) ===")
    print(f"  LR warmup: {warmup_steps} steps, LR: {args.lr}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        epoch_positions = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            if "legal_indices" in batch:
                legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])
            else:
                legal_mask = mask_builder(batch)

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
            val_metrics = evaluate(model, val_loader, mask_builder, device,
                                   use_amp=use_amp, precomputed_indices=val_legal_indices)

        report = weight_report_fn()

        logger.log_train(step=global_step, epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_top1=train_top1,
            val_loss=val_metrics["loss"],
            val_top1=val_metrics["top1_accuracy"],
            val_top5=val_metrics["top5_accuracy"],
            epoch_time_s=dt,
            **report,
        )

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
              f"val_top5={val_metrics['top5_accuracy']:.4%} | "
              f"{dt:.1f}s")

        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
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
                    try:
                        push_checkpoint_to_hf(ckpt_dir / "best", args.hf_repo, hf_branch,
                                              step=global_step)
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
        try:
            push_checkpoint_to_hf(ckpt_dir / "final", args.hf_repo, hf_branch,
                                  step=global_step)
            print(f"Pushed to HF: {args.hf_repo}@{hf_branch}")
        except Exception as e:
            print(f"WARNING: HF push failed: {e}")

    return best_val_loss


# ---------------------------------------------------------------------------
# Mode-specific setup
# ---------------------------------------------------------------------------

def setup_rosa(model, masks, args):
    """Standard RoSA: apply masks, optionally reinit LoRA, train jointly."""
    model.set_masks(masks)
    if args.restart_lora:
        model.reinit_lora()
    params = model.adapter_parameters()
    n_lora = sum(p.numel() for p in model.lora_parameters())
    n_sparse = model.n_active_sparse_params()
    n_total = sum(p.numel() for p in params)
    print(f"\nRoSA joint training: {n_total:,} trainable params")
    print(f"  LoRA: {n_lora:,}, Sparse active: {n_sparse:,}")
    return model, params


def _make_sparse_with_masks(masks, args, device):
    """Reload backbone, create SparseCLM, overwrite random masks with gradient-derived ones."""
    backbone = load_backbone(args.checkpoint, device)
    attn_targets = ATTN_PRESETS[args.lora_targets]

    sparse_model = SparseCLM(
        backbone, density=args.density,
        attn_targets=attn_targets,
        adapt_ffn=args.lora_ffn,
    )

    # Overwrite random masks with gradient-derived masks
    for layer_idx in range(len(backbone.layers)):
        block = backbone.get_block(layer_idx)
        for proj_name in attn_targets:
            module = getattr(block.attn, proj_name, None)
            if isinstance(module, SparseLinear):
                key = f"layer{layer_idx}.{proj_name}"
                if key in masks:
                    module.mask.copy_(masks[key])
        if args.lora_ffn:
            for proj_name in _FFN_TARGETS:
                module = getattr(block.ffn, proj_name, None)
                if isinstance(module, SparseLinear):
                    key = f"layer{layer_idx}.{proj_name}"
                    if key in masks:
                        module.mask.copy_(masks[key])

    return sparse_model


def setup_retro_sparse(masks, args, device):
    """Retrospective sparse-only: reload backbone, apply gradient masks."""
    print("\nReloading fresh backbone for retrospective sparse training...")
    sparse_model = _make_sparse_with_masks(masks, args, device)

    params = sparse_model.sparse_parameters()
    n_active = sparse_model.n_active_params()
    n_total = sum(p.numel() for p in params)
    print(f"Retro-sparse: {n_active:,} active / {n_total:,} total sparse params")
    return sparse_model, params


def setup_retro_bottleneck(masks, args, device):
    """Retrospective sparse + bottleneck: reload, apply masks, add bottlenecks."""
    print("\nReloading fresh backbone for retrospective sparse+bottleneck training...")
    sparse_model = _make_sparse_with_masks(masks, args, device)

    # Wrap with bottleneck adapters
    model = RetroBottleneckCLM(
        sparse_model.backbone,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    params = model.adapter_parameters()
    n_sparse = len(model.sparse_parameters())
    n_bottleneck = sum(p.numel() for p in model.bottleneck_parameters())
    n_total = sum(p.numel() for p in params)
    print(f"Retro-bottleneck: {n_total:,} trainable params")
    print(f"  Sparse deltas: {n_sparse}, Bottleneck params: {n_bottleneck:,}")
    return model, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = args.device
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parent.parent.parent / "logs"

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import psutil as _psutil
        logger = MetricsLogger.__new__(MetricsLogger)
        logger.slug = ""
        logger.run_dir = out_dir
        logger.metrics_path = out_dir / "metrics.jsonl"
        logger._file = open(logger.metrics_path, "a")
        logger._proc = _psutil.Process()
        logger._device = device
        logger._start_time = time.time()
    else:
        logger = MetricsLogger(str(log_dir), run_prefix=f"rosa-{args.mode}", device=device)
        out_dir = logger.run_dir

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # Write config record
    logger.log_config(
        run_type="rosa",
        mode=args.mode,
        checkpoint=str(args.checkpoint),
        pgn=str(args.pgn),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        warmup_frac=args.warmup_frac,
        max_grad_norm=args.max_grad_norm,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
        lora_targets=args.lora_targets,
        lora_ffn=args.lora_ffn,
        density=args.density,
        warmup_steps=args.warmup_steps,
        mask_samples=args.mask_samples,
        grad_alpha=args.grad_alpha,
        restart_lora=args.restart_lora,
        bottleneck_dim=args.bottleneck_dim if args.mode == "retro-bottleneck" else None,
    )

    # -----------------------------------------------------------------------
    # Prepare data
    # -----------------------------------------------------------------------
    print(f"\nPreparing Lichess data: {args.pgn}")
    data = prepare_lichess_dataset(
        args.pgn, max_ply=255, max_games=args.max_games, min_ply=args.min_ply,
    )
    n_total_games = data["n_games"]
    n_val = min(args.val_games, n_total_games // 5)
    n_train = n_total_games - n_val
    print(f"  Train: {n_train} games, Val: {n_val} games")

    train_ds = LichessDataset(data, start=0, end=n_train).share_memory()
    val_ds = LichessDataset(data, start=n_train, end=n_total_games)

    vocab_size = CLMConfig().vocab_size  # 4278
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

    mask_builder = LegalMaskBuilder(
        args.batch_size, max_ply=255, vocab_size=vocab_size, device=device,
    )

    # GPU config (don't compile yet -- save that for Phase 3)
    from pawn import model as model_module
    gpu_cfg = configure_gpu(
        device, no_compile=True, no_amp=args.no_amp,
        sdpa_math=args.sdpa_math,
    )
    use_amp = gpu_cfg["use_amp"]

    # Precompute val legal indices
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

    # -----------------------------------------------------------------------
    # Phase 1: LoRA warm-up
    # -----------------------------------------------------------------------
    print(f"\nLoading backbone: {args.checkpoint}")
    backbone = load_backbone(args.checkpoint, device)
    warmup_model = RoSACLM(
        backbone, rank=args.lora_rank, alpha=args.lora_alpha,
        attn_targets=args.lora_targets, adapt_ffn=args.lora_ffn,
        lora_enabled=True, sparse_enabled=False,
    ).to(device)

    run_warmup(warmup_model, train_loader, mask_builder, args, device, use_amp)

    # -----------------------------------------------------------------------
    # Phase 2: Mask generation
    # -----------------------------------------------------------------------
    masks = run_mask_generation(
        warmup_model, train_loader, mask_builder, args, device, use_amp,
    )

    # Save warm-up LoRA weights for posterity
    print("\nSaving warm-up LoRA weights...")
    from pawn.checkpoint import save_adapter_checkpoint  # used here and below
    save_adapter_checkpoint(
        ckpt_dir / "warmup",
        warmup_model.adapter_state_dict(),
        config=vars(args),
        epoch=-1,
        step=args.warmup_steps,
        val_metrics=None,
    )
    print(f"  Saved to {ckpt_dir / 'warmup'}")

    # -----------------------------------------------------------------------
    # Phase 3: Mode-dependent training
    # -----------------------------------------------------------------------

    # Re-enable compile for Phase 3
    gpu_cfg = configure_gpu(
        device, no_compile=args.no_compile, no_amp=args.no_amp,
        sdpa_math=args.sdpa_math,
    )

    if args.mode == "rosa":
        model, adapter_params = setup_rosa(warmup_model, masks, args)
        weight_report_fn = model.adapter_weight_report
    else:
        # Retrospective modes: free warm-up model, reload backbone
        del warmup_model
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

        if args.mode == "retro-sparse":
            model, adapter_params = setup_retro_sparse(masks, args, device)
            weight_report_fn = model.sparse_weight_report
        else:  # retro-bottleneck
            model, adapter_params = setup_retro_bottleneck(masks, args, device)
            weight_report_fn = model.adapter_weight_report

    best_val_loss = train_loop(
        model, adapter_params, train_loader, val_loader, mask_builder,
        val_legal_indices, logger, args, device, use_amp, gpu_cfg,
        weight_report_fn,
    )

    logger.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
