#!/usr/bin/env python3
"""Train a tiny standalone transformer on Lichess games.

Baseline experiment: how well can ~524K parameters do WITHOUT a pretrained
backbone? Uses the same data pipeline, loss, and legal mask setup as the
adapter experiments for a fair comparison.

Usage:
    uv run python scripts/train_tiny.py \
        --pgn /path/to/lichess_1000_1100.pgn \
        --d-model 84 --n-layers 2 --n-heads 4 --d-ff 336
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.config import PAD_TOKEN
from pawn.logging import MetricsLogger
from pawn.lichess_data import (
    compute_legal_indices,
    prepare_lichess_dataset,
    LegalMaskBuilder,
    LegalMaskCollate,
    LichessDataset,
)


# ---------------------------------------------------------------------------
# Tiny transformer
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.wo(attn_out)
        h = self.ffn_norm(x)
        x = x + self.w2(F.gelu(self.w1(h)))
        return x


def _apply_rope(x, rope_cos, rope_sin):
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_r.unbind(-1)
    out0 = x0 * rope_cos - x1 * rope_sin
    out1 = x0 * rope_sin + x1 * rope_cos
    return torch.stack([out0, out1], dim=-1).reshape(x.shape).to(x.dtype)


class TinyChessLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, max_seq_len: int = 256,
                 rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.layers = nn.ModuleList([
            TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        # Weight-tied output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # RoPE
        head_dim = d_model // n_heads
        freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer("rope_cos", angles.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("rope_sin", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward_hidden(self, input_ids):
        x = self.embed(input_ids)
        cos = self.rope_cos[:, :, :x.shape[1], :]
        sin = self.rope_sin[:, :, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def project_head(self, hidden):
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# Training utilities (same as bottleneck)
# ---------------------------------------------------------------------------

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
def evaluate(model, dataloader, mask_builder, device, use_amp=False,
             precomputed_indices=None):
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


def cosine_warmup_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tiny standalone chess LM baseline")
    p.add_argument("--pgn", type=str, required=True)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)

    # Architecture
    p.add_argument("--d-model", type=int, default=84)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=336)

    # Data
    p.add_argument("--max-games", type=int, default=100_000)
    p.add_argument("--val-games", type=int, default=10_000)
    p.add_argument("--min-ply", type=int, default=10)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-every", type=int, default=1)

    # Device
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--num-workers", type=int, default=3)

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="Push checkpoints to this HuggingFace repo (requires HF_TOKEN)")
    ckpt_group.add_argument("--local-checkpoints", action="store_true",
                            help="Save checkpoints locally only")

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    vocab_size = 4278
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
        logger = MetricsLogger(str(log_dir), run_prefix="tiny", device=device)
        out_dir = logger.run_dir

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    hf_branch = None
    if args.hf_repo:
        hf_branch = f"run/{out_dir.name}"

    # Build model
    model = TinyChessLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    # Weight tying means embed and lm_head share, so unique params:
    n_unique = n_params - model.embed.weight.numel()  # subtract double-counted
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"Model: d={args.d_model}, layers={args.n_layers}, heads={args.n_heads}, ff={args.d_ff}")
    print(f"Parameters: {n_params:,} total, {n_unique:,} unique (weight-tied)")

    # Config record
    logger.log_config({
        "run_type": "tiny",
        "pgn": str(args.pgn),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "n_params_unique": n_unique,
    })

    # Data
    print(f"\nPreparing data: {args.pgn}")
    data = prepare_lichess_dataset(
        args.pgn, max_ply=255, max_games=args.max_games, min_ply=args.min_ply,
    )
    n_total = data["n_games"]
    n_val = min(args.val_games, n_total // 5)
    n_train = n_total - n_val
    print(f"  Train: {n_train}, Val: {n_val}")

    train_ds = LichessDataset(data, start=0, end=n_train).share_memory()
    val_ds = LichessDataset(data, start=n_train, end=n_total)

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

    mask_builder = LegalMaskBuilder(args.batch_size, max_ply=255,
                                    vocab_size=vocab_size, device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    use_amp = not args.no_amp and device.startswith('cuda')
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP enabled (fp16)")

    # Precompute val legal indices
    val_legal_indices = []
    for batch in val_loader:
        move_ids = batch["move_ids"]
        if isinstance(move_ids, torch.Tensor):
            move_ids = move_ids.numpy()
        game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
        indices = compute_legal_indices(move_ids, game_lengths, mask_builder.T, vocab_size)
        val_legal_indices.append(torch.from_numpy(indices).pin_memory())
    print(f"  Precomputed legal masks for {len(val_legal_indices)} val batches")

    # Baseline
    print("\nBaseline (random init):")
    baseline = evaluate(model, val_loader, mask_builder, device, use_amp=use_amp,
                        precomputed_indices=val_legal_indices)
    print(f"  loss={baseline['loss']:.4f}, top1={baseline['top1_accuracy']:.4%}, "
          f"top5={baseline['top5_accuracy']:.4%}")

    logger.log({
        "train_loss": baseline["loss"], "train_top1": baseline["top1_accuracy"],
        "val_loss": baseline["loss"], "val_top1": baseline["top1_accuracy"],
        "val_top5": baseline["top5_accuracy"],
    }, step=0, epoch=-1)

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
            legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])

            valid_logits = sparse_forward(model, ids, msk, legal_mask, use_amp, device)
            valid_targets = tgt[msk]
            loss = F.cross_entropy(valid_logits, valid_targets)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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

        logger.log({
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "train_top1": train_top1,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "epoch_time_s": dt,
        }, step=global_step, epoch=epoch)

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
              f"val_top5={val_metrics['top5_accuracy']:.4%} | {dt:.1f}s")

        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                from pawn.checkpoint import save_pretrain_checkpoint
                save_pretrain_checkpoint(
                    ckpt_dir / "best",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    model_config={
                        "d_model": args.d_model,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "d_ff": args.d_ff,
                    },
                    training_config=vars(args),
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

    logger.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
