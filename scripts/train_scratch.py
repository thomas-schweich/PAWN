#!/usr/bin/env python3
"""Train a causal LM from scratch on Lichess games (no pretraining).

Uses the same transformer architecture as PAWN (factored embeddings, SwiGLU,
RoPE) but initializes from random weights and trains directly on a single Elo
band — no random-game pretraining, no adapter.  The resulting model is NOT
playstyle-agnostic; it learns one band's move distribution end-to-end.

Designed to answer: does PAWN pretraining help at ~10M param scale, or is the
move-sequence input representation itself the bottleneck vs board-tensor models
like MAIA?

Usage:
    # ~9.5M params on 1700-1800 Elo, 100K steps
    uv run python scripts/train_scratch.py \
        --variant small \
        --pgn thomas-schweich/pawn-lichess-full --elo-min 1700 --elo-max 1800 \
        --max-games 10000000 --total-steps 100000 --eval-interval 5000 \
        --batch-size 256 --local-checkpoints
"""

from __future__ import annotations

import argparse
import copy
import math
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.config import CLMConfig, PAD_TOKEN
from pawn.model import PAWNCLM
from pawn.logging import MetricsLogger
from pawn.gpu import configure_gpu, apply_gpu_config
from pawn.lichess_data import (
    compute_legal_indices,
    prepare_lichess_dataset,
    LegalMaskBuilder,
    LegalMaskCollate,
    LichessDataset,
)


# ---------------------------------------------------------------------------
# Wrapper for sparse training
# ---------------------------------------------------------------------------

class ScratchWrapper(nn.Module):
    """Expose forward_hidden/project_head for the sparse logit projection path.

    The base model's forward() returns full (B,T,V) logits.  For training we
    only need logits at loss-masked positions, so this wrapper exposes the
    hidden-state path used by the adapter training scripts.
    """

    def __init__(self, pawn: PAWNCLM):
        super().__init__()
        self.pawn = pawn
        self.cfg = pawn.cfg

    def forward_hidden(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.pawn.embed(input_ids)
        T = input_ids.shape[1]
        causal = self.pawn.causal_mask[:T, :T]
        padding = attn_mask.unsqueeze(1).unsqueeze(2)
        mask = causal.unsqueeze(0) & padding
        rope_cos = self.pawn.rope_cos[:, :, :T, :]
        rope_sin = self.pawn.rope_sin[:, :, :T, :]
        for layer in self.pawn.layers:
            x = layer(x, rope_cos, rope_sin, mask)
        return self.pawn.final_norm(x)

    def project_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.pawn.lm_head(hidden)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def sparse_forward(model, ids, loss_mask, legal_mask, use_amp, device):
    attn_mask = (ids != PAD_TOKEN)
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        hidden = model.forward_hidden(ids, attn_mask)
        valid_hidden = hidden[loss_mask]
        valid_logits = model.project_head(valid_hidden)
    valid_legal = legal_mask[loss_mask]
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
    p = argparse.ArgumentParser(description="Train a causal LM from scratch on Lichess games")
    p.add_argument("--pgn", type=str, required=True)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)

    # Architecture
    p.add_argument("--variant", type=str, default="small",
                    choices=["small", "base", "large", "toy"],
                    help="Architecture variant (default: small, ~9.5M params)")

    # Data
    p.add_argument("--max-games", type=int, default=10_000_000)
    p.add_argument("--val-games", type=int, default=10_000)
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--elo-min", type=int, default=None,
                    help="Min Elo (inclusive, enables shard-parallel loading)")
    p.add_argument("--elo-max", type=int, default=None,
                    help="Max Elo (exclusive)")
    p.add_argument("--cache-dir", type=str, default=None,
                    help="Prefetch filtered shards to this dir")

    # Training
    p.add_argument("--total-steps", type=int, default=None,
                    help="Stop after this many steps (overrides epoch count)")
    p.add_argument("--eval-interval", type=int, default=None,
                    help="Evaluate every N steps (default: every epoch)")
    p.add_argument("--log-interval", type=int, default=100,
                    help="Log training metrics every N steps")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
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
    p.add_argument("--sdpa-math", action="store_true",
                    help="Use MATH SDPA backend (required for ROCm + compile)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint directory to resume from")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="Push checkpoints to this HuggingFace repo")
    ckpt_group.add_argument("--local-checkpoints", action="store_true",
                            help="Save checkpoints locally only")

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parent.parent / "logs"

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
        logger = MetricsLogger(str(log_dir), run_prefix=f"scratch-{args.variant}",
                               device=device)
        out_dir = logger.run_dir

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    hf_branch = None
    if args.hf_repo:
        hf_branch = f"run/{out_dir.name}"

    # Build model
    cfg = getattr(CLMConfig, args.variant)()
    pawn = PAWNCLM(cfg).to(device)
    model = ScratchWrapper(pawn).to(device)
    vocab_size = cfg.vocab_size

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"Model: CLMConfig.{args.variant}() — {n_params:,} params (from scratch)")

    # GPU config
    from pawn import model as model_module
    gpu_cfg = configure_gpu(
        device, no_compile=args.no_compile, no_amp=args.no_amp,
        sdpa_math=args.sdpa_math,
    )
    model.forward_hidden = apply_gpu_config(gpu_cfg, model_module, model.forward_hidden)
    use_amp = gpu_cfg["use_amp"]

    # Config record
    logger.log_config(
        run_type=f"scratch-{args.variant}",
        pgn=str(args.pgn),
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_params=n_params,
        elo_min=args.elo_min,
        elo_max=args.elo_max,
    )

    # Data
    streaming = args.elo_min is not None or args.elo_max is not None

    if streaming:
        from pawn.shard_loader import ShardedLichessDataset, load_val_shards
        print(f"\nShard-parallel loading: {args.pgn} [{args.elo_min}, {args.elo_max})")

        val_data = load_val_shards(
            args.pgn, elo_min=args.elo_min, elo_max=args.elo_max,
            min_ply=args.min_ply, max_games=args.val_games,
            cache_dir=args.cache_dir,
        )
        val_ds = LichessDataset(val_data, start=0, end=val_data["n_games"])

        train_ds = ShardedLichessDataset(
            args.pgn, elo_min=args.elo_min, elo_max=args.elo_max,
            min_ply=args.min_ply, max_games=args.max_games,
            cache_dir=args.cache_dir,
        )
        print(f"  Val: {len(val_ds):,} games, Train: {len(train_ds.shard_files)} shards")
    else:
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
        train_ds, batch_size=args.batch_size,
        shuffle=not streaming,
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
    if args.total_steps is not None:
        total_steps = args.total_steps
    elif streaming:
        est_games = min(args.max_games or 1_000_000, len(train_ds.shard_files) * 60_000)
        total_steps = args.epochs * (est_games // args.batch_size)
        print(f"  Estimated ~{est_games:,} games, ~{total_steps:,} total steps")
    else:
        total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

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

    logger.log_train(step=0, epoch=-1,
        train_loss=baseline["loss"], train_top1=baseline["top1_accuracy"],
        val_loss=baseline["loss"], val_top1=baseline["top1_accuracy"],
        val_top5=baseline["top5_accuracy"],
    )

    def _do_eval():
        return evaluate(model, val_loader, mask_builder, device, use_amp=use_amp,
                        precomputed_indices=val_legal_indices)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    val_metrics = baseline

    if args.resume:
        from pawn.checkpoint import load_pretrain_checkpoint
        print(f"\nResuming from: {args.resume}")
        ckpt = load_pretrain_checkpoint(
            args.resume, model=model.pawn, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device,
        )
        global_step = ckpt["global_step"]
        tc = ckpt.get("training_config") or {}
        best_val_loss = tc.get("best_val_loss", float("inf"))
        patience_counter = tc.get("patience_counter", 0)
        start_epoch = tc.get("epoch", 0)
        print(f"  Resumed at step {global_step}, best_val_loss={best_val_loss:.4f}")
        del ckpt
        val_metrics = _do_eval()

    _shutdown_requested = False
    def _graceful_exit(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    eval_interval = args.eval_interval
    step_limit = args.total_steps

    print(f"\nTraining for up to {args.epochs} epochs ({total_steps} steps)")
    print(f"  Warmup: {warmup_steps} steps, LR: {args.lr}")
    if eval_interval:
        print(f"  Eval every {eval_interval} steps")

    _model_config = {
        "variant": args.variant,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
    }
    _save_thread: threading.Thread | None = None

    class _StateSnapshot:
        """Fake object whose .state_dict() returns a pre-computed snapshot."""
        def __init__(self, sd):
            self._sd = sd
        def state_dict(self):
            return self._sd

    def _save_checkpoint(tag, ep):
        """Snapshot state on main thread, write to disk in background."""
        nonlocal _save_thread
        if _save_thread is not None:
            _save_thread.join()

        from pawn.checkpoint import save_pretrain_checkpoint, push_checkpoint_to_hf

        # Snapshot everything now — training will mutate the live objects
        model_snap = _StateSnapshot(
            {k: v.cpu().clone() for k, v in model.pawn.state_dict().items()}
        )
        opt_snap = _StateSnapshot(copy.deepcopy(optimizer.state_dict()))
        sched_snap = _StateSnapshot(scheduler.state_dict().copy())
        scaler_snap = _StateSnapshot(scaler.state_dict().copy()) if scaler else None
        step = global_step
        train_cfg = {
            **vars(args),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "epoch": ep,
        }

        def _write():
            save_pretrain_checkpoint(
                ckpt_dir / tag,
                model=model_snap,
                optimizer=opt_snap,
                scheduler=sched_snap,
                scaler=scaler_snap,
                global_step=step,
                model_config=_model_config,
                training_config=train_cfg,
            )
            if args.hf_repo and hf_branch:
                try:
                    push_checkpoint_to_hf(ckpt_dir / tag, args.hf_repo, hf_branch,
                                          step=step)
                    print(f"Pushed to HF: {args.hf_repo}@{hf_branch}")
                except Exception as e:
                    print(f"WARNING: HF push failed: {e}")

        _save_thread = threading.Thread(target=_write, daemon=True)
        _save_thread.start()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        epoch_positions = 0
        log_loss = 0.0
        log_top1 = 0.0
        log_positions = 0
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
            log_loss += loss.item() * n_pos
            log_top1 += top1 * n_pos
            log_positions += n_pos
            global_step += 1

            # Step-level logging
            if global_step % args.log_interval == 0 and log_positions > 0:
                avg_loss = log_loss / log_positions
                avg_top1 = log_top1 / log_positions
                lr = optimizer.param_groups[0]["lr"]
                print(f"  step {global_step:6d} | loss={avg_loss:.4f} "
                      f"top1={avg_top1:.4%} lr={lr:.2e}")
                log_loss = log_top1 = log_positions = 0

            # Step-level eval
            if eval_interval and global_step % eval_interval == 0:
                val_metrics = _do_eval()
                print(f"  [eval @ step {global_step}] "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"val_top1={val_metrics['top1_accuracy']:.4%} "
                      f"val_top5={val_metrics['top5_accuracy']:.4%}")
                logger.log_train(step=global_step, epoch=epoch,
                    lr=optimizer.param_groups[0]["lr"],
                    val_loss=val_metrics["loss"],
                    val_top1=val_metrics["top1_accuracy"],
                    val_top5=val_metrics["top5_accuracy"],
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    _save_checkpoint("best", epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"\n  Early stopping at step {global_step} "
                              f"(patience={args.patience})")
                        break
                model.train()

            if step_limit is not None and global_step >= step_limit:
                break
            if _shutdown_requested:
                break

        dt = time.time() - t0
        train_loss = epoch_loss / max(epoch_positions, 1)
        train_top1 = epoch_top1 / max(epoch_positions, 1)

        # Epoch-level eval (when not using step-level eval)
        do_val = not eval_interval and (
            (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        )
        if do_val:
            val_metrics = _do_eval()

        logger.log_train(step=global_step, epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss, train_top1=train_top1,
            val_loss=val_metrics["loss"],
            val_top1=val_metrics["top1_accuracy"],
            val_top5=val_metrics["top5_accuracy"],
            epoch_time_s=dt,
        )

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
              f"val_top5={val_metrics['top5_accuracy']:.4%} | {dt:.1f}s")

        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                _save_best(val_metrics, epoch)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                    break

        if patience_counter >= args.patience:
            break
        if step_limit is not None and global_step >= step_limit:
            print(f"\n  Reached step limit ({step_limit})")
            break
        if _shutdown_requested:
            print("Shutdown requested, saving checkpoint...")
            break

    # Save final checkpoint (blocking — must complete before exit)
    _save_checkpoint("final", epoch)
    if _save_thread is not None:
        _save_thread.join()

    logger.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
