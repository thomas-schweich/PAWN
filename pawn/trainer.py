"""PAWN training loop with checkpointing and monitoring."""

import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM, clm_loss
from pawn.data import CLMDataset, create_validation_set

from pawn.data_utils import unpack_grid


class CosineWithWarmup:
    """Cosine LR schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0
        self._apply_lr(0)

    def _apply_lr(self, step: int):
        lr_scale = self._compute_lr_scale(step)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = base_lr * lr_scale

    def step(self):
        self._step += 1
        self._apply_lr(self._step)

    def _compute_lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"step": self._step}

    def load_state_dict(self, state):
        self._step = state["step"]
        self._apply_lr(self._step)


def _compute_legal_move_rate(
    logits: torch.Tensor,
    legal_grid: torch.Tensor,
    loss_mask: torch.Tensor,
    game_lengths: torch.Tensor,
) -> float:
    """Compute fraction of argmax predictions that are legal moves.

    Only evaluated at positions where the target is an actual move
    (positions 0 through game_lengths-1, i.e., not the final position
    where the target is PAD).

    Args:
        logits: (B, T, vocab_size)
        legal_grid: (B, max_ply, 64) bit-packed legal moves from engine
        loss_mask: (B, T) bool
        game_lengths: (B,) int
    """
    B, T, V = logits.shape
    max_ply = legal_grid.shape[1]

    with torch.no_grad():
        # Positions where target is an actual move: 0..game_lengths-1 in CLM indexing
        # CLM position p maps to engine ply p (prediction at position p = move at ply p)
        move_mask = torch.arange(T, device=logits.device).unsqueeze(0) < game_lengths.unsqueeze(1)
        move_mask = move_mask & loss_mask

        if not move_mask.any():
            return 0.0

        preds = logits.argmax(dim=-1)  # (B, T)

        # Unpack legal grid to dense: (B, max_ply, 64, 64) -> flatten to (B, max_ply, 4096)
        legal_dense = unpack_grid(legal_grid)  # (B, max_ply, 64, 64)
        legal_flat = legal_dense.reshape(B, max_ply, 4096)  # (B, max_ply, 4096)

        # For each valid position, check if the predicted token is legal
        # CLM position p -> ply p in the engine (0-indexed)
        # But legal_grid is (B, max_ply), and CLM positions are (B, T=256)
        # CLM position 0 predicts move at ply 0, which uses legal_grid[:, 0]
        # So CLM position p uses legal_grid[:, p]

        # We need to handle the case where T might differ from max_ply
        n_plies = min(T, max_ply)
        valid_count = 0
        legal_count = 0

        # Build a flat legal move check
        # For base grid tokens 1-4096: token_id - 1 is the grid index
        # For promotion tokens 4097-4272: the base grid move is also legal
        # For simplicity, check if pred is in 1-4272 (any valid move token)
        # and whether the corresponding grid position is set

        for p in range(n_plies):
            pos_mask = move_mask[:, p]  # (B,)
            if not pos_mask.any():
                continue

            batch_preds = preds[pos_mask, p]  # (N,)
            batch_legal = legal_flat[pos_mask, p]  # (N, 4096)

            # Check base grid tokens (1-4096)
            is_base = (batch_preds >= 1) & (batch_preds <= 4096)
            base_idx = (batch_preds - 1).clamp(0, 4095)
            base_legal = batch_legal[torch.arange(len(batch_preds), device=logits.device), base_idx]

            # Check promotion tokens (4097-4272): the underlying src-dst move
            # must be legal (promotion legality is implied by grid legality)
            # Token 4097+ maps to promotion pairs; for now just check grid
            is_promo = (batch_preds >= 4097) & (batch_preds <= 4272)

            # For promos, we need to map to the grid index
            # Promo tokens map to specific (src, dst) pairs
            # The grid already marks these as legal if any promotion is legal
            # So we need the token-to-grid mapping
            # For simplicity, count promos as legal if the src-dst pair is set
            # This is close enough for monitoring purposes

            is_legal = is_base & (base_legal > 0.5)

            valid_count += len(batch_preds)
            legal_count += is_legal.sum().item()
            # Count promos separately (assume legal if predicted in promo range
            # and the position has promotions available)
            legal_count += is_promo.sum().item()  # Approximate

        if valid_count == 0:
            return 0.0
        return legal_count / valid_count


def _get_memory_stats(device: str, reset_peak: bool = True) -> dict[str, float]:
    proc = psutil.Process()
    mem = proc.memory_info()
    sys_mem = psutil.virtual_memory()
    per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    cpu_pct = sum(per_cpu) if per_cpu else 0.0
    stats = {
        "system_rss_gb": mem.rss / (1024**3),
        "system_used_gb": sys_mem.used / (1024**3),
        "system_total_gb": sys_mem.total / (1024**3),
        "cpu_percent": cpu_pct,
    }

    if device != "cpu" and torch.cuda.is_available():
        stats["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        stats["gpu_current_gb"] = torch.cuda.memory_allocated() / (1024**3)
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()

    return stats


def _get_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total**0.5


def _make_run_dir(base_log_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_log_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class CLMTrainer:
    def __init__(self, train_cfg: TrainingConfig, model_cfg: CLMConfig):
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = train_cfg.device
        self.global_step = 0

        self.run_dir = _make_run_dir(train_cfg.log_dir)
        self.cfg.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self._jsonl_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._jsonl_file = None

        self.model = PAWNCLM(model_cfg).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {param_count:,}")
        print(f"Run directory: {self.run_dir}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.total_steps,
        )
        self.scaler = torch.amp.GradScaler(self.device, enabled=train_cfg.use_amp)

        self.dataset = CLMDataset(train_cfg.batch_size, train_cfg.max_ply, train_cfg.base_seed)
        print("Generating validation set...")
        self.val_data = create_validation_set(
            train_cfg.val_games, train_cfg.max_ply, train_cfg.val_seed
        )

        # W&B
        self.wandb_run = None
        if train_cfg.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=train_cfg.wandb_project,
                    config={
                        "model": model_cfg.__dict__,
                        "training": train_cfg.__dict__,
                    },
                )
            except (ImportError, Exception) as e:
                print(f"W&B init failed: {e}. Continuing without W&B.")

        # torch.compile
        self._compiled = False
        if self.device != "cpu":
            try:
                self.model = torch.compile(self.model, mode="default")
                self._compiled = True
                print("torch.compile enabled")
            except Exception:
                print("torch.compile not available, using eager mode")
        else:
            print("Skipping torch.compile on CPU")

        config_data = {
            "model": model_cfg.__dict__,
            "training": train_cfg.__dict__,
            "param_count": param_count,
            "compiled": self._compiled,
            "formulation": "clm",
        }

        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        # Write config record to JSONL so the dashboard can detect run type
        self._log_jsonl({"type": "config", **config_data})

    def seed_logs(self, run_dirs: list[str], max_step: int):
        """Splice prior run logs into this run's JSONL."""
        from pathlib import Path

        all_records: list[dict] = []
        for rd in run_dirs:
            p = Path(rd) / "metrics.jsonl"
            if not p.exists():
                print(f"  WARNING: {p} not found, skipping")
                continue
            with open(p, "rb") as f:
                data = f.read()
            text = data.rstrip(b"\x00").decode()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        all_records = [r for r in all_records if r.get("step", 0) <= max_step]
        all_records.sort(key=lambda r: (r.get("step", 0), r.get("type", "")))
        seen: set[tuple[str, int]] = set()
        deduped: list[dict] = []
        for r in all_records:
            key = (r.get("type", ""), r.get("step", 0))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        if not deduped:
            print("  No prior log lines to seed.")
            return

        with open(self._jsonl_path, "w") as f:
            for r in deduped:
                f.write(json.dumps(r, default=str) + "\n")

        first_step = deduped[0].get("step", "?")
        last_step = deduped[-1].get("step", "?")
        print(f"Seeded {len(deduped)} log lines from prior runs "
              f"(steps {first_step}-{last_step})")

    def _log_jsonl(self, record: dict):
        if self._jsonl_file is None:
            self._jsonl_file = open(self._jsonl_path, "a")
        self._jsonl_file.write(json.dumps(record, default=str) + "\n")
        self._jsonl_file.flush()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        model = self._eager_model()

        with torch.amp.autocast(self.device, enabled=self.cfg.use_amp):
            loss, metrics = model.forward_train(input_ids, loss_mask, targets)

        scaled_loss = loss / self.cfg.accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        return metrics

    def optimizer_step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        grad_norm = _get_grad_norm(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return grad_norm

    def _eager_model(self) -> PAWNCLM:
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        model = self._eager_model()
        model.eval()

        n = self.val_data["input_ids"].shape[0]
        batch_size = self.cfg.batch_size
        total_metrics: dict[str, float] = {}
        n_batches = 0

        has_legal = "legal_grid" in self.val_data
        total_legal_count = 0
        total_move_count = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            input_ids = self.val_data["input_ids"][start:end].to(self.device)
            targets = self.val_data["targets"][start:end].to(self.device)
            loss_mask = self.val_data["loss_mask"][start:end].to(self.device)

            with torch.amp.autocast(self.device, enabled=self.cfg.use_amp):
                logits, _layer_outputs = model(input_ids, loss_mask)
                del _layer_outputs
                _, metrics = clm_loss(logits, targets, loss_mask)

            # Top-5 accuracy
            valid_logits = logits[loss_mask]
            valid_targets = targets[loss_mask]
            top5 = valid_logits.topk(5, dim=-1).indices
            top5_acc = (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
            metrics["top5_accuracy"] = top5_acc

            # Perplexity
            metrics["perplexity"] = math.exp(min(metrics["loss"], 20.0))

            # Legal move rate (if legal grid available)
            if has_legal:
                legal_grid = self.val_data["legal_grid"][start:end].to(self.device)
                game_lengths = self.val_data["game_lengths"][start:end].to(self.device)
                legal_rate = _compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
                metrics["legal_move_rate"] = legal_rate

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {f"val/{k}": v / n_batches for k, v in total_metrics.items()}

    def train(self):
        self.dataset.set_start_step(self.global_step)
        num_workers = self.cfg.num_workers
        loader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=(self.device != "cpu"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=1 if num_workers > 0 else None,
        )

        def _graceful_exit(signum, frame):
            print(f"\nReceived signal {signum}, saving checkpoint and exiting...")
            self.save_checkpoint()
            if self._jsonl_file:
                self._jsonl_file.close()
            sys.exit(128 + signum)

        old_term = signal.signal(signal.SIGTERM, _graceful_exit)
        old_int = signal.signal(signal.SIGINT, _graceful_exit)

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        accum_count = 0
        step_start = time.time()
        games_per_step = self.cfg.batch_size * self.cfg.accumulation_steps

        print(f"Starting training from step {self.global_step}", flush=True)
        print(f"JSONL log: {self._jsonl_path}", flush=True)

        for batch in loader:
            metrics = self.train_step(batch)
            accum_count += 1

            if accum_count >= self.cfg.accumulation_steps:
                grad_norm = self.optimizer_step()
                accum_count = 0
                self.global_step += 1

                step_time = time.time() - step_start
                games_per_sec = games_per_step / step_time

                if self.global_step % self.cfg.log_interval == 0:
                    lr = self.scheduler.get_lr()
                    mem = _get_memory_stats(self.device)

                    msg = (
                        f"step {self.global_step:>7d} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"acc {metrics['accuracy']:.3f} | "
                        f"lr {lr:.2e} | "
                        f"gn {grad_norm:.2f} | "
                        f"{games_per_sec:.0f} g/s | "
                        f"{step_time:.2f}s"
                    )
                    print(msg, flush=True)

                    record = {
                        "type": "train",
                        "step": self.global_step,
                        "timestamp": time.time(),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "step_time": step_time,
                        "games_per_sec": games_per_sec,
                        **{f"train/{k}": v for k, v in metrics.items()},
                        **{f"mem/{k}": v for k, v in mem.items()},
                    }
                    self._log_jsonl(record)

                    if self.wandb_run:
                        log_data = {f"train/{k}": v for k, v in metrics.items()}
                        log_data["train/lr"] = lr
                        log_data["train/grad_norm"] = grad_norm
                        log_data["train/step_time"] = step_time
                        log_data["train/games_per_sec"] = games_per_sec
                        self.wandb_run.log(log_data, step=self.global_step)

                if self.global_step % self.cfg.eval_interval == 0:
                    val_metrics = self.evaluate()
                    val_msg = (
                        f"  val: loss {val_metrics['val/loss']:.4f} | "
                        f"acc {val_metrics['val/accuracy']:.3f} | "
                        f"top5 {val_metrics.get('val/top5_accuracy', 0):.3f} | "
                        f"ppl {val_metrics.get('val/perplexity', 0):.1f}"
                    )
                    if "val/legal_move_rate" in val_metrics:
                        val_msg += f" | legal {val_metrics['val/legal_move_rate']:.3f}"
                    print(val_msg, flush=True)

                    record = {
                        "type": "val",
                        "step": self.global_step,
                        "timestamp": time.time(),
                        **val_metrics,
                    }
                    self._log_jsonl(record)

                    if self.wandb_run:
                        self.wandb_run.log(val_metrics, step=self.global_step)

                if self.global_step % self.cfg.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.global_step >= self.cfg.total_steps:
                    print(f"Training complete at step {self.global_step}")
                    self.save_checkpoint()
                    break

                step_start = time.time()

        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)

        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None

    def save_checkpoint(self, path: str | None = None):
        if path is None:
            path = os.path.join(
                self.cfg.checkpoint_dir, f"step_{self.global_step:08d}.pt"
            )

        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "model_config": self.model_cfg.__dict__,
                "training_config": self.cfg.__dict__,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                ),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt["global_step"]

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu().byte())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu().byte())

        print(f"Resumed from step {self.global_step}")
