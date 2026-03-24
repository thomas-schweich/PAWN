#!/usr/bin/env python3
"""Train small, base, and large PAWN models simultaneously on shared data.

All three models see the exact same batches in the same order, eliminating
data generation overhead and ensuring comparable training conditions.

Usage:
    uv run python scripts/train_all.py --local-checkpoints
    uv run python scripts/train_all.py --hf-repo thomas-schweich/pawn-{variant}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM, clm_loss
from pawn.data import CLMDataset, create_validation_set
from pawn.gpu import configure_gpu
from pawn.checkpoint import save_pretrain_checkpoint, push_checkpoint_to_hf


# ---------------------------------------------------------------------------
# Per-model state
# ---------------------------------------------------------------------------

class ModelSlot:
    """Holds everything needed to train and checkpoint one model variant."""

    def __init__(
        self,
        name: str,
        model_cfg: CLMConfig,
        train_cfg: TrainingConfig,
        device: str,
        hf_repo: str | None,
        shm_checkpoints: bool = False,
        slug: str = "",
    ):
        self.name = name
        self.slug = slug
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.hf_repo = hf_repo
        self.shm_checkpoints = shm_checkpoints

        self.model = PAWNCLM(model_cfg).to(device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  {name}: {param_count:,} params ({model_cfg.d_model}d/{model_cfg.n_layers}L)")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        from pawn.trainer import CosineWithWarmup
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.total_steps,
        )

        self.scaler = torch.amp.GradScaler(device, enabled=train_cfg.use_amp)

        # Run directory (logs always on persistent disk)
        self.run_dir = _make_run_dir(train_cfg.log_dir, name, slug)

        # Checkpoint directory: /dev/shm if requested, else under run_dir
        if shm_checkpoints:
            self.checkpoint_dir = f"/dev/shm/pawn_checkpoints/{name}"
        else:
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._jsonl_file: open | None = None

        self.hf_branch = f"run/{os.path.basename(self.run_dir)}" if hf_repo else None
        self.global_step = 0
        self.best_val_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Background HF push (one thread per slot, so pushes don't block training)
        from concurrent.futures import ThreadPoolExecutor
        self._hf_push_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hf-{name}")
        self._hf_push_future = None

        # Write config
        import subprocess
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            git_hash = os.environ.get("PAWN_GIT_HASH")
        try:
            git_tag = subprocess.check_output(
                ["git", "tag", "--points-at", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip() or None
        except Exception:
            git_tag = os.environ.get("PAWN_GIT_TAG")

        config_data = {
            "model": model_cfg.__dict__,
            "training": train_cfg.__dict__,
            "param_count": param_count,
            "formulation": "clm",
            "multi_model": True,
            "variant": name,
            "slug": slug,
            "git_hash": git_hash,
            "git_tag": git_tag,
        }
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2, default=str)
        self._log_jsonl({"type": "config", **config_data})

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward + backward. Returns raw GPU tensor metrics (no .item() sync)."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        with torch.amp.autocast(self.device, enabled=self.train_cfg.use_amp):
            loss, metrics = self.model.forward_train(input_ids, loss_mask, targets)

        self.scaler.scale(loss).backward()
        return metrics

    def optimizer_step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.train_cfg.max_grad_norm
        ).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return grad_norm

    def _unwrapped_model(self):
        """Return the unwrapped model (strips torch.compile wrapper)."""
        m = self.model
        while hasattr(m, '_orig_mod'):
            m = m._orig_mod
        return m

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"step_{self.global_step:08d}")
        save_pretrain_checkpoint(
            path, self._unwrapped_model(), self.optimizer, self.scheduler, self.scaler,
            self.global_step, self.model_cfg.__dict__, self.train_cfg.__dict__,
        )
        print(f"  [{self.name}] Checkpoint saved: {path}")

        if self.hf_repo and self.hf_branch:
            self._push_to_hf_async(path, self.global_step)

    def _push_to_hf_async(self, ckpt_path: str, step: int):
        """Push checkpoint to HuggingFace in a background thread."""
        # Wait for any previous push to finish before starting a new one
        if self._hf_push_future is not None:
            self._hf_push_future.result()  # raises if previous push failed

        def _push():
            try:
                push_checkpoint_to_hf(
                    ckpt_path, self.hf_repo, self.hf_branch,
                    metrics_path=self.jsonl_path, step=step,
                )
                print(f"  [{self.name}] Pushed to HF: {self.hf_repo}@{self.hf_branch}")

                # On /dev/shm, clean up old checkpoints after successful push.
                # Keep the latest (just saved) and the best (for post-training evals).
                if self.shm_checkpoints:
                    keep = {Path(ckpt_path).name, f"step_{self.best_val_step:08d}"}
                    for old in sorted(Path(self.checkpoint_dir).glob("step_*")):
                        if old.name not in keep:
                            shutil.rmtree(old, ignore_errors=True)
            except Exception as e:
                print(f"  [{self.name}] WARNING: HF push failed: {e}")

        self._hf_push_future = self._hf_push_pool.submit(_push)

    def push_metrics_to_hf(self):
        """Push just metrics.jsonl to HF (lightweight, no checkpoint)."""
        if not self.hf_repo or not self.hf_branch:
            return

        def _push_metrics():
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_branch(self.hf_repo, repo_type="model",
                                  branch=self.hf_branch, exist_ok=True)
                api.upload_file(
                    path_or_fileobj=self.jsonl_path,
                    path_in_repo="metrics.jsonl",
                    repo_id=self.hf_repo,
                    repo_type="model",
                    revision=self.hf_branch,
                    commit_message=f"Metrics through step {self.global_step}",
                )
            except Exception as e:
                print(f"  [{self.name}] WARNING: metrics push failed: {e}")

        # Fire and forget on the push pool (queued behind any checkpoint push)
        self._hf_push_pool.submit(_push_metrics)

    def wait_for_push(self):
        """Block until any in-flight HF push completes."""
        if self._hf_push_future is not None:
            self._hf_push_future.result()
            self._hf_push_future = None

    @torch.no_grad()
    def evaluate(self, val_data: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.eval()
        n = val_data["input_ids"].shape[0]
        batch_size = self.train_cfg.batch_size
        total_metrics: dict[str, float] = {}
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            input_ids = val_data["input_ids"][start:end].to(self.device)
            targets = val_data["targets"][start:end].to(self.device)
            loss_mask = val_data["loss_mask"][start:end].to(self.device)

            with torch.amp.autocast(self.device, enabled=self.train_cfg.use_amp):
                logits, _ = self.model(input_ids, loss_mask)
                _, metrics = clm_loss(logits, targets, loss_mask)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        return {f"val/{k}": v / n_batches for k, v in total_metrics.items()}

    def _log_jsonl(self, record: dict):
        if self._jsonl_file is None:
            self._jsonl_file = open(self.jsonl_path, "a")
        self._jsonl_file.write(json.dumps(record, default=str) + "\n")
        self._jsonl_file.flush()

    def close(self):
        self.wait_for_push()
        self._hf_push_pool.shutdown(wait=True)
        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None


_ADJECTIVES = [
    "amber", "bold", "calm", "deft", "eager", "fair", "grim", "hale",
    "keen", "lush", "mild", "neat", "pale", "quick", "rare", "sly",
    "taut", "vast", "warm", "zesty", "brisk", "crisp", "dense", "fleet",
    "grand", "hardy", "jolly", "lucid", "noble", "prime", "stark", "vivid",
]
_ANIMALS = [
    "puma", "lynx", "hawk", "wolf", "bear", "deer", "fox", "owl",
    "pike", "wren", "crane", "otter", "raven", "cobra", "heron", "bison",
    "finch", "marten", "osprey", "falcon", "badger", "salmon", "condor",
    "coyote", "ferret", "jackal", "marmot", "parrot", "turtle", "walrus",
]


def _random_slug() -> str:
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}"


def _make_run_dir(log_dir: str, variant: str, slug: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}_{variant}_{slug}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train small/base/large PAWN models simultaneously")
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    p.add_argument("--total-steps", type=int, default=100_000, help="Total training steps")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size (shared across models)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=5000)
    p.add_argument("--discard-ply-limit", action="store_true")
    p.add_argument("--patience", type=int, default=10,
                    help="Stop if no val loss improvement for N eval intervals (0=disabled)")
    p.add_argument("--wandb", action="store_true")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="HF repo prefix (appends -{variant}). E.g. thomas-schweich/pawn")
    ckpt_group.add_argument("--local-checkpoints", action="store_true")

    p.add_argument("--shm-checkpoints", action="store_true",
                    help="Write checkpoints to /dev/shm (RAM-backed, instant writes). "
                         "Requires --hf-repo since /dev/shm is volatile.")

    p.add_argument("--run-evals", action="store_true",
                    help="Run probes, diagnostics, and Lichess eval after training completes")
    p.add_argument("--lichess-pgn", type=str, default=None,
                    help="Path to Lichess PGN file for eval (required with --run-evals)")
    p.add_argument("--publish-results", action="store_true",
                    help="Push eval results to HuggingFace (requires --hf-repo and --run-evals)")
    return p.parse_args()


def _run_post_training_evals(slots: list[ModelSlot], args):
    """Run probes, diagnostics, and Lichess eval on best checkpoint per variant."""
    import tempfile
    from pawn.eval_suite.probes import extract_probe_data, train_all_probes
    from pawn.eval_suite.corpus import generate_corpus, load_corpus
    from pawn.eval_suite.diagnostics import extract_diagnostic_positions, evaluate_diagnostic_positions

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for slot in slots:
        print(f"\n--- Evaluating {slot.name} ---")

        # Use tracked best val step (kept on /dev/shm if shm_checkpoints)
        best_step = slot.best_val_step
        best_loss = slot.best_val_loss

        ckpt_path = os.path.join(slot.checkpoint_dir, f"step_{best_step:08d}")
        if not os.path.isdir(ckpt_path):
            # Fall back to latest
            ckpts = sorted(Path(slot.checkpoint_dir).glob("step_*"))
            ckpt_path = str(ckpts[-1]) if ckpts else None

        if not ckpt_path:
            print(f"  No checkpoint found, skipping")
            continue

        print(f"  Best checkpoint: {ckpt_path} (val_loss={best_loss:.4f})")

        # Load model (unwrapped)
        from pawn.checkpoint import load_backbone_weights
        state_dict, _ = load_backbone_weights(ckpt_path)
        model = PAWNCLM(slot.model_cfg).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        results = {}

        # 1. Probes
        print("  Running probes...")
        train_data = extract_probe_data(2048, 256, seed=12345)
        val_data = extract_probe_data(512, 256, seed=54321)
        probe_results = train_all_probes(
            model, train_data, val_data, device=device,
            per_layer=True, n_epochs=20, verbose=True,
        )
        results["probes"] = probe_results
        del train_data, val_data

        # 2. Diagnostics
        print("  Running diagnostics...")
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = generate_corpus(tmpdir, n_games=2048, max_ply=255, seed=99999, batch_size=2048)
            corpus = load_corpus(corpus_path)
            positions = extract_diagnostic_positions(corpus, min_per_category=200, max_per_category=1000)
            diag_results = evaluate_diagnostic_positions(model, positions, corpus, device=device)
            results["diagnostics"] = diag_results

        # 3. Lichess eval (if PGN provided)
        if args.lichess_pgn:
            print("  Running Lichess eval...")
            from pawn.eval_suite.lichess import prepare_lichess_corpus, evaluate_on_lichess
            lichess_data = prepare_lichess_corpus(args.lichess_pgn, max_games_per_band=1000)
            lichess_results = evaluate_on_lichess(model, lichess_data, device=device)
            results["lichess"] = lichess_results

        # Save results
        results_path = os.path.join(slot.run_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved: {results_path}")

        # Publish to HF
        if args.publish_results and slot.hf_repo and slot.hf_branch:
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                api.upload_file(
                    path_or_fileobj=results_path,
                    path_in_repo="eval_results.json",
                    repo_id=slot.hf_repo,
                    repo_type="model",
                    revision=slot.hf_branch,
                    commit_message=f"Eval results (best step {best_step})",
                )
                print(f"  Published to {slot.hf_repo}@{slot.hf_branch}")
            except Exception as e:
                print(f"  WARNING: HF publish failed: {e}")

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()

    if args.shm_checkpoints and not args.hf_repo:
        print("ERROR: --shm-checkpoints requires --hf-repo (HF is the only durable store)")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        gpu_cfg = configure_gpu()
        import pawn.model as model_module
        if gpu_cfg.get("sdpa_backend"):
            model_module.SDPA_BACKEND = gpu_cfg["sdpa_backend"]

    # Build per-variant configs (shared training hyperparams, different model sizes)
    variants = {
        "small": CLMConfig.small(),
        "base": CLMConfig.base(),
        "large": CLMConfig.large(),
    }

    slug = _random_slug()

    print(f"=== Multi-Model Training [{slug}] ===")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total steps: {args.total_steps}")
    if args.shm_checkpoints:
        print("Checkpoints: /dev/shm (volatile, HF push is durable store)")
    print()

    # Linear LR scaling: lr = base_lr * (batch_size / base_batch_size)
    base_batch_size = 256
    base_lr = TrainingConfig.lr
    scaled_lr = base_lr * (args.batch_size / base_batch_size)
    print(f"LR: {scaled_lr:.2e} (scaled from {base_lr:.2e} for batch {args.batch_size})")

    slots: list[ModelSlot] = []
    for name, model_cfg in variants.items():
        train_cfg = TrainingConfig()
        train_cfg.lr = scaled_lr
        train_cfg.total_steps = args.total_steps
        train_cfg.batch_size = args.batch_size
        train_cfg.num_workers = args.num_workers
        train_cfg.device = device
        train_cfg.log_dir = args.log_dir
        train_cfg.log_interval = args.log_interval
        train_cfg.eval_interval = args.eval_interval
        train_cfg.checkpoint_interval = args.checkpoint_interval
        train_cfg.discard_ply_limit = args.discard_ply_limit
        train_cfg.use_wandb = args.wandb

        hf_repo = f"{args.hf_repo}-{name}" if args.hf_repo else None
        slots.append(ModelSlot(name, model_cfg, train_cfg, device, hf_repo,
                               shm_checkpoints=args.shm_checkpoints, slug=slug))

    # Shared dataset and validation set
    max_ply = 256
    dataset = CLMDataset(
        args.batch_size, max_ply, base_seed=42,
        discard_ply_limit=args.discard_ply_limit,
    )

    print("\nGenerating shared validation set...")
    val_data = create_validation_set(512, max_ply, seed=(2**63) - 1,
                                     discard_ply_limit=args.discard_ply_limit)

    # Compile models
    if device != "cpu":
        for slot in slots:
            try:
                slot.model = torch.compile(slot.model, mode="default")
                print(f"  [{slot.name}] torch.compile enabled")
            except Exception:
                print(f"  [{slot.name}] torch.compile not available")

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
    )

    # Signal handling
    _shutdown_requested = False
    _shutdown_signal = None

    def _graceful_exit(signum, frame):
        nonlocal _shutdown_requested, _shutdown_signal
        _shutdown_requested = True
        _shutdown_signal = signum

    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    # Training loop
    global_step = 0
    step_start = time.time()

    print(f"\nStarting training from step 0", flush=True)
    for slot in slots:
        print(f"  [{slot.name}] JSONL: {slot.jsonl_path}", flush=True)
    print()

    for batch in loader:
        # Forward + backward for each model on the same batch (no .item() sync)
        all_metrics: dict[str, dict[str, torch.Tensor]] = {}
        for slot in slots:
            metrics = slot.train_step(batch)
            all_metrics[slot.name] = metrics

        # Optimizer step for each model
        all_grad_norms: dict[str, float] = {}
        for slot in slots:
            gn = slot.optimizer_step()
            all_grad_norms[slot.name] = gn

        global_step += 1
        for slot in slots:
            slot.global_step = global_step

        step_time = time.time() - step_start
        games_per_sec = args.batch_size / step_time

        # Logging — .item() sync only at log intervals
        if global_step % args.log_interval == 0:
            print(f"step {global_step:>7d} | {games_per_sec:.0f} g/s | {step_time:.2f}s", flush=True)
            for slot in slots:
                m = all_metrics[slot.name]
                loss_val = m['loss'].item()
                acc_val = m['accuracy'].item()
                gn = all_grad_norms[slot.name]
                lr = slot.scheduler.get_lr()
                print(f"  {slot.name:>5s}: loss {loss_val:.4f} | acc {acc_val:.3f} | "
                      f"lr {lr:.2e} | gn {gn:.2f}", flush=True)

                record = {
                    "type": "train",
                    "step": global_step,
                    "timestamp": time.time(),
                    "lr": lr,
                    "grad_norm": gn,
                    "step_time": step_time,
                    "games_per_sec": games_per_sec,
                    "train/loss": loss_val,
                    "train/accuracy": acc_val,
                }
                slot._log_jsonl(record)

        # Eval
        if global_step % args.eval_interval == 0:
            for slot in slots:
                val_metrics = slot.evaluate(val_data)
                print(f"  {slot.name:>5s} val: loss {val_metrics['val/loss']:.4f} | "
                      f"acc {val_metrics['val/accuracy']:.3f}", flush=True)
                slot._log_jsonl({
                    "type": "val",
                    "step": global_step,
                    "timestamp": time.time(),
                    **val_metrics,
                })
                # Track best for eval, /dev/shm cleanup, and patience
                vl = val_metrics["val/loss"]
                if vl < slot.best_val_loss:
                    slot.best_val_loss = vl
                    slot.best_val_step = global_step
                    slot.patience_counter = 0
                else:
                    slot.patience_counter += 1

            # Push metrics to HF after eval (lightweight, background)
            for slot in slots:
                slot.push_metrics_to_hf()

            # Early stop when ALL slots have exhausted patience
            if args.patience > 0 and all(s.patience_counter >= args.patience for s in slots):
                print(f"\nEarly stopping at step {global_step} — no improvement "
                      f"for {args.patience} evals on any variant")
                for slot in slots:
                    slot.save_checkpoint()
                break

        # Checkpoint
        if global_step % args.checkpoint_interval == 0:
            for slot in slots:
                slot.save_checkpoint()

        # Done?
        if global_step >= args.total_steps:
            print(f"\nTraining complete at step {global_step}")
            for slot in slots:
                slot.save_checkpoint()
            break

        # Graceful shutdown
        if _shutdown_requested:
            print(f"\nShutdown requested (signal {_shutdown_signal}), "
                  f"saving checkpoints at step {global_step}...")
            for slot in slots:
                slot.save_checkpoint()
            break

        step_start = time.time()

    # Cleanup
    for slot in slots:
        slot.close()

    # Post-training evals
    if args.run_evals:
        print("\n" + "=" * 60)
        print("POST-TRAINING EVALUATION")
        print("=" * 60)
        _run_post_training_evals(slots, args)

    print("\nAll done.")


if __name__ == "__main__":
    try:
        mp.set_start_method("forkserver", force=True)
    except ValueError:
        mp.set_start_method("spawn", force=True)
    main()
