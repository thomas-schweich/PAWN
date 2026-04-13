"""Co-training: train multiple model variants on shared data batches.

Extracted from ``scripts/train_all.py`` so the lab MCP server and the CLI
script share the same implementation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.checkpoint import load_pretrain_checkpoint, save_pretrain_checkpoint, push_checkpoint_to_hf
from pawn.config import CLMConfig, LegacyVocab, TrainingConfig
from pawn.data import CLMDataset, create_validation_set
from pawn.gpu import configure_gpu
from pawn.logging import MetricsLogger, random_slug
from pawn.model import PAWNCLM
from pawn.run_config import CotrainConfig, CotrainVariant
from pawn.trainer import (
    CosineWithWarmup,
    compute_legal_move_rate_from_preds,
    eval_game_completion_metrics,
)


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
        resume_path: str | None = None,
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

        self.scheduler = CosineWithWarmup(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.total_steps,
        )

        self.scaler = torch.amp.GradScaler(device, enabled=train_cfg.use_amp)

        # Logger (creates run directory)
        self.logger = MetricsLogger(
            train_cfg.log_dir, run_prefix="run", device=device,
            slug=slug, suffix=name,
        )
        self.run_dir = str(self.logger.run_dir)
        self.jsonl_path = str(self.logger.metrics_path)

        # Checkpoint directory: /dev/shm if requested, else under run_dir
        if shm_checkpoints:
            self.checkpoint_dir = f"/dev/shm/pawn_checkpoints/{name}"
        else:
            self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.hf_branch = f"run/{os.path.basename(self.run_dir)}" if hf_repo else None
        self.global_step = 0
        self.best_val_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.stopped = False

        # Background HF push (one thread per slot, so pushes don't block training)
        from concurrent.futures import ThreadPoolExecutor
        self._hf_push_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hf-{name}")
        self._hf_push_future = None

        # Resume from checkpoint if requested
        if resume_path:
            meta = load_pretrain_checkpoint(
                resume_path, self.model, self.optimizer, self.scheduler,
                self.scaler, device=device,
            )
            self.global_step = meta["global_step"]
            if meta.get("best_val_loss") is not None:
                self.best_val_loss = meta["best_val_loss"]
            if meta.get("patience_counter") is not None:
                self.patience_counter = meta["patience_counter"]
            print(f"  [{name}] Resumed from step {self.global_step} "
                  f"(checkpoint: {resume_path})")

        self.logger.log_config(
            model=model_cfg.__dict__,
            training=train_cfg.__dict__,
            param_count=param_count,
            formulation="clm",
            multi_model=True,
            variant=name,
        )
        self.logger.write_config_json(
            model=model_cfg.__dict__,
            training=train_cfg.__dict__,
            param_count=param_count,
            formulation="clm",
            multi_model=True,
            variant=name,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward + backward. Returns raw GPU tensor metrics (no .item() sync)."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(self.device, non_blocking=True)

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

    def _unwrapped_model(self) -> PAWNCLM:
        """Return the unwrapped model (strips torch.compile wrapper)."""
        m: Any = self.model
        while hasattr(m, '_orig_mod'):
            m = m._orig_mod
        return m  # type: ignore[return-value]

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"step_{self.global_step:08d}")
        save_pretrain_checkpoint(
            path, self._unwrapped_model(), self.optimizer, self.scheduler, self.scaler,
            self.global_step, self.model_cfg.__dict__, self.train_cfg.__dict__,  # type: ignore[arg-type]
        )
        print(f"  [{self.name}] Checkpoint saved: {path}")

        if self.hf_repo and self.hf_branch:
            self._push_to_hf_async(path, self.global_step)

    def _push_to_hf_async(self, ckpt_path: str, step: int):
        """Push checkpoint to HuggingFace in a background thread."""
        # Wait for any previous push to finish before starting a new one
        if self._hf_push_future is not None:
            self._hf_push_future.result()  # blocks until previous push completes

        assert self.hf_repo is not None and self.hf_branch is not None
        hf_repo, hf_branch = self.hf_repo, self.hf_branch

        def _push():
            try:
                push_checkpoint_to_hf(
                    ckpt_path, hf_repo, hf_branch,
                    metrics_path=self.jsonl_path, step=step,
                )
                print(f"  [{self.name}] Pushed to HF: {hf_repo}@{hf_branch}")

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

        hf_repo, hf_branch = self.hf_repo, self.hf_branch

        def _push_metrics():
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_branch(hf_repo, repo_type="model",
                                  branch=hf_branch, exist_ok=True)
                api.upload_file(
                    path_or_fileobj=self.jsonl_path,
                    path_in_repo="metrics.jsonl",
                    repo_id=hf_repo,
                    repo_type="model",
                    revision=hf_branch,
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
        has_legal = "legal_grid" in val_data
        legality_late_ply = self.model_cfg.max_seq_len // 2
        n_actions = self.model.embed.n_actions

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            input_ids = val_data["input_ids"][start:end].to(self.device, non_blocking=True)
            targets = val_data["targets"][start:end].to(self.device, non_blocking=True)
            loss_mask = val_data["loss_mask"][start:end].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device, enabled=self.train_cfg.use_amp):
                # Get hidden states without materializing full (B,T,V) logits
                hidden = self.model.forward_eval(input_ids, loss_mask)

                # Sparse projection: only valid positions through lm_head
                valid_hidden = hidden[loss_mask]
                valid_logits = self.model.lm_head(valid_hidden)
                valid_targets = targets[loss_mask]

            loss = F.cross_entropy(valid_logits, valid_targets)
            accuracy = (valid_logits.argmax(-1) == valid_targets).float().mean().item()
            metrics: dict[str, float] = {"loss": loss.item(), "accuracy": accuracy}

            # Top-5 accuracy
            top5 = valid_logits.topk(5, dim=-1).indices
            metrics["top5_accuracy"] = (
                (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )

            # Legal move rate (full + late): reuse valid_logits argmax.
            if has_legal:
                legal_grid = val_data["legal_grid"][start:end].to(self.device, non_blocking=True)
                game_lengths = val_data["game_lengths"][start:end].to(self.device, non_blocking=True)
                preds = torch.zeros_like(loss_mask, dtype=torch.long)
                preds[loss_mask] = valid_logits.argmax(dim=-1)
                metrics["legal_move_rate"] = compute_legal_move_rate_from_preds(
                    preds, legal_grid, loss_mask, game_lengths,
                    n_actions=n_actions,
                )
                metrics["late_legal_move_rate"] = compute_legal_move_rate_from_preds(
                    preds, legal_grid, loss_mask, game_lengths,
                    min_ply=legality_late_ply,
                    n_actions=n_actions,
                )

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        avg = {f"val/{k}": v / n_batches for k, v in total_metrics.items()}
        avg["val/perplexity"] = math.exp(min(avg["val/loss"], 20.0))

        # Forfeit-ply / game-completion stats — same surface as pretraining
        # so the lab MCP server can fit a log-linear trend from each variant's
        # metrics.jsonl.
        avg.update(eval_game_completion_metrics(
            self.model, val_data,
            batch_size=self.train_cfg.batch_size,
            vocab_size=self.model_cfg.vocab_size,
            device=self.device,
            use_amp=self.train_cfg.use_amp,
        ))

        return avg

    def close(self):
        self.wait_for_push()
        self._hf_push_pool.shutdown(wait=True)
        self.logger.close()


# ---------------------------------------------------------------------------
# Variant config builder
# ---------------------------------------------------------------------------


def _resolve_cotrain_resume_prepend_outcome(config: CotrainConfig) -> None:
    """Peek at each variant's resume checkpoint and align config.prepend_outcome
    with what the checkpoints were actually trained on.

    Cotrain shares one data pipeline across variants, so a mismatch (some
    outcome-prefixed, some pure-moves) is unresolvable — exits with a
    loud error. Called BEFORE slot construction so each ModelSlot's
    write_config_json sees the corrected format; otherwise the new run
    directory would advertise a format inconsistent with its batches.

    No-op when no variant is resuming.
    """
    from pawn.checkpoint import infer_prepend_outcome, read_checkpoint_metadata

    resume_modes: dict[str, bool] = {}
    ambiguous_variants: list[tuple[str, str]] = []
    for variant_spec in config.variants:
        if not variant_spec.resume:
            continue
        try:
            saved = read_checkpoint_metadata(variant_spec.resume)
        except (FileNotFoundError, OSError) as e:
            print(f"WARNING: couldn't peek at {variant_spec.name} resume "
                  f"checkpoint ({e}); assuming current prepend_outcome")
            continue
        saved_prepend, reason = infer_prepend_outcome(
            saved.get("training_config"), saved.get("model_config"),
        )
        if saved_prepend is None:
            ambiguous_variants.append((variant_spec.name, reason))
            print(f"  [{variant_spec.name}] resume: ambiguous ({reason})")
            continue
        resume_modes[variant_spec.name] = saved_prepend
        print(f"  [{variant_spec.name}] resume: "
              f"{'outcome-prefixed' if saved_prepend else 'pure-moves'} "
              f"({reason})")

    if ambiguous_variants:
        if "prepend_outcome" in config.model_fields_set:
            print(
                f"WARNING: {len(ambiguous_variants)} resumed variant(s) "
                f"have ambiguous sequence format — trusting explicit "
                f"prepend_outcome={config.prepend_outcome} from the run "
                "config. Verify this matches what the checkpoints were "
                "trained with."
            )
            # Treat the user's explicit value as the ground-truth for
            # those variants so the mismatch-detection below still works.
            for name, _ in ambiguous_variants:
                resume_modes[name] = config.prepend_outcome
        else:
            names = ", ".join(n for n, _ in ambiguous_variants)
            print(
                f"ERROR: resumed variant(s) {names} have ambiguous sequence "
                "format. Pass `prepend_outcome: true` or "
                "`prepend_outcome: false` in the run config explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not resume_modes:
        return

    unique = set(resume_modes.values())
    if len(unique) > 1:
        print(
            "ERROR: resumed variants use incompatible sequence formats; "
            "cotrain shares one data pipeline, so all variants must "
            f"agree on prepend_outcome. Got: {resume_modes}",
            file=sys.stderr,
        )
        sys.exit(1)

    resumed_prepend = next(iter(unique))
    if resumed_prepend != config.prepend_outcome:
        print(f"Resume: overriding prepend_outcome="
              f"{config.prepend_outcome} → {resumed_prepend} to match "
              f"resumed checkpoints.")
        config.prepend_outcome = resumed_prepend


def _build_variant_configs(
    variant_spec: CotrainVariant,
    shared: CotrainConfig,
    device: str,
) -> tuple[CLMConfig, TrainingConfig]:
    """Build internal CLMConfig + TrainingConfig for one variant."""
    variant_factory = {
        "small": CLMConfig.small,
        "base": CLMConfig.base,
        "large": CLMConfig.large,
        "toy": CLMConfig.toy,
    }
    if variant_spec.variant == "custom":
        # CotrainVariant._check_custom_arch ensures all four are set.
        assert variant_spec.d_model is not None
        assert variant_spec.n_layers is not None
        assert variant_spec.n_heads is not None
        assert variant_spec.d_ff is not None
        model_cfg = CLMConfig(
            d_model=variant_spec.d_model,
            n_layers=variant_spec.n_layers,
            n_heads=variant_spec.n_heads,
            d_ff=variant_spec.d_ff,
        )
    else:
        model_cfg = variant_factory[variant_spec.variant]()
        # Architecture overrides on top of the preset
        if variant_spec.d_model is not None:
            model_cfg.d_model = variant_spec.d_model
        if variant_spec.n_layers is not None:
            model_cfg.n_layers = variant_spec.n_layers
        if variant_spec.n_heads is not None:
            model_cfg.n_heads = variant_spec.n_heads
        if variant_spec.d_ff is not None:
            model_cfg.d_ff = variant_spec.d_ff
    model_cfg.max_seq_len = variant_spec.max_seq_len

    if variant_spec.legacy_vocab:
        model_cfg.vocab_size = LegacyVocab.VOCAB_SIZE
        model_cfg.max_seq_len = 256

    train_cfg = TrainingConfig()
    train_cfg.lr = shared.lr
    train_cfg.total_steps = shared.total_steps or train_cfg.total_steps
    train_cfg.batch_size = shared.batch_size
    train_cfg.num_workers = shared.num_workers
    train_cfg.device = device
    train_cfg.log_dir = shared.log_dir or train_cfg.log_dir
    train_cfg.log_interval = shared.log_interval
    if shared.eval_interval is not None:
        train_cfg.eval_interval = shared.eval_interval
    train_cfg.checkpoint_interval = shared.checkpoint_interval
    train_cfg.discard_ply_limit = shared.discard_ply_limit
    train_cfg.prepend_outcome = shared.prepend_outcome
    train_cfg.mate_boost = shared.mate_boost
    train_cfg.use_wandb = shared.wandb
    train_cfg.use_amp = shared.amp_dtype != "none"
    train_cfg.max_ply = model_cfg.max_seq_len
    train_cfg.weight_decay = shared.weight_decay
    train_cfg.max_grad_norm = shared.max_grad_norm
    train_cfg.pause_after_steps = shared.pause_after_steps
    if shared.warmup_steps is not None:
        train_cfg.warmup_steps = shared.warmup_steps
    elif shared.total_steps is not None:
        train_cfg.warmup_steps = int(shared.warmup_frac * shared.total_steps)
    else:
        train_cfg.warmup_steps = int(shared.warmup_frac * train_cfg.total_steps)
    train_cfg.val_games = shared.val_games

    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_cotrain(config: CotrainConfig) -> list[ModelSlot]:
    """Run co-training from a ``CotrainConfig``. Returns the final slots."""
    device = config.device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available", file=sys.stderr)
            sys.exit(1)
        gpu_cfg = configure_gpu(
            device,
            no_compile=config.no_compile,
            no_amp=(config.amp_dtype == "none"),
            sdpa_math=config.sdpa_math,
        )
        import pawn.model as model_module
        if gpu_cfg.get("sdpa_backend"):
            model_module.SDPA_BACKEND = gpu_cfg["sdpa_backend"]

    total_steps = config.total_steps or 100_000

    slug = random_slug()
    variant_names = [v.name for v in config.variants]

    print(f"=== Co-Training [{slug}] ===")
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Total steps: {total_steps}")
    print(f"Variants: {', '.join(variant_names)}")
    if config.shm_checkpoints:
        print("Checkpoints: /dev/shm (volatile, HF push is durable store)")
    if config.no_outcome_token:
        import warnings
        warnings.warn(
            "no_outcome_token is deprecated and has no effect in cotraining. "
            "Sequences are pure moves by default; use prepend_outcome=True for "
            "outcome-conditioned training.",
            DeprecationWarning, stacklevel=2,
        )
    if config.prepend_outcome:
        print("Outcome token: PREPENDED at position 0 (outcome-conditioned training)")
    print(f"LR: {config.lr:.2e}")
    print()

    # Correct config.prepend_outcome BEFORE constructing slots. Each
    # ModelSlot writes its own config.json during __init__, so a
    # post-construction override would leave the new run directory
    # advertising a format inconsistent with what it actually trained on.
    _resolve_cotrain_resume_prepend_outcome(config)

    # Build slots (now uses the corrected config.prepend_outcome, so each
    # slot's TrainingConfig and config.json reflect the actual sequence
    # format).
    slots: list[ModelSlot] = []
    for variant_spec in config.variants:
        model_cfg, train_cfg = _build_variant_configs(
            variant_spec, config, device,
        )
        hf_repo = f"{config.hf_repo}-{variant_spec.name}" if config.hf_repo else None
        slots.append(ModelSlot(
            variant_spec.name, model_cfg, train_cfg, device, hf_repo,
            shm_checkpoints=config.shm_checkpoints, slug=slug,
            resume_path=variant_spec.resume,
        ))

    # Verify all resumed slots agree on global_step
    resumed_steps = {s.global_step for s in slots if s.global_step > 0}
    if len(resumed_steps) > 1:
        per_slot = {s.name: s.global_step for s in slots}
        print(f"ERROR: Resumed variants disagree on global_step: {per_slot}",
              file=sys.stderr)
        sys.exit(1)
    start_step = max(resumed_steps) if resumed_steps else 0

    # Shared dataset and validation set — use the max_seq_len from the first variant
    # All variants must produce compatible sequence lengths for the shared DataLoader.
    # Use the maximum max_seq_len across all variants so shorter models can mask off.
    max_ply = max(v.max_seq_len for v in config.variants)
    any_legacy = any(v.legacy_vocab for v in config.variants)
    if any_legacy:
        max_ply = 256

    dataset = CLMDataset(
        config.batch_size, max_ply, base_seed=42,
        discard_ply_limit=config.discard_ply_limit,
        mate_boost=config.mate_boost,
        prepend_outcome=config.prepend_outcome,
    )

    print("\nGenerating shared validation set...")
    val_data = create_validation_set(
        config.val_games, max_ply, seed=(2**63) - 1,
        discard_ply_limit=config.discard_ply_limit,
        mate_boost=config.mate_boost,
        prepend_outcome=config.prepend_outcome,
    )

    # Compile models
    if device != "cpu" and not config.no_compile:
        for slot in slots:
            try:
                slot.model = torch.compile(slot.model, mode="default")  # type: ignore[assignment]
                print(f"  [{slot.name}] torch.compile enabled")
            except Exception:
                print(f"  [{slot.name}] torch.compile not available")

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=2 if config.num_workers > 0 else None,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
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
    patience = config.patience or 0
    global_step = start_step
    step_start = time.time()

    print(f"\nStarting training from step {global_step}", flush=True)
    for slot in slots:
        print(f"  [{slot.name}] JSONL: {slot.jsonl_path}", flush=True)
    print()

    active_slots = [s for s in slots if not s.stopped]
    log_interval = config.log_interval
    eval_interval = slots[0].train_cfg.eval_interval
    checkpoint_interval = config.checkpoint_interval

    for batch in loader:
        # Forward + backward + optimizer step per model so CUDA can overlap
        # Adam updates (memory-bound) with the next model's forward (compute-bound)
        all_metrics: dict[str, dict[str, torch.Tensor]] = {}
        all_grad_norms: dict[str, float] = {}
        for slot in active_slots:
            metrics = slot.train_step(batch)
            all_metrics[slot.name] = metrics
            gn = slot.optimizer_step()
            all_grad_norms[slot.name] = gn

        global_step += 1
        for slot in slots:
            slot.global_step = global_step

        step_time = time.time() - step_start
        games_per_sec = config.batch_size / step_time

        # Logging — .item() sync only at log intervals
        if global_step % log_interval == 0:
            active_names = ", ".join(s.name for s in active_slots)
            print(f"step {global_step:>7d} | {games_per_sec:.0f} g/s | {step_time:.2f}s | active: {active_names}", flush=True)
            for slot in active_slots:
                m = all_metrics[slot.name]
                loss_val = m['loss'].item()
                acc_val = m['accuracy'].item()
                gn = all_grad_norms[slot.name]
                lr = slot.scheduler.get_lr()
                print(f"  {slot.name:>5s}: loss {loss_val:.4f} | acc {acc_val:.3f} | "
                      f"lr {lr:.2e} | gn {gn:.2f}", flush=True)

                slot.logger.log_train(
                    step=global_step,
                    lr=lr, grad_norm=gn,
                    step_time=step_time, games_per_sec=games_per_sec,
                    **{"train/loss": loss_val, "train/accuracy": acc_val},
                )

        # Eval
        if global_step % eval_interval == 0:
            for slot in active_slots:
                val_metrics = slot.evaluate(val_data)
                print(f"  {slot.name:>5s} val: loss {val_metrics['val/loss']:.4f} | "
                      f"acc {val_metrics['val/accuracy']:.3f}", flush=True)
                # Track best for eval, /dev/shm cleanup, and patience
                vl = val_metrics["val/loss"]
                if vl < slot.best_val_loss:
                    slot.best_val_loss = vl
                    slot.best_val_step = global_step
                    slot.patience_counter = 0
                else:
                    slot.patience_counter += 1

                slot.logger.log_val(
                    step=global_step,
                    patience=slot.patience_counter,
                    best_val_loss=slot.best_val_loss,
                    best_val_step=slot.best_val_step,
                    **val_metrics,
                )

                # Per-model early stopping
                if patience > 0 and slot.patience_counter >= patience:
                    print(f"  [{slot.name}] Early stopping — no improvement "
                          f"for {patience} evals (best step {slot.best_val_step})")
                    slot.stopped = True
                    slot.save_checkpoint()

            active_slots = [s for s in active_slots if not s.stopped]

            # Push metrics to HF after eval (lightweight, background)
            for slot in slots:
                slot.push_metrics_to_hf()

            if not active_slots:
                print(f"\nAll models stopped at step {global_step}")
                break

        # Checkpoint
        if global_step % checkpoint_interval == 0:
            for slot in active_slots:
                slot.save_checkpoint()

        # Done?
        if global_step >= total_steps:
            print(f"\nTraining complete at step {global_step}")
            for slot in active_slots:
                slot.save_checkpoint()
            break

        # Pause
        if config.pause_after_steps and global_step >= config.pause_after_steps:
            print(f"\nPause requested at step {global_step}, saving checkpoints...")
            for slot in active_slots:
                slot.save_checkpoint()
            break

        # Graceful shutdown
        if _shutdown_requested:
            print(f"\nShutdown requested (signal {_shutdown_signal}), "
                  f"saving checkpoints at step {global_step}...")
            for slot in active_slots:
                slot.save_checkpoint()
            break

        step_start = time.time()

    # Cleanup
    for slot in slots:
        slot.close()

    print("\nAll done.")
    return slots
