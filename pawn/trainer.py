"""PAWN training loop with checkpointing and monitoring.

Uses `AdamW <https://arxiv.org/abs/1711.05101>`_ (Loshchilov & Hutter,
2017) with cosine LR decay (`Loshchilov & Hutter, 2016
<https://arxiv.org/abs/1608.03983>`_) and mixed-precision training
(`Micikevicius et al., 2017 <https://arxiv.org/abs/1710.03740>`_).
"""

import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import chess_engine as engine
from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM
from pawn.data import (
    CLMDataset,
    align_legal_to_preds,
    create_validation_set,
)
from pawn.logging import MetricsLogger

from pawn.data_utils import unpack_grid


class CosineWithWarmup:
    """Cosine LR schedule with linear warmup.

    Based on SGDR (`Loshchilov & Hutter, 2016
    <https://arxiv.org/abs/1608.03983>`_).
    """

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

    def _apply_lr(self, step: int) -> None:
        lr_scale = self._compute_lr_scale(step)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = base_lr * lr_scale

    def step(self) -> None:
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

    def state_dict(self) -> dict[str, int]:
        return {"step": self._step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._step = state["step"]
        self._apply_lr(self._step)


def _build_action_grid_index(n_actions: int) -> list[int]:
    """Build a mapping from action token to grid index (src*64 + dst).

    For the searchless_chess vocab (n_actions=1968), each action maps to a
    (src, dst) pair via the engine's vocabulary export.

    For the legacy PAWN vocab (n_actions=4272), tokens 1-4096 use the formula
    src*64+dst+1, and tokens 4097-4272 are promotions looked up via engine.
    Token 0 (legacy PAD) maps to grid index 0.
    """
    import chess_engine
    from pawn.config import NUM_ACTIONS, LegacyVocab

    vocab = chess_engine.export_move_vocabulary()
    token_to_move = vocab["token_to_move"]
    square_names = vocab["square_names"]
    name_to_idx = {name: i for i, name in enumerate(square_names)}

    if n_actions == NUM_ACTIONS:
        grid_indices = []
        for action in range(n_actions):
            uci = token_to_move[action]
            src_sq = name_to_idx[uci[:2]]
            dst_sq = name_to_idx[uci[2:4]]
            grid_indices.append(src_sq * 64 + dst_sq)
        return grid_indices

    # Legacy vocab: token 0 = PAD (map to 0), tokens 1-4096 = grid, 4097-4272 = promos
    assert n_actions == LegacyVocab.NUM_ACTIONS
    grid_indices = [0]  # token 0 (PAD)
    for token in range(1, 4097):
        t = token - 1
        grid_indices.append((t // 64) * 64 + (t % 64))
    # Promotions: look up via searchless_to_pawn
    for token in range(4097, n_actions + 1):
        action = chess_engine.pawn_to_searchless(token)
        if action >= 0:
            uci = token_to_move[action]
            src_sq = name_to_idx[uci[:2]]
            dst_sq = name_to_idx[uci[2:4]]
            grid_indices.append(src_sq * 64 + dst_sq)
        else:
            grid_indices.append(0)
    return grid_indices


# Cache: n_actions -> grid_indices list
_ACTION_GRID_INDEX_CACHE: dict[int, list[int]] = {}
# Cache: (n_actions, device) -> tensor
_ACTION_GRID_TENSOR_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _get_action_grid_index(device: str | torch.device, n_actions: int | None = None) -> torch.Tensor:
    """Get the action-token-to-grid-index mapping as a tensor on the given device."""
    from pawn.config import NUM_ACTIONS
    if n_actions is None:
        n_actions = NUM_ACTIONS
    dev_key = (n_actions, str(device))
    cached = _ACTION_GRID_TENSOR_CACHE.get(dev_key)
    if cached is not None:
        return cached
    if n_actions not in _ACTION_GRID_INDEX_CACHE:
        _ACTION_GRID_INDEX_CACHE[n_actions] = _build_action_grid_index(n_actions)
    t = torch.tensor(_ACTION_GRID_INDEX_CACHE[n_actions], dtype=torch.long, device=device)
    _ACTION_GRID_TENSOR_CACHE[dev_key] = t
    return t


def compute_legal_move_rate(
    logits: torch.Tensor,
    legal_grid: torch.Tensor,
    loss_mask: torch.Tensor,
    game_lengths: torch.Tensor,
    n_actions: int | None = None,
) -> float:
    """Compute fraction of argmax predictions that are legal moves.

    Wrapper that computes preds from logits for backward compatibility.
    Prefer compute_legal_move_rate_from_preds when hidden states are available.
    """
    preds = logits.argmax(dim=-1)
    return compute_legal_move_rate_from_preds(
        preds, legal_grid, loss_mask, game_lengths, n_actions=n_actions,
    )


def compute_legal_move_rate_from_preds(
    preds: torch.Tensor,
    legal_grid: torch.Tensor,
    loss_mask: torch.Tensor,
    game_lengths: torch.Tensor,
    min_ply: int = 0,
    max_ply_limit: int | None = None,
    n_actions: int | None = None,
) -> float:
    """Compute fraction of argmax predictions that are legal moves.

    Evaluated at positions where loss_mask is True, intersected with the
    ply range [min_ply, max_ply_limit).

    Args:
        preds: (B, T) argmax token predictions
        legal_grid: (B, max_ply, 64) bit-packed legal moves from engine
        loss_mask: (B, T) bool
        game_lengths: (B,) int
        min_ply: inclusive lower bound on the ply range (default 0)
        max_ply_limit: exclusive upper bound on the ply range (default None = all)
        n_actions: number of action tokens in the vocab (default: NUM_ACTIONS=1968)
    """
    from pawn.config import NUM_ACTIONS
    if n_actions is None:
        n_actions = NUM_ACTIONS

    B, T = preds.shape
    max_ply = legal_grid.shape[1]

    with torch.no_grad():
        move_mask = loss_mask.clone()

        if not move_mask.any():
            return 0.0

        # Unpack legal grid to dense: (B, max_ply, 64, 64) -> flatten to (B, max_ply, 4096)
        legal_dense = unpack_grid(legal_grid)  # (B, max_ply, 64, 64)
        legal_flat = legal_dense.reshape(B, max_ply, 4096)  # (B, max_ply, 4096)

        n_plies = min(T, max_ply)
        upper = min(n_plies, max_ply_limit) if max_ply_limit is not None else n_plies
        valid_count = 0
        legal_acc = torch.tensor(0, dtype=torch.long, device=preds.device)

        # Action token -> grid index lookup (lazily built, cached per n_actions)
        action_grid_idx = _get_action_grid_index(preds.device, n_actions)

        for p in range(min_ply, upper):
            pos_mask = move_mask[:, p]  # (B,)
            if not pos_mask.any():
                continue

            batch_preds = preds[pos_mask, p]  # (N,)
            batch_legal = legal_flat[pos_mask, p]  # (N, 4096)
            n = len(batch_preds)
            arange_n = torch.arange(n, device=preds.device)

            # Action tokens: look up grid index
            max_idx = len(action_grid_idx) - 1
            is_action = batch_preds <= max_idx
            grid_idx = action_grid_idx[batch_preds.clamp(0, max_idx)]
            action_legal = batch_legal[arange_n, grid_idx] > 0.5
            legal_action = is_action & action_legal

            valid_count += n
            legal_acc += legal_action.sum()

        if valid_count == 0:
            return 0.0
        return legal_acc.item() / valid_count



def _game_completion_chunk(
    preds: torch.Tensor,
    legal_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    game_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized per-game forfeit lookup for a single chunk.

    Returns (has_forfeit, first_forfeit, gl) each of shape (B,).
    """
    B, T = preds.shape
    V = legal_mask.shape[2]

    has_legal = legal_mask.any(dim=-1)                # (B, T)
    checked = loss_mask.bool() & has_legal            # (B, T)

    preds_clamped = preds.clamp(min=0, max=V - 1).long()
    in_range = preds < V                              # (B, T)

    legal_at_pred = legal_mask.gather(
        dim=-1, index=preds_clamped.unsqueeze(-1)
    ).squeeze(-1)                                      # (B, T)

    illegal = checked & (~in_range | ~legal_at_pred)  # (B, T)

    has_forfeit = illegal.any(dim=-1)                 # (B,)
    first_forfeit = illegal.int().argmax(dim=-1)      # (B,)
    gl = game_lengths.long().clamp(max=T)             # (B,)
    return has_forfeit, first_forfeit, gl


def _aggregate_game_completion(
    has_forfeit: torch.Tensor,
    first_forfeit: torch.Tensor,
    gl: torch.Tensor,
) -> dict[str, float]:
    """Reduce per-game tensors to scalar summary statistics."""
    n_games = int(has_forfeit.shape[0])
    if n_games == 0:
        return {
            "game_completion_rate": 0.0,
            "avg_pct_completion": 0.0,
            "avg_plies_completed": 0.0,
            "min_forfeit_ply": 0.0,
            "max_forfeit_ply": 0.0,
            "median_forfeit_ply": 0.0,
        }

    plies_completed = torch.where(has_forfeit, first_forfeit, gl).float()
    pct = torch.where(
        has_forfeit & (gl > 0),
        first_forfeit.float() / gl.clamp(min=1).float(),
        torch.ones_like(plies_completed),
    )

    n_complete = int((~has_forfeit).sum().item())
    forfeit_only = first_forfeit[has_forfeit].float()
    if forfeit_only.numel() > 0:
        min_forfeit = float(forfeit_only.min().item())
        max_forfeit = float(forfeit_only.max().item())
        median_forfeit = float(forfeit_only.median().item())
    else:
        min_forfeit = 0.0
        max_forfeit = 0.0
        median_forfeit = 0.0

    return {
        "game_completion_rate": n_complete / n_games,
        "avg_pct_completion": float(pct.mean().item()),
        "avg_plies_completed": float(plies_completed.mean().item()),
        "min_forfeit_ply": min_forfeit,
        "max_forfeit_ply": max_forfeit,
        "median_forfeit_ply": median_forfeit,
    }


def compute_game_completion(
    preds: torch.Tensor,
    legal_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    game_lengths: torch.Tensor,
) -> dict[str, float]:
    """Measure how often the model gets through a full game without illegal moves.

    Vectorized: for each game, finds the first ply where the argmax prediction
    is either out-of-range or marked illegal by the legal_mask.  Games with no
    illegal prediction are "completed".

    Args:
        preds: (B, T) argmax token predictions (aligned with targets)
        legal_mask: (B, T, V) bool — legal token mask (shifted to align with targets)
        loss_mask: (B, T) bool — which positions are valid
        game_lengths: (B,) int — number of valid plies per game

    Returns dict with:
        game_completion_rate: fraction of games with zero illegal moves
        avg_pct_completion: mean fraction of game completed before forfeit
        avg_plies_completed: mean plies completed before first illegal move.
            Games with no illegal moves contribute their full game_length.
        min_forfeit_ply / max_forfeit_ply / median_forfeit_ply: forfeit ply
            statistics across games that actually forfeited (0 if none).
    """
    with torch.no_grad():
        has_forfeit, first_forfeit, gl = _game_completion_chunk(
            preds, legal_mask, loss_mask, game_lengths,
        )
    return _aggregate_game_completion(has_forfeit, first_forfeit, gl)


def _get_grad_norm(model: nn.Module) -> float:
    grads = [p.grad.data for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.stack([g.float().norm() for g in grads]).square().sum()
    return total.sqrt().item()


class CLMTrainer:
    def __init__(
        self,
        train_cfg: TrainingConfig,
        model_cfg: CLMConfig,
        hf_repo: str | None = None,
        patience: int | None = None,
        legality_late_ply: int | None = None,
    ):
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = train_cfg.device
        self.global_step = 0
        self.hf_repo = hf_repo
        self.hf_branch: str | None = None

        # Compound early stopping state
        self.patience = patience
        self.legality_late_ply = (
            legality_late_ply if legality_late_ply is not None
            else model_cfg.max_seq_len // 2
        )
        self.best_val_loss: float = float("inf")
        self.best_late_legality: float = 0.0
        self.best_game_completion: float = 0.0
        self.best_avg_plies_completed: float = 0.0
        self.patience_counter: int = 0

        self.logger = MetricsLogger(
            train_cfg.log_dir, run_prefix="run", device=self.device,
        )
        self.run_dir = str(self.logger.run_dir)
        self.cfg.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self._jsonl_path = str(self.logger.metrics_path)

        if self.hf_repo:
            self.hf_branch = f"run/{os.path.basename(self.run_dir)}"

        self._model = PAWNCLM(model_cfg).to(self.device)
        self.model = self._model
        param_count = sum(p.numel() for p in self._model.parameters())
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

        if train_cfg.no_outcome_token:
            import warnings
            warnings.warn(
                "no_outcome_token is deprecated and has no effect. "
                "Sequences are pure moves by default (no outcome prefix). "
                "Use prepend_outcome=True if you need outcome-conditioned training.",
                DeprecationWarning, stacklevel=2,
            )

        self.dataset = CLMDataset(
            train_cfg.batch_size, train_cfg.max_ply, train_cfg.base_seed,
            discard_ply_limit=train_cfg.discard_ply_limit,
            mate_boost=train_cfg.mate_boost,
            prepend_outcome=train_cfg.prepend_outcome,
        )
        print("Generating validation set...")
        self.val_data = create_validation_set(
            train_cfg.val_games, train_cfg.max_ply, train_cfg.val_seed,
            discard_ply_limit=train_cfg.discard_ply_limit,
            mate_boost=train_cfg.mate_boost,
            prepend_outcome=train_cfg.prepend_outcome,
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

        # Patience and legality_late_ply aren't in TrainingConfig — pass them
        # alongside so the dashboard and other consumers can see them.
        training_log = {
            **train_cfg.__dict__,
            "patience": self.patience,
            "legality_late_ply": self.legality_late_ply,
        }
        self.logger.log_config(
            model=model_cfg.__dict__,
            training=training_log,
            param_count=param_count,
            compiled=self._compiled,
            formulation="clm",
        )
        self.logger.write_config_json(
            model=model_cfg.__dict__,
            training=training_log,
            param_count=param_count,
            compiled=self._compiled,
            formulation="clm",
        )

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
            text = data.rstrip(b"\x00").decode(errors="replace")
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
        """Low-level JSONL write for seed_logs compatibility."""
        self.logger._write(record)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(self.device, non_blocking=True)

        model = self._eager_model()

        with torch.amp.autocast(self.device, enabled=self.cfg.use_amp):
            loss, metrics = model.forward_train(input_ids, loss_mask, targets)

        scaled_loss = loss / self.cfg.accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        return metrics

    def optimizer_step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        grad_norm = _get_grad_norm(self._model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return grad_norm

    def _eager_model(self) -> PAWNCLM:
        return self._model

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
            input_ids = self.val_data["input_ids"][start:end].to(self.device, non_blocking=True)
            targets = self.val_data["targets"][start:end].to(self.device, non_blocking=True)
            loss_mask = self.val_data["loss_mask"][start:end].to(self.device, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(self.device, enabled=self.cfg.use_amp):
                    # Get hidden states without materializing full (B,T,V) logits
                    hidden = model.forward_eval(input_ids, loss_mask)

                    # Sparse projection: only valid positions through lm_head
                    valid_hidden = hidden[loss_mask]
                    valid_logits = model.lm_head(valid_hidden)
                    valid_targets = targets[loss_mask]

                loss = F.cross_entropy(valid_logits, valid_targets)
                accuracy = (valid_logits.argmax(-1) == valid_targets).float().mean().item()
                metrics: dict[str, float] = {"loss": loss.item(), "accuracy": accuracy}

                # Top-5 accuracy
                top5 = valid_logits.topk(5, dim=-1).indices
                top5_acc = (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
                metrics["top5_accuracy"] = top5_acc

                # Legal move rate: reuse already-computed valid_logits argmax
                if has_legal:
                    legal_grid = self.val_data["legal_grid"][start:end].to(self.device, non_blocking=True)
                    game_lengths = self.val_data["game_lengths"][start:end].to(self.device, non_blocking=True)
                    preds = torch.zeros_like(loss_mask, dtype=torch.long)
                    preds[loss_mask] = valid_logits.argmax(dim=-1)
                    n_act = self._model.embed.n_actions
                    legal_rate = compute_legal_move_rate_from_preds(
                        preds, legal_grid, loss_mask, game_lengths,
                        n_actions=n_act,
                    )
                    metrics["legal_move_rate"] = legal_rate

                    # Late-game legality: only plies >= legality_late_ply
                    late_legal_rate = compute_legal_move_rate_from_preds(
                        preds, legal_grid, loss_mask, game_lengths,
                        min_ply=self.legality_late_ply,
                        n_actions=n_act,
                    )
                    metrics["late_legal_move_rate"] = late_legal_rate

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        avg = {f"val/{k}": v / n_batches for k, v in total_metrics.items()}
        avg["val/perplexity"] = math.exp(min(avg["val/loss"], 20.0))

        # Game completion eval: can the model get through entire games
        # without picking an illegal move?  Fully vectorized, chunked to keep
        # the dense (B, T, vocab) legal mask within VRAM limits.
        if "game_lengths" in self.val_data:
            gc_batch = max(1, self.cfg.batch_size)  # same chunk size as loss eval
            vocab_size = self.model_cfg.vocab_size
            has_forfeit_all: list[torch.Tensor] = []
            first_forfeit_all: list[torch.Tensor] = []
            gl_all: list[torch.Tensor] = []

            prepend_outcome = bool(
                self.val_data.get(
                    "prepend_outcome", torch.tensor(False)
                ).item()
            )

            for start in range(0, n, gc_batch):
                end = min(start + gc_batch, n)
                gc_input = self.val_data["input_ids"][start:end].to(self.device)
                gc_loss_mask = self.val_data["loss_mask"][start:end].to(self.device)
                gc_game_lengths = self.val_data["game_lengths"][start:end].to(self.device)
                raw_move_ids = (
                    self.val_data["move_ids"][start:end].numpy().astype(np.int16)
                )
                gl_np = self.val_data["game_lengths"][start:end].numpy().astype(np.int16)

                with torch.no_grad():
                    with torch.amp.autocast(self.device, enabled=self.cfg.use_amp):
                        hidden = model.forward_eval(gc_input, gc_loss_mask)
                        gc_logits = model.lm_head(hidden)
                    gc_preds = gc_logits.argmax(dim=-1)

                    legal_tokens = engine.compute_legal_token_masks(
                        raw_move_ids, gl_np, vocab_size,
                    )
                    legal_mask_t = torch.from_numpy(
                        align_legal_to_preds(legal_tokens, prepend_outcome)
                    ).to(self.device)

                    has_forfeit, first_forfeit, gl = _game_completion_chunk(
                        gc_preds, legal_mask_t, gc_loss_mask, gc_game_lengths,
                    )

                has_forfeit_all.append(has_forfeit.cpu())
                first_forfeit_all.append(first_forfeit.cpu())
                gl_all.append(gl.cpu())

                del gc_input, gc_loss_mask, gc_game_lengths, gc_logits, gc_preds
                del legal_tokens, legal_mask_t

            gc = _aggregate_game_completion(
                torch.cat(has_forfeit_all),
                torch.cat(first_forfeit_all),
                torch.cat(gl_all),
            )
            avg["val/game_completion_rate"] = gc["game_completion_rate"]
            avg["val/avg_pct_completion"] = gc["avg_pct_completion"]
            avg["val/avg_plies_completed"] = gc["avg_plies_completed"]
            avg["val/min_forfeit_ply"] = gc["min_forfeit_ply"]
            avg["val/max_forfeit_ply"] = gc["max_forfeit_ply"]
            avg["val/median_forfeit_ply"] = gc["median_forfeit_ply"]

            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return avg

    def train(self):
        self.dataset.set_start_step(self.global_step)
        num_workers = self.cfg.num_workers
        loader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=(self.device != "cpu"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        _shutdown_requested = False
        _shutdown_signal = None

        def _graceful_exit(signum, frame):
            nonlocal _shutdown_requested, _shutdown_signal
            _shutdown_requested = True
            _shutdown_signal = signum

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
                    # .item() sync only at log intervals (metrics are tensors here)
                    loss_val = metrics['loss'].item()
                    acc_val = metrics['accuracy'].item()
                    lr = self.scheduler.get_lr()

                    print(
                        f"step {self.global_step:>7d} | "
                        f"loss {loss_val:.4f} | "
                        f"acc {acc_val:.3f} | "
                        f"lr {lr:.2e} | "
                        f"gn {grad_norm:.2f} | "
                        f"{games_per_sec:.0f} g/s | "
                        f"{step_time:.2f}s",
                        flush=True,
                    )

                    self.logger.log_train(
                        step=self.global_step,
                        lr=lr, grad_norm=grad_norm,
                        step_time=step_time, games_per_sec=games_per_sec,
                        **{"train/loss": loss_val, "train/accuracy": acc_val},  # type: ignore[arg-type]
                    )

                    if self.wandb_run:
                        self.wandb_run.log({
                            "train/loss": loss_val, "train/accuracy": acc_val,
                            "train/lr": lr, "train/grad_norm": grad_norm,
                            "train/step_time": step_time, "train/games_per_sec": games_per_sec,
                        }, step=self.global_step)

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
                    if "val/late_legal_move_rate" in val_metrics:
                        val_msg += f" | late_legal {val_metrics['val/late_legal_move_rate']:.3f}"
                    if "val/game_completion_rate" in val_metrics:
                        val_msg += (
                            f" | complete {val_metrics['val/game_completion_rate']:.3f}"
                            f" | avg_ply {val_metrics['val/avg_plies_completed']:.0f}"
                        )
                        if "val/min_forfeit_ply" in val_metrics:
                            val_msg += (
                                f" | forfeit [{val_metrics['val/min_forfeit_ply']:.0f}"
                                f"-{val_metrics['val/max_forfeit_ply']:.0f}"
                                f" med {val_metrics['val/median_forfeit_ply']:.0f}]"
                            )

                    # Compound early stopping
                    extra_log: dict[str, object] = {}
                    if self.patience is not None:
                        val_loss = val_metrics["val/loss"]
                        late_legality = val_metrics.get("val/late_legal_move_rate", 0.0)
                        game_completion = val_metrics.get("val/game_completion_rate", 0.0)
                        avg_plies = val_metrics.get("val/avg_plies_completed", 0.0)

                        improved = False
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            improved = True
                        if late_legality > self.best_late_legality:
                            self.best_late_legality = late_legality
                            improved = True
                        if game_completion > self.best_game_completion:
                            self.best_game_completion = game_completion
                            improved = True
                        if avg_plies > self.best_avg_plies_completed:
                            self.best_avg_plies_completed = avg_plies
                            improved = True

                        if improved:
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1

                        val_msg += f" | pat {self.patience_counter}/{self.patience}"
                        extra_log = {
                            "patience_counter": self.patience_counter,
                            "best_val_loss": self.best_val_loss,
                            "best_late_legality": self.best_late_legality,
                            "best_game_completion": self.best_game_completion,
                            "best_avg_plies_completed": self.best_avg_plies_completed,
                        }

                    print(val_msg, flush=True)

                    self.logger.log_val(step=self.global_step, **val_metrics, **extra_log)  # type: ignore[arg-type]

                    if self.wandb_run:
                        self.wandb_run.log({**val_metrics, **extra_log}, step=self.global_step)

                if self.global_step % self.cfg.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.global_step >= self.cfg.total_steps:
                    print(f"Training complete at step {self.global_step}")
                    self.save_checkpoint()
                    break

                if (self.patience is not None
                        and self.patience_counter >= self.patience):
                    print(f"\nEarly stopping at step {self.global_step} "
                          f"(no improvement for {self.patience} evals)")
                    self.save_checkpoint()
                    break

                if (self.cfg.pause_after_steps
                        and self.global_step >= self.cfg.pause_after_steps):
                    print(f"\n  Paused at step {self.global_step} "
                          f"(pause_after_steps={self.cfg.pause_after_steps})")
                    self.save_checkpoint()
                    break

                if _shutdown_requested:
                    print(f"\nShutdown requested (signal {_shutdown_signal}), "
                          f"saving checkpoint at step {self.global_step}...")
                    self.save_checkpoint()
                    break

                step_start = time.time()

        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)

        self.logger.close()

    def save_checkpoint(self, path: str | None = None):
        from pawn.checkpoint import save_pretrain_checkpoint

        if path is None:
            path = os.path.join(
                self.cfg.checkpoint_dir, f"step_{self.global_step:08d}"
            )

        model: PAWNCLM = self._eager_model()

        save_pretrain_checkpoint(
            path,
            model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.global_step,
            self.model_cfg.__dict__,
            self.cfg.__dict__,
            extra={
                "best_val_loss": self.best_val_loss,
                "best_late_legality": self.best_late_legality,
                "best_game_completion": self.best_game_completion,
                "best_avg_plies_completed": self.best_avg_plies_completed,
                "patience_counter": self.patience_counter,
            },
        )
        print(f"Checkpoint saved: {path}")

        if self.hf_repo and self.hf_branch:
            from pawn.checkpoint import push_checkpoint_to_hf
            try:
                push_checkpoint_to_hf(
                    path, self.hf_repo, self.hf_branch,
                    metrics_path=self._jsonl_path,
                    step=self.global_step,
                )
                print(f"Pushed to HF: {self.hf_repo}@{self.hf_branch}")
            except Exception as e:
                print(f"WARNING: HF push failed: {e}")

    def load_checkpoint(self, path: str):
        from pawn.checkpoint import load_pretrain_checkpoint

        model: PAWNCLM = self._eager_model()

        meta = load_pretrain_checkpoint(
            path, model, self.optimizer, self.scheduler, self.scaler,
            device=self.device,
        )
        self.global_step = meta["global_step"]
        if meta.get("best_val_loss") is not None:
            self.best_val_loss = meta["best_val_loss"]
        if meta.get("best_late_legality") is not None:
            self.best_late_legality = meta["best_late_legality"]
        if meta.get("best_game_completion") is not None:
            self.best_game_completion = meta["best_game_completion"]
        if meta.get("best_avg_plies_completed") is not None:
            self.best_avg_plies_completed = meta["best_avg_plies_completed"]
        if meta.get("patience_counter") is not None:
            self.patience_counter = meta["patience_counter"]
        print(f"Resumed from step {self.global_step}")
