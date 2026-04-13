"""Optuna hyperparameter sweep integration for PAWN training scripts.

Wraps existing training scripts as Optuna objectives with:
- Automated search space definition per adapter type
- Pruning via val_loss reported at each evaluation
- Trial isolation (unique run dir per trial)
- MetricsLogger integration for consistent logging

Usage:
    # From CLI
    uv run python scripts/sweep.py --adapter lora --checkpoint ... --pgn ...

    # Programmatic
    from pawn.sweep import create_study, AdapterObjective
    study = create_study("lora", storage="sqlite:///sweeps/lora.db")
    study.optimize(AdapterObjective("lora", checkpoint, pgn, device), n_trials=50)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

_optuna = None
try:
    import optuna as _optuna
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Search spaces per adapter type
# ---------------------------------------------------------------------------

def suggest_common(trial: "optuna.trial.BaseTrial") -> dict:
    """Hyperparameters shared across all adapter types."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.15),
        "patience": trial.suggest_int("patience", 5, 20),
    }


def suggest_lora(trial: "optuna.trial.BaseTrial") -> dict:
    """LoRA-specific hyperparameters."""
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_categorical("lora_rank", [2, 4, 8, 16, 32])
    params["lora_targets"] = trial.suggest_categorical("lora_targets", ["qkvo", "qv", "qkv"])
    params["lora_ffn"] = trial.suggest_categorical("lora_ffn", [True, False])
    return params


def suggest_bottleneck(trial: "optuna.trial.BaseTrial") -> dict:
    """Bottleneck adapter hyperparameters."""
    params = suggest_common(trial)
    params["bottleneck_dim"] = trial.suggest_categorical("bottleneck_dim", [4, 8, 16, 32, 64, 128])
    params["no_adapt_attn"] = trial.suggest_categorical("no_adapt_attn", [True, False])
    params["no_adapt_ffn"] = trial.suggest_categorical("no_adapt_ffn", [True, False])
    return params


def suggest_film(trial: "optuna.trial.BaseTrial") -> dict:
    """FiLM hyperparameters."""
    params = suggest_common(trial)
    params["no_output_film"] = trial.suggest_categorical("no_output_film", [True, False])
    return params


def suggest_sparse(trial: "optuna.trial.BaseTrial") -> dict:
    """Sparse adapter hyperparameters."""
    params = suggest_common(trial)
    params["density"] = trial.suggest_float("density", 0.001, 0.1, log=True)
    params["sparse_targets"] = trial.suggest_categorical("sparse_targets", ["qkvo", "qv", "qkv"])
    params["sparse_ffn"] = trial.suggest_categorical("sparse_ffn", [True, False])
    return params


def suggest_hybrid(trial: "optuna.trial.BaseTrial") -> dict:
    """Hybrid (LoRA + FiLM) hyperparameters."""
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.15),
        "patience": trial.suggest_int("patience", 5, 20),
        "lora_lr": trial.suggest_float("lora_lr", 1e-5, 1e-2, log=True),
        "film_lr": trial.suggest_float("film_lr", 1e-5, 1e-2, log=True),
        "lora_rank": trial.suggest_categorical("lora_rank", [2, 4, 8, 16]),
        "lora_targets": trial.suggest_categorical("lora_targets", ["qkvo", "qv", "qkv"]),
        "no_film": trial.suggest_categorical("no_film", [True, False]),
    }
    return params


def suggest_tiny(trial: "optuna.trial.BaseTrial") -> dict:
    """Tiny standalone model hyperparameters."""
    params = suggest_common(trial)
    params["d_model"] = trial.suggest_categorical("d_model", [32, 64, 84, 128])
    params["n_layers"] = trial.suggest_int("n_layers", 1, 4)
    params["n_heads"] = trial.suggest_categorical("n_heads", [1, 2, 4])
    return params


def suggest_pretrain(trial: "optuna.trial.BaseTrial") -> dict:
    """Pretraining hyperparameters (fixed architecture, tune training)."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_steps": trial.suggest_int("warmup_steps", 500, 5000, step=500),
        "total_steps": trial.suggest_categorical("total_steps", [50000, 100000]),
    }


def suggest_architecture(trial: "optuna.trial.BaseTrial") -> dict:
    """Architecture search space for pretraining.

    Explores model size, depth/width tradeoff, and training hyperparameters.
    Target budget: 150M-500M parameters on 80GB GPUs.
    """
    d_model = trial.suggest_categorical("d_model", [512, 640, 768, 896, 1024, 1280])
    n_layers = trial.suggest_int("n_layers", 8, 24, step=2)
    n_heads = trial.suggest_categorical("n_heads", [8, 16])
    d_ff_mult = trial.suggest_categorical("d_ff_mult", [3, 4, 5])

    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_model * d_ff_mult,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_steps": trial.suggest_int("warmup_steps", 500, 3000, step=500),
    }


def _suggest_rosa_common(trial: "optuna.trial.BaseTrial") -> dict:
    """Shared search space for all RoSA modes."""
    params = suggest_common(trial)
    params["density"] = trial.suggest_float("density", 0.001, 0.1, log=True)
    params["lora_rank"] = trial.suggest_categorical("lora_rank", [2, 4, 8, 16])
    params["lora_targets"] = trial.suggest_categorical("lora_targets", ["qkvo", "qv", "qkv"])
    params["warmup_steps"] = trial.suggest_int("warmup_steps", 32, 256, step=32)
    params["mask_samples"] = trial.suggest_categorical("mask_samples", [16, 32, 64])
    params["grad_alpha"] = trial.suggest_categorical("grad_alpha", [1, 2])
    return params


def suggest_rosa(trial: "optuna.trial.BaseTrial") -> dict:
    """Standard RoSA: joint LoRA + gradient-informed sparse."""
    params = _suggest_rosa_common(trial)
    params["mode"] = "rosa"
    return params


def suggest_retro_sparse(trial: "optuna.trial.BaseTrial") -> dict:
    """Retrospective sparse-only with gradient-informed masks."""
    params = _suggest_rosa_common(trial)
    params["mode"] = "retro-sparse"
    return params


def suggest_retro_bottleneck(trial: "optuna.trial.BaseTrial") -> dict:
    """Retrospective sparse + bottleneck adapters."""
    params = _suggest_rosa_common(trial)
    params["mode"] = "retro-bottleneck"
    params["bottleneck_dim"] = trial.suggest_categorical("bottleneck_dim", [4, 8, 16])
    return params


# ---------------------------------------------------------------------------
# RoSA ratio sweep: bottleneck vs sparse parameter allocation
# ---------------------------------------------------------------------------

# Architecture constants for pawn-base (d_model=512, n_layers=8)
# Bottleneck: 2 positions (attn+ffn) * n_layers * 2 projections (down+up) * d_model
_BASE_BOTTLENECK_PARAMS_PER_DIM = 4 * 8 * 512  # 16_384
# Sparse maskable: 4 attn projections (qkvo) * n_layers * d_model^2
_BASE_SPARSE_MASKABLE_PARAMS = 4 * 8 * 512 * 512  # 8_388_608


def suggest_rosa_ratio(trial: "optuna.trial.BaseTrial") -> dict:
    """Retro-bottleneck ratio sweep: vary bottleneck vs sparse param allocation.

    For a fixed total parameter budget, sweeps the fraction allocated to
    bottleneck adapters (rest goes to gradient-informed sparse masks).
    Nuisance hyperparameters are fixed to focus trials on the ratio.
    """
    total_budget = trial.suggest_categorical("total_budget", [100_000, 250_000, 500_000])
    bottleneck_ratio = trial.suggest_float("bottleneck_ratio", 0.05, 0.95)

    # Derive bottleneck_dim and sparse density from budget split
    bottleneck_budget = bottleneck_ratio * total_budget
    sparse_budget = (1.0 - bottleneck_ratio) * total_budget

    bottleneck_dim = max(1, round(bottleneck_budget / _BASE_BOTTLENECK_PARAMS_PER_DIM))
    density = max(1e-5, sparse_budget / _BASE_SPARSE_MASKABLE_PARAMS)

    # Log realized param counts (rounding causes deviation from target)
    actual_bn = bottleneck_dim * _BASE_BOTTLENECK_PARAMS_PER_DIM
    actual_sp = int(density * _BASE_SPARSE_MASKABLE_PARAMS)
    trial.set_user_attr("actual_bottleneck_params", actual_bn)
    trial.set_user_attr("actual_sparse_params", actual_sp)
    trial.set_user_attr("actual_total_params", actual_bn + actual_sp)

    return {
        "mode": "retro-bottleneck",
        "bottleneck_dim": bottleneck_dim,
        "density": density,
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "weight_decay": 0.01,
        "warmup_frac": 0.05,
        "patience": 10,
        "lora_rank": 4,
        "lora_targets": "qkvo",
        "warmup_steps": trial.suggest_int("warmup_steps", 64, 128, step=64),
        "mask_samples": 32,
        "grad_alpha": 2,
    }


SUGGEST_FNS = {
    "lora": suggest_lora,
    "bottleneck": suggest_bottleneck,
    "film": suggest_film,
    "sparse": suggest_sparse,
    "hybrid": suggest_hybrid,
    "rosa": suggest_rosa,
    "retro-sparse": suggest_retro_sparse,
    "retro-bottleneck": suggest_retro_bottleneck,
    "rosa-ratio": suggest_rosa_ratio,
    "tiny": suggest_tiny,
    "architecture": suggest_architecture,
    "pretrain": suggest_pretrain,
}

ADAPTER_SCRIPTS = {
    "lora": "scripts/train.py",
    "bottleneck": "scripts/train.py",
    "film": "scripts/train.py",
    "sparse": "scripts/train.py",
    "hybrid": "scripts/train.py",
    "rosa": "scripts/train.py",
    "retro-sparse": "scripts/train.py",
    "retro-bottleneck": "scripts/train.py",
    "rosa-ratio": "scripts/train.py",
    "tiny": "scripts/train.py",
    "pretrain": "scripts/train.py",
    "architecture": "scripts/train.py",
}

_ADAPTER_STRATEGY = {
    "lora": "lora",
    "bottleneck": "bottleneck",
    "film": "film",
    "sparse": "sparse",
    "hybrid": "hybrid",
    "rosa": "rosa",
    "retro-sparse": "rosa",
    "retro-bottleneck": "rosa",
    "rosa-ratio": "rosa",
    "tiny": "specialized_clm",
}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _params_to_cli_args(params: dict) -> list[str]:
    """Convert a dict of hyperparameters to CLI argument list."""
    args = []
    for k, v in params.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.extend([flag, str(v)])
    return args


def _extract_best_val_loss(metrics_dir: Path) -> float:
    """Read metrics.jsonl and return the best val loss."""
    best = float("inf")
    for metrics_path in metrics_dir.glob("*/metrics.jsonl"):
        with open(metrics_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Adapter scripts log val_loss in train records
                vl = r.get("val_loss") or r.get("val/loss")
                if vl is not None and vl < best:
                    best = vl
    return best


def _extract_val_losses_by_epoch(metrics_dir: Path) -> list[tuple[int, float]]:
    """Read metrics.jsonl and return (epoch_or_step, val_loss) pairs for pruning."""
    losses = []
    for metrics_path in metrics_dir.glob("*/metrics.jsonl"):
        with open(metrics_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vl = r.get("val_loss") or r.get("val/loss")
                step = r.get("epoch", r.get("step"))
                if vl is not None and step is not None:
                    losses.append((step, vl))
    losses.sort()
    return losses


class AdapterObjective:
    """Optuna objective that runs an adapter training script as a subprocess.

    Each trial gets a unique output directory. The objective reads metrics.jsonl
    after training to extract the best validation loss.
    """

    def __init__(
        self,
        adapter_type: str,
        checkpoint: str,
        pgn: str,
        device: str = "cuda",
        output_base: str = "sweeps",
        epochs: int = 50,
        n_gpus: int = 1,
        extra_args: list[str] | None = None,
    ):
        self.adapter_type = adapter_type
        self.checkpoint = checkpoint
        self.pgn = pgn
        self.device = device
        self.output_base = Path(output_base) / adapter_type
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.n_gpus = n_gpus
        self.extra_args = extra_args or []
        self.script = ADAPTER_SCRIPTS[adapter_type]

    def __call__(self, trial: "optuna.trial.BaseTrial") -> float:
        suggest_fn = SUGGEST_FNS[self.adapter_type]
        params = suggest_fn(trial)

        trial_dir = self.output_base / f"trial_{trial.number:04d}"

        # Build command
        cmd = [sys.executable, self.script]

        # All sweep targets dispatch through scripts/train.py via --run-type.
        if self.adapter_type in ("pretrain", "architecture"):
            cmd.extend(["--run-type", "pretrain"])
            cmd.extend(["--log-dir", str(trial_dir)])
            cmd.extend(["--local-checkpoints"])
        else:
            cmd.extend(["--run-type", "adapter"])
            cmd.extend(["--strategy", _ADAPTER_STRATEGY[self.adapter_type]])
            cmd.extend(["--checkpoint", self.checkpoint])
            cmd.extend(["--pgn", self.pgn])
            cmd.extend(["--log-dir", str(trial_dir)])
            cmd.extend(["--local-checkpoints"])
            if "epochs" not in params:
                cmd.extend(["--epochs", str(self.epochs)])
        cmd.extend(["--device", self.device])

        # Suggested hyperparameters
        cmd.extend(_params_to_cli_args(params))

        # Extra user-provided args
        cmd.extend(self.extra_args)

        # GPU affinity: pin trial to GPU (trial.number % n_gpus)
        env = os.environ.copy()
        if self.n_gpus > 1:
            gpu_id = trial.number % self.n_gpus
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            # Log failure but don't crash the study
            print(f"  Trial {trial.number} FAILED (exit {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            raise _optuna.TrialPruned()  # type: ignore[union-attr]

        # Extract best val loss
        best_loss = _extract_best_val_loss(trial_dir)
        if best_loss == float("inf"):
            raise _optuna.TrialPruned()  # type: ignore[union-attr]

        return best_loss


class InProcessRoSAObjective:
    """Optuna objective that runs RoSA training in-process.

    Loads backbone weights and parses PGN data once at construction time.
    Each trial gets a fresh model wrapping the shared backbone state, with
    epoch-level Optuna pruning via ``trial.report()`` / ``trial.should_prune()``.

    Usage::

        objective = InProcessRoSAObjective("rosa", checkpoint, pgn, device="cuda")
        study = create_study("rosa")
        study.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        adapter_type: str,
        checkpoint: str,
        pgn: str,
        device: str = "cuda",
        output_base: str = "sweeps",
        epochs: int = 50,
        max_games: int = 12_000,
        val_games: int = 2_000,
        min_ply: int = 10,
        val_batch_size: int = 64,
        no_amp: bool = False,
        no_compile: bool = False,
        sdpa_math: bool = False,
        max_grad_norm: float = 1.0,
        elo_min: int | None = None,
        elo_max: int | None = None,
    ):
        _supported = ("rosa", "retro-sparse", "retro-bottleneck", "rosa-ratio")
        if adapter_type not in _supported:
            raise ValueError(f"InProcessRoSAObjective does not support adapter_type={adapter_type!r}")
        self.adapter_type = adapter_type
        self.checkpoint = checkpoint
        self.device = device
        self.output_base = Path(output_base) / adapter_type
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.val_batch_size = val_batch_size

        # Lazy imports -- only pay the cost when actually used
        import torch
        from pawn.config import CLMConfig
        from pawn.model import PAWNCLM
        from pawn.checkpoint import load_backbone_weights
        from pawn.gpu import configure_gpu
        from pawn.lichess_data import (
            prepare_lichess_dataset,
            LegalMaskBuilder,
            LichessDataset,
            compute_legal_indices,
        )
        import numpy as np
        from torch.utils.data import DataLoader

        # --- Load backbone weights once (kept on CPU) ---
        state_dict, model_config = load_backbone_weights(checkpoint, "cpu")
        self._cfg = CLMConfig(**model_config) if model_config else CLMConfig()
        self._backbone_state = state_dict

        # --- Parse PGN once ---
        data = prepare_lichess_dataset(
            pgn, max_ply=255, max_games=max_games, min_ply=min_ply,
            elo_min=elo_min, elo_max=elo_max,
        )
        n_total = data["n_games"]
        n_val = min(val_games, n_total // 5)
        self._n_train = n_total - n_val
        self._train_ds = LichessDataset(data, start=0, end=self._n_train).share_memory()
        self._val_ds = LichessDataset(data, start=self._n_train, end=n_total)

        # --- GPU config ---
        self._gpu_cfg = configure_gpu(
            device, no_compile=no_compile, no_amp=no_amp, sdpa_math=sdpa_math,
        )
        self._gpu_cfg_no_compile = configure_gpu(
            device, no_compile=True, no_amp=no_amp, sdpa_math=sdpa_math,
        )
        self._use_amp = self._gpu_cfg["use_amp"]

        # --- Precompute val legal indices (fixed batch size) ---
        vocab_size = self._cfg.vocab_size
        self._mask_builder = LegalMaskBuilder(
            val_batch_size, max_ply=255, vocab_size=vocab_size, device=device,
        )
        val_loader = DataLoader(
            self._val_ds, batch_size=val_batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        self._val_legal_indices = []
        for batch in val_loader:
            move_ids = batch["move_ids"]
            if isinstance(move_ids, torch.Tensor):
                move_ids = move_ids.numpy()
            game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
            indices = compute_legal_indices(
                move_ids, game_lengths, self._mask_builder.T, vocab_size,
            )
            self._val_legal_indices.append(torch.from_numpy(indices).pin_memory())

        print(f"InProcessRoSAObjective ready: {self._n_train} train / "
              f"{n_val} val games, {len(self._val_legal_indices)} val batches")

    def _make_backbone(self):
        """Create a fresh backbone from the cached state dict."""
        import torch
        from pawn.model import PAWNCLM
        model = PAWNCLM(self._cfg).to(self.device)
        model.load_state_dict(self._backbone_state)
        model.eval()
        return model

    def __call__(self, trial: "optuna.trial.BaseTrial") -> float:
        import gc
        import math
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from pawn.adapters.rosa import RoSACLM, RetroBottleneckCLM, generate_gradient_masks
        from pawn.adapters.sparse import SparseCLM, SparseLinear
        from pawn.adapters.lora import ATTN_PRESETS, _FFN_TARGETS
        from pawn.gpu import apply_gpu_config
        from pawn.lichess_data import LegalMaskCollate

        assert _optuna is not None

        suggest_fn = SUGGEST_FNS[self.adapter_type]
        params = suggest_fn(trial)
        mode = params["mode"]
        density = params["density"]
        lr = params["lr"]
        lora_rank = params["lora_rank"]
        lora_targets = params["lora_targets"]
        batch_size = params["batch_size"]
        warmup_steps = params["warmup_steps"]
        mask_samples = params["mask_samples"]
        grad_alpha = params["grad_alpha"]
        weight_decay = params["weight_decay"]
        warmup_frac = params["warmup_frac"]
        patience = params["patience"]
        bottleneck_dim = params.get("bottleneck_dim", 8)

        vocab_size = self._cfg.vocab_size
        use_amp = self._use_amp
        device = self.device

        # --- Data loaders (batch_size varies per trial) ---
        collate = LegalMaskCollate(seq_len=256, vocab_size=vocab_size)
        train_loader = DataLoader(
            self._train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate,
        )
        val_loader = DataLoader(
            self._val_ds, batch_size=self.val_batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        # Ensure mask builder capacity matches train batch size
        mask_builder = self._mask_builder
        if batch_size > mask_builder._mask_gpu.shape[0]:
            from pawn.lichess_data import LegalMaskBuilder
            mask_builder = LegalMaskBuilder(
                batch_size, max_ply=255, vocab_size=vocab_size, device=device,
            )

        # ---------------------------------------------------------------
        # Phase 1: LoRA warm-up
        # ---------------------------------------------------------------
        backbone = self._make_backbone()
        warmup_model = RoSACLM(
            backbone, rank=lora_rank, alpha=None,
            attn_targets=lora_targets, adapt_ffn=False,
            lora_enabled=True, sparse_enabled=False,
        ).to(device)

        lora_params = warmup_model.lora_parameters()
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

        warmup_model.train()
        step = 0
        while step < warmup_steps:
            for batch in train_loader:
                if step >= warmup_steps:
                    break
                ids = batch["input_ids"].to(device, non_blocking=True)
                tgt = batch["targets"].to(device, non_blocking=True)
                msk = batch["loss_mask"].to(device, non_blocking=True)
                legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    hidden = warmup_model.forward_hidden(ids, msk)
                    valid_logits = warmup_model.project_head(hidden[msk])
                valid_logits = valid_logits.float()
                valid_logits.masked_fill_(~legal_mask[msk], float("-inf"))
                valid_targets = tgt[msk]
                if valid_targets.shape[0] == 0:
                    continue

                loss = F.cross_entropy(valid_logits, valid_targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, self.max_grad_norm)
                optimizer.step()
                step += 1

        del optimizer

        # ---------------------------------------------------------------
        # Phase 2: Mask generation
        # ---------------------------------------------------------------
        masks = generate_gradient_masks(
            warmup_model, train_loader, mask_builder,
            density=density, alpha=grad_alpha,
            device=device, use_amp=use_amp, max_batches=mask_samples,
        )

        # ---------------------------------------------------------------
        # Phase 3: Set up model per mode
        # ---------------------------------------------------------------
        attn_target_tuple = ATTN_PRESETS[lora_targets]

        if mode == "rosa":
            warmup_model.set_masks(masks)
            warmup_model.reinit_lora()
            model = warmup_model
            adapter_params = model.adapter_parameters()
        else:
            # Retrospective modes: discard warm-up, reload backbone
            del warmup_model
            gc.collect()
            torch.cuda.empty_cache()

            backbone = self._make_backbone()
            sparse_model = SparseCLM(
                backbone, density=density, attn_targets=attn_target_tuple,
            )
            # Overwrite random masks with gradient-derived masks
            for layer_idx in range(len(backbone.layers)):
                block = backbone.get_block(layer_idx)
                for proj_name in attn_target_tuple:
                    module = getattr(block.attn, proj_name, None)
                    if isinstance(module, SparseLinear):
                        key = f"layer{layer_idx}.{proj_name}"
                        if key in masks:
                            module.mask.copy_(masks[key])

            if mode == "retro-sparse":
                model = sparse_model
                adapter_params = model.sparse_parameters()
            else:  # retro-bottleneck
                model = RetroBottleneckCLM(
                    sparse_model.backbone, bottleneck_dim=bottleneck_dim,
                ).to(device)
                adapter_params = model.adapter_parameters()

        # Compile for Phase 3
        from pawn import model as model_module
        model.forward_hidden = apply_gpu_config(
            self._gpu_cfg, model_module, model.forward_hidden,
        )

        # ---------------------------------------------------------------
        # Phase 3: Training with epoch-level pruning
        # ---------------------------------------------------------------
        optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=weight_decay)
        total_steps = self.epochs * len(train_loader)
        sched_warmup = int(warmup_frac * total_steps)

        def lr_lambda(s):
            if s < sched_warmup:
                return s / max(sched_warmup, 1)
            progress = (s - sched_warmup) / max(total_steps - sched_warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler() if use_amp else None

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            for batch in train_loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                tgt = batch["targets"].to(device, non_blocking=True)
                msk = batch["loss_mask"].to(device, non_blocking=True)
                legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    hidden = model.forward_hidden(ids, msk)
                    valid_logits = model.project_head(hidden[msk])
                valid_logits = valid_logits.float()
                valid_logits.masked_fill_(~legal_mask[msk], float("-inf"))
                valid_targets = tgt[msk]
                if valid_targets.shape[0] == 0:
                    continue

                loss = F.cross_entropy(valid_logits, valid_targets)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(adapter_params, self.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter_params, self.max_grad_norm)
                    optimizer.step()
                scheduler.step()

            # --- Validation ---
            model.eval()
            total_loss = 0.0
            total_pos = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    tgt = batch["targets"].to(device, non_blocking=True)
                    msk = batch["loss_mask"].to(device, non_blocking=True)
                    lm = self._mask_builder.scatter(
                        self._val_legal_indices[i], ids.shape[0],
                    )
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        hidden = model.forward_hidden(ids, msk)
                        vl = model.project_head(hidden[msk])
                    vl = vl.float()
                    vl.masked_fill_(~lm[msk], float("-inf"))
                    vt = tgt[msk]
                    n = vt.shape[0]
                    if n == 0:
                        continue
                    total_loss += F.cross_entropy(vl, vt).item() * n
                    total_pos += n

            val_loss = total_loss / max(total_pos, 1)

            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                self._cleanup(model, optimizer, scaler)
                raise _optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self._cleanup(model, optimizer, scaler)
        return best_val_loss

    @staticmethod
    def _cleanup(model, optimizer, scaler):
        """Free GPU memory between trials."""
        import gc
        import torch
        del model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Study creation helpers
# ---------------------------------------------------------------------------

def create_study(
    name: str,
    storage: str | None = None,
    direction: str = "minimize",
    pruner: str = "hyperband",
) -> "optuna.Study":
    """Create an Optuna study with sensible defaults.

    Args:
        name: Study name (also used as DB table name).
        storage: SQLite path (e.g. "sqlite:///sweeps/study.db"). None = in-memory.
        direction: "minimize" for val_loss, "maximize" for accuracy.
        pruner: "hyperband", "median", or "none".
    """
    if _optuna is None:
        raise ImportError("optuna is required for sweeps: pip install optuna")

    assert _optuna is not None
    if pruner == "hyperband":
        pruner_obj = _optuna.pruners.HyperbandPruner()
    elif pruner == "median":
        pruner_obj = _optuna.pruners.MedianPruner()
    else:
        pruner_obj = _optuna.pruners.NopPruner()

    return _optuna.create_study(
        study_name=name,
        storage=storage,
        direction=direction,
        pruner=pruner_obj,
        load_if_exists=True,
    )
