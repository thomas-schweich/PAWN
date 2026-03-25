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

def suggest_common(trial: "optuna.Trial") -> dict:
    """Hyperparameters shared across all adapter types."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.15),
        "patience": trial.suggest_int("patience", 5, 20),
    }


def suggest_lora(trial: "optuna.Trial") -> dict:
    """LoRA-specific hyperparameters."""
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_categorical("lora_rank", [2, 4, 8, 16, 32])
    params["lora_targets"] = trial.suggest_categorical("lora_targets", ["qkvo", "qv", "qkv"])
    params["lora_ffn"] = trial.suggest_categorical("lora_ffn", [True, False])
    return params


def suggest_bottleneck(trial: "optuna.Trial") -> dict:
    """Bottleneck adapter hyperparameters."""
    params = suggest_common(trial)
    params["bottleneck_dim"] = trial.suggest_categorical("bottleneck_dim", [4, 8, 16, 32, 64, 128])
    params["no_adapt_attn"] = trial.suggest_categorical("no_adapt_attn", [True, False])
    params["no_adapt_ffn"] = trial.suggest_categorical("no_adapt_ffn", [True, False])
    return params


def suggest_film(trial: "optuna.Trial") -> dict:
    """FiLM hyperparameters."""
    params = suggest_common(trial)
    params["no_output_film"] = trial.suggest_categorical("no_output_film", [True, False])
    return params


def suggest_sparse(trial: "optuna.Trial") -> dict:
    """Sparse adapter hyperparameters."""
    params = suggest_common(trial)
    params["density"] = trial.suggest_float("density", 0.001, 0.1, log=True)
    params["sparse_targets"] = trial.suggest_categorical("sparse_targets", ["qkvo", "qv", "qkv"])
    params["sparse_ffn"] = trial.suggest_categorical("sparse_ffn", [True, False])
    return params


def suggest_hybrid(trial: "optuna.Trial") -> dict:
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


def suggest_tiny(trial: "optuna.Trial") -> dict:
    """Tiny standalone model hyperparameters."""
    params = suggest_common(trial)
    params["d_model"] = trial.suggest_categorical("d_model", [32, 64, 84, 128])
    params["n_layers"] = trial.suggest_int("n_layers", 1, 4)
    params["n_heads"] = trial.suggest_categorical("n_heads", [1, 2, 4])
    return params


def suggest_pretrain(trial: "optuna.Trial") -> dict:
    """Pretraining hyperparameters (fixed architecture, tune training)."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_steps": trial.suggest_int("warmup_steps", 500, 5000, step=500),
        "total_steps": trial.suggest_categorical("total_steps", [50000, 100000]),
    }


def suggest_architecture(trial: "optuna.Trial") -> dict:
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


SUGGEST_FNS = {
    "lora": suggest_lora,
    "bottleneck": suggest_bottleneck,
    "film": suggest_film,
    "sparse": suggest_sparse,
    "hybrid": suggest_hybrid,
    "tiny": suggest_tiny,
    "architecture": suggest_architecture,
    "pretrain": suggest_pretrain,
}

ADAPTER_SCRIPTS = {
    "lora": "scripts/train_lora.py",
    "bottleneck": "scripts/train_bottleneck.py",
    "film": "scripts/train_film.py",
    "sparse": "scripts/train_sparse.py",
    "hybrid": "scripts/train_hybrid.py",
    "tiny": "scripts/train_tiny.py",
    "pretrain": "scripts/train.py",
    "architecture": "scripts/train.py",
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

    def __call__(self, trial: "optuna.Trial") -> float:
        suggest_fn = SUGGEST_FNS[self.adapter_type]
        params = suggest_fn(trial)

        trial_dir = self.output_base / f"trial_{trial.number:04d}"

        # Build command
        cmd = [sys.executable, self.script]

        # Fixed args — architecture and pretrain sweeps use train.py directly
        if self.adapter_type not in ("pretrain", "architecture"):
            cmd.extend(["--checkpoint", self.checkpoint])
            cmd.extend(["--pgn", self.pgn])
        cmd.extend(["--device", self.device])

        # output-dir for adapters, log-dir for pretraining
        if self.adapter_type in ("pretrain", "architecture"):
            cmd.extend(["--log-dir", str(trial_dir)])
            cmd.extend(["--local-checkpoints"])
        else:
            cmd.extend(["--output-dir", str(trial_dir)])
            # Only pass --epochs for adapter scripts (train.py uses --total-steps)
            if "epochs" not in params:
                cmd.extend(["--epochs", str(self.epochs)])

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


class InProcessAdapterObjective:
    """Optuna objective that runs training in-process for tighter pruning integration.

    Imports the training function directly and hooks into the validation loop
    for epoch-level Optuna pruning.
    """

    def __init__(
        self,
        adapter_type: str,
        checkpoint: str,
        pgn: str,
        device: str = "cuda",
        output_base: str = "sweeps",
        epochs: int = 50,
    ):
        self.adapter_type = adapter_type
        self.checkpoint = checkpoint
        self.pgn = pgn
        self.device = device
        self.output_base = Path(output_base) / adapter_type
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs

    def __call__(self, trial: "optuna.Trial") -> float:
        suggest_fn = SUGGEST_FNS[self.adapter_type]
        params = suggest_fn(trial)

        trial_dir = self.output_base / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Build args namespace (same as argparse would produce)
        args = argparse.Namespace(
            checkpoint=self.checkpoint,
            pgn=self.pgn,
            device=self.device,
            output_dir=str(trial_dir),
            log_dir=str(self.output_base),
            epochs=self.epochs,
            val_every=1,
            no_amp=False,
            max_games=12000,
            val_games=2000,
            min_ply=10,
            max_grad_norm=params.get("max_grad_norm", 1.0),
            **params,
        )

        # Import the training function
        # The actual in-process integration requires modifying each training
        # script to accept an optuna.Trial and call trial.report()/should_prune().
        # For now, fall back to subprocess.
        raise NotImplementedError(
            "In-process objectives require training scripts to accept a Trial parameter. "
            "Use AdapterObjective (subprocess) or add trial hooks to the training script."
        )


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
