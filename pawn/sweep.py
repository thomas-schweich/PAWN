"""Optuna hyperparameter sweep driver for v2 adapter training.

`AdapterObjective` runs `scripts/train_jax_adapter.py` as a subprocess
per trial and parses `metrics.jsonl` for the best `val_loss`.
`InProcessRoSAObjective` is the v1 in-process variant that skips
per-trial JAX startup for big RoSA sweeps (kept for parity with v1
sweep tooling). Per-strategy `suggest_*` functions match the v1
search-space contract.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

__all__ = [
    "AdapterObjective",
    "InProcessRoSAObjective",
    "suggest_lora",
    "suggest_film",
    "suggest_bottleneck",
    "suggest_hybrid",
    "suggest_sparse",
    "suggest_rosa",
    "suggest_rosa_retro_sparse",
    "suggest_rosa_retro_bottleneck",
    "suggest_rosa_ratio",
    "suggest_unfreeze",
    "suggest_specialized_clm",
    "STRATEGY_SUGGESTERS",
]


# ---------------------------------------------------------------------------
# Per-strategy suggesters — match the v1 search-space contract
# ---------------------------------------------------------------------------


def suggest_lora(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lora_rank": trial.suggest_int("lora_rank", 1, 16),
        "lora_targets": trial.suggest_categorical(
            "lora_targets", ["qkvo", "qv", "qkv"]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_film(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "use_output_film": trial.suggest_categorical(
            "use_output_film", [True, False]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_bottleneck(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "bottleneck_dim": trial.suggest_int("bottleneck_dim", 2, 32),
        "bottleneck_n_hidden": trial.suggest_int("bottleneck_n_hidden", 0, 2),
        "no_adapt_attn": trial.suggest_categorical("no_adapt_attn", [True, False]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_hybrid(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        "use_output_film": trial.suggest_categorical(
            "use_output_film", [True, False]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_sparse(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "sparse_targets": trial.suggest_categorical(
            "sparse_targets", ["qkvo", "qv", "qkv"]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_rosa(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "rosa_mode": "rosa",
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "rosa_warmup_steps": trial.suggest_int("rosa_warmup_steps", 32, 512),
        "mask_samples": trial.suggest_int("mask_samples", 8, 64),
        "grad_alpha": trial.suggest_categorical("grad_alpha", [1, 2]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_rosa_retro_sparse(trial: optuna.Trial) -> dict[str, Any]:
    return {**suggest_rosa(trial), "rosa_mode": "retro-sparse"}


def suggest_rosa_retro_bottleneck(trial: optuna.Trial) -> dict[str, Any]:
    return {**suggest_rosa(trial), "rosa_mode": "retro-bottleneck"}


def suggest_rosa_ratio(trial: optuna.Trial) -> dict[str, Any]:
    """Sweep over the bottleneck-vs-sparse parameter split (RoSA-specific
    v1 sweep)."""
    return {
        "rosa_mode": "rosa",
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "bottleneck_ratio": trial.suggest_float("bottleneck_ratio", 0.1, 0.9),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_unfreeze(trial: optuna.Trial) -> dict[str, Any]:
    n = trial.suggest_int("n_layers_unfrozen", 1, 4)
    # Always unfreeze the last N layers — a reasonable v1-style default.
    layers = ",".join(str(i) for i in range(10 - n, 10))
    return {
        "unfreeze_layers": layers,
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
    }


def suggest_specialized_clm(trial: optuna.Trial) -> dict[str, Any]:
    d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128, 192])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    # Round d_model up to a multiple of n_heads if needed.
    if d_model % n_heads != 0:
        d_model = ((d_model + n_heads - 1) // n_heads) * n_heads
    return {
        "d_model": d_model,
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "n_heads": n_heads,
        "d_ff": 4 * d_model,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }


STRATEGY_SUGGESTERS: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "lora": suggest_lora,
    "film": suggest_film,
    "bottleneck": suggest_bottleneck,
    "hybrid": suggest_hybrid,
    "sparse": suggest_sparse,
    "rosa": suggest_rosa,
    "rosa-retro-sparse": suggest_rosa_retro_sparse,
    "rosa-retro-bottleneck": suggest_rosa_retro_bottleneck,
    "rosa-ratio": suggest_rosa_ratio,
    "unfreeze": suggest_unfreeze,
    "specialized_clm": suggest_specialized_clm,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params_to_argv(params: dict[str, Any]) -> list[str]:
    """Convert a suggested-params dict into CLI argv (kebab-case flags).

    `{"lora_rank": 4}` becomes `["--lora-rank", "4"]`. Boolean True
    becomes a flag with no value (e.g. `["--use-output-film"]`); False
    is omitted (the v1 contract).
    """
    args: list[str] = []
    for k, v in params.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.extend([flag, str(v)])
    return args


def _read_best_val_loss(logs_dir: Path) -> float:
    """Walk `logs_dir/**/metrics.jsonl`, find the smallest `val_loss`
    on a `type=val` record."""
    best = float("inf")
    for jsonl in logs_dir.rglob("metrics.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "val":
                continue
            v = rec.get("val_loss") or rec.get("loss")
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if vf < best:
                best = vf
    return best


# ---------------------------------------------------------------------------
# Objective: subprocess-per-trial
# ---------------------------------------------------------------------------


@dataclass
class AdapterObjective:
    """Optuna objective that runs `scripts/train_jax_adapter.py` per trial.

    The subprocess writes `metrics.jsonl` to a per-trial log dir; this
    objective parses the file for the best `val_loss` and returns it
    (Optuna minimises by default).
    """

    strategy: str
    base_args: list[str]  # supernet/variant/total-steps/etc.
    logs_dir: Path
    script: str = "scripts/train_jax_adapter.py"
    python: str = "python"
    timeout: float | None = None

    def __call__(self, trial: optuna.Trial) -> float:
        suggester = STRATEGY_SUGGESTERS.get(self.strategy)
        if suggester is None:
            raise ValueError(f"no suggester for strategy {self.strategy!r}")
        params = suggester(trial)
        trial_logs = self.logs_dir / f"trial_{trial.number:05d}"
        cmd = [
            self.python, self.script,
            "--strategy", self.strategy,
            "--logs-dir", str(trial_logs),
        ] + self.base_args + _params_to_argv(params)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout
        )
        if result.returncode != 0:
            raise optuna.TrialPruned(
                f"trial {trial.number} failed: {result.stderr[-500:]}"
            )
        best = _read_best_val_loss(trial_logs)
        if best == float("inf"):
            raise optuna.TrialPruned(
                f"trial {trial.number} produced no val_loss"
            )
        return best


# ---------------------------------------------------------------------------
# Objective: in-process (skips per-trial JAX startup for big RoSA sweeps)
# ---------------------------------------------------------------------------


@dataclass
class InProcessRoSAObjective:
    """In-process RoSA objective — caller provides the trainer entry
    point as a Python callable that takes a params dict and returns
    val_loss directly. Skips the JAX startup overhead of subprocess
    per trial."""

    train_fn: Callable[[dict[str, Any]], float]
    strategy: str = "rosa"

    def __call__(self, trial: optuna.Trial) -> float:
        suggester = STRATEGY_SUGGESTERS[self.strategy]
        params = suggester(trial)
        return self.train_fn(params)
