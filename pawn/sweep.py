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
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
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
    "adapter_strategy_for",
    "params_to_config_json",
]


# ---------------------------------------------------------------------------
# Per-strategy suggesters — match the v1 search-space contract
# ---------------------------------------------------------------------------


# RoSA retro-bottleneck extra-(Linear+GELU)-stage depth choices — matches
# the v1 ``BOTTLENECK_N_HIDDEN_CHOICES`` search-space constant.
BOTTLENECK_N_HIDDEN_CHOICES: tuple[int, ...] = (0, 1, 2)


def suggest_lora(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {
        "lora_rank": trial.suggest_int("lora_rank", 1, 16),
        "lora_targets": trial.suggest_categorical(
            "lora_targets", ["qkvo", "qv", "qkv"]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_film(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {
        "use_output_film": trial.suggest_categorical(
            "use_output_film", [True, False]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_bottleneck(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {
        "bottleneck_dim": trial.suggest_int("bottleneck_dim", 2, 32),
        "bottleneck_n_hidden": trial.suggest_int("bottleneck_n_hidden", 0, 2),
        "no_adapt_attn": trial.suggest_categorical("no_adapt_attn", [True, False]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_hybrid(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        "use_output_film": trial.suggest_categorical(
            "use_output_film", [True, False]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_sparse(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "sparse_targets": trial.suggest_categorical(
            "sparse_targets", ["qkvo", "qv", "qkv"]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_rosa(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {
        "rosa_mode": "rosa",
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        # `lora_targets` is part of the v1 RoSA search space
        # (``_suggest_rosa_common``): RoSA's LoRA warmup can adapt the
        # ``qv`` / ``qkv`` projection subsets, not just the default
        # ``qkvo``, so the sweep must be free to pick the preset.
        "lora_targets": trial.suggest_categorical(
            "lora_targets", ["qkvo", "qv", "qkv"]
        ),
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "rosa_warmup_steps": trial.suggest_int("rosa_warmup_steps", 32, 512),
        "mask_samples": trial.suggest_int("mask_samples", 8, 64),
        "grad_alpha": trial.suggest_categorical("grad_alpha", [1, 2]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


def suggest_rosa_retro_sparse(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {**suggest_rosa(trial), "rosa_mode": "retro-sparse"}


def suggest_rosa_retro_bottleneck(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    # The retro-bottleneck mode instantiates a Houlsby bottleneck in
    # Phase 3, so (per v1 ``suggest_retro_bottleneck``) the sweep extends
    # the shared RoSA space with the bottleneck width *and* the extra-stage
    # depth — the prior v2 code locked both to their RoSAConfig defaults.
    return {
        **suggest_rosa(trial),
        "rosa_mode": "retro-bottleneck",
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", [4, 8, 16]),
        "bottleneck_n_hidden": trial.suggest_categorical(
            "bottleneck_n_hidden", list(BOTTLENECK_N_HIDDEN_CHOICES)
        ),
    }


def suggest_rosa_ratio(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    """Sweep over the bottleneck-vs-sparse parameter split (RoSA-specific
    v1 sweep).

    H9: ``rosa-ratio`` is a *sweep-only* strategy key — it is not a valid
    adapter ``--strategy`` (``STRATEGIES`` has no ``rosa-ratio`` entry). The
    objective maps it onto the real ``rosa`` strategy via
    :func:`adapter_strategy_for`; the sub-mode this suggester sweeps is
    ``retro-bottleneck`` (the only RoSA mode that actually instantiates a
    Houlsby bottleneck), so the bottleneck-vs-sparse split is a meaningful
    knob here. The historical ``bottleneck_ratio`` ∈ (0, 1) is translated
    into the *consumed* ``bottleneck_dim`` AdapterConfig field — the prior
    code emitted a raw ``bottleneck_ratio`` key that no config or adapter
    consumed, so every trial was rejected by ``extra="forbid"``.

    ``bottleneck_dim`` is derived as ``round(ratio · _ROSA_RATIO_DIM_SPAN)``
    (clamped to ≥1) so the swept ratio maps onto a small positive integer
    bottleneck width.
    """
    ratio = trial.suggest_float("bottleneck_ratio", 0.1, 0.9)
    bottleneck_dim = max(1, round(ratio * _ROSA_RATIO_DIM_SPAN))
    return {
        "rosa_mode": "retro-bottleneck",
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "bottleneck_dim": bottleneck_dim,
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


# The integer bottleneck width that a swept ``bottleneck_ratio`` of 1.0
# would map to (``bottleneck_dim = round(ratio · span)``). Picked so the
# (0.1, 0.9) ratio range spans a small-but-non-degenerate set of widths.
_ROSA_RATIO_DIM_SPAN = 32


def suggest_unfreeze(
    trial: optuna.Trial, *, n_layers: int = 10
) -> dict[str, Any]:
    """Sample an unfreeze-layers spec.

    ``n_layers`` is the total depth of the targeted backbone. Defaults
    to 10 (production SUPERNET), but the sweep driver should pass the
    actual variant depth so the suggester doesn't propose layer indices
    that ``init_unfreeze_adapter`` will reject — e.g. for the tiny
    supernet (n_layers=4) suggesting "7,8,9" kills every trial.
    """
    max_unfrozen = min(4, n_layers)
    n = trial.suggest_int("n_layers_unfrozen", 1, max_unfrozen)
    # Always unfreeze the last N layers — a reasonable v1-style default.
    layers = ",".join(str(i) for i in range(n_layers - n, n_layers))
    return {
        "unfreeze_layers": layers,
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
    }


def suggest_specialized_clm(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
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


STRATEGY_SUGGESTERS: dict[str, Callable[..., dict[str, Any]]] = {
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


# Sweep-only strategy keys that are NOT valid adapter `--strategy` values —
# they map onto a real `STRATEGIES` strategy whose `rosa_mode` the suggester
# sets explicitly (H9). `rosa-ratio` sweeps the bottleneck-vs-sparse split of
# the `rosa` strategy in its `retro-bottleneck` sub-mode.
_SWEEP_STRATEGY_ALIASES: dict[str, str] = {
    "rosa-ratio": "rosa",
}


def adapter_strategy_for(sweep_strategy: str) -> str:
    """Map a `STRATEGY_SUGGESTERS` key onto the adapter `--strategy` it runs.

    Most sweep keys are identical to the adapter strategy; sweep-only
    aliases (currently just `rosa-ratio`) resolve to the real strategy the
    `train_jax_adapter` argparse + pydantic accept (H9)."""
    return _SWEEP_STRATEGY_ALIASES.get(sweep_strategy, sweep_strategy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def params_to_config_json(strategy: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON `--config` body for one suggested trial (H8).

    The previous design rendered the suggested-params dict into kebab-cased
    CLI argv (`{"lora_rank": 4}` → `["--lora-rank", "4"]`). Many suggested
    keys (`bottleneck_n_hidden`, `sparse_targets`, `rosa_warmup_steps`,
    `mask_samples`, `grad_alpha`, …) have **no** registered argparse flag in
    `scripts/train_jax_adapter.py`, so argparse exited 2 and every trial was
    pruned. Routing the params through a temp-JSON `--config` instead lets
    them round-trip through `AdapterConfig` (pydantic) directly — the
    `--config` path in `_build_config` `json.loads`es this body and merges
    CLI flags on top.

    The returned dict carries the suggested params verbatim (every key is an
    `AdapterConfig` field — `extra="forbid"` would reject a stray one) plus
    `run_type="adapter"` and the resolved adapter `strategy` (sweep-only
    aliases like `rosa-ratio` are mapped to their real strategy via
    :func:`adapter_strategy_for`). Boolean and int values are preserved as
    native JSON types rather than stringified, so the config validates with
    the right field types.
    """
    body: dict[str, Any] = dict(params)
    body["run_type"] = "adapter"
    body["strategy"] = adapter_strategy_for(strategy)
    return body


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
            # `or` would silently fall back to `loss` if `val_loss` is
            # 0.0 (legitimate) or NaN-sanitised null. Use explicit
            # presence checks so a trial with `val_loss=0.0` isn't
            # mis-keyed on the training loss.
            if "val_loss" in rec:
                v = rec["val_loss"]
            elif "loss" in rec:
                v = rec["loss"]
            else:
                v = None
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
    # Use the interpreter running the sweep (which has the `pawn` package +
    # the ROCm JAX plugin resolved) rather than a bare `python` off PATH,
    # which may resolve to a different env (§8.4-low). `field(...)` because
    # `sys.executable` is evaluated at class-definition import time.
    python: str = field(default_factory=lambda: sys.executable)
    timeout: float | None = None
    # Backbone depth hint forwarded to suggesters that need it
    # (currently only `suggest_unfreeze`). Defaults to the production
    # supernet depth; sweep CLI should override when targeting tiny.
    n_layers: int = 10

    def __call__(self, trial: optuna.Trial) -> float:
        suggester = STRATEGY_SUGGESTERS.get(self.strategy)
        if suggester is None:
            raise ValueError(f"no suggester for strategy {self.strategy!r}")
        params = suggester(trial, n_layers=self.n_layers)
        trial_logs = self.logs_dir / f"trial_{trial.number:05d}"
        trial_logs.mkdir(parents=True, exist_ok=True)
        # H8: route the suggested params through a temp-JSON `--config` so
        # they round-trip through `AdapterConfig` (pydantic) rather than
        # through kebab-cased CLI flags the adapter argparse never
        # registered (which exited 2 → every trial pruned). The resolved
        # adapter `--strategy` (sweep-only aliases mapped) is set on the
        # config body *and* passed on the CLI so it wins over the config.
        config_body = params_to_config_json(self.strategy, params)
        config_path = trial_logs / "sweep_trial_config.json"
        config_path.write_text(json.dumps(config_body), encoding="utf-8")
        cmd = [
            self.python, self.script,
            "--config", str(config_path),
            "--strategy", adapter_strategy_for(self.strategy),
            "--logs-dir", str(trial_logs),
        ] + self.base_args
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
