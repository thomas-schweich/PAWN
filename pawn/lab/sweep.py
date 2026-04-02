"""Optuna sweep helpers: search spaces, study management, result reporting."""

from __future__ import annotations

import logging
from typing import Any

from pawn.lab.state import Trial

log = logging.getLogger("pawn.lab")


def builtin_distributions(strategy: str) -> dict[str, Any]:
    """Return Optuna distributions for a PAWN adapter strategy."""
    import optuna.distributions as d
    Cat = d.CategoricalDistribution
    Float = d.FloatDistribution
    Int = d.IntDistribution

    common: dict[str, Any] = {
        "lr": Float(1e-5, 1e-2, log=True),
        "batch_size": Cat([32, 64, 128, 256]),
        "weight_decay": Float(0.0, 0.1),
        "warmup_frac": Float(0.0, 0.15),
    }
    spaces: dict[str, dict[str, Any]] = {
        "lora": {**common,
            "lora_rank": Cat([2, 4, 8, 16, 32]),
            "lora_targets": Cat(["qkvo", "qv", "qkv"]),
            "lora_ffn": Cat([True, False]),
        },
        "bottleneck": {**common,
            "bottleneck_dim": Cat([4, 8, 16, 32, 64, 128, 256]),
            "no_adapt_attn": Cat([True, False]),
            "no_adapt_ffn": Cat([True, False]),
        },
        "film": {**common,
            "use_output_film": Cat([True, False]),
        },
        "sparse": {**common,
            "density": Float(0.001, 0.1, log=True),
            "sparse_targets": Cat(["qkvo", "qv", "qkv"]),
            "sparse_ffn": Cat([True, False]),
        },
        "hybrid": {
            "batch_size": Cat([32, 64, 128, 256]),
            "weight_decay": Float(0.0, 0.1),
            "warmup_frac": Float(0.0, 0.15),
            "lr": Float(1e-5, 1e-2, log=True),
            "film_lr": Float(1e-5, 1e-2, log=True),
            "lora_rank": Cat([2, 4, 8, 16]),
            "lora_targets": Cat(["qkvo", "qv", "qkv"]),
        },
        "specialized_clm": {**common,
            "d_model": Cat([32, 48, 84, 128, 192]),
            "n_layers": Int(1, 4),
            "n_heads": Cat([1, 2, 4, 8]),
        },
        "unfreeze": {**common,
            "unfreeze_layers": Cat(["6,7", "5,6,7", "4,5,6,7"]),
        },
    }
    rosa_common: dict[str, Any] = {**common,
        "density": Float(0.001, 0.1, log=True),
        "lora_rank": Cat([2, 4, 8, 16]),
        "lora_targets": Cat(["qkvo", "qv", "qkv"]),
        "rosa_warmup_steps": Int(32, 256, step=32),
        "mask_samples": Cat([16, 32, 64]),
        "grad_alpha": Cat([1, 2]),
    }
    spaces["rosa"] = rosa_common
    spaces["retro-sparse"] = rosa_common
    spaces["retro-bottleneck"] = {**rosa_common, "bottleneck_dim": Cat([4, 8, 16])}

    if strategy not in spaces:
        raise ValueError(f"No built-in search space for strategy '{strategy}'. "
                         f"Available: {sorted(spaces)}")
    return spaces[strategy]


def parse_distribution(spec: dict[str, Any]) -> Any:
    """Convert a JSON distribution spec to an Optuna distribution object."""
    import optuna.distributions as d

    t = spec["type"]
    if t == "float":
        return d.FloatDistribution(spec["low"], spec["high"], log=spec.get("log", False))
    elif t == "int":
        return d.IntDistribution(spec["low"], spec["high"], step=spec.get("step", 1))
    elif t == "categorical":
        return d.CategoricalDistribution(spec["choices"])
    raise ValueError(f"Unknown distribution type: {t}")


def get_or_create_study(
    workspace: str,
    study_name: str,
    directions: list[str],
) -> Any:
    """Create or load an Optuna study backed by SQLite."""
    import optuna
    storage = f"sqlite:///{workspace}/optuna-storage/lab.db"
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=directions,
        load_if_exists=True,
    )


def pick_strategy(
    strategies: list[str],
    trials: dict[int, Trial],
    sweep_launched: int,
) -> str:
    """Pick next strategy: round-robin biased toward least-explored."""
    if len(strategies) == 1:
        return strategies[0]
    counts: dict[str, int] = {s: 0 for s in strategies}
    for t in trials.values():
        if t.strategy in counts:
            counts[t.strategy] += 1
    min_count = min(counts.values())
    candidates = [s for s, c in counts.items() if c == min_count]
    return candidates[sweep_launched % len(candidates)]


def tell_optuna(
    study: Any,
    trial: Trial,
    directions: list[str],
) -> None:
    """Report trial results to Optuna."""
    if trial.optuna_number is None:
        return
    values: list[float] = []
    if trial.best_val_loss is not None:
        values.append(trial.best_val_loss)
    else:
        values.append(float("inf"))
    # Multi-objective: add param_count if directions has 2 entries
    if len(directions) > 1 and trial.actual_param_count is not None:
        values.append(float(trial.actual_param_count))
    elif len(directions) > 1:
        values.append(float("inf"))
    try:
        study.tell(trial.optuna_number, values)
    except Exception as e:
        log.warning("Optuna tell failed for trial %d: %s", trial.trial_id, e)
