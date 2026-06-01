"""Optuna search space definitions for PAWN adapter strategies."""

from __future__ import annotations

import optuna.distributions as d
from optuna.distributions import BaseDistribution

from pawn.sweep import BOTTLENECK_N_HIDDEN_CHOICES


def builtin_distributions(strategy: str) -> dict[str, BaseDistribution]:
    """Return Optuna distributions for a PAWN adapter strategy."""
    Cat = d.CategoricalDistribution
    Float = d.FloatDistribution
    Int = d.IntDistribution

    common: dict[str, BaseDistribution] = {
        "lr": Float(1e-5, 1e-2, log=True),
        "batch_size": Cat([32, 64, 128, 256]),
        "weight_decay": Float(0.0, 0.1),
        "warmup_frac": Float(0.0, 0.15),
    }
    spaces: dict[str, dict[str, BaseDistribution]] = {
        "lora": {**common,
            "lora_rank": Cat([2, 4, 8, 16, 32]),
            "lora_targets": Cat(["qkvo", "qv", "qkv"]),
            "lora_ffn": Cat([True, False]),
        },
        "bottleneck": {**common,
            "bottleneck_dim": Cat([4, 8, 16, 32, 64, 128, 256]),
            "bottleneck_n_hidden": Cat(list(BOTTLENECK_N_HIDDEN_CHOICES)),
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
        "hybrid": {**common,
            "lora_rank": Cat([2, 4, 8, 16]),
            "lora_targets": Cat(["qkvo", "qv", "qkv"]),
            "use_output_film": Cat([True, False]),
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
    rosa_common: dict[str, BaseDistribution] = {**common,
        "density": Float(0.001, 0.1, log=True),
        "lora_rank": Cat([2, 4, 8, 16]),
        "lora_targets": Cat(["qkvo", "qv", "qkv"]),
        "rosa_warmup_steps": Int(32, 256, step=32),
        "mask_samples": Cat([16, 32, 64]),
        "grad_alpha": Cat([1, 2]),
    }
    spaces["rosa"] = rosa_common
    spaces["retro-sparse"] = rosa_common
    spaces["retro-bottleneck"] = {
        **rosa_common,
        "bottleneck_dim": Cat([4, 8, 16]),
        "bottleneck_n_hidden": Cat(list(BOTTLENECK_N_HIDDEN_CHOICES)),
    }
    spaces["pretrain"] = {
        "lr": Float(1e-5, 1e-3, log=True),
        "batch_size": Cat([64, 128, 256]),
        "weight_decay": Float(0.0, 0.1),
        "warmup_frac": Float(0.0, 0.15),
        "d_model": Cat([256, 384, 512, 640]),
        "n_layers": Cat([6, 8, 10, 12]),
        "n_heads": Cat([4, 8]),
    }

    if strategy not in spaces:
        raise ValueError(f"No built-in search space for strategy '{strategy}'. "
                         f"Available: {sorted(spaces)}")
    return spaces[strategy]


def parse_distribution(spec: dict[str, object]) -> BaseDistribution:
    """Convert a JSON distribution spec to an Optuna distribution object."""
    t = spec["type"]
    if t == "float":
        low = spec["low"]
        high = spec["high"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError("float distribution requires numeric 'low' and 'high'")
        log = bool(spec.get("log", False))
        return d.FloatDistribution(float(low), float(high), log=log)
    elif t == "int":
        low = spec["low"]
        high = spec["high"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError("int distribution requires numeric 'low' and 'high'")
        step_raw = spec.get("step", 1)
        if not isinstance(step_raw, (int, float)):
            raise ValueError("'step' must be numeric")
        return d.IntDistribution(int(low), int(high), step=int(step_raw))
    elif t == "categorical":
        choices = spec["choices"]
        if not isinstance(choices, list):
            raise ValueError("categorical distribution requires a 'choices' list")
        return d.CategoricalDistribution(choices)
    raise ValueError(f"Unknown distribution type: {t}")
