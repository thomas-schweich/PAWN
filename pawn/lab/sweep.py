"""Optuna search space definitions for PAWN adapter strategies."""

from __future__ import annotations

from typing import Any

from pawn.sweep import BOTTLENECK_N_HIDDEN_CHOICES


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
