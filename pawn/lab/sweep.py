"""Optuna search space definitions for PAWN adapter strategies.

Distributions are keyed by the v2 ``--strategy`` flag names + use the
v2 trainer's argparse flag shapes (see ``scripts/train_jax_adapter.py``):
- ``lora_targets`` / ``rosa_targets`` / ``sparse_targets`` are
  space-separated *lists* (``["q", "v"]``), not concatenated strings.
- ``rank`` replaces ``lora_rank`` (the v2 CLI flag is ``--rank``).
- ``rosa`` is a single strategy now — the v1 ``retro-sparse`` /
  ``retro-bottleneck`` ablation modes were not ported.
"""

from __future__ import annotations

from typing import Any

# In-sync with the v2 trainer's argparse choice list. v1 lived in
# ``pawn.sweep`` which was deleted; restating here so the lab is
# self-contained.
BOTTLENECK_N_HIDDEN_CHOICES: tuple[int, ...] = (0, 1, 2)


def builtin_distributions(strategy: str) -> dict[str, Any]:
    """Return Optuna distributions for a PAWN v2 adapter strategy."""
    import optuna.distributions as d
    Cat = d.CategoricalDistribution
    Float = d.FloatDistribution
    Int = d.IntDistribution

    common: dict[str, Any] = {
        "lr": Float(1e-5, 1e-2, log=True),
        "batch_size": Cat([32, 64, 128, 256]),
        # ``warmup_steps`` rather than v1's ``warmup_frac`` — v2 takes
        # an absolute step count via ``--warmup-steps``.
        "warmup_steps": Int(0, 1000, step=50),
    }
    spaces: dict[str, dict[str, Any]] = {
        "lora": {**common,
            "rank": Cat([2, 4, 8, 16, 32]),
            "lora_targets": Cat(["qv", "qkv", "qkvo"]),  # concatenated letters; runner splits
            "lora_alpha": Cat([None, 8.0, 16.0, 32.0]),
        },
        "bottleneck": {**common,
            "bottleneck_dim": Cat([4, 8, 16, 32, 64, 128]),
            "bottleneck_n_hidden": Cat(list(BOTTLENECK_N_HIDDEN_CHOICES)),
            "bottleneck_no_attn": Cat([True, False]),
            "bottleneck_no_ffn": Cat([True, False]),
        },
        "film": {**common,
            "film_output": Cat([True, False]),
        },
        "sparse": {**common,
            "sparse_density": Float(0.001, 0.1, log=True),
            "sparse_targets": Cat(["qv", "qkv", "qkvo"]),  # concatenated letters; runner splits
            "sparse_hard": Cat([True, False]),
        },
        "hybrid": {**common,
            "rank": Cat([2, 4, 8, 16]),
            "lora_targets": Cat(["qv", "qkv", "qkvo"]),  # concatenated letters; runner splits
            "film_output": Cat([True, False]),
        },
        "specialized_clm": {**common,
            "specialized_d_model": Cat([32, 48, 64, 128, 192]),
            "specialized_n_layers": Int(1, 4),
            "specialized_n_heads": Cat([1, 2, 4, 8]),
            "specialized_d_ff": Cat([128, 256, 512, 768]),
        },
        "unfreeze": {**common,
            "n_unfreeze": Int(1, 4),
            "include_lm_head": Cat([True, False]),
            "include_embeddings": Cat([True, False]),
        },
        "rosa": {**common,
            "rank": Cat([2, 4, 8, 16]),
            "rosa_targets": Cat(["qv", "qkv", "qkvo"]),  # concatenated letters; runner splits
            "rosa_warmup_frac": Float(0.1, 0.6),
            "rosa_top_k_frac": Float(0.001, 0.1, log=True),
        },
        "pretrain": {
            "lr": Float(1e-5, 1e-3, log=True),
            "batch_size": Cat([64, 128, 256]),
            "warmup_steps": Int(0, 2000, step=100),
            # supernet selection (variant size lives in the variant
            # validator inside the trainer, not as a sweep axis).
            "supernet": Cat(["tiny", "supernet"]),
        },
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
