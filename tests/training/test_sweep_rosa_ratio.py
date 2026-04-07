"""Tests for the rosa-ratio sweep suggest function."""

from __future__ import annotations

import optuna

from pawn.sweep import (
    suggest_rosa_ratio,
    _BASE_BOTTLENECK_PARAMS_PER_DIM,
    _BASE_SPARSE_MASKABLE_PARAMS,
    SUGGEST_FNS,
    ADAPTER_SCRIPTS,
)


def _make_fixed_trial(params: dict) -> optuna.trial.FixedTrial:
    """Create a FixedTrial with the given parameter values."""
    return optuna.trial.FixedTrial(params)


# ---------------------------------------------------------------------------
# Required keys for InProcessRoSAObjective.__call__
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "mode", "density", "bottleneck_dim", "lr", "batch_size",
    "weight_decay", "warmup_frac", "patience",
    "lora_rank", "lora_targets", "warmup_steps",
    "mask_samples", "grad_alpha",
}


def test_suggest_keys():
    """All keys expected by InProcessRoSAObjective.__call__ are present."""
    trial = _make_fixed_trial({
        "total_budget": 250_000,
        "bottleneck_ratio": 0.5,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)
    assert _REQUIRED_KEYS.issubset(params.keys()), (
        f"Missing keys: {_REQUIRED_KEYS - params.keys()}"
    )


def test_mode_is_retro_bottleneck():
    trial = _make_fixed_trial({
        "total_budget": 250_000,
        "bottleneck_ratio": 0.5,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)
    assert params["mode"] == "retro-bottleneck"


def test_fixed_targets():
    trial = _make_fixed_trial({
        "total_budget": 250_000,
        "bottleneck_ratio": 0.5,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)
    assert params["lora_targets"] == "qkvo"


def test_budget_derivation_half_split():
    """50/50 split at 250K budget gives expected bottleneck_dim and density."""
    trial = _make_fixed_trial({
        "total_budget": 250_000,
        "bottleneck_ratio": 0.5,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)

    # bottleneck: 125K / 16384 = 7.63 -> round to 8
    assert params["bottleneck_dim"] == 8
    # sparse: 125K / 8_388_608 ≈ 0.0149
    expected_density = 125_000 / _BASE_SPARSE_MASKABLE_PARAMS
    assert abs(params["density"] - expected_density) < 1e-8


def test_budget_derivation_500k():
    """500K budget with 0.3 ratio."""
    trial = _make_fixed_trial({
        "total_budget": 500_000,
        "bottleneck_ratio": 0.3,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)

    # bottleneck: 150K / 16384 = 9.16 -> round to 9
    assert params["bottleneck_dim"] == 9
    # sparse: 350K / 8_388_608
    expected_density = 350_000 / _BASE_SPARSE_MASKABLE_PARAMS
    assert abs(params["density"] - expected_density) < 1e-8


def test_edge_low_ratio_clamps_bottleneck_dim():
    """Very low ratio still produces bottleneck_dim >= 1."""
    trial = _make_fixed_trial({
        "total_budget": 100_000,
        "bottleneck_ratio": 0.05,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)
    # 5K / 16384 = 0.305 -> round to 0 -> clamp to 1
    assert params["bottleneck_dim"] >= 1
    assert params["density"] > 0


def test_edge_high_ratio_positive_density():
    """Very high ratio still produces density > 0."""
    trial = _make_fixed_trial({
        "total_budget": 100_000,
        "bottleneck_ratio": 0.95,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    params = suggest_rosa_ratio(trial)
    assert params["density"] > 0
    assert params["bottleneck_dim"] >= 1


def test_user_attrs_logged():
    """Realized param counts are stored as trial user attributes."""
    trial = _make_fixed_trial({
        "total_budget": 250_000,
        "bottleneck_ratio": 0.5,
        "lr": 1e-3,
        "batch_size": 64,
        "warmup_steps": 64,
    })
    suggest_rosa_ratio(trial)

    attrs = trial.user_attrs
    assert "actual_bottleneck_params" in attrs
    assert "actual_sparse_params" in attrs
    assert "actual_total_params" in attrs
    assert attrs["actual_bottleneck_params"] == 8 * _BASE_BOTTLENECK_PARAMS_PER_DIM
    assert attrs["actual_total_params"] == (
        attrs["actual_bottleneck_params"] + attrs["actual_sparse_params"]
    )


def test_registered_in_suggest_fns():
    assert "rosa-ratio" in SUGGEST_FNS
    assert SUGGEST_FNS["rosa-ratio"] is suggest_rosa_ratio


def test_registered_in_adapter_scripts():
    assert "rosa-ratio" in ADAPTER_SCRIPTS
    assert ADAPTER_SCRIPTS["rosa-ratio"] == "scripts/legacy/train_rosa.py"
