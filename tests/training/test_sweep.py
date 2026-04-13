"""Tests for pawn.sweep: suggest_* search-space functions and create_study.

Uses ``optuna.trial.FixedTrial`` for deterministic testing — no real study needed.
"""

from __future__ import annotations

import pytest

import optuna

from pawn.sweep import (
    SUGGEST_FNS,
    TRAIN_SCRIPT,
    create_study,
    suggest_architecture,
    suggest_bottleneck,
    suggest_common,
    suggest_film,
    suggest_hybrid,
    suggest_lora,
    suggest_pretrain,
    suggest_retro_bottleneck,
    suggest_retro_sparse,
    suggest_rosa,
    suggest_sparse,
    suggest_tiny,
)


# ---------------------------------------------------------------------------
# suggest_common
# ---------------------------------------------------------------------------


_COMMON_PARAMS = {
    "lr": 1e-3,
    "batch_size": 64,
    "weight_decay": 0.01,
    "warmup_frac": 0.05,
    "patience": 10,
}


class TestSuggestCommon:
    def test_returns_dict(self):
        trial = optuna.trial.FixedTrial(_COMMON_PARAMS)
        out = suggest_common(trial)
        assert isinstance(out, dict)

    def test_has_required_keys(self):
        trial = optuna.trial.FixedTrial(_COMMON_PARAMS)
        out = suggest_common(trial)
        assert {"lr", "batch_size", "weight_decay", "warmup_frac", "patience"} <= out.keys()

    def test_passes_through_values(self):
        trial = optuna.trial.FixedTrial(_COMMON_PARAMS)
        out = suggest_common(trial)
        assert out["lr"] == pytest.approx(1e-3)
        assert out["batch_size"] == 64
        assert out["weight_decay"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# suggest_lora
# ---------------------------------------------------------------------------


class TestSuggestLora:
    def test_has_lora_keys(self):
        params = {**_COMMON_PARAMS, "lora_rank": 4, "lora_targets": "qkvo", "lora_ffn": False}
        trial = optuna.trial.FixedTrial(params)
        out = suggest_lora(trial)
        assert "lora_rank" in out
        assert "lora_targets" in out
        assert "lora_ffn" in out
        assert out["lora_rank"] == 4
        assert out["lora_targets"] == "qkvo"
        assert out["lora_ffn"] is False

    def test_inherits_common_keys(self):
        params = {**_COMMON_PARAMS, "lora_rank": 8, "lora_targets": "qv", "lora_ffn": True}
        trial = optuna.trial.FixedTrial(params)
        out = suggest_lora(trial)
        assert "lr" in out and "batch_size" in out


# ---------------------------------------------------------------------------
# suggest_bottleneck
# ---------------------------------------------------------------------------


class TestSuggestBottleneck:
    def test_has_bottleneck_keys(self):
        params = {
            **_COMMON_PARAMS,
            "bottleneck_dim": 16,
            "no_adapt_attn": False,
            "no_adapt_ffn": False,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_bottleneck(trial)
        assert out["bottleneck_dim"] == 16
        assert out["no_adapt_attn"] is False
        assert out["no_adapt_ffn"] is False

    def test_includes_common(self):
        params = {
            **_COMMON_PARAMS,
            "bottleneck_dim": 8,
            "no_adapt_attn": True,
            "no_adapt_ffn": True,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_bottleneck(trial)
        assert out["lr"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# suggest_film
# ---------------------------------------------------------------------------


class TestSuggestFilm:
    def test_has_no_output_film_key(self):
        params = {**_COMMON_PARAMS, "no_output_film": True}
        trial = optuna.trial.FixedTrial(params)
        out = suggest_film(trial)
        assert "no_output_film" in out
        assert out["no_output_film"] is True


# ---------------------------------------------------------------------------
# suggest_sparse
# ---------------------------------------------------------------------------


class TestSuggestSparse:
    def test_density_and_targets(self):
        params = {
            **_COMMON_PARAMS,
            "density": 0.02,
            "sparse_targets": "qkvo",
            "sparse_ffn": False,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_sparse(trial)
        assert out["density"] == pytest.approx(0.02)
        assert out["sparse_targets"] == "qkvo"
        assert out["sparse_ffn"] is False

    def test_density_is_positive(self):
        params = {
            **_COMMON_PARAMS,
            "density": 0.001,
            "sparse_targets": "qv",
            "sparse_ffn": True,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_sparse(trial)
        assert out["density"] > 0


# ---------------------------------------------------------------------------
# suggest_hybrid
# ---------------------------------------------------------------------------


class TestSuggestHybrid:
    def test_has_required_hybrid_fields(self):
        params = {
            "lr": 1e-3,
            "batch_size": 64,
            "weight_decay": 0.01,
            "warmup_frac": 0.05,
            "patience": 10,
            "lora_rank": 4,
            "lora_targets": "qkvo",
            "use_output_film": False,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_hybrid(trial)
        assert "lr" in out
        assert "lora_rank" in out
        assert "use_output_film" in out
        assert out["lr"] == pytest.approx(1e-3)
        assert out["use_output_film"] is False


# ---------------------------------------------------------------------------
# suggest_rosa / retro-*
# ---------------------------------------------------------------------------


_ROSA_PARAMS = {
    **_COMMON_PARAMS,
    "density": 0.01,
    "lora_rank": 4,
    "lora_targets": "qkvo",
    "rosa_warmup_steps": 64,
    "mask_samples": 32,
    "grad_alpha": 2,
}


class TestSuggestRosa:
    def test_has_mode_rosa(self):
        trial = optuna.trial.FixedTrial(_ROSA_PARAMS)
        out = suggest_rosa(trial)
        assert out["rosa_mode"] == "rosa"

    def test_has_required_rosa_fields(self):
        trial = optuna.trial.FixedTrial(_ROSA_PARAMS)
        out = suggest_rosa(trial)
        assert "density" in out
        assert "lora_rank" in out
        assert "lora_targets" in out
        assert "rosa_warmup_steps" in out
        assert "mask_samples" in out
        assert "grad_alpha" in out


class TestSuggestRetroSparse:
    def test_has_mode_retro_sparse(self):
        trial = optuna.trial.FixedTrial(_ROSA_PARAMS)
        out = suggest_retro_sparse(trial)
        assert out["rosa_mode"] == "retro-sparse"


class TestSuggestRetroBottleneck:
    def test_has_mode_retro_bottleneck(self):
        params = {**_ROSA_PARAMS, "bottleneck_dim": 8}
        trial = optuna.trial.FixedTrial(params)
        out = suggest_retro_bottleneck(trial)
        assert out["rosa_mode"] == "retro-bottleneck"
        assert out["bottleneck_dim"] == 8


# ---------------------------------------------------------------------------
# suggest_tiny
# ---------------------------------------------------------------------------


class TestSuggestTiny:
    def test_arch_keys(self):
        params = {
            **_COMMON_PARAMS,
            "d_model": 84,
            "n_layers": 2,
            "n_heads": 4,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_tiny(trial)
        assert out["d_model"] == 84
        assert out["n_layers"] == 2
        assert out["n_heads"] == 4


# ---------------------------------------------------------------------------
# suggest_pretrain
# ---------------------------------------------------------------------------


class TestSuggestPretrain:
    def test_expected_keys(self):
        params = {
            "lr": 3e-4,
            "batch_size": 256,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
            "total_steps": 100_000,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_pretrain(trial)
        assert out["lr"] == pytest.approx(3e-4)
        assert out["batch_size"] == 256
        assert out["warmup_steps"] == 1000
        assert out["total_steps"] == 100_000


# ---------------------------------------------------------------------------
# suggest_architecture
# ---------------------------------------------------------------------------


class TestSuggestArchitecture:
    def test_derives_d_ff_from_d_ff_mult(self):
        params = {
            "d_model": 512,
            "n_layers": 8,
            "n_heads": 8,
            "d_ff_mult": 4,
            "lr": 3e-4,
            "batch_size": 128,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_architecture(trial)
        assert out["d_ff"] == 512 * 4
        assert out["d_model"] == 512
        assert out["n_layers"] == 8
        assert out["n_heads"] == 8

    def test_all_required_keys(self):
        params = {
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 16,
            "d_ff_mult": 3,
            "lr": 1e-4,
            "batch_size": 128,
            "weight_decay": 0.05,
            "warmup_steps": 2000,
        }
        trial = optuna.trial.FixedTrial(params)
        out = suggest_architecture(trial)
        for k in ("d_model", "n_layers", "n_heads", "d_ff", "lr", "batch_size", "weight_decay", "warmup_steps"):
            assert k in out


# ---------------------------------------------------------------------------
# SUGGEST_FNS registry
# ---------------------------------------------------------------------------


class TestSuggestFnsRegistry:
    def test_registry_is_dict(self):
        assert isinstance(SUGGEST_FNS, dict)

    @pytest.mark.parametrize(
        "name",
        [
            "lora", "bottleneck", "film", "sparse", "hybrid",
            "rosa", "retro-sparse", "retro-bottleneck", "rosa-ratio",
            "tiny", "architecture", "pretrain",
        ],
    )
    def test_key_present(self, name):
        assert name in SUGGEST_FNS

    def test_all_values_callable(self):
        for fn in SUGGEST_FNS.values():
            assert callable(fn)


class TestAdapterScriptsRegistry:
    @pytest.mark.parametrize(
        "name",
        ["lora", "bottleneck", "film", "sparse", "hybrid", "rosa", "tiny", "pretrain"],
    )
    def test_key_present(self, name):
        assert name in SUGGEST_FNS

    def test_unified_entry_point(self):
        assert TRAIN_SCRIPT == "scripts/train.py"


# ---------------------------------------------------------------------------
# create_study
# ---------------------------------------------------------------------------


class TestCreateStudy:
    def test_returns_study_object(self):
        study = create_study("test_study_inmem", storage=None, pruner="none")
        assert isinstance(study, optuna.Study)

    def test_minimize_direction_default(self):
        study = create_study("test_mini", storage=None, pruner="none")
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_maximize_direction(self):
        study = create_study(
            "test_maxi", storage=None, direction="maximize", pruner="none",
        )
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_pruner_hyperband(self):
        study = create_study("test_hb", storage=None, pruner="hyperband")
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)

    def test_pruner_median(self):
        study = create_study("test_med", storage=None, pruner="median")
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_pruner_none_is_nop(self):
        study = create_study("test_nop", storage=None, pruner="none")
        assert isinstance(study.pruner, optuna.pruners.NopPruner)

    def test_run_tiny_trial(self):
        """Sanity check: a create_study() can actually run a trivial objective."""
        study = create_study("test_tiny_trial", storage=None, pruner="none")
        study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=2)
        assert len(study.trials) == 2
