"""Tests for pawn.lab.sweep — Optuna search space helpers."""

from __future__ import annotations

import pytest

from pawn.lab.sweep import builtin_distributions, parse_distribution


class TestBuiltinDistributions:
    def test_lora_has_lr(self):
        dists = builtin_distributions("lora")
        assert "lr" in dists
        assert "lora_rank" in dists
        assert "lora_targets" in dists

    def test_bottleneck_has_bottleneck_dim(self):
        dists = builtin_distributions("bottleneck")
        assert "bottleneck_dim" in dists

    def test_film_has_use_output_film(self):
        dists = builtin_distributions("film")
        assert "use_output_film" in dists

    def test_sparse_has_density(self):
        dists = builtin_distributions("sparse")
        assert "density" in dists
        assert "sparse_targets" in dists

    def test_hybrid_has_film_lr(self):
        dists = builtin_distributions("hybrid")
        assert "film_lr" in dists
        assert "lora_rank" in dists

    def test_rosa_has_rosa_warmup_steps(self):
        dists = builtin_distributions("rosa")
        assert "rosa_warmup_steps" in dists
        assert "density" in dists
        assert "lora_rank" in dists

    def test_retro_sparse_alias(self):
        dists = builtin_distributions("retro-sparse")
        assert "density" in dists
        assert "lora_rank" in dists

    def test_retro_bottleneck_has_bottleneck_dim(self):
        dists = builtin_distributions("retro-bottleneck")
        assert "bottleneck_dim" in dists
        assert "lora_rank" in dists

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="No built-in search space"):
            builtin_distributions("not_a_real_strategy")

    def test_common_fields_in_all_spaces(self):
        for strat in ["lora", "bottleneck", "film", "sparse", "rosa"]:
            dists = builtin_distributions(strat)
            assert "lr" in dists, f"{strat} missing lr"
            assert "batch_size" in dists, f"{strat} missing batch_size"
            assert "warmup_frac" in dists, f"{strat} missing warmup_frac"


class TestParseDistribution:
    def test_parse_float(self):
        dist = parse_distribution({"type": "float", "low": 0.0, "high": 1.0})
        import optuna.distributions as d
        assert isinstance(dist, d.FloatDistribution)
        assert dist.low == 0.0
        assert dist.high == 1.0

    def test_parse_float_log(self):
        dist = parse_distribution({"type": "float", "low": 1e-5, "high": 1e-1, "log": True})
        import optuna.distributions as d
        assert isinstance(dist, d.FloatDistribution)
        assert dist.log is True

    def test_parse_int(self):
        dist = parse_distribution({"type": "int", "low": 1, "high": 10})
        import optuna.distributions as d
        assert isinstance(dist, d.IntDistribution)
        assert dist.low == 1
        assert dist.high == 10

    def test_parse_int_with_step(self):
        dist = parse_distribution({"type": "int", "low": 0, "high": 100, "step": 10})
        import optuna.distributions as d
        assert isinstance(dist, d.IntDistribution)
        assert dist.step == 10

    def test_parse_categorical(self):
        dist = parse_distribution({"type": "categorical", "choices": ["a", "b", "c"]})
        import optuna.distributions as d
        assert isinstance(dist, d.CategoricalDistribution)
        assert list(dist.choices) == ["a", "b", "c"]

    def test_parse_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution type"):
            parse_distribution({"type": "bogus"})
