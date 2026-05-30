"""Tests for `pawn.sweep` / `pawn.lab` / `pawn.wandb_utils` / `pawn.dashboard.metrics`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import optuna

from pawn.dashboard.metrics import MetricsBundle, discover_runs, load_metrics
from pawn.lab.runner import lab_launch, lab_schema, validate_config
from pawn.sweep import (
    STRATEGY_SUGGESTERS,
    _params_to_argv,
    _read_best_val_loss,
    suggest_lora,
    suggest_rosa,
)
from pawn.wandb_utils import finish_wandb, init_wandb, log_metrics


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def test_strategy_suggesters_present_for_all_strategies() -> None:
    """Every adapter strategy has a `suggest_*` function in the
    dispatcher table."""
    assert set(STRATEGY_SUGGESTERS.keys()) >= {
        "lora", "film", "bottleneck", "hybrid",
        "sparse", "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
        "unfreeze", "specialized_clm",
    }


def test_suggest_lora_returns_valid_params() -> None:
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_lora(trial)
    assert "lora_rank" in params
    assert "lora_targets" in params
    assert "lr" in params
    assert 1 <= params["lora_rank"] <= 16
    assert params["lora_targets"] in ("qkvo", "qv", "qkv")


def test_suggest_rosa_includes_v1_hyperparams() -> None:
    """Plan §6 + §10 S3: rosa_warmup_steps / mask_samples / grad_alpha
    are non-negotiable v1 hyperparameters; the sweep must search over
    them."""
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_rosa(trial)
    assert "rosa_warmup_steps" in params
    assert "mask_samples" in params
    assert "grad_alpha" in params
    assert params["grad_alpha"] in (1, 2)


def test_params_to_argv_handles_bool_and_int() -> None:
    """Bool True → flag with no value; False → omitted; int → flag + value."""
    argv = _params_to_argv({"lora_rank": 4, "use_output_film": True, "no_adapt_attn": False})
    assert "--lora-rank" in argv
    assert "4" in argv
    assert "--use-output-film" in argv
    assert "--no-adapt-attn" not in argv  # False omitted


def test_read_best_val_loss_finds_minimum(tmp_path: Path) -> None:
    """`_read_best_val_loss` walks the dir and returns the smallest
    val_loss across all metrics.jsonl files."""
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "train", "loss": 5.0}),
        json.dumps({"type": "val", "val_loss": 3.0}),
        json.dumps({"type": "val", "val_loss": 2.0}),
        json.dumps({"type": "val", "loss": 2.5}),  # accept "loss" too
    ]))
    assert _read_best_val_loss(tmp_path) == 2.0


def test_read_best_val_loss_returns_inf_on_no_records(tmp_path: Path) -> None:
    assert _read_best_val_loss(tmp_path) == float("inf")


# ---------------------------------------------------------------------------
# Lab — pydantic validation
# ---------------------------------------------------------------------------


def test_lab_schema_returns_all_run_types() -> None:
    schema = lab_schema()
    assert set(schema.keys()) == {
        "pretrain", "adapter", "specialized_clm", "distill",
    }
    # Each schema is a JSON Schema dict.
    for k, s in schema.items():
        assert "properties" in s


def test_validate_config_dispatches_by_run_type() -> None:
    from pawn.run_config import AdapterConfig, PretrainConfig

    assert isinstance(
        validate_config({
            "run_type": "pretrain", "local_checkpoints": True,
            "total_steps": 100,
        }),
        PretrainConfig,
    )
    assert isinstance(
        validate_config({
            "run_type": "adapter", "local_checkpoints": True,
            "total_steps": 100, "strategy": "lora", "lora_rank": 4,
        }),
        AdapterConfig,
    )


def test_validate_config_rejects_unknown_field() -> None:
    """The lab's pydantic boundary refuses stale field names per
    acceptance criterion 19."""
    with pytest.raises(ValueError, match="extra"):
        validate_config({
            "run_type": "pretrain",
            "local_checkpoints": True,
            "legacy_vocab": True,  # stale v1 field
        })


def test_validate_config_rejects_missing_run_type() -> None:
    with pytest.raises(ValueError, match="run_type"):
        validate_config({"local_checkpoints": True})


def test_validate_config_rejects_unknown_run_type() -> None:
    with pytest.raises(ValueError, match="unknown run_type"):
        validate_config({"run_type": "cotrain", "local_checkpoints": True})


def test_lab_launch_dry_run_validates_without_spawning() -> None:
    """`dry_run=True` validates the config but doesn't actually spawn
    the subprocess."""
    result = lab_launch(
        {
            "run_type": "adapter",
            "local_checkpoints": True,
            "total_steps": 100,
            "strategy": "lora",
            "lora_rank": 4,
        },
        dry_run=True,
    )
    assert result["status"] == "validated"
    assert result["pid"] is None
    assert result["run_type"] == "adapter"


# ---------------------------------------------------------------------------
# Dashboard — metrics loader
# ---------------------------------------------------------------------------


def test_load_metrics_splits_by_type_discriminator(tmp_path: Path) -> None:
    """The dashboard's metrics loader splits records on the `type`
    discriminator (plan §10 S9)."""
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "config", "run_type": "pretrain", "slug": "test"}),
        json.dumps({"type": "train", "step": 1, "loss": 5.0}),
        json.dumps({"type": "train", "step": 2, "loss": 4.0}),
        json.dumps({"type": "val", "step": 2, "loss": 4.5}),
    ]))
    bundle = load_metrics(run)
    assert isinstance(bundle, MetricsBundle)
    assert bundle.config is not None
    assert bundle.config["run_type"] == "pretrain"
    assert bundle.slug == "test"
    assert len(bundle.train_records) == 2
    assert len(bundle.val_records) == 1


def test_load_metrics_tolerates_malformed_lines(tmp_path: Path) -> None:
    """Partial writes / corrupted lines are skipped — the dashboard
    can tail a running file safely."""
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "train", "loss": 1.0}),
        "not valid json {",
        json.dumps({"type": "train", "loss": 0.9}),
    ]))
    bundle = load_metrics(run)
    assert len(bundle.train_records) == 2  # malformed line skipped


def test_discover_runs_finds_all_metrics_jsonl(tmp_path: Path) -> None:
    """`discover_runs` finds every subdirectory with a metrics.jsonl."""
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a/metrics.jsonl").write_text("{}")
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_b/metrics.jsonl").write_text("{}")
    (tmp_path / "no_metrics").mkdir()
    runs = discover_runs(tmp_path)
    assert len(runs) == 2
    assert all(r.name in ("run_a", "run_b") for r in runs)


def test_load_metrics_handles_missing_file_gracefully(tmp_path: Path) -> None:
    bundle = load_metrics(tmp_path / "nonexistent")
    assert bundle.train_records == []
    assert bundle.val_records == []
    assert bundle.config is None


# ---------------------------------------------------------------------------
# W&B — disabled-mode no-ops
# ---------------------------------------------------------------------------


def test_init_wandb_disabled_returns_none() -> None:
    run = init_wandb(
        project="pawn", slug="test", run_config={}, enabled=False
    )
    assert run is None


def test_log_metrics_no_op_on_none() -> None:
    """`log_metrics(None, ...)` is a no-op for disabled W&B."""
    log_metrics(None, {"loss": 1.0})  # must not raise


def test_finish_wandb_no_op_on_none() -> None:
    finish_wandb(None)  # must not raise


def test_init_wandb_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PAWN_WANDB_MODE=disabled` short-circuits even when `enabled=True`."""
    monkeypatch.setenv("PAWN_WANDB_MODE", "disabled")
    run = init_wandb(
        project="pawn", slug="test", run_config={}, enabled=True
    )
    assert run is None
