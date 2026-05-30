"""Tests for `pawn.sweep` / `pawn.lab` / `pawn.wandb_utils` / `pawn.dashboard.metrics`."""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import pytest
import optuna

from pawn.dashboard.metrics import MetricsBundle, discover_runs, load_metrics
from pawn.lab.runner import (
    audit_schedule_health,
    lab_launch,
    lab_schema,
    read_schedule_health,
    validate_config,
)
from pawn.sweep import (
    STRATEGY_SUGGESTERS,
    _params_to_argv,
    _read_best_val_loss,
    suggest_lora,
    suggest_rosa,
)
from pawn.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    require_wandb_available,
    wandb_available,
)


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


# ---------------------------------------------------------------------------
# W&B — gating (H7: --wandb without the extra is a hard error)
# ---------------------------------------------------------------------------


def test_require_wandb_available_errors_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--wandb` requested but the `wandb` extra absent → SystemExit with
    an actionable message. (Simulate the missing extra by patching the
    availability probe.)"""
    monkeypatch.setattr("pawn.wandb_utils.wandb_available", lambda: False)
    with pytest.raises(SystemExit, match="wandb"):
        require_wandb_available()


def test_require_wandb_available_passes_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the extra is installed, the gate is a no-op."""
    monkeypatch.setattr("pawn.wandb_utils.wandb_available", lambda: True)
    require_wandb_available()  # must not raise


def test_init_wandb_invokes_mirror_with_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the extra present, init_wandb forwards to `wandb.init` and
    `log_metrics` forwards to the run's `.log`. Uses a mocked wandb module
    so no network/login is required."""
    import sys
    import types

    # The test conftest pins PAWN_WANDB_MODE=disabled globally (so no real
    # W&B run is ever created); override to "offline" here so init_wandb
    # reaches the (mocked) wandb.init call rather than short-circuiting.
    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    fake_run = mock.MagicMock(name="wandb_run")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = mock.MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    run = init_wandb(
        project="pawn", slug="run-xyz", run_config={"lr": 1e-3},
        git_hash="deadbeef", enabled=True,
    )
    assert run is fake_run
    fake_wandb.init.assert_called_once()  # type: ignore[attr-defined]
    _, kwargs = fake_wandb.init.call_args  # type: ignore[attr-defined]
    assert kwargs["name"] == "run-xyz"
    assert kwargs["config"]["lr"] == 1e-3
    assert kwargs["config"]["git_hash"] == "deadbeef"

    log_metrics(run, {"loss": 0.5}, step=10)
    fake_run.log.assert_called_once_with({"loss": 0.5}, step=10)
    finish_wandb(run)
    fake_run.finish.assert_called_once()


def test_wandb_available_returns_bool() -> None:
    assert isinstance(wandb_available(), bool)


# ---------------------------------------------------------------------------
# H7 — lab runner reads schedule_health.json + flags structural mismatch
# ---------------------------------------------------------------------------


def _write_health(run_dir: Path, **fields: object) -> None:
    base = {
        "format_version": 1,
        "schedule": "cosine",
        "should_reach_zero": True,
        "planned_total_steps": 1000,
        "actual_total_steps": 1000,
        "completion_ratio": 1.0,
        "lr_peak": 3e-4,
        "actual_final_lr": 0.0,
        "reason_for_stop": "completed",
    }
    base.update(fields)
    (run_dir / "schedule_health.json").write_text(json.dumps(base))


def test_read_schedule_health_absent_returns_none(tmp_path: Path) -> None:
    assert read_schedule_health(tmp_path) is None


def test_audit_schedule_health_clean_full_run(tmp_path: Path) -> None:
    """A `completed` run whose actual == planned is healthy: no banner."""
    _write_health(tmp_path)
    audit = audit_schedule_health(tmp_path)
    assert audit["present"] is True
    assert audit["structural_mismatch"] is False
    assert audit["banner"] is None


def test_audit_schedule_health_flags_structural_mismatch(
    tmp_path: Path,
) -> None:
    """`actual != planned` AND reason_for_stop == 'completed' is the
    structural-bug signal: the lab runner raises the flag + banner."""
    _write_health(tmp_path, actual_total_steps=500, reason_for_stop="completed")
    audit = audit_schedule_health(tmp_path)
    assert audit["structural_mismatch"] is True
    assert audit["banner"] is not None
    assert "STRUCTURAL MISMATCH" in audit["banner"]


def test_audit_schedule_health_sigterm_is_not_mismatch(tmp_path: Path) -> None:
    """A SIGTERM early exit with actual != planned is a *legitimate*
    early stop, not the structural-bug signal."""
    _write_health(tmp_path, actual_total_steps=500, reason_for_stop="sigterm")
    audit = audit_schedule_health(tmp_path)
    assert audit["structural_mismatch"] is False
    assert audit["banner"] is None


def test_audit_schedule_health_absent_file(tmp_path: Path) -> None:
    audit = audit_schedule_health(tmp_path)
    assert audit["present"] is False
    assert audit["structural_mismatch"] is False
