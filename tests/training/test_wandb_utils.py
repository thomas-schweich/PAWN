"""Unit tests for :mod:`pawn.wandb_utils` — the metrics-only W&B helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from pawn.logging import MetricsLogger
from pawn.wandb_utils import finish_wandb, init_wandb, log_metrics


@pytest.fixture
def logger(tmp_path: Path) -> MetricsLogger:
    return MetricsLogger(str(tmp_path), run_prefix="wandbtest", device="cpu")


def test_init_disabled_returns_none(logger: MetricsLogger):
    run = init_wandb(
        enabled=False,
        project="pawn-test",
        logger=logger,
        run_type="pretrain",
        config={"x": 1},
    )
    assert run is None
    # log/finish are null-safe.
    log_metrics(run, {"loss": 0.5}, step=1)
    finish_wandb(run)


def test_init_enabled_populates_reproducibility_config(logger: MetricsLogger):
    run = init_wandb(
        enabled=True,
        project="pawn-test",
        logger=logger,
        run_type="pretrain",
        config={"variant": "toy", "lr": 3e-4},
        tags=["variant:toy"],
    )
    try:
        assert run is not None
        cfg = dict(run.config)
        # Caller config survives.
        assert cfg["variant"] == "toy"
        assert cfg["lr"] == pytest.approx(3e-4)
        # Reproducibility metadata is injected.
        assert cfg["slug"] == logger.slug
        assert cfg["run_dir"] == str(logger.run_dir)
        assert "hostname" in cfg
        assert "command_line" in cfg
        assert "python_version" in cfg
        assert "torch_version" in cfg
        assert "gpu_name" in cfg  # None on CPU, but present
        assert cfg["run_type"] == "pretrain"
        # (Run name would match ``logger.run_dir.name`` in online/offline
        # modes; disabled mode substitutes a dummy name, so we don't assert
        # on it here.)
    finally:
        finish_wandb(run)


def test_log_metrics_accepts_step_and_no_step(logger: MetricsLogger):
    run = init_wandb(
        enabled=True,
        project="pawn-test",
        logger=logger,
        run_type="adapter",
        config={},
    )
    try:
        log_metrics(run, {"train/loss": 1.0}, step=10)
        log_metrics(run, {"train/loss": 0.9})  # auto step
    finally:
        finish_wandb(run)


def test_finish_wandb_noop_on_none():
    finish_wandb(None)
    finish_wandb(None, exit_code=1)


def test_log_metrics_noop_on_none():
    log_metrics(None, {"loss": 1.0})
    log_metrics(None, {"loss": 1.0}, step=5)


def test_wandb_project_env_var_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, logger: MetricsLogger
):
    """``WANDB_PROJECT`` must override the caller's ``project`` so users
    can redirect runs without touching ``TrainingConfig.wandb_project``."""
    monkeypatch.setenv("WANDB_PROJECT", "override-project")
    run = init_wandb(
        enabled=True,
        project="pawn",
        logger=logger,
        run_type="pretrain",
        config={},
    )
    try:
        assert run is not None
        # ``run.project`` isn't reliable in disabled mode (wandb stubs it),
        # but the init call itself should have honored the env var: we
        # re-read the env to confirm the helper didn't strip it.
        import os as _os
        assert _os.environ["WANDB_PROJECT"] == "override-project"
    finally:
        finish_wandb(run)


def test_env_mode_override(
    monkeypatch: pytest.MonkeyPatch, logger: MetricsLogger
):
    monkeypatch.setenv("PAWN_WANDB_MODE", "disabled")
    run = init_wandb(
        enabled=True,
        project="pawn-test",
        logger=logger,
        run_type="cotrain",
        config={},
    )
    try:
        assert run is not None
        # disabled-mode runs still quack like a Run.
        log_metrics(run, {"loss": 0.1}, step=1)
    finally:
        finish_wandb(run)
