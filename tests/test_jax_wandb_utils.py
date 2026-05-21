"""Unit tests for :mod:`pawn.wandb_utils` — the metrics-only W&B helper.

The v2 ``wandb_utils`` API takes ``slug`` + ``run_dir`` directly
instead of a ``MetricsLogger`` object (the v1 logger was removed in
Phase 4). All other contracts — disabled-returns-None, repro config
shape, log/finish null-safety — are preserved.

These tests force ``PAWN_WANDB_MODE=disabled`` so they never reach
the wandb cloud. ``wandb.init(mode='disabled')`` returns a
``RunDisabled`` proxy that quacks like ``Run`` (same ``.log`` /
``.finish`` / ``.config`` surface).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("wandb")

from pawn.wandb_utils import finish_wandb, get_git_info, init_wandb, log_metrics


@pytest.fixture(autouse=True)
def _force_wandb_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled-mode runs don't touch the wandb cloud; the rest of
    the API surface (config inspection, log, finish) still works."""
    monkeypatch.setenv("PAWN_WANDB_MODE", "disabled")
    # Make sure ``WANDB_PROJECT`` doesn't leak in from the dev shell.
    monkeypatch.delenv("WANDB_PROJECT", raising=False)


def test_init_disabled_arg_returns_none(tmp_path: Path) -> None:
    """``enabled=False`` short-circuits without touching wandb. The
    null-safe ``log_metrics`` + ``finish_wandb`` accept the None."""
    run = init_wandb(
        enabled=False,
        project="pawn-test",
        slug="run1",
        run_dir=tmp_path,
        run_type="pretrain",
        config={"x": 1},
    )
    assert run is None
    log_metrics(run, {"loss": 0.5}, step=1)
    finish_wandb(run)


def test_init_enabled_populates_reproducibility_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "jax_run_test"
    run_dir.mkdir()
    run = init_wandb(
        enabled=True,
        project="pawn-test",
        slug="jax_run_test",
        run_dir=run_dir,
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
        assert cfg["slug"] == "jax_run_test"
        assert cfg["run_dir"] == str(run_dir)
        assert "hostname" in cfg
        assert "command_line" in cfg
        assert "python_version" in cfg
        # v2 reads JAX backend instead of torch CUDA info.
        assert "jax_backend" in cfg
        assert "gpu_name" in cfg  # None on CPU, but present
        assert "gpu_count" in cfg
        assert cfg["run_type"] == "pretrain"
    finally:
        finish_wandb(run)


def test_log_metrics_accepts_step_and_no_step(tmp_path: Path) -> None:
    run = init_wandb(
        enabled=True,
        project="pawn-test",
        slug="jax_run_test",
        run_dir=tmp_path,
        run_type="adapter",
        config={},
    )
    try:
        log_metrics(run, {"train/loss": 1.0}, step=10)
        log_metrics(run, {"train/loss": 0.9})  # auto step
    finally:
        finish_wandb(run)


def test_finish_wandb_noop_on_none() -> None:
    finish_wandb(None)
    finish_wandb(None, exit_code=1)


def test_get_git_info_returns_keys() -> None:
    """``get_git_info`` is a shared helper exposed for callers that
    want the same env-var precedence as wandb_utils.init."""
    info = get_git_info()
    assert set(info.keys()) == {"git_hash", "git_tag"}
    # Both fields are ``str | None`` — no exception on a non-git
    # checkout (PAWN_GIT_HASH env var path) and no exception when
    # ``git`` is in PATH.


def test_get_git_info_honours_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PAWN_GIT_HASH`` takes precedence over the live git checkout
    so the Docker image (which builds outside a git tree) can still
    surface a hash."""
    # Clear the cache so the env-var path is taken.
    import pawn.wandb_utils

    monkeypatch.setattr(pawn.wandb_utils, "_git_info_cache", None)
    monkeypatch.setenv("PAWN_GIT_HASH", "deadbeef")
    monkeypatch.setenv("PAWN_GIT_TAG", "v2.0.0")
    info = get_git_info()
    assert info["git_hash"] == "deadbeef"
    assert info["git_tag"] == "v2.0.0"
    # Clear the cache again so the next test isn't poisoned by ours.
    monkeypatch.setattr(pawn.wandb_utils, "_git_info_cache", None)


def test_pawn_wandb_mode_offline_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PAWN_WANDB_MODE=offline`` produces a real run dir under
    ``run_dir/wandb/offline-run-.../`` instead of phoning home."""
    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    run_dir = tmp_path / "jax_run_offline"
    run_dir.mkdir()
    run = init_wandb(
        enabled=True,
        project="pawn-test",
        slug=run_dir.name,
        run_dir=run_dir,
        run_type="pretrain",
        config={"x": 1},
    )
    try:
        assert run is not None
        log_metrics(run, {"loss": 0.5}, step=1)
    finally:
        finish_wandb(run)
    offline_dirs = list((run_dir / "wandb").glob("offline-run-*"))
    assert offline_dirs, (
        f"PAWN_WANDB_MODE=offline should have produced an "
        f"offline-run-* dir under {run_dir}/wandb/, but found "
        f"{list((run_dir / 'wandb').iterdir()) if (run_dir / 'wandb').exists() else '(none)'}"
    )
