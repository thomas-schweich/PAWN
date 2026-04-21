"""Integration tests for W&B plumbing.

Complements the unit tests in ``test_wandb_utils.py`` by verifying that
the lifecycle (init → log → finish) stays connected through the trainer
and adapter code paths. The end-to-end smoke test uses a real subprocess
with ``PAWN_WANDB_MODE=disabled`` so no network calls are made.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pawn.logging import MetricsLogger


REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Direct helper tests
# ---------------------------------------------------------------------------


class TestLoggerWandbRoundTrip:
    """Exercise the helper contract used by every training entry point."""

    def test_slug_and_git_propagate_into_config(self, tmp_path: Path):
        from pawn.wandb_utils import finish_wandb, init_wandb, log_metrics

        logger = MetricsLogger(str(tmp_path), run_prefix="roundtrip")
        run = init_wandb(
            enabled=True,
            project="pawn-test",
            logger=logger,
            run_type="adapter",
            config={"strategy": "lora", "epochs": 1},
            group=logger.slug,
            job_type="adapter-lora",
            tags=["strategy:lora"],
        )
        try:
            assert run is not None
            cfg = dict(run.config)
            assert cfg["slug"] == logger.slug
            assert cfg["strategy"] == "lora"
            assert cfg["run_type"] == "adapter"
            # Metric forwarding is null-safe both ways.
            log_metrics(run, {"train/loss": 1.23}, step=1)
            log_metrics(run, {"val/loss": 1.10}, step=2)
        finally:
            finish_wandb(run, exit_code=0)

    def test_init_disabled_short_circuits_all_calls(self, tmp_path: Path):
        from pawn.wandb_utils import finish_wandb, init_wandb, log_metrics

        logger = MetricsLogger(str(tmp_path), run_prefix="disabled")
        run = init_wandb(
            enabled=False,
            project="pawn-test",
            logger=logger,
            run_type="pretrain",
            config={},
        )
        assert run is None
        # These must not raise on None.
        log_metrics(run, {"loss": 0.0})
        log_metrics(run, {"loss": 0.0}, step=0)
        finish_wandb(run)
        finish_wandb(run, exit_code=1)


# ---------------------------------------------------------------------------
# ModelSlot lifecycle (spy init_wandb + finish_wandb)
# ---------------------------------------------------------------------------


class TestModelSlotLifecycle:
    """``ModelSlot.close()`` must call ``finish_wandb``; ``__init__`` must
    call ``init_wandb`` with ``enabled=train_cfg.use_wandb``."""

    def test_init_and_close_call_wandb_helpers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        import pawn.cotrain as cotrain_mod

        init_spy = MagicMock(return_value=MagicMock(name="WandbRun"))
        finish_spy = MagicMock()
        monkeypatch.setattr(cotrain_mod, "init_wandb", init_spy)
        monkeypatch.setattr(cotrain_mod, "finish_wandb", finish_spy)

        from pawn.config import CLMConfig, TrainingConfig

        model_cfg = CLMConfig.toy()
        train_cfg = TrainingConfig.toy()
        train_cfg.device = "cpu"
        train_cfg.use_wandb = True
        train_cfg.log_dir = str(tmp_path / "logs")

        slot = cotrain_mod.ModelSlot(
            "toy", model_cfg, train_cfg, device="cpu", hf_repo=None,
            slug="test-slug",
            wandb_group="cotrain-test-slug",
            run_config={"total_steps": 100, "variants": []},
        )

        assert init_spy.call_count == 1
        kw = init_spy.call_args.kwargs
        assert kw["enabled"] is True
        assert kw["run_type"] == "cotrain"
        assert kw["group"] == "cotrain-test-slug"
        assert kw["job_type"] == "cotrain-toy"
        assert "variant:toy" in kw["tags"]
        assert kw["config"]["variant"] == "toy"

        slot.close(wandb_exit_code=0)
        assert finish_spy.call_count == 1
        assert finish_spy.call_args.kwargs["exit_code"] == 0

    def test_init_disabled_when_flag_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        import pawn.cotrain as cotrain_mod

        init_spy = MagicMock(return_value=None)
        finish_spy = MagicMock()
        monkeypatch.setattr(cotrain_mod, "init_wandb", init_spy)
        monkeypatch.setattr(cotrain_mod, "finish_wandb", finish_spy)

        from pawn.config import CLMConfig, TrainingConfig

        model_cfg = CLMConfig.toy()
        train_cfg = TrainingConfig.toy()
        train_cfg.device = "cpu"
        train_cfg.use_wandb = False
        train_cfg.log_dir = str(tmp_path / "logs")

        slot = cotrain_mod.ModelSlot(
            "toy", model_cfg, train_cfg, device="cpu", hf_repo=None,
            slug="x",
        )
        assert init_spy.call_args.kwargs["enabled"] is False
        assert slot.wandb_run is None
        slot.close()
        assert finish_spy.call_count == 1


# ---------------------------------------------------------------------------
# End-to-end subprocess smoke: toy pretrain completes with --wandb on
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_toy_pretrain_with_wandb_flag_runs_to_completion(tmp_path: Path):
    """Real subprocess: ``scripts/train.py --wandb`` must not crash even
    when W&B is disabled at the env level. Guards against regressions in
    the init/finish plumbing."""
    log_dir = tmp_path / "logs"
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PAWN_ALLOW_CPU": "1",
        "PAWN_WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
        "WANDB_MODE": "disabled",
    }

    result = subprocess.run(
        [
            sys.executable, "scripts/train.py",
            "--run-type", "pretrain", "--variant", "toy",
            "--local-checkpoints",
            "--total-steps", "20",
            "--batch-size", "8",
            "--device", "cpu",
            "--num-workers", "0",
            "--log-dir", str(log_dir),
            "--wandb",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Training exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    # A metrics.jsonl should exist for the run.
    runs = list(log_dir.glob("run_*"))
    assert runs, f"No run directory created under {log_dir}"
    assert (runs[0] / "metrics.jsonl").exists()
