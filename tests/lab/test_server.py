"""Tests for pawn.lab.server — MCP tool functions.

The tools are decorated with @mcp.tool, so we test the underlying
coroutine functions directly by constructing a fake Context that carries
a TrialRunner in lifespan_context.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from pawn.lab.runner import TrialRunner
from pawn.lab.server import (
    lab_events,
    lab_kill,
    lab_launch,
    lab_log,
    lab_notes,
    lab_resume,
    lab_results,
    lab_schema,
    lab_set_cost,
    lab_status,
)
from pawn.lab.state import Trial


class FakeContext:
    """Minimal stand-in for FastMCP's Context."""

    def __init__(self, runner: TrialRunner):
        self.lifespan_context = {"runner": runner}


@pytest.fixture
def runner(tmp_path):
    r = TrialRunner(workspace=str(tmp_path))
    return r


@pytest.fixture
def ctx(runner):
    return FakeContext(runner)


# =====================================================================
# lab_status
# =====================================================================


class TestLabStatus:
    def test_returns_dict_with_required_keys(self, ctx):
        result = asyncio.run(lab_status(ctx))
        assert isinstance(result, dict)
        assert "total_trials" in result
        assert "running" in result
        assert "completed" in result
        assert "failed" in result
        assert "elapsed" in result

    def test_empty_status(self, ctx):
        result = asyncio.run(lab_status(ctx))
        assert result["total_trials"] == 0
        assert result["running"] == []
        assert result["completed"] == 0
        assert result["failed"] == 0


# =====================================================================
# lab_launch
# =====================================================================


class TestLabLaunch:
    def test_invalid_config_returns_error(self, ctx):
        # Missing run_type
        result = asyncio.run(lab_launch({}, ctx))
        assert "error" in result

    def test_invalid_run_type_returns_error(self, ctx):
        result = asyncio.run(lab_launch({"run_type": "bogus"}, ctx))
        assert "error" in result

    def test_no_free_gpu_returns_error(self, ctx, runner):
        # No GPUs discovered at all → _find_free_gpu returns None
        runner._gpus_discovered = True
        runner.gpu_count = 0
        result = asyncio.run(lab_launch(
            {"run_type": "pretrain", "variant": "base", "local_checkpoints": True},
            ctx,
        ))
        assert "error" in result


# =====================================================================
# lab_kill
# =====================================================================


class TestLabKill:
    def test_kill_missing_trial(self, ctx):
        result = asyncio.run(lab_kill(999, ctx))
        assert "error" in result

    def test_kill_non_running_trial(self, ctx, runner):
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[], status="completed",
        )
        result = asyncio.run(lab_kill(0, ctx))
        assert "error" in result

    def test_kill_running_sends_sigterm(self, ctx, runner):
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[], status="running",
            pid=99999,
        )
        with patch("os.kill") as mock_kill:
            result = asyncio.run(lab_kill(0, ctx))
        assert result["killed"] == 0
        mock_kill.assert_called_once()


# =====================================================================
# lab_resume
# =====================================================================


class TestLabResume:
    def test_resume_missing_trial_returns_error(self, ctx):
        result = asyncio.run(lab_resume(999, ctx))
        assert "error" in result

    def test_resume_trial_without_run_dir_returns_error(self, ctx, runner):
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[], status="completed",
            run_dir=None,
        )
        result = asyncio.run(lab_resume(0, ctx))
        assert "error" in result


# =====================================================================
# lab_results
# =====================================================================


class TestLabResults:
    def test_empty_results_structure(self, ctx):
        result = asyncio.run(lab_results(ctx))
        assert "trials" in result
        assert "pareto_front" in result
        assert "suggestions" in result
        assert result["trials"] == []

    def test_results_shape_with_trials(self, ctx, runner):
        runner.trials[0] = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            status="completed", best_val_loss=0.5, actual_param_count=1000,
        )
        result = asyncio.run(lab_results(ctx))
        assert len(result["trials"]) == 1
        assert result["trials"][0]["trial"] == 0

    def test_results_tag_filter(self, ctx, runner):
        runner.trials[0] = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            status="completed", tags=["phase1"], best_val_loss=0.5,
            actual_param_count=1000,
        )
        runner.trials[1] = Trial(
            trial_id=1, strategy="lora", params={}, cli_command=[],
            status="completed", tags=["phase2"], best_val_loss=0.4,
            actual_param_count=2000,
        )
        result = asyncio.run(lab_results(ctx, tag="phase1"))
        assert len(result["trials"]) == 1
        assert result["trials"][0]["trial"] == 0


# =====================================================================
# lab_events
# =====================================================================


class TestLabEvents:
    def test_empty_events(self, ctx):
        result = asyncio.run(lab_events(ctx))
        assert "events" in result
        assert "latest_seq" in result
        assert result["events"] == []
        assert result["latest_seq"] == 0

    def test_events_returned_in_order(self, ctx, runner):
        runner._emit("one")
        runner._emit("two")
        result = asyncio.run(lab_events(ctx, since=0))
        assert len(result["events"]) == 2
        assert result["events"][0]["type"] == "one"
        assert result["events"][1]["type"] == "two"
        assert result["latest_seq"] == 2

    def test_events_since_n_returns_only_new(self, ctx, runner):
        runner._emit("one")
        runner._emit("two")
        runner._emit("three")
        result = asyncio.run(lab_events(ctx, since=1))
        assert len(result["events"]) == 2
        types = [e["type"] for e in result["events"]]
        assert types == ["two", "three"]


# =====================================================================
# lab_log
# =====================================================================


class TestLabLog:
    def test_log_missing_trial(self, ctx):
        result = asyncio.run(lab_log(999, ctx))
        assert "error" in result

    def test_log_returns_last_n_lines(self, ctx, runner, tmp_path):
        log_path = tmp_path / "t.log"
        log_path.write_text("line1\nline2\nline3\nline4\nline5\n")
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[],
            log_path=str(log_path),
        )
        result = asyncio.run(lab_log(0, ctx, lines=3))
        assert result["trial"] == 0
        assert len(result["lines"]) == 3
        assert result["lines"][-1] == "line5"


# =====================================================================
# lab_notes
# =====================================================================


class TestLabNotes:
    def test_notes_missing_trial(self, ctx):
        result = asyncio.run(lab_notes(999, "note", ctx))
        assert "error" in result

    def test_notes_updates_trial(self, ctx, runner):
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[],
        )
        result = asyncio.run(lab_notes(0, "looks good", ctx))
        assert result["ok"] is True
        assert runner.trials[0].notes == "looks good"


# =====================================================================
# lab_set_cost
# =====================================================================


class TestLabSetCost:
    def test_sets_cost_per_hour(self, ctx, runner):
        result = asyncio.run(lab_set_cost(3.59, ctx))
        assert result["cost_per_hour"] == 3.59
        assert runner.cost_per_hour == 3.59

    def test_set_cost_persists(self, ctx, runner, tmp_path):
        asyncio.run(lab_set_cost(2.00, ctx))
        # Reload runner from disk
        runner2 = TrialRunner(workspace=str(tmp_path))
        runner2._load_state()
        assert runner2.cost_per_hour == 2.00


# =====================================================================
# lab_schema
# =====================================================================


class TestLabSchema:
    def test_returns_both_schemas(self, ctx):
        result = asyncio.run(lab_schema(ctx))
        assert "pretrain" in result
        assert "adapter" in result

    def test_schemas_are_json_schema_dicts(self, ctx):
        result = asyncio.run(lab_schema(ctx))
        assert isinstance(result["pretrain"], dict)
        assert isinstance(result["adapter"], dict)
        # JSON Schema has a `properties` key
        assert "properties" in result["pretrain"]
        assert "properties" in result["adapter"]


# =====================================================================
# lab_audit
# =====================================================================


class TestLabAudit:
    def test_empty_lab_no_failures(self, ctx):
        from pawn.lab.server import lab_audit

        result = asyncio.run(lab_audit(ctx))
        assert result == {"trials": [], "any_failure": False}

    def test_passes_through_trial_id_and_check_hf(self, ctx, runner):
        from pawn.lab.server import lab_audit

        # Wire a trial through the runner so audit has something to read.
        runner.trials[0] = Trial(
            trial_id=0, strategy="bottleneck", params={},
            cli_command=[], status="completed",
        )

        with patch.object(
            runner, "audit", wraps=runner.audit
        ) as wrapped:
            asyncio.run(lab_audit(ctx, trial_id=0, check_hf=True))
        wrapped.assert_called_once_with(trial_id=0, check_hf=True)
