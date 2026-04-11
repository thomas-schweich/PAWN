"""Tests for pawn.lab.runner.TrialRunner."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pawn.lab.runner import TrialRunner
from pawn.lab.state import Trial


# =====================================================================
# __init__ and directory setup
# =====================================================================


class TestInit:
    def test_creates_log_dir(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        assert runner.log_dir.exists()
        assert runner.log_dir == tmp_path / "logs"

    def test_creates_results_dir(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        assert runner.results_dir.exists()
        assert runner.results_dir == tmp_path / "sweep_results"

    def test_state_path_under_workspace(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        assert runner.state_path == tmp_path / "lab_state.json"

    def test_events_path_under_workspace(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        assert runner.events_path == tmp_path / "lab_events.jsonl"

    def test_initial_state(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        assert runner.trials == {}
        assert runner.next_trial_id == 0
        assert runner.gpu_count == 0
        assert runner.event_seq == 0
        assert runner.events == []
        assert runner.cost_per_hour is None

    def test_workspace_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAWN_WORKSPACE", str(tmp_path))
        runner = TrialRunner()
        assert runner.workspace == tmp_path


# =====================================================================
# GPU discovery (mocked)
# =====================================================================


class TestGpuDiscovery:
    def test_discover_gpus_parses_json(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        fake_output = json.dumps([
            {"name": "NVIDIA A100", "vram_mb": 80000},
            {"name": "NVIDIA A100", "vram_mb": 80000},
        ])
        with patch("subprocess.check_output", return_value=fake_output):
            runner._discover_gpus()
        assert runner.gpu_count == 2
        assert runner.gpu_names == ["NVIDIA A100", "NVIDIA A100"]
        assert runner.gpu_vram_mb == [80000, 80000]

    def test_discover_gpus_idempotent(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        fake_output = json.dumps([{"name": "A100", "vram_mb": 80000}])
        with patch("subprocess.check_output", return_value=fake_output) as mock:
            runner._discover_gpus()
            runner._discover_gpus()  # second call
        assert mock.call_count == 1
        assert runner.gpu_count == 1

    def test_discover_gpus_on_failure_sets_zero(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        with patch("subprocess.check_output", side_effect=Exception("boom")):
            runner._discover_gpus()
        assert runner.gpu_count == 0

    def test_discover_gpus_empty(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        with patch("subprocess.check_output", return_value="[]"):
            runner._discover_gpus()
        assert runner.gpu_count == 0
        assert runner.gpu_names == []


# =====================================================================
# GPU assignment
# =====================================================================


class TestGpuAssignment:
    def _make_runner_with_gpus(self, tmp_path, n: int) -> TrialRunner:
        runner = TrialRunner(workspace=str(tmp_path))
        runner._gpus_discovered = True
        runner._mps_active = False
        runner.gpu_count = n
        runner.gpu_names = [f"GPU-{i}" for i in range(n)]
        runner.gpu_vram_mb = [80000] * n
        runner.gpu_assignments = {i: None for i in range(n)}
        return runner

    def test_find_free_gpu_returns_first_free(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 3)
        assert runner._find_free_gpu() == 0

    def test_find_free_gpu_skips_assigned(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 3)
        runner.gpu_assignments[0] = 42  # trial 42 on gpu 0
        assert runner._find_free_gpu() == 1

    def test_find_free_gpu_returns_none_when_all_busy(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 2)
        runner.gpu_assignments[0] = 1
        runner.gpu_assignments[1] = 2
        assert runner._find_free_gpu() is None

    def test_find_free_gpu_returns_none_when_no_gpus(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 0)
        assert runner._find_free_gpu() is None

    def test_assign_gpu_sets_mapping(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 2)
        runner._assign_gpu(5, 1)
        assert runner.gpu_assignments[1] == 5

    def test_release_gpu_clears_mapping(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 2)
        runner._assign_gpu(5, 1)
        runner._release_gpu(1)
        assert runner.gpu_assignments[1] is None

    def test_gpu_utilization_shape(self, tmp_path):
        runner = self._make_runner_with_gpus(tmp_path, 2)
        runner.gpu_assignments[0] = 7
        util = runner.gpu_utilization()
        assert len(util) == 2
        assert util[0]["gpu"] == 0
        assert util[0]["total_mb"] == 80000
        assert util[0]["assigned_trial"] == 7
        assert util[1]["assigned_trial"] is None


# =====================================================================
# State persistence
# =====================================================================


class TestStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.next_trial_id = 5
        runner.event_seq = 12
        runner.cost_per_hour = 3.59
        trial = Trial(
            trial_id=0, strategy="lora", params={"lr": 1e-3},
            cli_command=["python", "x.py"], status="running",
            current_step=100, total_steps=1000,
        )
        runner.trials[0] = trial
        runner._save_state()

        assert runner.state_path.exists()

        # New runner loads
        runner2 = TrialRunner(workspace=str(tmp_path))
        runner2._load_state()
        assert runner2.next_trial_id == 5
        assert runner2.event_seq == 12
        assert runner2.cost_per_hour == 3.59
        assert 0 in runner2.trials
        assert runner2.trials[0].strategy == "lora"
        assert runner2.trials[0].current_step == 100

    def test_load_state_missing_file_is_noop(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        # no state file yet
        runner._load_state()
        assert runner.trials == {}

    def test_load_state_corrupt_json_logs_and_continues(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.state_path.write_text("{corrupt")
        runner._load_state()
        # Should not crash
        assert runner.trials == {}

    def test_save_state_is_atomic(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._save_state()
        # No stale .tmp file left
        assert not runner.state_path.with_suffix(".tmp").exists()


# =====================================================================
# Events
# =====================================================================


class TestEvents:
    def test_emit_increments_seq(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        assert runner.event_seq == 0
        runner._emit("test_event", trial_id=1, data={"x": 1})
        assert runner.event_seq == 1
        assert len(runner.events) == 1
        runner._emit("another", trial_id=2)
        assert runner.event_seq == 2
        assert len(runner.events) == 2

    def test_emit_writes_to_jsonl(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("test_event", trial_id=1, data={"x": 1})
        assert runner.events_path.exists()
        lines = runner.events_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["type"] == "test_event"
        assert event["trial_id"] == 1
        assert event["data"] == {"x": 1}

    def test_emit_event_has_timestamp_and_seq(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("hello")
        ev = runner.events[0]
        assert "seq" in ev
        assert "timestamp" in ev
        assert ev["seq"] == 1

    def test_events_since_returns_tail(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("a")
        runner._emit("b")
        runner._emit("c")
        events, latest = runner.events_since(1)
        assert len(events) == 2
        assert events[0]["type"] == "b"
        assert events[1]["type"] == "c"
        assert latest == 3

    def test_events_since_zero_returns_all(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("a")
        runner._emit("b")
        events, latest = runner.events_since(0)
        assert len(events) == 2
        assert latest == 2

    def test_events_since_none_auto_tracks(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("a")
        runner._emit("b")
        events, latest = runner.events_since(None)
        assert len(events) == 2
        # Second call with None should return 0 new events
        runner._emit("c")
        events, latest = runner.events_since(None)
        assert len(events) == 1
        assert events[0]["type"] == "c"

    def test_events_since_seq_at_or_above_latest(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("a")
        events, latest = runner.events_since(10)
        assert events == []
        assert latest == 1


# =====================================================================
# status()
# =====================================================================


class TestStatus:
    def test_empty_status(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        s = runner.status()
        assert s["total_trials"] == 0
        assert s["running"] == []
        assert s["completed"] == 0
        assert s["failed"] == 0
        assert "elapsed" in s

    def test_status_with_running_trial(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="lora", params={"lr": 1e-3, "lora_rank": 4},
            cli_command=[], status="running", current_step=50, total_steps=100,
            steps_per_sec=10.0, last_train_loss=1.5, best_val_loss=2.0,
            pid=1234, gpu_id=0,
        )
        s = runner.status()
        assert s["total_trials"] == 1
        assert len(s["running"]) == 1
        r = s["running"][0]
        assert r["trial"] == 0
        assert r["strategy"] == "lora"
        assert r["step"] == 50
        assert r["total"] == 100
        assert r["train_loss"] == 1.5
        assert r["val_loss"] == 2.0
        assert r["pid"] == 1234
        assert r["gpu"] == 0
        assert "key_hp" in r
        assert r["key_hp"].get("lora_rank") == 4
        assert r["key_hp"].get("lr") == 1e-3

    def test_status_counts(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(trial_id=0, strategy="a", params={}, cli_command=[], status="completed")
        runner.trials[1] = Trial(trial_id=1, strategy="b", params={}, cli_command=[], status="failed")
        runner.trials[2] = Trial(trial_id=2, strategy="c", params={}, cli_command=[], status="running")
        s = runner.status()
        assert s["total_trials"] == 3
        assert s["completed"] == 1
        assert s["failed"] == 1
        assert len(s["running"]) == 1

    def test_status_with_cost(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.cost_per_hour = 3.00
        runner.start_time = time.time() - 3600  # 1 hour ago
        s = runner.status()
        assert s["cost_per_hour"] == 3.00
        assert s["estimated_cost"] is not None
        assert s["estimated_cost"] == pytest.approx(3.0, abs=0.1)

    def test_status_attaches_pretrain_block_for_pretrain_runs(self, tmp_path):
        """Running pretrain trials get a `pretrain` block with latest val
        metrics and the log-linear forfeit fit."""
        import json
        import math
        runner = TrialRunner(workspace=str(tmp_path))
        run_dir = tmp_path / "run_x"
        run_dir.mkdir()
        # Enough val records with known exponential forfeit decay for a fit.
        k = 1e-5
        records = []
        for i in range(1, 13):
            step = i * 1000
            forfeit = math.exp(-k * step + math.log(0.5))
            records.append({
                "type": "val", "step": step, "val/loss": 3.0,
                "val/game_completion_rate": 1.0 - forfeit,
                "val/avg_plies_completed": 300.0,
                "val/min_forfeit_ply": 20.0, "val/max_forfeit_ply": 400.0,
                "val/median_forfeit_ply": 100.0,
                "val/legal_move_rate": 0.997,
                "val/late_legal_move_rate": 0.993,
            })
        with open(run_dir / "metrics.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        runner.trials[0] = Trial(
            trial_id=0, strategy="base", params={"run_type": "pretrain"},
            cli_command=[], status="running", current_step=12_000,
            total_steps=100_000, pid=1, gpu_id=0,
            config={"run_type": "pretrain", "variant": "base"},
            run_dir=str(run_dir),
        )

        s = runner.status()
        assert len(s["running"]) == 1
        row = s["running"][0]
        assert "pretrain" in row
        pretrain = row["pretrain"]
        assert "latest" in pretrain
        assert pretrain["latest"]["step"] == 12_000
        assert "forfeit_fit" in pretrain
        assert pretrain["forfeit_fit"]["slope_per_step"] == pytest.approx(-k, rel=1e-6)

    def test_status_omits_pretrain_block_for_adapter_runs(self, tmp_path):
        """Adapter runs don't get a pretrain block even if they log val records."""
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="lora", params={"run_type": "adapter"},
            cli_command=[], status="running",
            config={"run_type": "adapter", "strategy": "lora"},
            pid=1, gpu_id=0,
        )
        s = runner.status()
        assert len(s["running"]) == 1
        assert "pretrain" not in s["running"][0]


# =====================================================================
# results()
# =====================================================================


class TestResults:
    def _make_completed(self, trial_id, params, val_loss, strategy="lora", tags=None):
        t = Trial(
            trial_id=trial_id, strategy=strategy, params={},
            cli_command=[], status="completed",
            config={"lr": 1e-3, **{k: v for k, v in params.items()}},
            best_val_loss=val_loss,
            actual_param_count=params.get("param_count", 10_000),
            tags=tags or [],
        )
        return t

    def test_empty_results(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        r = runner.results()
        assert r["trials"] == []
        assert r["pareto_front"] == []
        assert r["suggestions"] == []

    def test_results_sorted_by_trial_id(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[2] = self._make_completed(2, {"lora_rank": 4}, 1.0)
        runner.trials[0] = self._make_completed(0, {"lora_rank": 2}, 1.5)
        runner.trials[1] = self._make_completed(1, {"lora_rank": 8}, 1.2)
        r = runner.results()
        ids = [row["trial"] for row in r["trials"]]
        assert ids == [0, 1, 2]

    def test_results_tag_filter(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = self._make_completed(0, {}, 1.0, tags=["p1"])
        runner.trials[1] = self._make_completed(1, {}, 1.2, tags=["p2"])
        runner.trials[2] = self._make_completed(2, {}, 1.5, tags=["p1", "p2"])
        r = runner.results(tag="p1")
        ids = sorted(row["trial"] for row in r["trials"])
        assert ids == [0, 2]

    def test_results_pareto_front(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        # trial 0: 1000 params, val_loss 2.0
        runner.trials[0] = self._make_completed(0, {"param_count": 1000}, 2.0)
        # trial 1: 2000 params, val_loss 1.8
        runner.trials[1] = self._make_completed(1, {"param_count": 2000}, 1.8)
        # trial 2: 2000 params, val_loss 2.2 (dominated by 1)
        runner.trials[2] = self._make_completed(2, {"param_count": 2000}, 2.2)
        # trial 3: 500 params, val_loss 3.0 (not dominated: fewest params)
        runner.trials[3] = self._make_completed(3, {"param_count": 500}, 3.0)
        # trial 4: 3000 params, val_loss 3.0 (dominated by 1: more params, worse loss)
        runner.trials[4] = self._make_completed(4, {"param_count": 3000}, 3.0)
        r = runner.results()
        all_completed = [row for row in r["trials"]
                         if row["status"] == "completed"
                         and row["val_loss"] is not None
                         and row["params"] is not None]
        pareto = r["pareto_front"]
        pareto_ids = {row["trial"] for row in pareto}
        non_pareto_ids = {row["trial"] for row in all_completed} - pareto_ids

        # Verify non-domination: for every Pareto-optimal trial, no other
        # completed trial dominates it (both strictly better on at least one axis)
        for p_row in pareto:
            for other in all_completed:
                if other["trial"] == p_row["trial"]:
                    continue
                both_le = (other["params"] <= p_row["params"]
                           and other["val_loss"] <= p_row["val_loss"])
                one_strict = (other["params"] < p_row["params"]
                              or other["val_loss"] < p_row["val_loss"])
                assert not (both_le and one_strict), (
                    f"Pareto trial {p_row['trial']} is dominated by trial {other['trial']}"
                )

        # Verify every non-Pareto trial IS dominated by at least one other trial
        for np_id in non_pareto_ids:
            np_row = next(r for r in all_completed if r["trial"] == np_id)
            dominated = False
            for other in all_completed:
                if other["trial"] == np_id:
                    continue
                if (other["params"] <= np_row["params"]
                        and other["val_loss"] <= np_row["val_loss"]
                        and (other["params"] < np_row["params"]
                             or other["val_loss"] < np_row["val_loss"])):
                    dominated = True
                    break
            assert dominated, (
                f"Non-Pareto trial {np_id} is not dominated by any other trial"
            )

        # Spot-check expected membership
        assert 0 in pareto_ids
        assert 1 in pareto_ids
        assert 3 in pareto_ids
        assert 2 not in pareto_ids
        assert 4 not in pareto_ids


# =====================================================================
# trial_log / add_notes
# =====================================================================


class TestTrialLog:
    def test_trial_log_missing_trial(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        r = runner.trial_log(999)
        assert "error" in r

    def test_trial_log_missing_file(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[],
            log_path=str(tmp_path / "nonexistent.log"),
        )
        r = runner.trial_log(0)
        assert "error" in r

    def test_trial_log_returns_last_n_lines(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        log_path = tmp_path / "log.txt"
        log_path.write_text("\n".join(f"line {i}" for i in range(100)))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[],
            log_path=str(log_path),
        )
        r = runner.trial_log(0, lines=5)
        assert r["trial"] == 0
        assert len(r["lines"]) == 5
        assert r["lines"][-1] == "line 99"


class TestAddNotes:
    def test_add_notes_updates_trial(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(trial_id=0, strategy="x", params={}, cli_command=[])
        r = runner.add_notes(0, "cool trial")
        assert r["ok"] is True
        assert runner.trials[0].notes == "cool trial"

    def test_add_notes_missing_trial(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        r = runner.add_notes(999, "foo")
        assert "error" in r


# =====================================================================
# kill()
# =====================================================================


class TestKill:
    def test_kill_missing_trial(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        r = asyncio.run(runner.kill(999))
        assert "error" in r

    def test_kill_not_running(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[], status="completed",
        )
        r = asyncio.run(runner.kill(0))
        assert "error" in r

    def test_kill_sends_sigterm(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[], status="running",
            pid=99999,
        )
        with patch("os.kill") as mock_kill:
            r = asyncio.run(runner.kill(0))
        assert runner.trials[0].status == "killed"
        assert "killed" in r
        assert r["killed"] == 0
        mock_kill.assert_called_once()
        # Signal must be SIGTERM
        import signal
        assert mock_kill.call_args[0][1] == signal.SIGTERM

    def test_kill_handles_process_already_gone(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[], status="running",
            pid=99999,
        )
        with patch("os.kill", side_effect=ProcessLookupError):
            r = asyncio.run(runner.kill(0))
        # Should not raise
        assert runner.trials[0].status == "killed"


# =====================================================================
# render_progress_log
# =====================================================================


class TestRenderProgressLog:
    def test_render_writes_file(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.gpu_count = 2
        runner.gpu_names = ["A100", "A100"]
        runner.gpu_vram_mb = [80000, 80000]
        content = runner.render_progress_log()
        assert runner.progress_log_path.exists()
        assert "Pod Manager Log" in content

    def test_render_shows_environment(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.gpu_count = 2
        runner.gpu_names = ["A100", "A100"]
        runner.gpu_vram_mb = [80000, 80000]
        content = runner.render_progress_log()
        assert "A100" in content
        assert "80000" in content

    def test_render_shows_running_trials(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            status="running", current_step=50, total_steps=100,
            steps_per_sec=5.0, pid=1234, gpu_id=0,
        )
        content = runner.render_progress_log()
        assert "Active Processes" in content
        assert "lora" in content

    def test_render_shows_completed(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            status="completed", best_val_loss=0.5, best_accuracy=0.9,
            actual_param_count=100_000, notes="first run",
        )
        content = runner.render_progress_log()
        assert "Results" in content
        assert "first run" in content


# =====================================================================
# recover()
# =====================================================================


class TestRecover:
    def test_recover_empty_state(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        asyncio.run(runner.recover())
        assert runner.trials == {}

    def test_recover_marks_dead_trial_failed(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[],
            status="running", pid=99999,  # likely dead
            best_val_loss=None,
        )
        runner._save_state()
        # Create a fresh runner and recover
        runner2 = TrialRunner(workspace=str(tmp_path))
        with patch("pawn.lab.runner.is_alive", return_value=(False, None)):
            asyncio.run(runner2.recover())
        assert runner2.trials[0].status == "failed"

    def test_recover_marks_dead_trial_completed_if_has_val(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.trials[0] = Trial(
            trial_id=0, strategy="x", params={}, cli_command=[],
            status="running", pid=99999,
            best_val_loss=0.5,
        )
        runner._save_state()
        runner2 = TrialRunner(workspace=str(tmp_path))
        with patch("pawn.lab.runner.is_alive", return_value=(False, None)):
            asyncio.run(runner2.recover())
        assert runner2.trials[0].status == "completed"

    def test_recover_reads_events(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner._emit("a")
        runner._emit("b")
        runner._save_state()
        runner2 = TrialRunner(workspace=str(tmp_path))
        asyncio.run(runner2.recover())
        assert len(runner2.events) == 2
        assert runner2.event_seq == 2


# =====================================================================
# shutdown()
# =====================================================================


class TestShutdown:
    def test_shutdown_saves_state(self, tmp_path):
        runner = TrialRunner(workspace=str(tmp_path))
        runner.shutdown()
        assert runner.state_path.exists()
