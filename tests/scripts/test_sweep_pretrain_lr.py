"""Unit tests for the PAWN-large teacher LR-sweep driver.

These exercise the pure helpers (config build, loss scoring, refinement
grid, command construction) and the orchestration via an injected
``launch_fn`` that writes synthetic ``metrics.jsonl`` files — no JAX, no
GPU, no subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sweep_pretrain_lr",
    Path(__file__).resolve().parent.parent.parent
    / "scripts" / "sweep_pretrain_lr.py",
)
assert _SPEC is not None and _SPEC.loader is not None
swp = importlib.util.module_from_spec(_SPEC)
# Register before exec so the dataclass machinery can resolve stringized
# annotations (`from __future__ import annotations`) via the module's
# namespace in sys.modules.
sys.modules[_SPEC.name] = swp
_SPEC.loader.exec_module(swp)

from pawn.run_config import PretrainConfig  # noqa: E402


# ---------------------------------------------------------------------------
# base_config → valid PretrainConfig
# ---------------------------------------------------------------------------


def test_base_config_validates_as_large_teacher_pretrain() -> None:
    settings = swp.SweepSettings()
    payload = base = swp.base_config(settings)
    # lr + total_steps are supplied per-trial on the CLI; add them here to
    # validate the merged config the way train_jax would.
    cfg = PretrainConfig(**{**payload, "lr": 2e-4, "total_steps": 1500})
    assert cfg.variants == ("large",)
    assert cfg.lr_schedule == "infinite"
    assert cfg.stable_lr_ratio == 0.5
    assert cfg.batch_size == 64
    assert cfg.accumulation_steps == 1
    assert cfg.local_checkpoints is True
    # checkpoint_interval is above the proxy budget → only the final save.
    assert cfg.checkpoint_interval > 1500
    # base_config deliberately omits lr / total_steps (CLI supplies them).
    assert "lr" not in base
    assert "total_steps" not in base


def test_base_config_respects_settings_overrides() -> None:
    settings = swp.SweepSettings(
        batch_size=32, stable_lr_ratio=0.75, conditioning=("outcome",),
        cooldown_frac=0.3,
    )
    cfg = PretrainConfig(
        **{**swp.base_config(settings), "lr": 3e-4, "total_steps": 800}
    )
    assert cfg.batch_size == 32
    assert cfg.stable_lr_ratio == 0.75
    assert cfg.cooldown_frac == 0.3
    assert list(cfg.conditioning) == ["outcome"]


# ---------------------------------------------------------------------------
# trial_command
# ---------------------------------------------------------------------------


def test_trial_command_shape(tmp_path: Path) -> None:
    cmd = swp.trial_command(tmp_path / "cfg.json", 2e-4, 1500, tmp_path / "t")
    assert "--config" in cmd
    assert "--lr" in cmd and "0.0002" in cmd
    assert cmd[cmd.index("--total-steps") + 1] == "1500"
    assert "--local-checkpoints" in cmd
    assert cmd[cmd.index("--logs-dir") + 1] == str(tmp_path / "t")
    # Routes through the in-repo train_jax entry point.
    assert cmd[1].endswith("train_jax.py")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_score_picks_smoothed_floor() -> None:
    # Loss decays from 7 to a 3.0 plateau then noise.
    rows = [(i * 20, 7.0 - 4.0 * min(1.0, i / 30) + (0.02 if i % 2 else -0.02))
            for i in range(1, 61)]
    res = swp.score_losses(rows, 2e-4, stable_lr_ratio=0.5, smooth_window=10)
    assert not res.diverged
    assert res.plateau_lr == pytest.approx(1e-4)
    assert res.smoothed_min == pytest.approx(3.0, abs=0.1)
    assert res.n_records == 60


def test_score_flags_nonfinite_as_diverged() -> None:
    rows = [(20, 7.0), (40, 6.5), (60, None), (80, float("nan"))]
    res = swp.score_losses(rows, 1e-3, stable_lr_ratio=0.5, smooth_window=3)
    assert res.diverged
    assert res.n_nonfinite == 2


def test_score_flags_rising_loss_as_diverged() -> None:
    rows = [(i * 20, 4.0 + 0.05 * i) for i in range(1, 41)]  # monotonic rise
    res = swp.score_losses(rows, 1e-3, stable_lr_ratio=0.5, smooth_window=5)
    assert res.diverged


def test_score_empty_is_diverged_inf() -> None:
    res = swp.score_losses([], 5e-4, stable_lr_ratio=0.5, smooth_window=5)
    assert res.diverged
    assert math.isinf(res.smoothed_min)
    assert res.n_records == 0


# ---------------------------------------------------------------------------
# refinement grid
# ---------------------------------------------------------------------------


def test_refine_interior_winner_geometric_means() -> None:
    grid = [6e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
    pts, edge = swp.refine_grid(grid, 2e-4, n_points=2)
    assert not edge
    # geomean(1e-4, 2e-4) and geomean(2e-4, 3e-4)
    assert pts[0] == pytest.approx(math.sqrt(1e-4 * 2e-4))
    assert pts[1] == pytest.approx(math.sqrt(2e-4 * 3e-4))
    # None coincide with existing grid points.
    assert all(p not in set(grid) for p in pts)


def test_refine_endpoint_winner_extends_and_flags_edge() -> None:
    grid = [6e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
    pts, edge = swp.refine_grid(grid, 1e-3, n_points=2)
    assert edge
    # One geomean inside (toward 5e-4), one extension above 1e-3.
    assert any(p > 1e-3 for p in pts)
    assert any(5e-4 < p < 1e-3 for p in pts)


def test_refine_winner_not_in_grid_raises() -> None:
    with pytest.raises(ValueError):
        swp.refine_grid([1e-4, 2e-4], 7e-4, n_points=2)


# ---------------------------------------------------------------------------
# orchestration with an injected launcher
# ---------------------------------------------------------------------------


def _fake_launcher(best_lr: float, floor_at_best: float = 2.5):
    """Return a launch_fn that writes a synthetic metrics.jsonl per trial.

    The loss floor is a parabola in log-LR minimised at ``best_lr``, so the
    sweep should select ``best_lr`` (or the refinement point nearest it).
    """
    def launch(cmd: list[str]) -> None:
        lr = float(cmd[cmd.index("--lr") + 1])
        logs_dir = Path(cmd[cmd.index("--logs-dir") + 1])
        run_dir = logs_dir / "run-slug"
        run_dir.mkdir(parents=True, exist_ok=True)
        gap = (math.log10(lr) - math.log10(best_lr)) ** 2
        floor = floor_at_best + 1.5 * gap
        lines = []
        for i in range(1, 61):
            loss = floor + 4.0 * math.exp(-i / 12.0)  # decay to floor
            lines.append(json.dumps(
                {"type": "train", "step": i * 20, "loss": loss}
            ))
        (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    return launch


def test_run_sweep_selects_best_and_refines(tmp_path: Path) -> None:
    settings = swp.SweepSettings(steps=1500, smooth_window=8)
    grid = [6e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
    report = swp.run_sweep(
        settings, grid, refine=True, refine_points=2,
        logs_root=tmp_path, launch_fn=_fake_launcher(2e-4),
        progress=lambda _msg: None,
    )
    assert len(report.wide) == 6
    assert len(report.refined) >= 1  # interior winner → 2 geomeans
    best = report.best
    assert best is not None
    # Winner is the true minimum or a refinement point adjacent to it.
    assert best.lr == pytest.approx(2e-4, rel=0.6)
    assert best.smoothed_min <= min(r.smoothed_min for r in report.wide)
    # The base config was written.
    assert (tmp_path / "base_config.json").is_file()


def test_run_sweep_all_diverged_returns_no_best(tmp_path: Path) -> None:
    def nan_launcher(cmd: list[str]) -> None:
        logs_dir = Path(cmd[cmd.index("--logs-dir") + 1])
        run_dir = logs_dir / "run-slug"
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"type": "train", "step": i * 20, "loss": None})
                 for i in range(1, 11)]
        (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    report = swp.run_sweep(
        swp.SweepSettings(), [1e-4, 1e-3], refine=True, refine_points=2,
        logs_root=tmp_path, launch_fn=nan_launcher, progress=lambda _m: None,
    )
    assert report.best is None
    assert report.refined == []  # no winner → no refinement pass
    table = swp.format_table(report)
    assert "NO USABLE TRIAL" in table


def test_format_table_and_report_json(tmp_path: Path) -> None:
    settings = swp.SweepSettings(steps=1500)
    report = swp.run_sweep(
        settings, [1e-4, 2e-4, 3e-4], refine=False, refine_points=2,
        logs_root=tmp_path, launch_fn=_fake_launcher(2e-4),
        progress=lambda _m: None,
    )
    table = swp.format_table(report)
    assert "BEST peak_lr" in table
    assert "train_jax.py --variants large" in table
    swp.write_report_json(report, tmp_path / "report.json")
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["best"]["lr"] == pytest.approx(2e-4, rel=0.6)
    assert len(payload["trials"]) == 3
