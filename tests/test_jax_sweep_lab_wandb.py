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
    AdapterObjective,
    _read_best_val_loss,
    adapter_strategy_for,
    params_to_config_json,
    suggest_lora,
    suggest_rosa,
    suggest_rosa_retro_bottleneck,
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
    # `lora_targets` is part of the v1 RoSA search space — the LoRA warmup
    # must be free to adapt a non-default projection subset.
    assert "lora_targets" in params
    assert params["lora_targets"] in ("qkvo", "qv", "qkv")


def test_suggest_rosa_retro_bottleneck_sweeps_bottleneck_axes() -> None:
    """v1 ``suggest_retro_bottleneck`` extends the shared RoSA space with the
    Houlsby width *and* the extra-stage depth; the prior v2 code locked both
    to their RoSAConfig defaults. Both must now appear in the search space."""
    study = optuna.create_study()
    params = suggest_rosa_retro_bottleneck(study.ask())
    assert params["rosa_mode"] == "retro-bottleneck"
    assert "bottleneck_dim" in params
    assert params["bottleneck_dim"] in (4, 8, 16)
    assert "bottleneck_n_hidden" in params
    assert params["bottleneck_n_hidden"] in (0, 1, 2)
    # The shared RoSA axes are still present (it builds on suggest_rosa).
    assert "lora_targets" in params
    assert "density" in params


def test_params_to_config_json_preserves_native_types() -> None:
    """H8: suggested params round-trip through a JSON `--config` body as
    native types (bool stays bool, int stays int) — not kebab CLI flags
    the adapter argparse never registered. The body carries `run_type` and
    the resolved adapter `strategy`."""
    body = params_to_config_json(
        "bottleneck",
        {"lora_rank": 4, "use_output_film": True, "no_adapt_attn": False},
    )
    assert body["run_type"] == "adapter"
    assert body["strategy"] == "bottleneck"
    assert body["lora_rank"] == 4
    assert body["use_output_film"] is True
    assert body["no_adapt_attn"] is False  # native False preserved, not dropped
    # The body must be JSON-serialisable (this is what gets written to
    # `--config`); native bool/int survive the round-trip unchanged.
    assert json.loads(json.dumps(body)) == body


def test_rosa_ratio_maps_to_consumed_rosa_strategy() -> None:
    """H9: `rosa-ratio` is a sweep-only key with no `--strategy` of its
    own. `adapter_strategy_for` resolves it to the real `rosa` strategy,
    and its suggester emits only consumed `AdapterConfig` fields (a concrete
    `bottleneck_dim`, never the unconsumed `bottleneck_ratio` key that the
    old code emitted and `extra=forbid` rejected)."""
    assert adapter_strategy_for("rosa-ratio") == "rosa"
    # Non-alias strategies pass through unchanged.
    assert adapter_strategy_for("bottleneck") == "bottleneck"
    study = optuna.create_study()
    params = STRATEGY_SUGGESTERS["rosa-ratio"](study.ask())
    assert "bottleneck_ratio" not in params  # raw ratio is not a config field
    assert params["rosa_mode"] == "retro-bottleneck"
    assert isinstance(params["bottleneck_dim"], int)
    assert params["bottleneck_dim"] >= 1


def _load_adapter_script():
    """Import `scripts/train_jax_adapter.py` by path (scripts/ isn't a
    package). Imports JAX at module load — used only by the argparse +
    pydantic acceptance test and the GPU sweep smokes below."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_adapter_accept", Path("scripts/train_jax_adapter.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("sweep_strategy", sorted(STRATEGY_SUGGESTERS))
def test_suggested_params_accepted_by_adapter_argparse_and_pydantic(
    sweep_strategy: str, tmp_path: Path
) -> None:
    """H8/H9 acceptance gate: every `STRATEGY_SUGGESTERS` entry's suggested
    params, serialized to a `--config` JSON exactly as `AdapterObjective`
    does, are accepted by `train_jax_adapter.py`'s argparse + `AdapterConfig`
    pydantic boundary — no `SystemExit` (argparse exit-2), no
    `pydantic.ValidationError` (`extra=forbid` / required-field).

    This is the previously-broken path: kebab CLI flags for
    `bottleneck_n_hidden` / `sparse_targets` / `rosa_*` / `mask_samples` /
    `grad_alpha` were never registered (exit-2), and `bottleneck_ratio` was
    not a config field (`extra=forbid`). The JSON `--config` round-trip plus
    the `rosa-ratio`→`rosa` alias fix both regimes.
    """
    adapter = _load_adapter_script()
    study = optuna.create_study()
    # Mirror AdapterObjective: suggest params, render the config body, write
    # it to disk, and build the argv the objective would run.
    params = STRATEGY_SUGGESTERS[sweep_strategy](study.ask(), n_layers=4)
    body = params_to_config_json(sweep_strategy, params)
    config_path = tmp_path / "trial_config.json"
    config_path.write_text(json.dumps(body))
    argv = [
        "--config", str(config_path),
        "--strategy", adapter_strategy_for(sweep_strategy),
        "--logs-dir", str(tmp_path / "logs"),
        # base_args the sweep CLI supplies (scripts/sweep.py); these don't
        # collide with the suggested params and satisfy the required
        # total_steps / checkpoint-mode pydantic gates.
        "--supernet", "tiny", "--variant", "base",
        "--total-steps", "10", "--log-interval", "2",
        "--no-pgn", "--batch-size", "8", "--seq-len", "32", "--k", "5",
        "--local-checkpoints",
    ]
    # Must not raise SystemExit (argparse) or ValidationError (pydantic).
    args = adapter._parse_args(argv)
    cfg = adapter._build_config(args)
    # The resolved strategy must be a real adapter strategy, and the config
    # must carry the suggested values (spot-check the non-default knobs).
    from pawn.adapter_trainer import STRATEGIES

    assert cfg.strategy in STRATEGIES
    assert cfg.strategy == adapter_strategy_for(sweep_strategy)
    for key, val in params.items():
        # `rosa_mode` is the sub-mode selector; every other suggested key is
        # a direct AdapterConfig field that must survive the round-trip.
        assert getattr(cfg, key) == val, (
            f"{sweep_strategy}: config dropped suggested {key}={val!r}"
        )


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
# Sweep — end-to-end tiny-trial smoke (subprocess per trial; GPU-only)
# ---------------------------------------------------------------------------


def _gpu_only() -> None:
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("sweep subprocess trains via train_jax_adapter (GPU-only)")


def _local_tiny_backbone(ckpt_dir: Path) -> Path:
    """Persist a TINY_SUPERNET-shaped local backbone checkpoint so the sweep
    smokes don't depend on the (unpublished) `pawn-base-v2` HF repo — the
    spec's "local backbone" smoke setup."""
    from pawn.checkpoint import save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    save_model(
        backbone, ckpt_dir, training_state={"step": 0},
        run_config={"conditioning": []},
    )
    return ckpt_dir


def _run_tiny_sweep(strategy: str, logs_dir: Path, n_trials: int = 2) -> float:
    """Drive a tiny in-process Optuna sweep through `AdapterObjective`, which
    subprocesses `scripts/train_jax_adapter.py` per trial. Returns
    `study.best_value` — a finite value proves the previously-broken
    argparse/pydantic path now produces non-pruned trials (no all-prune)."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _local_tiny_backbone(logs_dir / "backbone")
    study = optuna.create_study(direction="minimize")
    base_args = [
        "--supernet", "tiny", "--variant", "small",
        "--checkpoint", str(ckpt),
        "--total-steps", "10", "--log-interval", "2",
        "--no-pgn", "--batch-size", "8", "--seq-len", "32", "--k", "5",
        "--local-checkpoints",
    ]
    obj = AdapterObjective(
        strategy=strategy, base_args=base_args, logs_dir=logs_dir, n_layers=4,
    )
    study.optimize(obj, n_trials=n_trials)
    return study.best_value


@pytest.mark.parametrize(
    "strategy", ["bottleneck", "sparse", "rosa", "rosa-ratio"]
)
def test_tiny_sweep_yields_finite_best_value(
    strategy: str, tmp_path: Path
) -> None:
    """E1 smoke: a 2-trial sweep for each previously-broken strategy
    (`bottleneck`/`sparse` had unregistered kebab flags; a `rosa` sub-mode
    had unregistered `rosa_*` flags; `rosa-ratio` emitted a non-field key)
    completes with a finite `best_value` rather than 100% pruned (H8/H9)."""
    _gpu_only()
    best = _run_tiny_sweep(strategy, tmp_path / "sweep", n_trials=2)
    assert best != float("inf")
    import math

    assert math.isfinite(best), f"{strategy}: best_value not finite: {best}"


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


def test_init_wandb_forwards_job_type_group_and_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`init_wandb` exposes the v1 `job_type` / `group` / run-name knobs so
    a single project can separate pretrain vs adapter runs and resumed
    siblings join one group. `run_dir_name` overrides the W&B run name
    (v1 used `logger.run_dir.name`); `group` defaults to the slug."""
    import sys
    import types

    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    fake_run = mock.MagicMock(name="wandb_run")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = mock.MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    run = init_wandb(
        project="pawn", slug="bold-fox", run_config={"lr": 1e-3},
        git_hash="deadbeef", enabled=True,
        job_type="adapter", group="sweep-42",
        run_dir_name="lora_20260530_000000_000000_bold-fox",
    )
    assert run is fake_run
    _, kwargs = fake_wandb.init.call_args  # type: ignore[attr-defined]
    assert kwargs["name"] == "lora_20260530_000000_000000_bold-fox"
    assert kwargs["group"] == "sweep-42"
    assert kwargs["job_type"] == "adapter"
    assert "job_type:adapter" in kwargs["tags"]
    assert "git:deadbeef" in kwargs["tags"]


def test_init_wandb_group_defaults_to_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit `group`, resumed sibling processes still join
    one group via the slug (v1 Option-A resume)."""
    import sys
    import types

    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    fake_run = mock.MagicMock(name="wandb_run")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = mock.MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    init_wandb(project="pawn", slug="bold-fox", run_config={}, enabled=True)
    _, kwargs = fake_wandb.init.call_args  # type: ignore[attr-defined]
    assert kwargs["group"] == "bold-fox"
    assert kwargs["name"] == "bold-fox"


def test_finish_wandb_forwards_exit_code() -> None:
    """`finish_wandb` records a non-zero exit code so a crashed / SIGTERM'd
    run surfaces as failed in the W&B UI (v1 parity)."""
    fake_run = mock.MagicMock(name="wandb_run")
    finish_wandb(fake_run, exit_code=1)
    fake_run.finish.assert_called_once_with(exit_code=1)


def test_finish_wandb_default_exit_code_zero() -> None:
    fake_run = mock.MagicMock(name="wandb_run")
    finish_wandb(fake_run)
    fake_run.finish.assert_called_once_with(exit_code=0)


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
