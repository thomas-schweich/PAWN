"""Smoke tests for scripts/train_jax_adapter.py multi-epoch cadence.

Regression guard for the schedule-sizing blocker: ``train_schedule`` must be
sized by the *effective* step budget (``epochs * steps_per_epoch``), not
``cfg.total_steps`` (the LR-schedule decay timeline). When
``effective_total_steps > cfg.total_steps`` the undersized schedule made
``_gather_chunk`` slice past the end of the index stream, so
``.reshape(n, batch_size)`` raised ``ValueError: cannot reshape`` partway
through the run.

The cadence knobs (``epochs`` / ``steps_per_epoch`` / ``checkpoint_interval``)
have no dedicated CLI flag on ``train_jax_adapter.py`` — they are set via a
``--config`` JSON (the same mechanism
``test_adapter_patience_early_stops_and_writes_best_step`` uses). These tests
drive ``main()`` in-process against a *local* TINY_SUPERNET backbone with
``--no-pgn`` (random-game corpus, no HF dependency) so they run self-contained
on the active GPU venv.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_train_jax_adapter():  # type: ignore[no-untyped-def]
    """Import ``scripts/train_jax_adapter.py`` as a module to reach ``main``."""
    script_path = Path("scripts") / "train_jax_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_adapter_smoke", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _local_backbone(tmp_path: Path) -> Path:
    """Save a tiny loadable v2 backbone checkpoint (C=1) under ``tmp_path``."""
    from pawn.checkpoint import save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    ckpt_dir = tmp_path / "backbone"
    save_model(backbone, ckpt_dir, training_state={"step": 0}, run_config={})
    return ckpt_dir


def test_adapter_multi_epoch_runs_full_budget(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A multi-epoch run where ``effective_total_steps > total_steps`` drives
    every step to completion without an IndexError / reshape failure.

    Regression for the schedule-sizing blocker: with ``epochs=5``,
    ``steps_per_epoch=4``, ``total_steps=4`` the loop drives 20 steps but the
    old schedule held only ``total_steps * batch_size`` indices, so once
    ``_gather_chunk``'s ``flat_start`` passed ``total_steps * batch_size`` the
    numpy slice came back short and ``.reshape(n, batch_size)`` raised
    ``ValueError: cannot reshape``. Sizing by ``cadence.effective_total_steps``
    fixes it. The existing patience smoke sets ``epochs * steps_per_epoch ==
    total_steps`` so it never exposed this.

    An *integer* ``steps_per_epoch`` makes the effective budget independent of
    the random-game corpus size, so the expected final step is exactly 20.
    Asserts ``main()`` returns 0 (ran to completion), the step-20 checkpoint
    lands, and ``schedule_health.json`` confirms the full 20-step budget ran to
    ``completed`` (``planned > total_steps``).
    """
    backbone = _local_backbone(tmp_path)
    cfg_path = tmp_path / "adapter.json"
    cfg_path.write_text(json.dumps({
        "run_type": "adapter",
        "strategy": "lora",
        "lora_rank": 2,
        "supernet": "tiny",
        "variant": "small",
        "checkpoint": str(backbone),
        "total_steps": 4,        # LR-schedule horizon (< effective budget)
        "epochs": 5,
        "steps_per_epoch": 4,    # effective budget = 5 * 4 = 20 steps
        "checkpoint_interval": 20,
        "batch_size": 2,
        "seq_len": 16,
        "k": 1,
        "lr": 1e-4,
        "local_checkpoints": True,
    }))
    logs_dir = tmp_path / "logs"
    main = _load_train_jax_adapter().main
    rc = main(["--config", str(cfg_path), "--no-pgn", "--logs-dir", str(logs_dir)])
    assert rc == 0, "adapter multi-epoch run did not complete cleanly"
    run_dirs = list(logs_dir.glob("adapter-*"))
    assert run_dirs, "no run directory created"
    run_dir = run_dirs[0]
    ckpt = run_dir / "adapter_step_00000020"
    assert ckpt.is_dir(), (
        "step-20 checkpoint missing — the multi-epoch budget did not run to "
        "completion (the schedule was undersized and the run died mid-epoch)"
    )
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["planned_total_steps"] == 20
    assert health["actual_total_steps"] == 20
    assert health["reason_for_stop"] == "completed"
    # The effective budget genuinely exceeds the LR-schedule horizon — this is
    # the configuration that tripped the undersized-schedule crash.
    assert health["planned_total_steps"] > 4


def test_adapter_steps_per_epoch_all_runs_full_budget(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """``steps_per_epoch="all"`` with ``epochs > 1`` also runs without a
    reshape failure.

    ``"all"`` resolves to ``n_train_games // batch_size`` once the corpus
    materialises; with two epochs ``effective_total_steps`` is twice that,
    again exceeding the small ``total_steps`` the LR schedule is sized to — the
    canonical adapter cadence CLAUDE.md documents. ``--no-pgn`` builds a
    random-game corpus of ``max(batch_size*10, 1000)`` games; with
    ``batch_size=128`` that is 1280 games → ``1280 // 128 = 10`` steps/epoch →
    ``2 * 10 = 20`` effective steps, still > the 4-step LR horizon. Rather than
    hard-code the resolved count, assert via ``schedule_health.json`` that the
    *planned* (effective) budget exceeds ``total_steps`` and that the run
    reached it (``actual == planned``, ``completed``) — proving the schedule
    covered the whole multi-epoch budget.
    """
    backbone = _local_backbone(tmp_path)
    cfg_path = tmp_path / "adapter.json"
    cfg_path.write_text(json.dumps({
        "run_type": "adapter",
        "strategy": "lora",
        "lora_rank": 2,
        "supernet": "tiny",
        "variant": "small",
        "checkpoint": str(backbone),
        "total_steps": 4,           # LR horizon < effective budget
        "epochs": 2,
        "steps_per_epoch": "all",   # 1280 // 128 = 10 → effective = 20 steps
        "checkpoint_interval": 1000,
        "batch_size": 128,
        "seq_len": 16,
        "k": 2,
        "lr": 1e-4,
        "local_checkpoints": True,
    }))
    logs_dir = tmp_path / "logs"
    main = _load_train_jax_adapter().main
    rc = main([
        "--config", str(cfg_path), "--no-pgn", "--logs-dir", str(logs_dir),
    ])
    assert rc == 0, "steps_per_epoch='all' multi-epoch run did not complete"
    run_dirs = list(logs_dir.glob("adapter-*"))
    assert run_dirs, "no run directory created"
    health = json.loads((run_dirs[0] / "schedule_health.json").read_text())
    # The resolved budget is 2 epochs * (1280 // 128) = 20 steps, > total_steps.
    assert health["planned_total_steps"] > 4, (
        "steps_per_epoch='all' did not expand the budget past total_steps"
    )
    assert health["actual_total_steps"] == health["planned_total_steps"], (
        "the run stopped short of the resolved multi-epoch budget — the "
        "schedule was undersized and the loop died mid-run"
    )
    assert health["reason_for_stop"] == "completed"
