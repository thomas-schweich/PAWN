"""Lab trial runner — pydantic-validated dispatch into the JAX trainer.

The runner is intentionally thin: it validates the incoming trial
config through pydantic and dispatches to the appropriate training
script. The plan §10 S9 contract is:

- ``validate_config(config: dict, run_type: str) -> RunConfig`` —
  raise `ValueError` (pydantic's) on stale/typo'd fields.
- ``lab_schema()`` — return the JSON Schema of every supported
  run_type. Clients use this to render forms / pre-validate.
- ``lab_launch(config: dict)`` — validate + dispatch.

The full subprocess lifecycle (start/monitor/kill) is owned by
:mod:`pawn.lab.server` (which wraps this in FastMCP tool handlers).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pawn.run_config import (
    AdapterConfig,
    DistillConfig,
    PretrainConfig,
    SpecializedCLMConfig,
)

__all__ = [
    "lab_launch",
    "lab_schema",
    "validate_config",
    "read_schedule_health",
    "audit_schedule_health",
]


def lab_schema() -> dict[str, Any]:
    """Return JSON Schema for every supported run_type.

    The lab client uses this to render trial forms and pre-validate
    user input. The dict shape is::

        {
            "pretrain": {<JSON Schema dict>},
            "adapter": {<JSON Schema dict>},
            "specialized_clm": {<JSON Schema dict>},
            "distill": {<JSON Schema dict>},
        }
    """
    return {
        "pretrain": PretrainConfig.model_json_schema(),
        "adapter": AdapterConfig.model_json_schema(),
        "specialized_clm": SpecializedCLMConfig.model_json_schema(),
        "distill": DistillConfig.model_json_schema(),
    }


def validate_config(config: Mapping[str, Any]) -> Any:
    """Dispatch on ``config["run_type"]`` and validate through pydantic.

    `extra="forbid"` rejects unknown / stale field names at the lab
    boundary — the v1 contract per plan §10 S9. Returns the validated
    Config instance.
    """
    run_type = config.get("run_type")
    if run_type is None:
        raise ValueError("missing 'run_type' in trial config")
    config = dict(config)
    if run_type == "pretrain":
        return PretrainConfig(**config)
    if run_type == "adapter":
        return AdapterConfig(**config)
    if run_type == "specialized_clm":
        return SpecializedCLMConfig(**config)
    if run_type == "distill":
        return DistillConfig(**config)
    raise ValueError(
        f"unknown run_type {run_type!r}; valid: pretrain / adapter / "
        f"specialized_clm / distill"
    )


def read_schedule_health(run_dir: Path | str) -> dict[str, Any] | None:
    """Read a run's ``schedule_health.json`` (H7) or ``None`` if absent.

    The trainers write this file at every exit path; the lab reads it to
    answer "did the LR schedule run to completion?" post-hoc without
    replaying the run.
    """
    path = Path(run_dir) / "schedule_health.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def audit_schedule_health(run_dir: Path | str) -> dict[str, Any]:
    """Audit a run's schedule health and flag the structural-bug signal.

    Returns ``{"present": bool, "structural_mismatch": bool,
    "banner": str | None, "health": dict | None}``. The structural-bug
    signal (plan §8.3 H7) is ``actual_total_steps != planned_total_steps``
    AND ``reason_for_stop == "completed"``: the loop fell off the end of
    the schedule without an early-exit, yet the step counts disagree.
    With cache-first that's unreachable, so a True here is a regression
    tripwire and gets a banner the lab surfaces to the operator. SIGTERM /
    patience / pause / resume_no_op are legitimate early exits and do NOT
    raise the flag.
    """
    health = read_schedule_health(run_dir)
    if health is None:
        return {
            "present": False,
            "structural_mismatch": False,
            "banner": None,
            "health": None,
        }
    planned = health.get("planned_total_steps")
    actual = health.get("actual_total_steps")
    reason = health.get("reason_for_stop")
    mismatch = (
        reason == "completed"
        and isinstance(planned, int)
        and isinstance(actual, int)
        and actual != planned
    )
    banner: str | None = None
    if mismatch:
        banner = (
            "\033[31m[lab] STRUCTURAL MISMATCH: schedule reported "
            f"reason_for_stop='completed' but actual_total_steps={actual} "
            f"!= planned_total_steps={planned}. The training loop fell off "
            "the end of the LR schedule without an early-exit yet the step "
            "counts disagree — this is a structural bug, not a normal early "
            "stop.\033[0m"
        )
    return {
        "present": True,
        "structural_mismatch": mismatch,
        "banner": banner,
        "health": health,
    }


def lab_launch(
    config: Mapping[str, Any], *, dry_run: bool = False
) -> dict[str, Any]:
    """Validate ``config`` and dispatch to the appropriate trainer script.

    Returns a dict ``{"status": "launched"|"validated", "config": ..., "pid": int|None}``.
    ``dry_run=True`` validates the config but doesn't actually spawn
    the subprocess — useful for the lab's pre-flight smoke check.
    """
    validated = validate_config(config)
    run_type = config["run_type"]
    if dry_run:
        return {
            "status": "validated",
            "run_type": run_type,
            "config": validated.model_dump(),
            "pid": None,
        }

    # Dispatch table: run_type → script entry point.
    script_map = {
        "pretrain": "scripts/train_jax.py",
        "adapter": "scripts/train_jax_adapter.py",
        "specialized_clm": "scripts/train_jax_adapter.py",
        "distill": "scripts/train_jax_distill.py",
    }
    script = script_map[run_type]
    # Spawn (the script reads its own --config JSON; we hand it the
    # validated config as a stdin-compatible JSON file). The runner
    # detaches; lab monitoring picks up metrics via the run-dir.
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(validated.model_dump(), f)
        cfg_path = f.name
    proc = subprocess.Popen(
        ["python", script, "--config", cfg_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "launched",
        "run_type": run_type,
        "config": validated.model_dump(),
        "pid": proc.pid,
    }
