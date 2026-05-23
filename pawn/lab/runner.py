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

import subprocess
from collections.abc import Mapping
from typing import Any

from pawn.run_config import (
    AdapterConfig,
    PretrainConfig,
    SpecializedCLMConfig,
)

__all__ = [
    "lab_launch",
    "lab_schema",
    "validate_config",
]


def lab_schema() -> dict[str, Any]:
    """Return JSON Schema for every supported run_type.

    The lab client uses this to render trial forms and pre-validate
    user input. The dict shape is::

        {
            "pretrain": {<JSON Schema dict>},
            "adapter": {<JSON Schema dict>},
            "specialized_clm": {<JSON Schema dict>},
        }
    """
    return {
        "pretrain": PretrainConfig.model_json_schema(),
        "adapter": AdapterConfig.model_json_schema(),
        "specialized_clm": SpecializedCLMConfig.model_json_schema(),
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
    raise ValueError(
        f"unknown run_type {run_type!r}; valid: pretrain / adapter / "
        f"specialized_clm"
    )


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
