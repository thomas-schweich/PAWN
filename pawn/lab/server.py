"""MCP server for pawn-lab: exposes the trial runner as tools via FastMCP.

The server is a thin wrapper around :mod:`pawn.lab.runner`. A single
:class:`~pawn.lab.runner.TrialRunner` is created per server lifespan and shared
across tool calls via the FastMCP request context. Run via
``python -m pawn.lab``.

Every tool's inputs are validated by FastMCP against the JSON Schema it derives
from the tool signature (unknown fields rejected), and config-bearing tools
(``lab_launch`` / ``lab_schema``) additionally round-trip through the pydantic
``RunConfig`` models, which forbid extra fields.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Context, FastMCP

from pawn.lab.runner import TrialRunner

log = logging.getLogger("pawn.lab")


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    runner = TrialRunner()
    await runner.recover()
    log.info("pawn-lab MCP server starting (workspace=%s)", runner.workspace)
    yield {"runner": runner}
    runner.shutdown()


mcp: FastMCP = FastMCP("pawn-lab", lifespan=_lifespan)


def _runner(ctx: Context) -> TrialRunner:
    runner = ctx.lifespan_context["runner"]
    assert isinstance(runner, TrialRunner)
    return runner


# -----------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------


@mcp.tool
async def lab_status(ctx: Context) -> dict[str, Any]:
    """Compact lab status: running trials, counts, elapsed time, cost.

    Returns total/running/completed/failed trial counts. Running trials list
    their id, strategy, status, current step, and configured total steps. Use
    ``lab_log`` for real-time stdout from an individual trial.
    """
    return _runner(ctx).status()


@mcp.tool
async def lab_launch(
    config: dict[str, Any], ctx: Context, tags: list[str] | None = None
) -> dict[str, Any]:
    """Launch a trial from a RunConfig dict.

    Use ``lab_schema`` to discover all fields. The config must include
    ``run_type`` (``"pretrain"``, ``"adapter"``, ``"specialized_clm"``, or
    ``"distill"``). Unknown fields are rejected. Optionally pass ``tags`` for
    grouping (e.g. ``["phase1", "mate-boost"]``).
    """
    try:
        tid = await _runner(ctx).launch(config, tags=tags)
        return _runner(ctx).trials[tid].to_dict()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
async def lab_kill(trial_id: int, ctx: Context) -> dict[str, Any]:
    """Kill a running trial by id (sends SIGTERM for graceful shutdown)."""
    return await _runner(ctx).kill(trial_id)


@mcp.tool
async def lab_resume(
    trial_id: int,
    ctx: Context,
    total_steps: int | None = None,
    pause_after_steps: int | None = None,
) -> dict[str, Any]:
    """Resume a completed/paused trial from its best checkpoint.

    Creates a new trial with the same config plus ``--resume``. Override
    ``total_steps`` or ``pause_after_steps`` for iterative narrowing.
    """
    try:
        new_id = await _runner(ctx).resume_trial(
            trial_id,
            total_steps=total_steps,
            pause_after_steps=pause_after_steps,
        )
        return _runner(ctx).trials[new_id].to_dict()
    except RuntimeError as e:
        return {"error": str(e)}


@mcp.tool
async def lab_results(
    ctx: Context, strategy: str | None = None, tag: str | None = None
) -> dict[str, Any]:
    """All trials with val_loss, params, status, notes, tags.

    Includes a Pareto front and Optuna suggestions. Filter by ``strategy``
    and/or ``tag`` (e.g. ``tag="phase2"``).
    """
    return _runner(ctx).results(strategy, tag=tag)


@mcp.tool
async def lab_events(ctx: Context, since: int | None = None) -> dict[str, Any]:
    """Events since a sequence number.

    Omit ``since`` to get events since the last call (auto-tracked). Pass
    ``since=0`` for all events.
    """
    runner = _runner(ctx)
    events, latest_seq = runner.events_since(since)
    return {"events": events, "latest_seq": latest_seq}


@mcp.tool
async def lab_log(trial_id: int, ctx: Context, lines: int = 50) -> dict[str, Any]:
    """Last N lines of a trial's stdout/stderr log.

    Use to debug failures or check training output.
    """
    return _runner(ctx).trial_log(trial_id, lines)


@mcp.tool
async def lab_notes(trial_id: int, notes: str, ctx: Context) -> dict[str, Any]:
    """Add notes to a trial. Notes appear in the results table."""
    return _runner(ctx).add_notes(trial_id, notes)


@mcp.tool
async def lab_set_cost(cost_per_hour: float, ctx: Context) -> dict[str, Any]:
    """Set the $/hr rate for cost tracking (e.g. 3.59 for an H200 SXM on RunPod)."""
    return _runner(ctx).set_cost(cost_per_hour)


@mcp.tool
async def lab_schema(ctx: Context) -> dict[str, Any]:
    """Return the JSON Schema for every supported run_type.

    Keys: ``pretrain`` (:class:`PretrainConfig`), ``adapter``
    (:class:`AdapterConfig`), ``specialized_clm``
    (:class:`SpecializedCLMConfig`), and ``distill`` (:class:`DistillConfig`).
    Use this to discover all available parameters before calling ``lab_launch``.
    Delegates to :func:`pawn.lab.runner.lab_schema` so the schema surface stays
    the single source of truth shared with the in-process helper.
    """
    from pawn.lab.runner import lab_schema as _lab_schema

    return _lab_schema()


@mcp.tool
async def lab_audit(
    ctx: Context,
    trial_id: int | None = None,
    check_hf: bool = False,
) -> dict[str, Any]:
    """Per-trial pass/fail on completion invariants.

    Without arguments: audits every trial. With ``trial_id``: just that one.
    Returns ``{trials: [...], any_failure: bool}`` where each row carries a
    ``checks`` dict covering ``schedule_complete``, ``checkpoint_complete``,
    and (when ``check_hf=True``) ``checkpoint_on_hf``. Each check is
    ``{pass: true|false|null, ...}`` where ``null`` means "not applicable" or
    "couldn't verify".
    """
    return _runner(ctx).audit(trial_id=trial_id, check_hf=check_hf)


def build_server() -> FastMCP:
    """Return the module-level FastMCP server.

    ``python -m pawn.lab`` (see :mod:`pawn.lab.__main__`) calls this to obtain
    the configured server. All tools are registered at import time via the
    ``@mcp.tool`` decorators above, and the lifespan creates / recovers the
    :class:`~pawn.lab.runner.TrialRunner`.
    """
    return mcp
