"""MCP server for pawn-lab: exposes the trial runner as tools via FastMCP."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Context, FastMCP

from pawn.lab.runner import TrialRunner

log = logging.getLogger("pawn.lab")


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


@asynccontextmanager
async def _lifespan(server: FastMCP):
    runner = TrialRunner()
    await runner.recover()
    log.info("pawn-lab MCP server starting (workspace=%s)", runner.workspace)
    yield {"runner": runner}
    runner.shutdown()


mcp = FastMCP("pawn-lab", lifespan=_lifespan)


def _runner(ctx: Context) -> TrialRunner:
    return ctx.lifespan_context["runner"]


# -----------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------

@mcp.tool
async def lab_status(ctx: Context) -> str:
    """Compact lab status: GPUs, running trials (ID, strategy, key HPs, step/total, ETA, val_loss), counts, elapsed time, cost."""
    return _json(_runner(ctx).status())


@mcp.tool
async def lab_launch(strategy: str, ctx: Context, params: dict[str, Any] | None = None, base_args: dict[str, Any] | None = None) -> str:
    """Launch a single trial on the next free GPU. Params use underscores, converted to CLI --flags (e.g. lora_rank -> --lora-rank, no_compile -> --no-compile). Returns trial_id and config."""
    try:
        tid = await _runner(ctx).launch(strategy, params, base_args)
        return _json(_runner(ctx).trials[tid].to_dict())
    except RuntimeError as e:
        return _json({"error": str(e)})


@mcp.tool
async def lab_kill(trial_id: int, ctx: Context) -> str:
    """Kill a running trial by ID (sends SIGTERM for graceful shutdown)."""
    return _json(await _runner(ctx).kill(trial_id))


@mcp.tool
async def lab_results(strategy: str, ctx: Context) -> str:
    """All trials with val_loss, accuracy, param count, wall time, key HPs, status, notes. Includes Pareto front and 3 Optuna suggestions for what to try next. Strategy determines the search space for suggestions: bottleneck, lora, film, sparse, hybrid, specialized_clm, unfreeze, rosa, retro-sparse, retro-bottleneck."""
    return _json(_runner(ctx).results(strategy))


@mcp.tool
async def lab_events(ctx: Context, since: int = 0) -> str:
    """Events since a sequence number. Types: trial_started, trial_completed, trial_failed, trial_killed, gpu_idle, health_warning. Use since=0 for all."""
    runner = _runner(ctx)
    events = runner.events_since(since)
    return _json({"events": events, "latest_seq": runner.event_seq})


@mcp.tool
async def lab_log(trial_id: int, ctx: Context, lines: int = 50) -> str:
    """Last N lines of a trial's stdout/stderr log. Use to debug failures or check training output."""
    return _json(_runner(ctx).trial_log(trial_id, lines))


@mcp.tool
async def lab_notes(trial_id: int, notes: str, ctx: Context) -> str:
    """Add notes to a trial. Notes appear in results table and progress log."""
    return _json(_runner(ctx).add_notes(trial_id, notes))


@mcp.tool
async def lab_set_cost(cost_per_hour: float, ctx: Context) -> str:
    """Set $/hr rate for cost tracking (e.g. 3.59 for H200 SXM on RunPod)."""
    runner = _runner(ctx)
    runner.cost_per_hour = cost_per_hour
    runner._save_state()
    return _json({"cost_per_hour": runner.cost_per_hour})
