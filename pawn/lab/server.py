"""MCP server for pawn-lab: exposes the trial runner as tools via FastMCP."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Context, FastMCP

from pawn.lab.runner import TrialRunner

log = logging.getLogger("pawn.lab")


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
async def lab_status(ctx: Context) -> dict[str, Any]:
    """Compact lab status: GPUs, running trials (ID, strategy, key HPs, step/total, ETA, val_loss), counts, elapsed time, cost."""
    return _runner(ctx).status()


@mcp.tool
async def lab_launch(config: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """Launch a trial from a RunConfig dict. Use lab_schema to discover all fields. The config must include run_type ('pretrain' or 'adapter'). For adapter runs, include 'strategy'. Example: {"run_type": "adapter", "strategy": "lora", "lora_rank": 4, "lr": 3e-4, "local_checkpoints": true}."""
    try:
        tid = await _runner(ctx).launch(config)
        return _runner(ctx).trials[tid].to_dict()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
async def lab_kill(trial_id: int, ctx: Context) -> dict[str, Any]:
    """Kill a running trial by ID (sends SIGTERM for graceful shutdown)."""
    return await _runner(ctx).kill(trial_id)


@mcp.tool
async def lab_resume(trial_id: int, ctx: Context, total_steps: int | None = None, pause_after_steps: int | None = None) -> dict[str, Any]:
    """Resume a completed/paused trial from its best checkpoint. Creates a new trial with the same config plus --resume. Override total_steps or pause_after_steps for iterative narrowing."""
    try:
        new_id = await _runner(ctx).resume_trial(trial_id, total_steps=total_steps, pause_after_steps=pause_after_steps)
        return _runner(ctx).trials[new_id].to_dict()
    except RuntimeError as e:
        return {"error": str(e)}


@mcp.tool
async def lab_results(ctx: Context, strategy: str | None = None) -> dict[str, Any]:
    """All trials with val_loss, accuracy, param count, wall time, key HPs, status, notes. Includes Pareto front and Optuna suggestions. Optionally filter by strategy."""
    return _runner(ctx).results(strategy)


@mcp.tool
async def lab_events(ctx: Context, since: int = 0) -> dict[str, Any]:
    """Events since a sequence number. Types: trial_started, trial_completed, trial_failed, trial_killed, gpu_idle, health_warning. Use since=0 for all."""
    runner = _runner(ctx)
    events = runner.events_since(since)
    return {"events": events, "latest_seq": runner.event_seq}


@mcp.tool
async def lab_log(trial_id: int, ctx: Context, lines: int = 50) -> dict[str, Any]:
    """Last N lines of a trial's stdout/stderr log. Use to debug failures or check training output."""
    return _runner(ctx).trial_log(trial_id, lines)


@mcp.tool
async def lab_notes(trial_id: int, notes: str, ctx: Context) -> dict[str, Any]:
    """Add notes to a trial. Notes appear in results table and progress log."""
    return _runner(ctx).add_notes(trial_id, notes)


@mcp.tool
async def lab_set_cost(cost_per_hour: float, ctx: Context) -> dict[str, Any]:
    """Set $/hr rate for cost tracking (e.g. 3.59 for H200 SXM on RunPod)."""
    runner = _runner(ctx)
    runner.cost_per_hour = cost_per_hour
    runner._save_state()
    return {"cost_per_hour": runner.cost_per_hour}


@mcp.tool
async def lab_schema(ctx: Context) -> dict[str, Any]:
    """Return the JSON Schema for RunConfig (PretrainConfig and AdapterConfig). Use this to discover all available parameters before calling lab_launch."""
    from pawn.run_config import AdapterConfig, PretrainConfig

    return {
        "pretrain": PretrainConfig.model_json_schema(),
        "adapter": AdapterConfig.model_json_schema(),
    }
