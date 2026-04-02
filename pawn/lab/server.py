"""MCP server for pawn-lab: exposes the trial runner as tools via FastMCP."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Context, FastMCP

from pawn.lab.runner import TrialRunner

log = logging.getLogger("pawn.lab")

_session: Any = None  # low-level MCP session, captured for background notifications


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


async def _notify_via_session(message: str) -> None:
    """Send a notification via the captured low-level MCP session.

    ctx.info() only works inside a tool handler. Background monitor tasks
    use this instead, which writes directly to the stored session object.
    """
    if _session is None:
        return
    try:
        await _session.send_log_message(level="info", data=message, logger="pawn-lab")
    except Exception as e:
        log.debug("Failed to send notification: %s", e)


@asynccontextmanager
async def _lifespan(server: FastMCP):
    runner = TrialRunner()
    await runner.recover()
    log.info("pawn-lab MCP server starting (workspace=%s)", runner.workspace)
    yield {"runner": runner}
    runner.shutdown()


mcp = FastMCP("pawn-lab", lifespan=_lifespan)


def _wire_notifications(ctx: Context, runner: TrialRunner) -> None:
    """Capture the low-level session for background notifications."""
    global _session
    if runner._notify is not None:
        return
    try:
        _session = ctx.session
        runner.set_notify(_notify_via_session)
        log.info("Captured MCP session for notifications")
    except Exception:
        pass


@mcp.tool
async def lab_status(ctx: Context) -> str:
    """Current lab status: GPU count/type/VRAM, running trials with step progress and ETAs, sweep progress, elapsed time, estimated cost. Call this first to orient."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    _wire_notifications(ctx, runner)
    return _json(runner.status())


@mcp.tool
async def lab_launch(strategy: str, ctx: Context, params: dict[str, Any] | None = None, base_args: dict[str, Any] | None = None) -> str:
    """Launch a single trial on the next free GPU. Returns trial_id and full config. The process runs in background with CUDA_VISIBLE_DEVICES isolation. Use base_args for data/training config (checkpoint, pgn, total_steps, etc.) and params for hyperparameters (lr, bottleneck_dim, etc.). Params use underscores -- they're converted to CLI --flags automatically (e.g. lora_rank -> --lora-rank, no_compile -> --no-compile)."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    _wire_notifications(ctx, runner)
    try:
        tid = await runner.launch(strategy, params, base_args)
        return _json(runner.trials[tid].to_dict())
    except RuntimeError as e:
        return _json({"error": str(e)})


@mcp.tool
async def lab_sweep(
    strategies: list[str],
    n_trials: int,
    base_args: dict[str, Any],
    ctx: Context,
    pinned_params: dict[str, Any] | None = None,
    search_space: dict[str, Any] | None = None,
    study_name: str = "sweep",
    directions: list[str] | None = None,
) -> str:
    """Configure and start an Optuna-driven autopilot sweep. Trials auto-launch on free GPUs and auto-advance when completed. Use search_space to define what Optuna searches over and pinned_params for values to fix. If search_space is omitted, built-in distributions for the strategy are used."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    _wire_notifications(ctx, runner)
    result = await runner.configure_sweep(
        strategies=strategies,
        n_trials=n_trials,
        base_args=base_args,
        pinned_params=pinned_params,
        search_space=search_space,
        study_name=study_name,
        directions=directions,
    )
    return _json(result)


@mcp.tool
async def lab_seed(strategy: str, params: str, values: str, ctx: Context) -> str:
    """Seed a prior result into the Optuna study so the sampler can learn from it. Use to import results from previous experiments. Pass params as JSON object string and values as JSON array string."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    _wire_notifications(ctx, runner)
    parsed_params = json.loads(params) if isinstance(params, str) else params
    parsed_values = json.loads(values) if isinstance(values, str) else values
    result = await runner.seed_trial(strategy, parsed_params, parsed_values)
    return _json(result)


@mcp.tool
async def lab_kill(trial_id: int, ctx: Context) -> str:
    """Kill a running trial by its trial_id. Sends SIGTERM for graceful shutdown (the training script saves a checkpoint before exiting). If autopilot is on, a replacement trial is auto-launched."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    _wire_notifications(ctx, runner)
    return _json(await runner.kill(trial_id))


@mcp.tool
async def lab_pause(ctx: Context) -> str:
    """Pause autopilot. Running trials continue to completion but no new trials are launched. Use when changing strategy or reviewing results."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    return _json(await runner.pause())


@mcp.tool
async def lab_resume(ctx: Context) -> str:
    """Resume autopilot. Immediately fills all free GPUs with new trials."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    _wire_notifications(ctx, runner)
    return _json(await runner.resume())


@mcp.tool
async def lab_pin(params: dict[str, Any], ctx: Context) -> str:
    """Pin parameters for all future trials. Pinned params override Optuna suggestions and are excluded from the search space. Useful after discovering a value is clearly best (e.g. batch_size=256). Cumulative -- call multiple times to pin more params."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    return _json(runner.pin(params))


@mcp.tool
async def lab_results(ctx: Context) -> str:
    """Results table for all trials (completed, failed, killed) plus the Pareto front (non-dominated set sorted by param_count). Each row includes trial_id, strategy, param_count, val_loss, accuracy, status, notes, and key hyperparameters."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    return _json(runner.results())


@mcp.tool
async def lab_events(ctx: Context, since: int = 0) -> str:
    """Get events since a sequence number. Event types: trial_started, trial_completed, trial_failed, trial_killed, gpu_idle, health_warning, sweep_started, sweep_complete. Use since=0 for all events, or pass the latest_seq from a prior call to get only new events."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    events = runner.events_since(since)
    return _json({"events": events, "latest_seq": runner.event_seq})


@mcp.tool
async def lab_notes(trial_id: int, notes: str, ctx: Context) -> str:
    """Add agent notes to a trial. Use during check-ins to record your assessment: is this trial promising? Dominated? Surprising? Notes appear in the results table and progress log."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    return _json(runner.add_notes(trial_id, notes))


@mcp.tool
async def lab_set_cost(cost_per_hour: float, ctx: Context) -> str:
    """Set the $/hr rate for cost tracking. Shows estimated spend in lab_status. Example: 3.59 for H200 SXM on RunPod."""
    runner: TrialRunner = ctx.lifespan_context["runner"]
    runner.cost_per_hour = cost_per_hour
    runner._save_state()
    return _json({"cost_per_hour": runner.cost_per_hour})
