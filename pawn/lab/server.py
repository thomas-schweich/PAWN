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
async def lab_suggest(ctx: Context, search_space: str | None = None, strategy: str | None = None, study_name: str = "suggest", directions: str = '["minimize"]') -> str:
    """Get an Optuna suggestion based on completed trial results. Creates an ephemeral study, seeds it with all completed trials, and asks for the next config to try. Returns suggested params — you decide whether to launch.

    search_space: JSON string of {param: {type, low, high, log?, choices?}}. Omit to use built-in distributions for the strategy.
    directions: JSON string of optimization directions, e.g. '["minimize"]' or '["minimize", "minimize"]'.
    """
    from pawn.lab.sweep import builtin_distributions, parse_distribution
    import optuna

    runner = _runner(ctx)
    parsed_directions = json.loads(directions)

    # Build distributions
    if search_space:
        specs = json.loads(search_space) if isinstance(search_space, str) else search_space
        dists = {k: parse_distribution(v) for k, v in specs.items()}
    elif strategy:
        dists = builtin_distributions(strategy)
    else:
        return _json({"error": "Provide search_space or strategy"})

    # Create ephemeral study and seed with completed results
    study = optuna.create_study(
        study_name=study_name,
        directions=parsed_directions,
    )

    # Seed from completed trials
    seeded = 0
    for t in runner.trials.values():
        if t.status != "completed" or t.best_val_loss is None:
            continue
        trial_dists = {k: v for k, v in dists.items() if k in t.params}
        if not trial_dists:
            continue
        trial_params = {k: v for k, v in t.params.items() if k in dists}
        values = [t.best_val_loss]
        if len(parsed_directions) > 1 and t.actual_param_count is not None:
            values.append(float(t.actual_param_count))
        try:
            frozen = optuna.trial.create_trial(
                params=trial_params, distributions=trial_dists,
                values=values, state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(frozen)
            seeded += 1
        except Exception:
            pass  # skip incompatible trials

    # Ask for suggestion
    trial = study.ask(dists)
    return _json({
        "suggested_params": trial.params,
        "seeded_from": seeded,
        "study_trials": len(study.trials),
    })


@mcp.tool
async def lab_kill(trial_id: int, ctx: Context) -> str:
    """Kill a running trial by ID (sends SIGTERM for graceful shutdown)."""
    return _json(await _runner(ctx).kill(trial_id))


@mcp.tool
async def lab_results(ctx: Context) -> str:
    """All trials with val_loss, accuracy, param count, wall time, key HPs, status, notes. Includes Pareto front."""
    return _json(_runner(ctx).results())


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
