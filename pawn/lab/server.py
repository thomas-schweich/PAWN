"""MCP server for pawn-lab: exposes the trial runner as tools.

The server is a thin layer over TrialRunner. All state lives in the runner;
the server translates MCP tool calls to runner method calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

log = logging.getLogger("pawn.lab")

# Late import — runner is always available, MCP may not be
from pawn.lab.runner import TrialRunner

_runner: TrialRunner | None = None


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "lab_status",
        "description": (
            "Current lab status: GPU count/type/VRAM, running trials with step "
            "progress and ETAs, sweep progress, elapsed time, estimated cost. "
            "Call this first to orient."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_launch",
        "description": (
            "Launch a single trial on the next free GPU. Returns trial_id and "
            "full config. The process runs in background with CUDA_VISIBLE_DEVICES "
            "isolation. Use base_args for data/training config (checkpoint, pgn, "
            "total_steps, etc.) and params for hyperparameters (lr, bottleneck_dim, "
            "etc.). Params use underscores — they're converted to CLI --flags "
            "automatically (e.g. lora_rank -> --lora-rank, no_compile -> --no-compile)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "description": (
                        "Adapter strategy: bottleneck, lora, film, sparse, "
                        "rosa, hybrid, specialized_clm, or unfreeze"
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Hyperparameters as {name: value}. Examples: "
                        "{lr: 5e-4, bottleneck_dim: 32, batch_size: 64, "
                        "adapter_layers: '8,9', no_compile: true}"
                    ),
                },
                "base_args": {
                    "type": "object",
                    "description": (
                        "Fixed CLI args for data and training. Examples: "
                        "{checkpoint: 'thomas-schweich/pawn-base', "
                        "pgn: 'thomas-schweich/pawn-lichess-full', "
                        "elo_min: 1800, elo_max: 1900, total_steps: 2000, "
                        "eval_interval: 500, max_games: 1000000, "
                        "sdpa_math: true, amp_dtype: 'bfloat16', num_workers: 2}"
                    ),
                },
            },
            "required": ["strategy"],
        },
    },
    {
        "name": "lab_sweep",
        "description": (
            "Configure and start an Optuna-driven autopilot sweep. Trials auto-launch "
            "on free GPUs and auto-advance when completed. Use search_space to define "
            "what Optuna searches over and pinned_params for values to fix. If "
            "search_space is omitted, built-in distributions for the strategy are used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Strategies to sweep. Example: ['bottleneck', 'lora']"
                    ),
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Total trials to run (sequentially on 1 GPU, parallel on N GPUs)",
                },
                "base_args": {
                    "type": "object",
                    "description": (
                        "Fixed CLI args applied to every trial. Must include "
                        "checkpoint, pgn. Example: {checkpoint: 'thomas-schweich/pawn-large', "
                        "pgn: 'thomas-schweich/pawn-lichess-full', elo_min: 1800, "
                        "elo_max: 1900, total_steps: 2000, eval_interval: 500, "
                        "sdpa_math: true, amp_dtype: 'bfloat16'}"
                    ),
                },
                "pinned_params": {
                    "type": "object",
                    "description": (
                        "Params to fix for all trials (excluded from Optuna search). "
                        "Example: {bottleneck_dim: 1953, adapter_layers: '8,9'}"
                    ),
                },
                "search_space": {
                    "type": "object",
                    "description": (
                        "Custom search space as {param: distribution_spec}. "
                        "Distribution types: "
                        "{type: 'float', low: 1e-4, high: 2e-3, log: true}, "
                        "{type: 'int', low: 1, high: 10, step: 1}, "
                        "{type: 'categorical', choices: [64, 128, 256]}. "
                        "Omit to use built-in distributions for the strategy."
                    ),
                },
                "study_name": {
                    "type": "string",
                    "default": "sweep",
                    "description": "Optuna study name (persisted to SQLite)",
                },
                "directions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optimization directions. ['minimize'] for val_loss only, "
                        "['minimize', 'minimize'] for val_loss + param_count (Pareto)"
                    ),
                },
            },
            "required": ["strategies", "n_trials", "base_args"],
        },
    },
    {
        "name": "lab_seed",
        "description": (
            "Seed a prior result into the Optuna study so the sampler can learn "
            "from it. Use to import results from previous experiments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "description": "Strategy that was used"},
                "params": {
                    "type": "object",
                    "description": "Hyperparameters used (e.g. {lr: 5e-4, bottleneck_dim: 64})",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Objective values matching study directions. "
                        "[val_loss] for single-objective, [val_loss, param_count] for Pareto"
                    ),
                },
            },
            "required": ["strategy", "params", "values"],
        },
    },
    {
        "name": "lab_kill",
        "description": (
            "Kill a running trial by its trial_id. Sends SIGTERM for graceful "
            "shutdown (the training script saves a checkpoint before exiting). "
            "If autopilot is on, a replacement trial is auto-launched."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "integer", "description": "Trial ID to kill"},
            },
            "required": ["trial_id"],
        },
    },
    {
        "name": "lab_pause",
        "description": (
            "Pause autopilot. Running trials continue to completion but no new "
            "trials are launched. Use when changing strategy or reviewing results."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_resume",
        "description": (
            "Resume autopilot. Immediately fills all free GPUs with new trials."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_pin",
        "description": (
            "Pin parameters for all future trials. Pinned params override Optuna "
            "suggestions and are excluded from the search space. Useful after "
            "discovering a value is clearly best (e.g. batch_size=256). "
            "Cumulative — call multiple times to pin more params."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Params to pin. Example: {batch_size: 256, warmup_frac: 0.05}",
                },
            },
            "required": ["params"],
        },
    },
    {
        "name": "lab_results",
        "description": (
            "Results table for all trials (completed, failed, killed) plus the "
            "Pareto front (non-dominated set sorted by param_count). Each row "
            "includes trial_id, strategy, param_count, val_loss, accuracy, status, "
            "notes, and key hyperparameters."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_events",
        "description": (
            "Get events since a sequence number. Event types: trial_started, "
            "trial_completed, trial_failed, trial_killed, gpu_idle, "
            "health_warning, sweep_started, sweep_complete. "
            "Use since=0 for all events, or pass the latest_seq from a prior "
            "call to get only new events."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {
                    "type": "integer",
                    "description": "Return events with seq > this value. 0 = all events.",
                    "default": 0,
                },
            },
        },
    },
    {
        "name": "lab_notes",
        "description": (
            "Add agent notes to a trial. Use during check-ins to record your "
            "assessment: is this trial promising? Dominated? Surprising? "
            "Notes appear in the results table and progress log."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "integer", "description": "Trial to annotate"},
                "notes": {
                    "type": "string",
                    "description": "Your assessment (replaces existing notes for this trial)",
                },
            },
            "required": ["trial_id", "notes"],
        },
    },
    {
        "name": "lab_set_cost",
        "description": (
            "Set the $/hr rate for cost tracking. Shows estimated spend in "
            "lab_status. Example: 3.59 for H200 SXM on RunPod."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cost_per_hour": {
                    "type": "number",
                    "description": "Cost in $/hr for this pod",
                },
            },
            "required": ["cost_per_hour"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

async def _handle_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call to the runner. Returns JSON string."""
    assert _runner is not None

    if name == "lab_status":
        return _json(_runner.status())

    elif name == "lab_launch":
        try:
            tid = await _runner.launch(
                args["strategy"],
                args.get("params"),
                args.get("base_args"),
            )
            return _json(_runner.trials[tid].to_dict())
        except RuntimeError as e:
            return _json({"error": str(e)})

    elif name == "lab_sweep":
        result = await _runner.configure_sweep(
            strategies=args["strategies"],
            n_trials=args["n_trials"],
            base_args=args["base_args"],
            pinned_params=args.get("pinned_params"),
            search_space=args.get("search_space"),
            study_name=args.get("study_name", "sweep"),
            directions=args.get("directions"),
        )
        return _json(result)

    elif name == "lab_seed":
        result = await _runner.seed_trial(
            args["strategy"], args["params"], args["values"],
        )
        return _json(result)

    elif name == "lab_kill":
        return _json(await _runner.kill(args["trial_id"]))

    elif name == "lab_pause":
        return _json(await _runner.pause())

    elif name == "lab_resume":
        return _json(await _runner.resume())

    elif name == "lab_pin":
        return _json(_runner.pin(args["params"]))

    elif name == "lab_results":
        return _json(_runner.results())

    elif name == "lab_events":
        since = args.get("since", 0)
        events = _runner.events_since(since)
        return _json({"events": events, "latest_seq": _runner.event_seq})

    elif name == "lab_notes":
        return _json(_runner.add_notes(args["trial_id"], args["notes"]))

    elif name == "lab_set_cost":
        _runner.cost_per_hour = args["cost_per_hour"]
        _runner._save_state()
        return _json({"cost_per_hour": _runner.cost_per_hour})

    return _json({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

async def run_server(workspace: str | None = None) -> None:
    """Start the MCP server over stdio."""
    global _runner

    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
        from mcp.server.stdio import stdio_server
    except ImportError:
        log.error(
            "mcp package not installed. Install with: "
            "uv add 'mcp>=1.0.0' or pip install 'mcp>=1.0.0'"
        )
        sys.exit(1)

    _runner = TrialRunner(workspace=workspace)
    await _runner.recover()

    app = Server("pawn-lab")

    @app.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @app.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        result = await _handle_tool(name, arguments or {})
        return [TextContent(type="text", text=result)]

    log.info("pawn-lab MCP server starting (workspace=%s)", _runner.workspace)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

    _runner.shutdown()
