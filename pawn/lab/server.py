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
            "Current lab status: GPU utilization, running trials with ETAs, "
            "sweep progress, elapsed time, and estimated cost."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_launch",
        "description": (
            "Launch a single trial on the next free GPU. "
            "Provide strategy name and optional hyperparameters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "description": "Adapter strategy (lora, bottleneck, film, sparse, etc.)",
                },
                "params": {
                    "type": "object",
                    "description": "Hyperparameters (lr, lora_rank, etc.)",
                },
                "base_args": {
                    "type": "object",
                    "description": "Base CLI args (checkpoint, pgn, total_steps, etc.)",
                },
            },
            "required": ["strategy"],
        },
    },
    {
        "name": "lab_sweep",
        "description": (
            "Configure and start an autopilot sweep. The runner will use Optuna "
            "to sample hyperparameters and auto-launch trials as GPUs free up."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of strategies to sweep over",
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Total number of trials to run",
                },
                "base_args": {
                    "type": "object",
                    "description": "Fixed CLI args for all trials (checkpoint, pgn, etc.)",
                },
                "pinned_params": {
                    "type": "object",
                    "description": "Params to fix (not searched by Optuna)",
                },
                "search_space": {
                    "type": "object",
                    "description": "Custom search space as {param: {type, low, high, ...}}. "
                    "Omit to use built-in distributions per strategy.",
                },
                "study_name": {"type": "string", "default": "sweep"},
                "directions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optimization directions (e.g. ['minimize', 'minimize'] for "
                    "val_loss + param_count)",
                },
            },
            "required": ["strategies", "n_trials", "base_args"],
        },
    },
    {
        "name": "lab_seed",
        "description": (
            "Seed an existing result into the Optuna study. Use this to prime "
            "the sampler with prior experiments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string"},
                "params": {
                    "type": "object",
                    "description": "The hyperparameters used",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Objective values (e.g. [val_loss] or [val_loss, param_count])",
                },
            },
            "required": ["strategy", "params", "values"],
        },
    },
    {
        "name": "lab_kill",
        "description": "Kill a running trial by ID (sends SIGTERM for graceful shutdown).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "integer"},
            },
            "required": ["trial_id"],
        },
    },
    {
        "name": "lab_pause",
        "description": "Pause autopilot — running trials continue but no new ones launch.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_resume",
        "description": "Resume autopilot — fill free GPUs with new trials.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_pin",
        "description": (
            "Pin parameters for all future trials. Pinned params override Optuna "
            "suggestions (e.g. pin batch_size=256 after discovering it's best)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Parameters to pin (e.g. {batch_size: 256})",
                },
            },
            "required": ["params"],
        },
    },
    {
        "name": "lab_results",
        "description": (
            "Get the results table and Pareto front for all completed trials."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lab_events",
        "description": (
            "Get events since a given sequence number. Returns trial completions, "
            "failures, health warnings, and other events the agent should act on."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {
                    "type": "integer",
                    "description": "Sequence number to read from (0 = all events)",
                    "default": 0,
                },
            },
        },
    },
    {
        "name": "lab_notes",
        "description": (
            "Add agent notes to a trial. Use during check-ins to record observations, "
            "findings, or decisions about specific trials."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "integer"},
                "notes": {
                    "type": "string",
                    "description": "Free-text notes (replaces existing notes for this trial)",
                },
            },
            "required": ["trial_id", "notes"],
        },
    },
    {
        "name": "lab_set_cost",
        "description": "Set the $/hr rate for cost tracking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cost_per_hour": {"type": "number"},
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
