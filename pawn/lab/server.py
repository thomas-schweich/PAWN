"""FastMCP server for pawn-lab — exposes `lab_launch` / `lab_schema` as MCP tools.

The server is a thin wrapper around :mod:`pawn.lab.runner`. Run via
``python -m pawn.lab``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pawn.lab.runner import lab_launch as _lab_launch
from pawn.lab.runner import lab_schema as _lab_schema

__all__ = ["build_server"]


def build_server() -> Any:
    """Construct the FastMCP server. Lazy import to keep the `lab` extra
    optional — when fastmcp isn't installed, the runner functions are
    still callable directly."""
    from fastmcp import FastMCP

    mcp = FastMCP("pawn-lab")

    @mcp.tool()
    def lab_schema() -> dict[str, Any]:
        """Return the JSON Schema for every supported run_type."""
        return _lab_schema()

    @mcp.tool()
    def lab_launch(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        """Validate the trial config (pydantic, extra='forbid') and
        dispatch to the appropriate training script. Pass `dry_run=True`
        to validate without spawning a process."""
        return _lab_launch(config, dry_run=dry_run)

    return mcp
