"""pawn-lab — trial-runner daemon + FastMCP dispatch for trial orchestration.

The lab exposes two MCP tools per plan §10 S9:

- ``lab_launch(config={...})`` — validate the incoming dict through
  pydantic (`extra="forbid"` rejects stale field names) and dispatch
  a trial via the runner.
- ``lab_schema()`` — return the JSON Schema generated from each run
  config's ``model_json_schema()``. Clients use it for client-side
  validation.

The full process-lifecycle daemon (:class:`TrialRunner` — GPU discovery
+ scheduling, subprocess spawn/monitor/kill, ``lab_state.json`` crash
recovery, the monotonic-seq event bus, cost tracking, and the
completion-invariant audit) and the schedule-health helpers live in
:mod:`pawn.lab.runner`; the FastMCP server wiring lives in
:mod:`pawn.lab.server`.
"""

from pawn.lab.runner import (
    TrialRunner,
    audit_schedule_health,
    lab_launch,
    lab_schema,
    read_schedule_health,
    validate_config,
)

__all__ = [
    "TrialRunner",
    "audit_schedule_health",
    "lab_launch",
    "lab_schema",
    "read_schedule_health",
    "validate_config",
]
