"""pawn-lab — FastMCP daemon for trial orchestration.

The lab exposes two MCP tools per plan §10 S9:

- ``lab_launch(config={...})`` — validate the incoming dict through
  pydantic (`extra="forbid"` rejects stale field names) and dispatch
  a trial via the runner.
- ``lab_schema()`` — return the JSON Schema generated from
  `PretrainConfig.model_json_schema()` /
  `AdapterConfig.model_json_schema()`. Clients use this for
  client-side validation.

The runner subprocess and dashboard wiring live in
:mod:`pawn.lab.runner` / :mod:`pawn.lab.server`.
"""

from pawn.lab.runner import lab_launch, lab_schema, validate_config

__all__ = ["lab_launch", "lab_schema", "validate_config"]
