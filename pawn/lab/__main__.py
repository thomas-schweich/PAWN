"""Entry point: python -m pawn.lab

Starts the pawn-lab MCP server over stdio.
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [pawn-lab] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from pawn.lab.server import run_server

workspace = None
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--workspace" and i < len(sys.argv) - 1:
        workspace = sys.argv[i + 1]

asyncio.run(run_server(workspace=workspace))
