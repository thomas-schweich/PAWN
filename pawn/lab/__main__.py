"""Entry point: python -m pawn.lab

Starts the pawn-lab MCP server over stdio.
"""

import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [pawn-lab] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from pawn.lab.server import mcp

mcp.run()
