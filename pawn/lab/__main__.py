"""Entry point: ``python -m pawn.lab`` — starts the FastMCP server over stdio."""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [pawn-lab] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from pawn.lab.server import build_server

server = build_server()
server.run()
