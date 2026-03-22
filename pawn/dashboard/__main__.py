"""Launch the Solara dashboard server via ``python -m pawn.dashboard``."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PAWN Dashboard (Solara)")
    parser.add_argument("--log-dir", default="../../logs", help="Path to logs directory")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    args = parser.parse_args()

    os.environ["PAWN_LOG_DIR"] = str(Path(args.log_dir).resolve())

    sys.exit(subprocess.call([
        sys.executable, "-m", "solara", "run",
        "pawn.dashboard.sol",
        "--host", args.host,
        "--port", str(args.port),
    ]))


main()
