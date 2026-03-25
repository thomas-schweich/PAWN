#!/usr/bin/env python3
"""Generate Lc0 self-play data as UCI move sequences.

Drives a single Lc0 process via UCI (GPU is the bottleneck, not CPU).
Output format: one game per line, space-separated UCI moves followed by the result.

    e2e4 e7e5 g1f3 b8c6 ... 1-0

Two modes per network:
  - Policy-only (nodes=1, Temperature > 0): raw NN policy sampling, no search.
  - MCTS (nodes=N): full search with temperature for the opening.

Tiers:
    t1_policy:    128K games, T1-256x10,         nodes=1   (policy-only)
    t1_mcts_128:  128K games, T1-256x10,         nodes=128
    t3_policy:    128K games, T3-512x15,         nodes=1   (policy-only)
    t3_mcts_128:  128K games, T3-512x15,         nodes=128
    bt4_policy:   128K games, BT4-1024x15,       nodes=1   (policy-only)
    bt4_mcts_128: 128K games, BT4-1024x15,       nodes=128

Usage:
    python scripts/generate_lc0_data.py --output data/lc0/
    python scripts/generate_lc0_data.py --output data/lc0/ --tier bt4_policy
    python scripts/generate_lc0_data.py --output data/lc0/ --games 1000 --backend cuda-auto
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_NET_DIR = "/opt/lc0_nets"

NETS = {
    "t1": "t1-256x10.pb.gz",
    "t3": "t3-512x15.pb.gz",
    "bt4": "bt4-1024x15.pb.gz",
}

# Hardcoded, non-conflicting seeds per tier.
# Lc0 doesn't have per-game seeds — diversity comes from Temperature sampling.
# These seeds control Python-side RNG for any supplementary randomness.
TIERS = [
    {"name": "t1_policy",    "net": "t1",  "nodes": 1,   "games": 128_000, "seed": 100_000},
    {"name": "t1_mcts_128",  "net": "t1",  "nodes": 128, "games": 128_000, "seed": 110_000},
    {"name": "t3_policy",    "net": "t3",  "nodes": 1,   "games": 128_000, "seed": 200_000},
    {"name": "t3_mcts_128",  "net": "t3",  "nodes": 128, "games": 128_000, "seed": 210_000},
    {"name": "bt4_policy",   "net": "bt4", "nodes": 1,   "games": 128_000, "seed": 300_000},
    {"name": "bt4_mcts_128", "net": "bt4", "nodes": 128, "games": 128_000, "seed": 310_000},
]

# Temperature settings: sample from policy during opening, greedy after.
OPENING_TEMP = 1.0       # temperature for the first TEMP_DECAY_MOVES plies
TEMP_DECAY_MOVES = 15    # after this many plies, temperature drops to 0 (greedy)


class Lc0Engine:
    def __init__(self, path: str, weights: str, backend: str = "cuda-auto"):
        self.proc = subprocess.Popen(
            [path, f"--weights={weights}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Backend value {backend}")
        self._send(f"setoption name Temperature value {OPENING_TEMP}")
        self._send(f"setoption name TempDecayMoves value {TEMP_DECAY_MOVES}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> list[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def best_move(self, moves: list[str], nodes: int) -> str | None:
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        self._send(f"go nodes {nodes}")
        lines = self._wait_for("bestmove")
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else None
                if move == "(none)":
                    return None
                return move
        return None

    def check_terminal(self, moves: list[str]) -> str | None:
        """Check if position after moves is terminal.

        Returns 'checkmate' or 'stalemate' or None.
        """
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        self._send("go nodes 1")
        lines = self._wait_for("bestmove")

        is_terminal = False
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else None
                if move == "(none)" or move is None:
                    is_terminal = True
                break

        if not is_terminal:
            return None

        # Distinguish mate vs stalemate from info score
        for line in lines:
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    si = parts.index("score")
                    if parts[si + 1] == "mate":
                        return "checkmate"
                except (ValueError, IndexError):
                    pass
        return "stalemate"

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(
    engine: Lc0Engine, nodes: int, max_ply: int = 500
) -> tuple[list[str], str]:
    """Play one self-play game. Returns (moves_uci, result)."""
    engine.new_game()
    moves: list[str] = []

    for _ in range(max_ply):
        move = engine.best_move(moves, nodes)
        if move is None:
            break
        moves.append(move)

    n = len(moves)
    if n == 0:
        return moves, "*"

    if n >= max_ply:
        return moves, "1/2-1/2"

    # Game ended mid-play — check why
    terminal = engine.check_terminal(moves)
    if terminal == "checkmate":
        result = "0-1" if n % 2 == 0 else "1-0"
    elif terminal == "stalemate":
        result = "1/2-1/2"
    else:
        result = "*"

    return moves, result


def generate_tier(
    lc0_path: str,
    net_dir: Path,
    output_dir: Path,
    tier: dict,
    backend: str,
):
    nodes = tier["nodes"]
    total_games = tier["games"]
    name = tier["name"]
    net_file = net_dir / NETS[tier["net"]]
    out_path = output_dir / f"{name}.txt"

    print(f"\n{'=' * 60}")
    print(f"Generating {total_games:,} games: {name}")
    print(f"Network: {net_file.name}")
    print(f"Nodes: {nodes} ({'policy-only' if nodes == 1 else 'MCTS'})")
    print(f"Backend: {backend}")
    print(f"Temperature: {OPENING_TEMP} for {TEMP_DECAY_MOVES} plies, then greedy")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}")

    if not net_file.exists():
        print(f"ERROR: Network not found: {net_file}")
        sys.exit(1)

    engine = Lc0Engine(lc0_path, str(net_file), backend=backend)

    t0 = time.perf_counter()
    with open(out_path, "w") as f:
        for i in range(1, total_games + 1):
            moves, result = play_game(engine, nodes)
            line = " ".join(moves) + " " + result
            f.write(line + "\n")

            if i % 100 == 0 or i == total_games:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                eta = (total_games - i) / rate if rate > 0 else 0
                print(
                    f"  {i:>7,}/{total_games:,}  ({i/total_games:.1%})  "
                    f"{rate:.1f} games/s  ETA {eta/60:.0f}m  "
                    f"last: {len(moves)} ply, {result}",
                    end="\r",
                )

    elapsed = time.perf_counter() - t0
    engine.close()

    print(f"\n  Done: {total_games:,} games in {elapsed / 60:.1f}m")
    print(f"  Rate: {total_games / elapsed:.1f} games/s")
    print(f"  File: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lc0 self-play UCI data (GPU)"
    )
    parser.add_argument(
        "--lc0",
        type=str,
        default="lc0",
        help="Path to lc0 binary (default: lc0 on PATH)",
    )
    parser.add_argument(
        "--net-dir",
        type=str,
        default=DEFAULT_NET_DIR,
        help="Directory containing .pb.gz network files",
    )
    parser.add_argument(
        "--output", type=str, default="data/lc0", help="Output directory"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Only generate this tier (e.g. bt4_policy)",
    )
    parser.add_argument(
        "--games", type=int, default=None, help="Override number of games per tier"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda-auto",
        help="Lc0 backend (cuda-auto, cudnn-fp16, opencl, eigen)",
    )
    args = parser.parse_args()

    # Resolve lc0 path
    lc0_path = args.lc0
    if not os.path.isfile(lc0_path):
        # Try PATH
        import shutil
        found = shutil.which(lc0_path)
        if not found:
            print(f"ERROR: lc0 not found at '{lc0_path}' or on PATH")
            sys.exit(1)
        lc0_path = found

    net_dir = Path(args.net_dir)
    if not net_dir.is_dir():
        print(f"ERROR: Network directory not found: {net_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = TIERS
    if args.tier is not None:
        matched = [t for t in TIERS if t["name"] == args.tier]
        if not matched:
            valid = ", ".join(t["name"] for t in TIERS)
            print(f"ERROR: unknown tier '{args.tier}'. Valid: {valid}")
            sys.exit(1)
        tiers = matched

    for tier in tiers:
        if args.games is not None:
            tier = {**tier, "games": args.games}
        generate_tier(lc0_path, net_dir, output_dir, tier, args.backend)

    print(f"\nAll done. Files in {output_dir}/")


if __name__ == "__main__":
    main()
