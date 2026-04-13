#!/usr/bin/env python3
"""Benchmark Stockfish self-play at varying node counts.

Runs a pool of single-threaded engines in parallel (one per core) and
extrapolates to find which node count allows 100K games within 1 hour.

Usage:
    python scripts/benchmark_stockfish_nodes.py --stockfish ~/bin/stockfish
    python scripts/benchmark_stockfish_nodes.py --stockfish ~/bin/stockfish --workers 14 --sample 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context


class StockfishEngine:
    def __init__(self, path: str, hash_mb: int = 16):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name Threads value 1")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        self._stdin.write(cmd + "\n")
        self._stdin.flush()

    def _wait_for(self, token: str) -> list[str]:
        lines = []
        while True:
            line = self._stdout.readline().strip()
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


def play_game(engine: StockfishEngine, nodes: int, max_ply: int = 500) -> tuple[int, str]:
    """Play one self-play game. Returns (num_plies, result)."""
    engine.new_game()
    moves: list[str] = []

    for _ in range(max_ply):
        move = engine.best_move(moves, nodes)
        if move is None:
            break
        moves.append(move)

    n = len(moves)
    if n == 0:
        return 0, "*"

    # Check if the final position has a legal move
    final = engine.best_move(moves, nodes)
    if final is None:
        result = "0-1" if n % 2 == 0 else "1-0"
    elif n >= max_ply:
        result = "1/2-1/2"
    else:
        result = "*"

    return n, result


def worker_play_games(
    stockfish_path: str, nodes: int, num_games: int, hash_mb: int
) -> tuple[float, int, int]:
    """Worker function: play num_games and return (elapsed, total_ply, num_games)."""
    engine = StockfishEngine(stockfish_path, hash_mb=hash_mb)
    total_ply = 0
    t0 = time.perf_counter()
    for _ in range(num_games):
        n_ply, _ = play_game(engine, nodes)
        total_ply += n_ply
    elapsed = time.perf_counter() - t0
    engine.close()
    return elapsed, total_ply, num_games


def benchmark(
    stockfish_path: str,
    node_counts: list[int],
    sample_size: int,
    num_workers: int,
    hash_mb: int,
):
    TARGET_GAMES = 100_000
    TARGET_SECONDS = 3600

    total_games = sample_size * num_workers

    print(f"Stockfish: {stockfish_path}")
    print(f"Workers: {num_workers} (1 engine per worker, 1 thread per engine)")
    print(f"Sample: {sample_size} games/worker x {num_workers} workers = {total_games} games per node count")
    print(f"Target: {TARGET_GAMES:,} games in {TARGET_SECONDS // 60} minutes")
    print()
    print(
        f"{'Nodes':>8}  {'Wall (s)':>9}  {'G/s (1w)':>9}  {'G/s (all)':>10}  "
        f"{'Avg Ply':>8}  {'Est 100K':>10}  {'Fits?':>5}"
    )
    print("-" * 75)

    ctx = get_context("spawn")

    for nodes in node_counts:
        # Distribute games across workers
        wall_t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(worker_play_games, stockfish_path, nodes, sample_size, hash_mb)
                for _ in range(num_workers)
            ]
            results = [f.result() for f in futures]
        wall_elapsed = time.perf_counter() - wall_t0

        # Aggregate
        total_ply = sum(r[1] for r in results)
        total_played = sum(r[2] for r in results)
        avg_worker_time = sum(r[0] for r in results) / num_workers
        avg_ply = total_ply / total_played

        # Rate: total games / wall clock time
        parallel_rate = total_played / wall_elapsed
        single_rate = total_played / sum(r[0] for r in results) * 1  # per-engine rate

        est_seconds = TARGET_GAMES / parallel_rate
        fits = est_seconds <= TARGET_SECONDS

        print(
            f"{nodes:>8,}  {wall_elapsed:>9.2f}  {single_rate:>9.2f}  {parallel_rate:>10.2f}  "
            f"{avg_ply:>8.1f}  {est_seconds / 60:>9.1f}m  {'YES' if fits else 'no':>5}"
        )

    print()
    print("G/s (1w)  = games/sec per single engine")
    print("G/s (all) = aggregate games/sec across all workers (wall clock)")
    print("Est 100K  = estimated minutes to generate 100,000 games at aggregate rate")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Stockfish node counts (parallel)")
    parser.add_argument("--stockfish", type=str, default=os.path.expanduser("~/bin/stockfish"))
    parser.add_argument("--sample", type=int, default=20, help="Games per worker")
    parser.add_argument("--workers", type=int, default=14, help="Number of parallel engines")
    parser.add_argument("--hash", type=int, default=16, help="Hash table MB per engine")
    parser.add_argument(
        "--nodes",
        type=str,
        default="1,5,10,25,50,100,200,500,1000",
        help="Comma-separated node counts to test",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.stockfish):
        print(f"ERROR: Stockfish not found at {args.stockfish}")
        sys.exit(1)

    node_counts = [int(x) for x in args.nodes.split(",")]
    benchmark(args.stockfish, node_counts, args.sample, args.workers, args.hash)


if __name__ == "__main__":
    main()
