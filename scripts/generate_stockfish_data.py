#!/usr/bin/env python3
"""Generate Stockfish self-play data as zstd-compressed Parquet.

Runs a pool of single-threaded Stockfish engines in parallel, one per core.
Each worker writes an incremental Parquet shard to the output directory.
After all workers finish, shards are merged into a single file.

Output schema matches the Lichess dataset format:
    uci (string)        — space-separated UCI moves
    result (string)     — 1-0, 0-1, 1/2-1/2, or *
    n_ply (int16)       — number of half-moves
    nodes (int32)       — Stockfish node budget per move
    worker_id (int16)   — worker index (for reproducibility)
    seed (int32)        — worker RNG seed

Each tier uses MultiPV + softmax temperature sampling to produce diverse games
from deterministic Stockfish search. Seeds are hardcoded per tier so runs are
reproducible. Worker seeds are derived as tier_seed + worker_id.

Tiers (by node count):
    nodes_0001:   1 node   (near-random)
    nodes_0032:  32 nodes
    nodes_0128: 128 nodes
    nodes_0256: 256 nodes
    nodes_1024: 1024 nodes (strongest)

Usage:
    python scripts/generate_stockfish_data.py --stockfish ~/bin/stockfish --output /dev/shm/stockfish
    python scripts/generate_stockfish_data.py --stockfish ~/bin/stockfish --output /dev/shm/stockfish --tier nodes_0001 --games 1000000
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Hardcoded, non-conflicting seeds per tier.  Worker i gets seed = tier_seed + i.
TIERS = [
    {"name": "nodes_0001", "nodes": 1,    "games": 128_000, "seed": 10_000},
    {"name": "nodes_0032", "nodes": 32,   "games": 128_000, "seed": 20_000},
    {"name": "nodes_0128", "nodes": 128,  "games": 128_000, "seed": 30_000},
    {"name": "nodes_0256", "nodes": 256,  "games": 128_000, "seed": 40_000},
    {"name": "nodes_1024", "nodes": 1024, "games": 128_000, "seed": 50_000},
]

MULTI_PV = 5        # candidates per move during opening
TEMPERATURE = 1.0   # softmax temperature (higher = more random)
SAMPLE_PLIES = 999  # use MultiPV+temperature for all plies by default

FLUSH_EVERY = 5_000  # games per Parquet row group

SCHEMA = pa.schema([
    ('uci', pa.string()),
    ('result', pa.string()),
    ('n_ply', pa.int16()),
    ('nodes', pa.int32()),
    ('worker_id', pa.int16()),
    ('seed', pa.int32()),
])


class StockfishEngine:
    def __init__(self, path: str, hash_mb: int = 16, multi_pv: int = 1):
        self.last_terminal: str | None = None  # "checkmate", "stalemate", or None
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name Threads value 1")
        if multi_pv > 1:
            self._send(f"setoption name MultiPV value {multi_pv}")
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

    def candidates(self, moves: list[str], nodes: int) -> list[tuple[str, float]]:
        """Return list of (uci_move, score_cp) from MultiPV search.

        Side effect: sets self.last_terminal to "checkmate" or "stalemate"
        when bestmove is (none), otherwise None.
        """
        self.last_terminal = None
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        self._send(f"go nodes {nodes}")
        lines = self._wait_for("bestmove")

        # Parse info lines for multipv results.  Keep the last (deepest) line
        # for each multipv index.
        best_by_pv: dict[int, tuple[str, float]] = {}
        for line in lines:
            if not line.startswith("info") or " multipv " not in line:
                continue
            parts = line.split()
            try:
                pv_idx = int(parts[parts.index("multipv") + 1])
                # Parse score
                si = parts.index("score")
                if parts[si + 1] == "cp":
                    score = float(parts[si + 2])
                elif parts[si + 1] == "mate":
                    mate_in = int(parts[si + 2])
                    score = 30_000.0 if mate_in > 0 else -30_000.0
                else:
                    continue
                # First move of the PV
                pv_start = parts.index("pv")
                move = parts[pv_start + 1]
                best_by_pv[pv_idx] = (move, score)
            except (ValueError, IndexError):
                continue

        if not best_by_pv:
            # Fallback: parse bestmove directly
            for line in lines:
                if line.startswith("bestmove"):
                    parts = line.split()
                    m = parts[1] if len(parts) > 1 else None
                    if m and m != "(none)":
                        return [(m, 0.0)]
                    # No legal moves — distinguish checkmate from stalemate
                    self.last_terminal = "stalemate"  # default
                    for info_line in lines:
                        if not info_line.startswith("info") or "score" not in info_line:
                            continue
                        info_parts = info_line.split()
                        try:
                            si = info_parts.index("score")
                            if info_parts[si + 1] == "mate":
                                self.last_terminal = "checkmate"
                        except (ValueError, IndexError):
                            pass
                    return []
            return []

        # Return sorted by pv index
        return [best_by_pv[k] for k in sorted(best_by_pv)]

    def set_multi_pv(self, n: int):
        self._send(f"setoption name MultiPV value {n}")
        self._send("isready")
        self._wait_for("readyok")

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


def softmax_sample(
    candidates: list[tuple[str, float]], temperature: float, rng: random.Random
) -> str | None:
    """Pick a move from candidates using softmax over centipawn scores."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    # temperature=0 → argmax (pick the highest-scoring move)
    if temperature <= 0:
        return max(candidates, key=lambda c: c[1])[0]

    scores = [s for _, s in candidates]
    # Shift for numerical stability
    max_s = max(scores)
    # Scale: 100 cp ~ 1 pawn.  temperature=1.0 means 1-pawn difference ≈ e fold.
    exps = [math.exp((s - max_s) / (100.0 * temperature)) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]

    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return candidates[i][0]
    return candidates[-1][0]


def play_game(
    engine: StockfishEngine,
    nodes: int,
    rng: random.Random,
    temperature: float,
    multi_pv: int,
    sample_plies: int,
    max_ply: int = 500,
) -> tuple[list[str], str]:
    """Play one self-play game with temperature sampling. Returns (moves_uci, result)."""
    engine.new_game()
    engine.set_multi_pv(multi_pv)
    moves: list[str] = []
    switched = False

    for ply in range(max_ply):
        # Switch to top-1 after the opening phase
        if not switched and ply >= sample_plies:
            engine.set_multi_pv(1)
            switched = True

        cands = engine.candidates(moves, nodes)
        if switched:
            # Top-1 mode: just take the best move
            move = cands[0][0] if cands else None
        else:
            move = softmax_sample(cands, temperature, rng)
        if move is None:
            break
        moves.append(move)

    n = len(moves)
    if n == 0:
        return moves, "*"

    if n >= max_ply:
        result = "1/2-1/2"
    elif engine.last_terminal == "checkmate":
        # Side to move was checkmated
        result = "0-1" if n % 2 == 0 else "1-0"
    elif engine.last_terminal == "stalemate":
        result = "1/2-1/2"
    else:
        result = "*"

    return moves, result


def worker_generate(
    stockfish_path: str,
    nodes: int,
    num_games: int,
    hash_mb: int,
    worker_id: int,
    seed: int,
    temperature: float,
    multi_pv: int,
    sample_plies: int,
    shard_path: str,
) -> tuple[int, int, float]:
    """Worker: play num_games, write Parquet shard incrementally.

    Returns (n_written, total_ply, elapsed).
    """
    rng = random.Random(seed)
    engine = StockfishEngine(stockfish_path, hash_mb=hash_mb, multi_pv=multi_pv)
    writer = pq.ParquetWriter(shard_path, SCHEMA, compression='zstd')

    batch: dict[str, list] = {name: [] for name in SCHEMA.names}
    total_ply = 0
    n_written = 0
    t0 = time.perf_counter()

    for i in range(num_games):
        moves, result = play_game(engine, nodes, rng, temperature, multi_pv, sample_plies)
        n_ply = len(moves)
        total_ply += n_ply

        batch['uci'].append(" ".join(moves))
        batch['result'].append(result)
        batch['n_ply'].append(n_ply)
        batch['nodes'].append(nodes)
        batch['worker_id'].append(worker_id)
        batch['seed'].append(seed)
        n_written += 1

        # Flush batch to Parquet periodically
        if n_written % FLUSH_EVERY == 0:
            table = pa.table(batch, schema=SCHEMA)
            writer.write_table(table)
            batch = {name: [] for name in SCHEMA.names}

            elapsed = time.perf_counter() - t0
            rate = n_written / elapsed
            print(
                f"  [worker {worker_id:>2}] {n_written:>6,}/{num_games:,}  "
                f"{rate:.1f} games/s  (flushed)",
                flush=True,
            )
        elif (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(
                f"  [worker {worker_id:>2}] {i + 1:>6,}/{num_games:,}  "
                f"{rate:.1f} games/s",
                flush=True,
            )

    # Flush remaining
    if batch['uci']:
        table = pa.table(batch, schema=SCHEMA)
        writer.write_table(table)

    writer.close()
    elapsed = time.perf_counter() - t0
    engine.close()
    return n_written, total_ply, elapsed


def generate_tier(
    stockfish_path: str,
    output_dir: Path,
    tier: dict,
    num_workers: int,
    hash_mb: int,
    temperature: float,
    multi_pv: int,
    sample_plies: int,
):
    nodes = tier["nodes"]
    total_games = tier["games"]
    tier_seed = tier["seed"]
    name = tier["name"]

    # Shard directory for worker output
    shard_dir = output_dir / f".shards_{name}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Distribute games across workers
    base = total_games // num_workers
    remainder = total_games % num_workers
    per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    print(f"\n{'=' * 60}")
    print(f"Generating {total_games:,} games: {name} (nodes={nodes})")
    print(f"Workers: {num_workers}, games/worker: ~{base}")
    print(f"MultiPV: {multi_pv} for first {sample_plies} plies, then top-1; temperature: {temperature}")
    print(f"Seed base: {tier_seed} (workers {tier_seed}..{tier_seed + num_workers - 1})")
    print(f"Output: {output_dir / f'{name}.parquet'}")
    print(f"{'=' * 60}")

    ctx = get_context("spawn")

    wall_t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                worker_generate,
                stockfish_path,
                nodes,
                per_worker[i],
                hash_mb,
                i,
                tier_seed + i,
                temperature,
                multi_pv,
                sample_plies,
                str(shard_dir / f"shard_{i:03d}.parquet"),
            )
            for i in range(num_workers)
        ]
        results = [f.result() for f in futures]
    wall_elapsed = time.perf_counter() - wall_t0

    total_written = sum(r[0] for r in results)
    total_ply = sum(r[1] for r in results)
    avg_ply = total_ply / total_written if total_written else 0
    rate = total_written / wall_elapsed

    print(f"\n  Generation done: {total_written:,} games in {wall_elapsed / 60:.1f}m")
    print(f"  Rate: {rate:.1f} games/s  Avg ply: {avg_ply:.0f}")

    # Merge shards into single file
    print("  Merging shards...")
    out_path = output_dir / f"{name}.parquet"
    shard_files = sorted(shard_dir.glob("shard_*.parquet"))
    tables = [pq.read_table(f) for f in shard_files]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, out_path, compression='zstd')

    # Clean up shards
    for f in shard_files:
        f.unlink()
    shard_dir.rmdir()

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Merged: {out_path} ({size_mb:.1f} MB, {len(merged):,} rows)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stockfish self-play data as Parquet (parallel)"
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=os.path.expanduser("~/bin/stockfish"),
        help="Path to stockfish binary",
    )
    parser.add_argument(
        "--output", type=str, default="data/stockfish", help="Output directory"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Only generate this tier (e.g. nodes_0128)",
    )
    parser.add_argument(
        "--workers", type=int, default=14, help="Number of parallel engines"
    )
    parser.add_argument(
        "--games", type=int, default=None, help="Override number of games per tier"
    )
    parser.add_argument(
        "--hash", type=int, default=16, help="Hash table MB per engine"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE, help="Softmax temperature"
    )
    parser.add_argument(
        "--multi-pv", type=int, default=MULTI_PV, help="MultiPV candidates per move"
    )
    parser.add_argument(
        "--sample-plies", type=int, default=SAMPLE_PLIES,
        help="Use MultiPV+temperature for the first N plies, then top-1"
    )
    args = parser.parse_args()

    sf_path = os.path.expanduser(args.stockfish)
    if not os.path.isfile(sf_path):
        print(f"ERROR: Stockfish not found at {sf_path}")
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
        generate_tier(
            sf_path, output_dir, tier, args.workers, args.hash,
            args.temperature, args.multi_pv, args.sample_plies,
        )

    print(f"\nAll done. Files in {output_dir}/")


if __name__ == "__main__":
    main()
