#!/usr/bin/env python3
"""Generate Stockfish self-play data as zstd-compressed Parquet.

Runs a pool of single-threaded Stockfish engines in parallel, one per core.
Each worker writes an incremental Parquet shard to the output directory.
After all workers finish, shards are merged into a single file.

Output schema mirrors the columns of the Lichess dataset that apply to
self-play games (no Elo, no clocks, no headers):

    tokens (List[Int16])    — searchless_chess action tokens (0..1967)
    san (List[Utf8])        — SAN with check/mate suffixes
    uci (List[Utf8])        — UCI move strings as Stockfish emitted them
    game_length (UInt16)    — number of plies (== len of each list above)
    outcome_token (UInt16)  — outcome in the 1969..1979 range (vocab.rs)
    result (Utf8)           — 1-0 / 0-1 / 1/2-1/2

Plus self-play-specific metadata:

    nodes (Int32)           — Stockfish node budget per move
    temperature (Float32)   — softmax temperature used for sampling
    sample_plies (Int32)    — # of opening plies sampled before going top-1
    worker_id (Int16)       — worker index (for reproducibility)
    seed (Int32)            — worker RNG seed

Token / SAN derivation goes through `chess_engine` (the Rust crate) — no
Python chess library. UCI is what Stockfish prints; the engine
re-derives tokens and SAN from those UCI strings, so all three columns
agree by construction.

Each tier uses MultiPV + softmax temperature sampling to produce diverse
games from deterministic Stockfish search. All per-tier and per-worker
RNG seeds are derived from a single ``--seed`` master via the hierarchy
master → per-tier seed → per-worker seed (each level a fresh
``random.Random(parent_seed).randrange(2**31)``). Per-tier derivation
iterates the full TIERS list in declaration order, so collisions across
tiers / workers are astronomically unlikely regardless of how many
workers or games each tier uses, and ``--tier nodes_0128`` alone yields
the same worker seeds as a full multi-tier run.

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

import chess_engine
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SEED = 42

TIERS = [
    {"name": "nodes_0001", "nodes": 1,    "games": 128_000},
    {"name": "nodes_0032", "nodes": 32,   "games": 128_000},
    {"name": "nodes_0128", "nodes": 128,  "games": 128_000},
    {"name": "nodes_0256", "nodes": 256,  "games": 128_000},
    {"name": "nodes_1024", "nodes": 1024, "games": 128_000},
]


def derive_tier_seeds(master_seed: int) -> dict[str, int]:
    """Derive a per-tier seed deterministically from the master seed.

    Always iterates the full TIERS list (in declaration order) so that
    ``--tier X`` selects the same seed for tier X as a full multi-tier run,
    independent of which other tiers were requested.
    """
    rng = random.Random(master_seed)
    return {tier["name"]: rng.randrange(2**31) for tier in TIERS}


def derive_worker_seeds(tier_seed: int, num_workers: int) -> list[int]:
    """Derive worker seeds deterministically from a tier seed."""
    rng = random.Random(tier_seed)
    return [rng.randrange(2**31) for _ in range(num_workers)]

MULTI_PV = 5        # candidates per move during opening
OPENING_MULTI_PV = 20  # widen to all reasonable first moves at ply 0
OPENING_PLIES = 1   # how many opening plies use OPENING_MULTI_PV
TEMPERATURE = 1.0   # softmax temperature (higher = more random)
SAMPLE_PLIES = 999  # use MultiPV+temperature for all plies by default
MAX_PLY = 512       # matches the Lichess pipeline default

FLUSH_EVERY = 5_000  # games per Parquet row group / token+SAN conversion batch

# Outcome token IDs — must match engine/src/vocab.rs. Duplicated here so this
# script stays runnable in slim images that only have chess_engine + pyarrow.
WHITE_CHECKMATES = 1969
BLACK_CHECKMATES = 1970
STALEMATE = 1971
DRAW_BY_RULE = 1972
PLY_LIMIT = 1973

SCHEMA = pa.schema([
    ('tokens', pa.list_(pa.int16())),
    ('san', pa.list_(pa.string())),
    ('uci', pa.list_(pa.string())),
    ('game_length', pa.uint16()),
    ('outcome_token', pa.uint16()),
    ('result', pa.string()),
    ('nodes', pa.int32()),
    ('temperature', pa.float32()),
    ('sample_plies', pa.int32()),
    ('multi_pv', pa.int32()),
    ('opening_multi_pv', pa.int32()),
    ('opening_plies', pa.int32()),
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
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name Threads value 1")
        if multi_pv > 1:
            self._send(f"setoption name MultiPV value {multi_pv}")
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


def classify_outcome(n_ply: int, terminal: str | None, max_ply: int) -> tuple[str, int]:
    """Return (result, outcome_token) for a finished self-play game.

    `terminal` is the engine's `last_terminal` after the loop exited:
    "checkmate", "stalemate", or None. Self-play has no resign / agreement /
    clock, so the only valid outcomes are the natural-termination subset
    that pretraining already uses (WHITE/BLACK_CHECKMATES, STALEMATE,
    DRAW_BY_RULE, PLY_LIMIT).
    """
    if n_ply >= max_ply:
        return "1/2-1/2", PLY_LIMIT
    if terminal == "checkmate":
        # Side to move was checkmated, so the *previous* mover won.
        if n_ply % 2 == 1:
            return "1-0", WHITE_CHECKMATES
        return "0-1", BLACK_CHECKMATES
    if terminal == "stalemate":
        return "1/2-1/2", STALEMATE
    # No terminal info and we're under the ply cap — treat as draw-by-rule.
    return "1/2-1/2", DRAW_BY_RULE


def play_game(
    engine: StockfishEngine,
    nodes: int,
    rng: random.Random,
    temperature: float,
    multi_pv: int,
    opening_multi_pv: int,
    opening_plies: int,
    sample_plies: int,
    max_ply: int = MAX_PLY,
) -> tuple[list[str], str, int]:
    """Play one self-play game with temperature sampling.

    Three MultiPV phases:
      1. plies [0, opening_plies)        → opening_multi_pv  (wide first move sampling)
      2. plies [opening_plies, sample_plies) → multi_pv      (normal exploration)
      3. plies [sample_plies, max_ply)   → 1                 (top-1, exploit)

    Returns (moves_uci, result, outcome_token). Empty-game (no legal first
    move) is reported as PLY_LIMIT to keep every emitted row classifiable;
    the caller filters n_ply == 0 anyway.
    """
    engine.new_game()
    moves: list[str] = []
    current_pv = -1  # force a set_multi_pv on the first ply

    for ply in range(max_ply):
        # Pick the right MultiPV bucket for this ply.
        if ply < opening_plies:
            target_pv = opening_multi_pv
        elif ply < sample_plies:
            target_pv = multi_pv
        else:
            target_pv = 1
        if target_pv != current_pv:
            engine.set_multi_pv(target_pv)
            current_pv = target_pv

        cands = engine.candidates(moves, nodes)
        if target_pv == 1:
            move = cands[0][0] if cands else None
        else:
            move = softmax_sample(cands, temperature, rng)
        if move is None:
            break
        moves.append(move)

    terminal = engine.last_terminal
    n = len(moves)
    result, outcome_token = classify_outcome(n, terminal, max_ply)
    return moves, result, outcome_token


def _flush_batch(
    writer: pq.ParquetWriter,
    uci_games: list[list[str]],
    metas: list[dict],
    max_ply: int,
) -> None:
    """Convert a batch of finished games to tokens + SAN and append to parquet.

    Drops any game where the engine couldn't tokenize every UCI move (should
    never happen for legal Stockfish output, but a single corrupt game
    shouldn't sink the whole shard).
    """
    if not uci_games:
        return
    san_per_game = chess_engine.uci_to_san(uci_games)
    tokens_arr, lens_arr = chess_engine.uci_to_tokens(uci_games, max_ply=max_ply)

    rows: dict[str, list] = {name: [] for name in SCHEMA.names}
    for i, meta in enumerate(metas):
        n = int(lens_arr[i])
        if n != len(uci_games[i]) or n == 0:
            # Tokenization rejected a move (or game empty) — skip the row
            # rather than emit mismatched columns.
            continue
        rows['tokens'].append(tokens_arr[i, :n].tolist())
        rows['san'].append(san_per_game[i][:n])
        rows['uci'].append(uci_games[i][:n])
        rows['game_length'].append(n)
        rows['outcome_token'].append(meta['outcome_token'])
        rows['result'].append(meta['result'])
        rows['nodes'].append(meta['nodes'])
        rows['temperature'].append(meta['temperature'])
        rows['sample_plies'].append(meta['sample_plies'])
        rows['multi_pv'].append(meta['multi_pv'])
        rows['opening_multi_pv'].append(meta['opening_multi_pv'])
        rows['opening_plies'].append(meta['opening_plies'])
        rows['worker_id'].append(meta['worker_id'])
        rows['seed'].append(meta['seed'])

    if not rows['tokens']:
        return
    table = pa.table(rows, schema=SCHEMA)
    writer.write_table(table)


def worker_generate(
    stockfish_path: str,
    nodes: int,
    num_games: int,
    hash_mb: int,
    worker_id: int,
    seed: int,
    temperature: float,
    multi_pv: int,
    opening_multi_pv: int,
    opening_plies: int,
    sample_plies: int,
    max_ply: int,
    shard_path: str,
) -> tuple[int, int, float]:
    """Worker: play num_games, write Parquet shard incrementally.

    Returns (n_written, total_ply, elapsed).
    """
    rng = random.Random(seed)
    # Initial MultiPV doesn't matter; play_game sets it explicitly per ply.
    engine = StockfishEngine(stockfish_path, hash_mb=hash_mb, multi_pv=max(multi_pv, opening_multi_pv))
    writer = pq.ParquetWriter(shard_path, SCHEMA, compression='zstd')

    pending_uci: list[list[str]] = []
    pending_meta: list[dict] = []
    total_ply = 0
    n_written = 0
    t0 = time.perf_counter()

    for i in range(num_games):
        moves, result, outcome_token = play_game(
            engine, nodes, rng, temperature, multi_pv,
            opening_multi_pv, opening_plies, sample_plies, max_ply,
        )
        n_ply = len(moves)
        if n_ply == 0:
            # Stockfish refused to make a first move — extremely rare, skip.
            continue
        total_ply += n_ply

        pending_uci.append(moves)
        pending_meta.append({
            'result': result,
            'outcome_token': outcome_token,
            'nodes': nodes,
            'temperature': temperature,
            'sample_plies': sample_plies,
            'multi_pv': multi_pv,
            'opening_multi_pv': opening_multi_pv,
            'opening_plies': opening_plies,
            'worker_id': worker_id,
            'seed': seed,
        })
        n_written += 1

        if n_written % FLUSH_EVERY == 0:
            _flush_batch(writer, pending_uci, pending_meta, max_ply)
            pending_uci.clear()
            pending_meta.clear()

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

    _flush_batch(writer, pending_uci, pending_meta, max_ply)
    writer.close()
    elapsed = time.perf_counter() - t0
    engine.close()
    return n_written, total_ply, elapsed


def generate_tier(
    stockfish_path: str,
    output_dir: Path,
    tier: dict,
    tier_seed: int,
    num_workers: int,
    hash_mb: int,
    temperature: float,
    multi_pv: int,
    opening_multi_pv: int,
    opening_plies: int,
    sample_plies: int,
    max_ply: int,
):
    nodes = tier["nodes"]
    total_games = tier["games"]
    name = tier["name"]

    # Shard directory for worker output
    shard_dir = output_dir / f".shards_{name}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Distribute games across workers
    base = total_games // num_workers
    remainder = total_games % num_workers
    per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]
    worker_seeds = derive_worker_seeds(tier_seed, num_workers)

    print(f"\n{'=' * 60}")
    print(f"Generating {total_games:,} games: {name} (nodes={nodes})")
    print(f"Workers: {num_workers}, games/worker: ~{base}, max_ply: {max_ply}")
    print(f"MultiPV: {opening_multi_pv} for first {opening_plies} plies, "
          f"{multi_pv} for next {sample_plies - opening_plies} plies, then top-1; "
          f"temperature: {temperature}")
    print(f"Tier seed: {tier_seed}  (worker seeds derived from it)")
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
                worker_seeds[i],
                temperature,
                multi_pv,
                opening_multi_pv,
                opening_plies,
                sample_plies,
                max_ply,
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
        "--opening-multi-pv", type=int, default=OPENING_MULTI_PV,
        help="Wider MultiPV used only for the first --opening-plies moves "
             "(broadens first-move coverage without slowing down the rest)",
    )
    parser.add_argument(
        "--opening-plies", type=int, default=OPENING_PLIES,
        help="Number of plies that use --opening-multi-pv before falling back to --multi-pv",
    )
    parser.add_argument(
        "--sample-plies", type=int, default=SAMPLE_PLIES,
        help="Use MultiPV+temperature for the first N plies, then top-1"
    )
    parser.add_argument(
        "--max-ply", type=int, default=MAX_PLY,
        help=f"Per-game ply cap (default {MAX_PLY}, matches the Lichess pipeline)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=(
            f"Master RNG seed (default {DEFAULT_SEED}). All per-tier and "
            "per-worker seeds are derived from this; the same --seed always "
            "produces the same per-tier seeds regardless of --workers / --games."
        ),
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

    tier_seeds = derive_tier_seeds(args.seed)
    for tier in tiers:
        if args.games is not None:
            tier = {**tier, "games": args.games}
        generate_tier(
            sf_path, output_dir, tier, tier_seeds[tier["name"]],
            args.workers, args.hash,
            args.temperature, args.multi_pv,
            args.opening_multi_pv, args.opening_plies,
            args.sample_plies, args.max_ply,
        )

    print(f"\nAll done. Files in {output_dir}/")


if __name__ == "__main__":
    main()
