#!/usr/bin/env python3
"""Play a PAWN adapter-on-backbone model against Stockfish (UCI_LimitStrength).

Usage:
    uv run --extra rocm python scripts/eval_vs_stockfish.py \\
        --checkpoint thomas-schweich/pawn-base \\
        --adapter-checkpoint runs/logs/trial_0022/bottleneck_.../checkpoints/step_00020000 \\
        --stockfish ~/bin/stockfish --stockfish-elo 1320 \\
        --games 100 --movetime-ms 1 \\
        --output runs/eval/stockfish_vs_dim512.json

Goal
----
Triangulate the effective playing strength of the adapted backbone: a
win rate near 50% against Stockfish at a given UCI_Elo approximates the
model's Elo at that time control.

Notes
-----
- PAWN plays greedy (argmax) over legal moves at every ply. The
  v1.0.0 ``prepend_outcome=False`` sequence format can't predict the
  very first move (no training supervision at position 0), so when
  PAWN plays white we open with ``e2e4`` and let the model take over
  from move 2.
- Stockfish is constrained via ``UCI_LimitStrength=true`` + ``UCI_Elo``
  and ``go movetime <ms>``. Both limits apply — per-move time stays
  low so games complete in reasonable wall time.
- Colors alternate per game so the result is symmetric.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Stockfish UCI wrapper (adapted from scripts/benchmark_stockfish_nodes.py)
# ---------------------------------------------------------------------------


class StockfishEngine:
    def __init__(
        self,
        path: str,
        elo: int = 1320,
        hash_mb: int = 16,
        threads: int = 1,
    ) -> None:
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
        self._send(f"setoption name Threads value {threads}")
        self._send("setoption name UCI_LimitStrength value true")
        self._send(f"setoption name UCI_Elo value {elo}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str) -> None:
        self._stdin.write(cmd + "\n")
        self._stdin.flush()

    def _wait_for(self, token: str) -> list[str]:
        lines: list[str] = []
        while True:
            line = self._stdout.readline().strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def best_move(self, moves_uci: list[str], movetime_ms: int) -> str | None:
        pos = "position startpos"
        if moves_uci:
            pos += " moves " + " ".join(moves_uci)
        self._send(pos)
        self._send(f"go movetime {movetime_ms}")
        for line in self._wait_for("bestmove"):
            if line.startswith("bestmove"):
                parts = line.split()
                mv = parts[1] if len(parts) > 1 else None
                return None if mv in (None, "(none)") else mv
        return None

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# PAWN adapter-backbone move selector
# ---------------------------------------------------------------------------


@torch.no_grad()
def pawn_pick_move(
    model: torch.nn.Module,
    history_tokens: list[int],
    legal_tokens: list[int],
    device: str,
    temperature: float,
    rng: torch.Generator,
) -> int:
    """Return the token id of a legal move under ``temperature`` sampling.

    ``temperature == 0`` is greedy (argmax over legal logits).
    ``temperature > 0`` samples from ``softmax(logits / T)`` restricted
    to legal moves.

    With ``prepend_outcome=False`` the v1.0.0 data layout supervises
    position 0 to predict move 2 from move 1. That means move 1 is
    never a training target, so this function requires ``history_tokens``
    to be non-empty; the caller handles the first-move case (book).
    """
    assert history_tokens, "pawn_pick_move needs at least one prior move"
    ids = torch.tensor([history_tokens], dtype=torch.long, device=device)
    if hasattr(model, "forward_hidden"):
        hidden = model.forward_hidden(ids)  # type: ignore[operator]
        logits = model.project_head(hidden[:, -1, :])  # type: ignore[operator]
    else:
        logits = model(ids)[:, -1, :]
    legal = torch.tensor(legal_tokens, dtype=torch.long, device=device)
    legal_logits = logits[0, legal].float()
    if temperature <= 0.0:
        return int(legal[legal_logits.argmax()].item())
    probs = torch.softmax(legal_logits / temperature, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
    return int(legal[idx].item())


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


# Result codes on PyGameState.check_termination():
#   0=Checkmate, 1=Stalemate, 2=SeventyFiveMoveRule,
#   3=FivefoldRepetition, 4=InsufficientMaterial, 5=PlyLimit
# Draw set: stalemate, 75-move, 5-fold, insufficient material, ply limit
DRAW_CODES = {1, 2, 3, 4, 5}


def play_one_game(
    model: torch.nn.Module,
    sf: StockfishEngine,
    pawn_plays_white: bool,
    device: str,
    uci_to_token: dict[str, int],
    token_to_uci: list[str],
    movetime_ms: int,
    max_ply: int,
    first_move_uci: str | None,
    temperature: float,
    rng: torch.Generator,
) -> dict[str, Any]:
    """Play one game. Return result dict with outcome from PAWN's POV."""
    import chess_engine

    state = chess_engine.PyGameState(max_ply=max_ply)
    sf.new_game()
    moves_uci: list[str] = []
    moves_tokens: list[int] = []

    while True:
        term = state.check_termination()
        if term != -1:
            if term == 0:  # Checkmate — current side is mated
                # side_to_move just lost. We know whose turn it is.
                white_to_move = state.is_white_to_move()
                # If white-to-move is in checkmate, black won.
                white_won = not white_to_move
                pawn_won = white_won == pawn_plays_white
                result = "win" if pawn_won else "loss"
            elif term in DRAW_CODES:
                result = "draw"
            else:
                result = "draw"
            return {
                "result": result,
                "term": int(term),
                "plies": len(moves_uci),
                "pawn_white": pawn_plays_white,
                "first_move": first_move_uci if pawn_plays_white else None,
                "moves": moves_uci,
            }

        white_to_move = state.is_white_to_move()
        pawn_to_move = white_to_move == pawn_plays_white

        if pawn_to_move:
            if len(moves_uci) == 0:
                # First move of the game, PAWN plays white: book opening.
                if first_move_uci is None:
                    raise RuntimeError(
                        "PAWN plays white but no book move was provided"
                    )
                uci = first_move_uci
                tok = uci_to_token.get(uci)
                if tok is None:
                    raise RuntimeError(
                        f"book move {uci} missing from move_to_token vocab"
                    )
            else:
                legal_tokens = state.legal_move_tokens()
                if not legal_tokens:
                    # No legal moves; termination should have fired.
                    break
                tok = pawn_pick_move(
                    model, moves_tokens, legal_tokens, device,
                    temperature=temperature, rng=rng,
                )
                uci = token_to_uci[tok]
        else:
            sf_uci = sf.best_move(moves_uci, movetime_ms)
            if sf_uci is None:
                # Stockfish thinks no legal move — let termination check handle it.
                break
            uci = sf_uci
            tok = uci_to_token.get(uci)
            if tok is None:
                # Stockfish produced a move outside our 1968-action vocab
                # (e.g. an underpromotion we don't cover). Apply it via UCI
                # by finding the matching legal token.
                tok = _find_token_for_uci(
                    state.legal_move_tokens(), uci, token_to_uci,
                )
                if tok is None:
                    raise RuntimeError(
                        f"Stockfish move {uci} has no matching token"
                    )

        ok = state.make_move(tok)
        if not ok:
            raise RuntimeError(
                f"Engine rejected move {uci} (token {tok}) at ply "
                f"{len(moves_uci)}"
            )
        moves_uci.append(uci)
        moves_tokens.append(tok)

    # Fallthrough (shouldn't happen if termination is checked first).
    return {
        "result": "draw",
        "term": -1,
        "plies": len(moves_uci),
        "pawn_white": pawn_plays_white,
        "first_move": first_move_uci if pawn_plays_white else None,
        "moves": moves_uci,
    }


def _find_token_for_uci(
    legal_tokens: list[int], uci: str, token_to_uci: list[str],
) -> int | None:
    for tok in legal_tokens:
        if token_to_uci[tok] == uci:
            return tok
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_match(
    model: torch.nn.Module,
    sf: StockfishEngine,
    device: str,
    uci_to_token: dict[str, int],
    token_to_uci: list[str],
    movetime_ms: int,
    max_ply: int,
    first_moves: list[str],
    n_games: int,
    temperature: float,
    rng: torch.Generator,
    label: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Play ``n_games`` PAWN-vs-Stockfish with alternating colors.

    When PAWN plays white, the book move rotates through
    ``first_moves`` so each opening is sampled equally. The rotation is
    deterministic on the white-game index, independent of the overall
    game index, so adding more temperatures or games doesn't reshuffle.
    """
    assert first_moves, "run_match needs at least one book opening"
    wins = draws = losses = 0
    w_w = w_d = w_l = 0
    b_w = b_d = b_l = 0
    per_opening: dict[str, dict[str, int]] = {
        mv: {"w": 0, "d": 0, "l": 0} for mv in first_moves
    }
    games: list[dict[str, Any]] = []
    t0 = time.time()

    white_game_idx = 0
    for i in range(n_games):
        pawn_white = (i % 2 == 0)
        if pawn_white:
            first_move = first_moves[white_game_idx % len(first_moves)]
            white_game_idx += 1
        else:
            first_move = None
        g = play_one_game(
            model=model, sf=sf, pawn_plays_white=pawn_white,
            device=device,
            uci_to_token=uci_to_token,
            token_to_uci=token_to_uci,
            movetime_ms=movetime_ms,
            max_ply=max_ply,
            first_move_uci=first_move,
            temperature=temperature,
            rng=rng,
        )
        g["temperature"] = temperature
        if g["result"] == "win":
            wins += 1
            if pawn_white:
                w_w += 1
                per_opening[first_move]["w"] += 1  # type: ignore[index]
            else:
                b_w += 1
        elif g["result"] == "loss":
            losses += 1
            if pawn_white:
                w_l += 1
                per_opening[first_move]["l"] += 1  # type: ignore[index]
            else:
                b_l += 1
        else:
            draws += 1
            if pawn_white:
                w_d += 1
                per_opening[first_move]["d"] += 1  # type: ignore[index]
            else:
                b_d += 1
        games.append(g)
        if (i + 1) % 10 == 0 or i == n_games - 1:
            score = wins + 0.5 * draws
            print(
                f"  [{label} {i+1}/{n_games}] "
                f"W/D/L={wins}/{draws}/{losses} "
                f"score_rate={(score) / (i + 1):.3f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    summary = {
        "temperature": temperature,
        "total_games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": wins + 0.5 * draws,
        "score_rate": (wins + 0.5 * draws) / max(n_games, 1),
        "winrate_strict": wins / max(n_games, 1),
        "as_white_wdl": [w_w, w_d, w_l],
        "as_black_wdl": [b_w, b_d, b_l],
        "per_opening": per_opening,
        "wall_seconds": time.time() - t0,
    }
    return summary, games


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="HF repo id or local path to PAWN backbone")
    p.add_argument("--adapter-checkpoint", required=True,
                   help="Path to adapter checkpoint directory")
    p.add_argument("--stockfish", default=str(Path.home() / "bin" / "stockfish"))
    p.add_argument("--stockfish-elo", type=int, default=1320)
    p.add_argument("--movetime-ms", type=int, default=1)
    p.add_argument("--games", type=int, default=100,
                   help="Games per temperature setting")
    p.add_argument("--max-ply", type=int, default=400)
    p.add_argument("--first-moves", default="e2e4,d2d4,g1f3",
                   help="Comma-separated list of PAWN's white-opening "
                        "book moves. When PAWN plays white the book "
                        "move rotates across this list so each opening "
                        "gets an equal share of the white games. Move 1 "
                        "is untrained with prepend_outcome=False.")
    p.add_argument("--temperatures", default="0.0,0.5,1.0",
                   help="Comma-separated list of sampling temperatures. "
                        "0 = greedy argmax.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None,
                   help="Write per-game JSON here (default: stdout summary only)")
    args = p.parse_args()

    temperatures = [float(x) for x in args.temperatures.split(",") if x.strip()]
    first_moves = [m.strip() for m in args.first_moves.split(",") if m.strip()]
    if not first_moves:
        raise SystemExit("--first-moves must contain at least one UCI move")

    torch.manual_seed(args.seed)
    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    print(f"loading backbone: {args.checkpoint}", flush=True)
    print(f"loading adapter : {args.adapter_checkpoint}", flush=True)
    from scripts.eval_accuracy import load_model
    from pawn.gpu import apply_gpu_config, configure_gpu

    gpu_cfg = configure_gpu(args.device, no_compile=False, no_amp=False)
    model, adapter_type = load_model(
        args.checkpoint, args.adapter_checkpoint, args.device,
    )
    print(f"adapter type: {adapter_type}", flush=True)

    from pawn import model as model_module
    model.forward_hidden = apply_gpu_config(  # type: ignore[method-assign]
        gpu_cfg, model_module, model.forward_hidden,  # type: ignore[attr-defined]
    )
    model.eval()

    import chess_engine
    vocab = chess_engine.export_move_vocabulary()
    uci_to_token: dict[str, int] = vocab["move_to_token"]
    token_to_uci: list[str] = vocab["token_to_move"]

    print(f"launching stockfish at {args.stockfish} (UCI_Elo={args.stockfish_elo})",
          flush=True)
    sf = StockfishEngine(args.stockfish, elo=args.stockfish_elo)

    print(f"first-move book (rotated across white games): {first_moves}",
          flush=True)

    all_summaries: list[dict[str, Any]] = []
    all_games: list[dict[str, Any]] = []
    try:
        for temp in temperatures:
            print(f"\n=== Temperature {temp} ===", flush=True)
            summary, games = run_match(
                model=model, sf=sf, device=args.device,
                uci_to_token=uci_to_token, token_to_uci=token_to_uci,
                movetime_ms=args.movetime_ms, max_ply=args.max_ply,
                first_moves=first_moves, n_games=args.games,
                temperature=temp, rng=rng, label=f"T={temp}",
            )
            all_summaries.append(summary)
            all_games.extend(games)
    finally:
        sf.close()

    print("\n=== Final Summary (per temperature) ===")
    print(f"{'T':>6} {'W':>4} {'D':>4} {'L':>4} {'score_rate':>12} {'wall_s':>8}")
    for s in all_summaries:
        print(
            f"{s['temperature']:>6.2f} "
            f"{s['wins']:>4} {s['draws']:>4} {s['losses']:>4} "
            f"{s['score_rate']:>12.3f} {s['wall_seconds']:>8.1f}"
        )
    print("\n=== Per-opening breakdown ===")
    print(f"{'T':>6} {'opening':>8} {'W':>4} {'D':>4} {'L':>4} {'score_rate':>12}")
    for s in all_summaries:
        for opening, w in s["per_opening"].items():
            n = w["w"] + w["d"] + w["l"]
            rate = (w["w"] + 0.5 * w["d"]) / max(n, 1)
            print(
                f"{s['temperature']:>6.2f} {opening:>8} "
                f"{w['w']:>4} {w['d']:>4} {w['l']:>4} {rate:>12.3f}"
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": {
                "checkpoint": args.checkpoint,
                "adapter_checkpoint": args.adapter_checkpoint,
                "stockfish_elo": args.stockfish_elo,
                "movetime_ms": args.movetime_ms,
                "games_per_temperature": args.games,
                "temperatures": temperatures,
                "first_moves": first_moves,
                "seed": args.seed,
            },
            "summaries": all_summaries,
            "games": all_games,
        }, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
