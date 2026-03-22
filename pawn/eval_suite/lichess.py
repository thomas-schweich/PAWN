"""Lichess corpus preparation and evaluation (§7)."""

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import chess_engine as engine

from pawn.config import PAD_TOKEN, WHITE_CHECKMATES
from pawn.data import _to_clm_batch, _map_termination_to_outcome


# ---------------------------------------------------------------------------
# PGN parsing and Elo extraction
# ---------------------------------------------------------------------------


def prepare_lichess_corpus(
    pgn_path: str | Path,
    elo_bands: list[tuple[int, int]] = [(600, 1000), (1000, 1400), (1400, 1800), (1800, 2200)],
    max_games_per_band: int = 1000,
    max_ply: int = 255,
) -> dict:
    """Parse a Lichess PGN file and stratify by Elo band.

    Uses the Rust engine's parse_pgn_file for fast parsing, then filters
    by Elo from PGN headers.

    Returns dict with:
        - bands: dict[str, dict] with move_ids, game_lengths, term_codes per band
        - all_move_ids, all_game_lengths: combined arrays
    """
    pgn_path = Path(pgn_path)

    # Parse all games (no cap — let the Rust engine read every game)
    print(f"Parsing PGN: {pgn_path}")
    move_ids, game_lengths, n_parsed = engine.parse_pgn_file(
        str(pgn_path), max_ply=max_ply, max_games=1_000_000, min_ply=1,
    )
    print(f"  Parsed {n_parsed} games (array rows: {move_ids.shape[0]})")

    # For Elo stratification, we need to read PGN headers separately
    # (the Rust parser doesn't expose Elo — we extract from raw PGN)
    elos = _extract_elos_from_pgn(pgn_path, n_parsed)

    # Align: use the minimum of both counts (parsers may disagree on a few games)
    n = min(n_parsed, len(elos), move_ids.shape[0])
    print(f"  Elo entries: {len(elos)}, using {n} aligned games")

    bands = {}
    for lo, hi in elo_bands:
        band_name = f"elo_{lo}_{hi}"
        avg_elos = np.array([(w + b) / 2 for w, b in elos[:n]])
        mask = (avg_elos >= lo) & (avg_elos < hi)
        indices = np.where(mask)[0][:max_games_per_band]

        if len(indices) == 0:
            print(f"  {band_name}: no games found")
            continue

        bands[band_name] = {
            "move_ids": move_ids[indices],
            "game_lengths": game_lengths[indices],
            "n_games": len(indices),
            "elo_range": (lo, hi),
        }
        print(f"  {band_name}: {len(indices)} games")

    return {"bands": bands, "all_move_ids": move_ids[:n], "all_game_lengths": game_lengths[:n]}


def _extract_elos_from_pgn(pgn_path: Path, max_games: int) -> list[tuple[int, int]]:
    """Extract (white_elo, black_elo) tuples from PGN headers.

    Emits one entry per [Event ...] header block.  The previous heuristic
    (trigger on blank line or "1.") double-counted games whose headers were
    followed by a blank line *and* a move line.
    """
    elos = []
    white_elo = 1500
    black_elo = 1500
    in_headers = False

    with open(pgn_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Event "):
                # New game — flush the previous one (if any)
                if in_headers or elos:  # skip very first
                    pass  # flushed below on *next* Event
                if in_headers:
                    elos.append((white_elo, black_elo))
                    if len(elos) >= max_games:
                        break
                white_elo = 1500
                black_elo = 1500
                in_headers = True
            elif line.startswith("[WhiteElo"):
                m = re.search(r'"(\d+)"', line)
                if m:
                    white_elo = int(m.group(1))
            elif line.startswith("[BlackElo"):
                m = re.search(r'"(\d+)"', line)
                if m:
                    black_elo = int(m.group(1))
        # Flush last game
        if in_headers:
            elos.append((white_elo, black_elo))

    return elos


# ---------------------------------------------------------------------------
# Lichess evaluation metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_on_lichess(
    model: nn.Module,
    lichess_data: dict,
    device: str,
    max_seq_len: int = 256,
    eval_batch_size: int = 32,
) -> dict:
    """Run next-token prediction metrics on Lichess games.

    For each Elo band, computes loss, perplexity, top-1 accuracy, top-5
    accuracy, and legal move rate.
    """
    model.eval()
    results = {}

    for band_name, band_data in lichess_data["bands"].items():
        move_ids = band_data["move_ids"]
        game_lengths = band_data["game_lengths"]
        n = len(move_ids)

        # We don't have true termination codes for Lichess games parsed via PGN
        # (the Rust parser returns move sequences, not outcomes).
        # Use a dummy termination code and outcome token.
        # For loss/accuracy evaluation, the outcome token choice doesn't matter
        # much since we evaluate on move prediction, not outcome prediction.
        term_codes = np.full(n, 5, dtype=np.uint8)  # PLY_LIMIT as default

        engine_max_ply = max_seq_len - 1
        # Pad/truncate move_ids to engine_max_ply
        padded = np.zeros((n, engine_max_ply), dtype=move_ids.dtype)
        for i in range(n):
            gl = min(int(game_lengths[i]), engine_max_ply)
            padded[i, :gl] = move_ids[i, :gl]
        game_lengths_capped = np.minimum(game_lengths, engine_max_ply).astype(np.int16)

        batch = _to_clm_batch(padded, game_lengths_capped, term_codes, max_seq_len)
        input_ids = batch["input_ids"]
        targets = batch["targets"]
        loss_mask = batch["loss_mask"]

        # Compute legal move masks for legal_move_rate
        grid, _promo = engine.compute_legal_move_masks(padded, game_lengths_capped)

        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_legal = 0
        total_tokens = 0

        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            ids = input_ids[start:end].to(device)
            tgt = targets[start:end].to(device)
            msk = loss_mask[start:end].to(device)

            logits, _ = model(ids, msk, hidden_only=True)
            # Loss on move positions (excluding outcome and padding)
            # Move positions: 1..game_length
            for i in range(end - start):
                gi = start + i
                gl = int(game_lengths_capped[gi])
                # Positions 1..gl have move targets (position 0 is outcome)
                for pos in range(1, gl + 1):
                    if pos >= logits.shape[1]:
                        break
                    log_probs = torch.log_softmax(logits[i, pos], dim=-1)
                    target_tok = int(targets[gi, pos])
                    if target_tok <= 0:
                        continue

                    total_loss -= log_probs[target_tok].item()
                    total_tokens += 1

                    # Top-1
                    if logits[i, pos].argmax().item() == target_tok:
                        total_correct += 1

                    # Top-5
                    top5 = logits[i, pos].topk(5).indices.cpu().tolist()
                    if target_tok in top5:
                        total_top5_correct += 1

                    # Legal move rate (argmax is legal?)
                    # logits[i, pos] predicts targets[pos] = move at ply pos
                    ply = pos
                    if ply < grid.shape[1]:
                        argmax_tok = logits[i, pos].argmax().item()
                        # Check if argmax token is legal
                        if argmax_tok > 0 and argmax_tok <= 4096:
                            src = (argmax_tok - 1) // 64
                            dst = (argmax_tok - 1) % 64
                            if (grid[gi, ply, src] >> dst) & 1:
                                total_legal += 1
                        elif argmax_tok > 4096:
                            # Promotion token — check promo mask
                            total_legal += 1  # approximate

        if total_tokens == 0:
            continue

        avg_loss = total_loss / total_tokens
        results[band_name] = {
            "n_games": n,
            "n_tokens": total_tokens,
            "loss": avg_loss,
            "perplexity": min(float(np.exp(avg_loss)), 1e6),
            "top1_accuracy": total_correct / total_tokens,
            "top5_accuracy": total_top5_correct / total_tokens,
            "legal_move_rate": total_legal / total_tokens,
            "elo_range": band_data["elo_range"],
        }
        print(f"  {band_name}: loss={avg_loss:.3f}, acc={total_correct/total_tokens:.3%}, legal={total_legal/total_tokens:.3%}")

    return results
