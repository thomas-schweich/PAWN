"""Lichess corpus preparation and evaluation (§7)."""

import re
from pathlib import Path

import numpy as np
import torch

import chess_engine as engine

from pawn.config import NUM_ACTIONS, PAD_TOKEN, WHITE_CHECKMATES, PLY_LIMIT
from pawn.data import pack_clm_sequences, _map_termination_to_outcome
from pawn.model import PAWNCLM
from pawn.trainer import _get_action_grid_index


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
                if in_headers:
                    elos.append((white_elo, black_elo))
                    if len(elos) >= max_games:
                        in_headers = False
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
    model: PAWNCLM,
    lichess_data: dict,
    device: str,
    max_seq_len: int | None = None,
    eval_batch_size: int = 32,
    prepend_outcome: bool = False,
) -> dict:
    """Run next-token prediction metrics on Lichess games.

    ``max_seq_len`` defaults to the loaded model's configured context
    length (``model.cfg.max_seq_len``) so the same eval function works
    for 256-ctx and 512-ctx checkpoints without callers having to
    thread the value through manually.

    ``prepend_outcome`` must match how the model was trained: when
    False (the current default, pure-moves), the dummy outcome token
    is stripped so predictions are made at the positions the model
    actually saw during training. Mis-setting this flag silently
    shifts every prediction by one position and degrades every metric.

    For each Elo band, computes loss, perplexity, top-1 accuracy,
    top-5 accuracy, and legal move rate.
    """
    if max_seq_len is None:
        max_seq_len = model.cfg.max_seq_len
    model.eval()
    results = {}

    # Searchless action → (src*64 + dst) lookup for legality decoding.
    action_grid_idx = _get_action_grid_index(device, NUM_ACTIONS)

    for band_name, band_data in lichess_data["bands"].items():
        move_ids = band_data["move_ids"]
        game_lengths = band_data["game_lengths"]
        n = len(move_ids)

        # We don't have true termination codes for Lichess games parsed via PGN.
        # Use PLY_LIMIT as a dummy outcome token for all games — for
        # loss/accuracy evaluation the outcome token choice doesn't matter
        # since we evaluate on move prediction, not outcome prediction.
        dummy_outcomes = torch.full((n,), PLY_LIMIT, dtype=torch.long)

        # In outcome-prefixed mode, slot 0 holds the outcome so only
        # ``max_seq_len - 1`` moves fit; in pure-moves mode all
        # ``max_seq_len`` slots hold moves.
        engine_max_ply = max_seq_len - 1 if prepend_outcome else max_seq_len
        padded = np.zeros((n, engine_max_ply), dtype=move_ids.dtype)
        for i in range(n):
            gl = min(int(game_lengths[i]), engine_max_ply)
            padded[i, :gl] = move_ids[i, :gl]
        game_lengths_capped = np.minimum(game_lengths, engine_max_ply).astype(np.int16)

        # `pack_clm_sequences` picks the right layout from `prepend_outcome`
        # — no post-hoc stripping.
        batch = pack_clm_sequences(
            padded, game_lengths_capped, dummy_outcomes, max_seq_len,
            prepend_outcome=prepend_outcome,
        )

        input_ids = batch["input_ids"]
        targets = batch["targets"]
        loss_mask = batch["loss_mask"]

        # Position `pos` in the sequence predicts the move at 0-based ply
        # `pos + grid_offset`:
        #   outcome-prefixed: logits[0] predicts m_1 (ply 0), so
        #       grid_offset = 0;
        #   pure-moves: logits[0] predicts m_2 (ply 1), so grid_offset = 1.
        grid_offset = 0 if prepend_outcome else 1

        # Per-game legality grid: (n, max_ply, 64) bit-packed dst masks.
        grid_np, _promo = engine.compute_legal_move_masks(padded, game_lengths_capped)
        grid = torch.from_numpy(grid_np).to(device)  # (n, max_ply, 64)

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

            # ``forward_eval`` returns final hidden states without the
            # ``lm_head`` projection, so we only pay for the
            # ``(N_valid, vocab_size)`` projection below instead of
            # materialising a full ``(B, T, vocab_size)`` logits tensor.
            hidden = model.forward_eval(ids, msk)  # (B, T, d_model)

            # Skip positions where the target is PAD (game end) — they
            # don't contribute to any metric but would still cost a
            # ``lm_head`` row otherwise.
            effective_mask = msk & (tgt != PAD_TOKEN)
            if not bool(effective_mask.any()):
                continue

            rows, cols = torch.where(effective_mask)
            valid_hidden = hidden[rows, cols]                # (N_valid, d_model)
            valid_logits = model.lm_head(valid_hidden)       # (N_valid, vocab_size)
            valid_targets = tgt[rows, cols]                  # (N_valid,)
            valid_log_probs = torch.log_softmax(valid_logits, dim=-1)

            # Gather loss contributions for the true targets in one shot.
            target_log_probs = valid_log_probs.gather(
                dim=-1, index=valid_targets.unsqueeze(-1)
            ).squeeze(-1)                                    # (N_valid,)
            total_loss += float((-target_log_probs).sum())
            total_tokens += int(valid_targets.shape[0])

            argmax = valid_logits.argmax(dim=-1)             # (N_valid,)
            total_correct += int((argmax == valid_targets).sum())

            top5_indices = valid_logits.topk(5, dim=-1).indices  # (N_valid, 5)
            total_top5_correct += int(
                (top5_indices == valid_targets.unsqueeze(-1)).any(dim=-1).sum()
            )

            # Legal-move rate: materialise argmax decisions on the host
            # for the per-position grid lookup. The masked tensors are
            # small (at most ``B * T`` but usually far fewer) so the
            # CPU trip doesn't dominate.
            argmax_cpu = argmax.cpu().tolist()
            rows_cpu = rows.cpu().tolist()
            cols_cpu = cols.cpu().tolist()
            grid_cpu = grid.cpu()
            for k, argmax_tok in enumerate(argmax_cpu):
                if argmax_tok >= NUM_ACTIONS:
                    continue
                gi = start + rows_cpu[k]
                ply = cols_cpu[k] + grid_offset
                if ply >= grid_cpu.shape[1]:
                    continue
                grid_idx = int(action_grid_idx[argmax_tok])
                src = grid_idx // 64
                dst = grid_idx % 64
                if int(grid_cpu[gi, ply, src]) >> dst & 1:
                    total_legal += 1

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
