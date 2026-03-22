"""Edge-case diagnostic evaluation (§5)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess_engine as engine

from pawn.config import PAD_TOKEN
from pawn.data import _to_clm_batch, _map_termination_to_outcome


# ---------------------------------------------------------------------------
# Edge-case bit constants
# ---------------------------------------------------------------------------

EDGE_BITS = engine.edge_case_bits()

DIAGNOSTIC_CATEGORIES = {
    "in_check":            EDGE_BITS["IN_CHECK"],
    "double_check":        EDGE_BITS["IN_DOUBLE_CHECK"],
    "pin_restricts":       EDGE_BITS["PIN_RESTRICTS_MOVEMENT"],
    "ep_available":        EDGE_BITS["EP_CAPTURE_AVAILABLE"],
    "castle_legal_k":      EDGE_BITS["CASTLE_LEGAL_KINGSIDE"],
    "castle_legal_q":      EDGE_BITS["CASTLE_LEGAL_QUEENSIDE"],
    "castle_blocked_check": EDGE_BITS["CASTLE_BLOCKED_CHECK"],
    "promotion_available": EDGE_BITS["PROMOTION_AVAILABLE"],
    "checkmate":           EDGE_BITS["CHECKMATE"],
    "stalemate":           EDGE_BITS["STALEMATE"],
}


# ---------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------


def extract_diagnostic_positions(
    corpus: dict,
    min_per_category: int = 2000,
    max_per_category: int = 5000,
) -> dict[str, list[dict]]:
    """Extract diagnostic positions from corpus.

    Returns dict[category_name] -> list of dicts with:
        - game_idx: int
        - ply: int
        - move_ids: array of moves up to that ply
        - game_length: int (original game length)
        - term_code: int
        - outcome_name: str
    """
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]
    n_games = len(move_ids)

    # Compute per-ply edge stats in batches
    batch_size = 50_000
    print("  Computing per-ply edge stats...")
    all_ply_stats = []
    for start in range(0, n_games, batch_size):
        end = min(start + batch_size, n_games)
        stats, _, _ = engine.compute_edge_stats_per_ply(
            move_ids[start:end], game_lengths[start:end]
        )
        all_ply_stats.append(stats)
    per_ply_stats = np.concatenate(all_ply_stats, axis=0)

    # Extract positions per category
    positions = {}

    # Terminal categories (checkmate, stalemate) come from termination codes,
    # not per-ply stats. The "position" is the final one (ply = game_length - 1).
    _TERMINAL_CATEGORIES = {"checkmate": 0, "stalemate": 1}

    for cat_name, bit_value in DIAGNOSTIC_CATEGORIES.items():
        print(f"  {cat_name}: scanning...", end="", flush=True)
        found = []

        if cat_name in _TERMINAL_CATEGORIES:
            # Terminal: use termination codes. The "ply" is game_length,
            # i.e., the position AFTER the last move. At this point the
            # game is over — the model should predict PAD.
            target_tc = _TERMINAL_CATEGORIES[cat_name]
            for g in range(n_games):
                if int(term_codes[g]) == target_tc:
                    gl = int(game_lengths[g])
                    outcome = _term_code_to_outcome_name(target_tc, gl)
                    found.append({
                        "game_idx": g,
                        "ply": gl,  # post-terminal position
                        "game_length": gl,
                        "term_code": target_tc,
                        "outcome_name": outcome,
                    })
                    if len(found) >= max_per_category:
                        break
        else:
            for g in range(n_games):
                gl = int(game_lengths[g])
                for t in range(gl):
                    if per_ply_stats[g, t] & bit_value:
                        tc = int(term_codes[g])
                        outcome = _term_code_to_outcome_name(tc, gl)
                        found.append({
                            "game_idx": g,
                            "ply": t,
                            "game_length": gl,
                            "term_code": tc,
                            "outcome_name": outcome,
                        })
                        if len(found) >= max_per_category:
                            break
                if len(found) >= max_per_category:
                    break

        positions[cat_name] = found
        print(f" {len(found)} positions")

    return positions


def _term_code_to_outcome_name(tc: int, gl: int) -> str:
    if tc == 0:
        return "WHITE_CHECKMATES" if gl % 2 == 1 else "BLACK_CHECKMATES"
    elif tc == 1:
        return "STALEMATE"
    elif tc in (2, 3, 4):
        return "DRAW_BY_RULE"
    elif tc == 5:
        return "PLY_LIMIT"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_diagnostic_positions(
    model: nn.Module,
    positions: dict,
    corpus: dict,
    device: str,
    n_samples: int = 100,
    batch_size: int = 32,
) -> dict:
    """Evaluate model behavior at diagnostic positions.

    For each position, collect the full softmax distribution and sample
    N moves. Check legality via the engine.

    Returns dict[category] -> metrics dict.
    """
    model.eval()
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    results = {}
    for cat_name, pos_list in positions.items():
        if not pos_list:
            continue
        print(f"  {cat_name}: evaluating {len(pos_list)} positions...", end="", flush=True)

        all_legal_rates = []
        all_pad_rates = []
        all_entropies = []

        for batch_start in range(0, len(pos_list), batch_size):
            batch_end = min(batch_start + batch_size, len(pos_list))
            batch_pos = pos_list[batch_start:batch_end]
            B = len(batch_pos)

            # Build input sequences for each position.
            # To predict move at ply t, input is [outcome, m_0, ..., m_{t-1}]
            # (t+1 tokens total). The model at position t predicts ply t's move.
            max_seq = max(p["ply"] + 1 for p in batch_pos)  # outcome + t moves
            seq_len = min(max_seq, 256)

            input_seqs = np.full((B, seq_len), PAD_TOKEN, dtype=np.int64)
            pred_positions = []
            for i, pos in enumerate(batch_pos):
                g = pos["game_idx"]
                t = pos["ply"]
                gl = pos["game_length"]
                tc = int(term_codes[g])

                # Outcome token at position 0
                outcome = _map_termination_to_outcome(
                    np.array([tc], dtype=np.uint8),
                    np.array([gl], dtype=np.int16),
                )
                input_seqs[i, 0] = outcome[0].item()

                # Moves 0..t-1 at positions 1..t
                n_moves = min(t, seq_len - 1)
                if n_moves > 0:
                    input_seqs[i, 1:n_moves + 1] = move_ids[g, :n_moves]
                pred_positions.append(min(n_moves, seq_len - 1))

            input_ids = torch.tensor(input_seqs, dtype=torch.long, device=device)
            attn_mask = input_ids != PAD_TOKEN  # attend to non-padding only

            logits, _ = model(input_ids, attn_mask, hidden_only=True)

            # For each position, get the logit at the prediction point
            for i, pos in enumerate(batch_pos):
                pred_pos = pred_positions[i]
                pos_logits = logits[i, pred_pos]  # (vocab_size,)

                # Softmax distribution
                probs = F.softmax(pos_logits, dim=-1)

                # Entropy
                log_probs = F.log_softmax(pos_logits, dim=-1)
                entropy = -(probs * log_probs).sum().item()
                all_entropies.append(entropy)

                # PAD probability
                pad_prob = probs[PAD_TOKEN].item()
                all_pad_rates.append(pad_prob)

                # Sample N moves and check legality
                sampled = torch.multinomial(probs.unsqueeze(0).expand(n_samples, -1), 1)
                sampled = sampled.squeeze(-1).cpu().numpy()

                # Replay game to this position and check legality
                gs = engine.PyGameState(max_ply=256)
                g = pos["game_idx"]
                for mv_idx in range(pos["ply"]):
                    gs.make_move(int(move_ids[g, mv_idx]))

                legal_tokens = set(gs.legal_move_tokens())
                n_legal = sum(1 for s in sampled if int(s) in legal_tokens)
                all_legal_rates.append(n_legal / n_samples)

        results[cat_name] = {
            "n_positions": len(pos_list),
            "mean_legal_rate": float(np.mean(all_legal_rates)),
            "std_legal_rate": float(np.std(all_legal_rates)),
            "mean_pad_prob": float(np.mean(all_pad_rates)),
            "mean_entropy": float(np.mean(all_entropies)),
            "std_entropy": float(np.std(all_entropies)),
        }
        print(f" legal_rate={results[cat_name]['mean_legal_rate']:.3f}")

    return results
