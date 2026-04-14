"""Edge-case diagnostic evaluation (§5).

Uses the Rust engine's quota-controlled generator to produce a corpus
with guaranteed coverage of rare edge cases (stalemate, double check,
en passant, etc.) rather than relying on random sampling.
"""

import numpy as np
import torch
import torch.nn.functional as F

import chess_engine as engine

from pawn.config import PAD_TOKEN
from pawn.data import _map_termination_to_outcome
from pawn.model import PAWNCLM


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

# Bit index for each diagnostic category
_CAT_BIT_INDEX = {
    name: bit_val.bit_length() - 1
    for name, bit_val in DIAGNOSTIC_CATEGORIES.items()
}

# Terminal categories use termination codes, not per-ply stats
_TERMINAL_CATEGORIES = {"checkmate": 0, "stalemate": 1}


# ---------------------------------------------------------------------------
# Corpus generation (quota-controlled)
# ---------------------------------------------------------------------------


def generate_diagnostic_corpus(
    n_per_category: int = 10_000,
    max_ply: int = 255,
    seed: int = 42,
    max_simulated_factor: float = 200.0,
) -> dict:
    """Generate a diagnostic corpus with quota-controlled edge case coverage.

    Uses the Rust engine's generate_diagnostic_sets() to produce games
    biased toward underrepresented edge cases. Each category gets at least
    n_per_category games in the corpus.

    Returns a corpus dict compatible with extract_diagnostic_positions()
    and evaluate_diagnostic_positions().
    """
    # Build quota arrays: request n_per_category for each diagnostic bit,
    # split evenly between white and black perspectives.
    quotas_white = np.zeros(64, dtype=np.int32)
    quotas_black = np.zeros(64, dtype=np.int32)

    half = n_per_category // 2
    for cat_name, bit_val in DIAGNOSTIC_CATEGORIES.items():
        bit_idx = bit_val.bit_length() - 1
        quotas_white[bit_idx] = half
        quotas_black[bit_idx] = n_per_category - half  # handle odd n

    # Total games is sum of all quotas (some games will fill multiple quotas)
    total_games = int(quotas_white.sum() + quotas_black.sum())

    print(f"  Generating diagnostic corpus: {n_per_category} per category, "
          f"~{total_games} games max, max_simulated_factor={max_simulated_factor}...")

    (move_ids, game_lengths, term_codes, per_ply_stats,
     _white_acc, _black_acc, _qa_w, _qa_b,
     filled_white, filled_black) = engine.generate_diagnostic_sets(
        quotas_white, quotas_black, total_games, max_ply, seed, max_simulated_factor,
    )

    n_games = len(game_lengths)
    print(f"  Generated {n_games} games")

    # Report quota fill rates
    for cat_name, bit_val in DIAGNOSTIC_CATEGORIES.items():
        bit_idx = bit_val.bit_length() - 1
        filled = int(filled_white[bit_idx]) + int(filled_black[bit_idx])
        requested = int(quotas_white[bit_idx]) + int(quotas_black[bit_idx])
        pct = filled / requested * 100 if requested > 0 else 0
        status = "OK" if filled >= requested else "SHORT"
        print(f"    {cat_name}: {filled}/{requested} ({pct:.0f}%) [{status}]")

    return {
        "move_ids": np.asarray(move_ids),
        "game_lengths": np.asarray(game_lengths),
        "termination_codes": np.asarray(term_codes),
        "per_ply_stats": np.asarray(per_ply_stats),
    }


# ---------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------


def extract_diagnostic_positions(
    corpus: dict,
    max_per_category: int = 10_000,
) -> dict[str, list[dict]]:
    """Extract diagnostic positions from a corpus.

    If the corpus contains pre-computed per_ply_stats (from
    generate_diagnostic_corpus), those are used directly. Otherwise,
    per-ply edge stats are computed on the fly.

    Returns dict[category_name] -> list of position dicts.
    """
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]
    n_games = len(game_lengths)

    # Use pre-computed per-ply stats if available, otherwise compute them
    if "per_ply_stats" in corpus:
        per_ply_stats = corpus["per_ply_stats"]
    else:
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

    positions = {}

    for cat_name, bit_value in DIAGNOSTIC_CATEGORIES.items():
        print(f"  {cat_name}: scanning...", end="", flush=True)
        found = []

        if cat_name in _TERMINAL_CATEGORIES:
            target_tc = _TERMINAL_CATEGORIES[cat_name]
            for g in range(n_games):
                if int(term_codes[g]) == target_tc:
                    gl = int(game_lengths[g])
                    outcome = _term_code_to_outcome_name(target_tc, gl)
                    found.append({
                        "game_idx": g,
                        "ply": gl,
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
    model: PAWNCLM,
    positions: dict,
    corpus: dict,
    device: str,
    n_samples: int = 100,
    batch_size: int = 32,
    prepend_outcome: bool = False,
) -> dict:
    """Evaluate model behavior at diagnostic positions.

    For each position, collect the full softmax distribution and sample
    N moves. Check legality via the engine. Sequences are capped at
    ``model.cfg.max_seq_len`` so the same routine works for 256-ctx and
    512-ctx checkpoints.

    ``prepend_outcome`` must match how the model was trained so the
    prompt layout at each diagnostic position is in-distribution. When
    True the prompt is ``[outcome, m_0, ..., m_{t-1}]`` and the
    prediction position is ``t``; when False (pure-moves default) the
    prompt is ``[m_0, ..., m_{t-1}]`` and the prediction position is
    ``t - 1``. Getting this wrong silently evaluates every position on
    an out-of-distribution layout.

    Returns dict[category] -> metrics dict.
    """
    model.eval()
    model_max_seq_len = model.cfg.max_seq_len
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

            # Build input sequences for each diagnostic position.
            # Outcome-prefixed: prompt = [outcome, m_0, ..., m_{t-1}],
            # length t + 1, and logits at position t predict ply t.
            # Pure-moves: prompt = [m_0, ..., m_{t-1}], length t, and
            # logits at position t - 1 predict ply t. Ply 0 has no
            # predictive context in pure-moves mode and is skipped.
            outcome_slots = 1 if prepend_outcome else 0
            max_seq = max(p["ply"] + outcome_slots for p in batch_pos)
            if max_seq == 0:
                # Only ply-0 positions in pure-moves mode — nothing to predict.
                continue
            seq_len = min(max_seq, model_max_seq_len)

            input_seqs = np.full((B, seq_len), PAD_TOKEN, dtype=np.int64)
            pred_positions: list[int] = []
            valid_mask: list[bool] = []
            for i, pos in enumerate(batch_pos):
                g = pos["game_idx"]
                t = pos["ply"]
                gl = pos["game_length"]
                tc = int(term_codes[g])

                if prepend_outcome:
                    outcome = _map_termination_to_outcome(
                        np.array([tc], dtype=np.uint8),
                        np.array([gl], dtype=np.int16),
                    )
                    input_seqs[i, 0] = outcome[0].item()
                    n_moves = min(t, seq_len - 1)
                    if n_moves > 0:
                        input_seqs[i, 1:n_moves + 1] = move_ids[g, :n_moves]
                    pred_positions.append(min(n_moves, seq_len - 1))
                    valid_mask.append(True)
                else:
                    if t < 1:
                        # No prior moves to condition on in pure-moves mode.
                        pred_positions.append(0)
                        valid_mask.append(False)
                        continue
                    n_moves = min(t, seq_len)
                    input_seqs[i, :n_moves] = move_ids[g, :n_moves]
                    pred_positions.append(n_moves - 1)
                    valid_mask.append(True)

            input_ids = torch.tensor(input_seqs, dtype=torch.long, device=device)
            attn_mask = input_ids != PAD_TOKEN  # attend to non-padding only

            logits, _ = model(input_ids, attn_mask, hidden_only=True)

            # For each position, get the logit at the prediction point
            for i, pos in enumerate(batch_pos):
                if not valid_mask[i]:
                    continue
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
                gs = engine.PyGameState(max_ply=model_max_seq_len)
                g = pos["game_idx"]
                for mv_idx in range(pos["ply"]):
                    gs.make_move(int(move_ids[g, mv_idx]))

                legal_tokens = set(gs.legal_move_tokens())
                n_legal = sum(1 for s in sampled if int(s) in legal_tokens)
                all_legal_rates.append(n_legal / n_samples)

        is_terminal = cat_name in ("checkmate", "stalemate")
        results[cat_name] = {
            "n_positions": len(pos_list),
            "terminal": is_terminal,
            "mean_legal_rate": float(np.mean(all_legal_rates)),
            "std_legal_rate": float(np.std(all_legal_rates)),
            "mean_pad_prob": float(np.mean(all_pad_rates)),
            "mean_entropy": float(np.mean(all_entropies)),
            "std_entropy": float(np.std(all_entropies)),
        }
        if is_terminal:
            print(f" pad_prob={results[cat_name]['mean_pad_prob']:.3f}")
        else:
            print(f" legal_rate={results[cat_name]['mean_legal_rate']:.3f}")

    return results
