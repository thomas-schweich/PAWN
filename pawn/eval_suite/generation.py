"""Autoregressive generation for outcome token signal tests (§6)."""

from __future__ import annotations

import gc
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess_engine as engine

from pawn.config import PAD_TOKEN, WHITE_CHECKMATES, PLY_LIMIT, CLMConfig
from pawn.data import _map_termination_to_outcome


class GenerativeModel(Protocol):
    """Structural type for models usable in autoregressive generation."""

    def eval(self) -> nn.Module: ...

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_only: bool = ...,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]: ...

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = ...,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]: ...


# ---------------------------------------------------------------------------
# Core autoregressive generation
# ---------------------------------------------------------------------------


def autoregressive_generate(
    model: GenerativeModel,
    outcome_token: int,
    n_games: int,
    device: str,
    mask_illegal: bool = False,
    prefix_moves: np.ndarray | None = None,
    prefix_lengths: np.ndarray | None = None,
    max_seq_len: int = 512,
    temperature: float = 1.0,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Generate games autoregressively from a trained PAWN.

    Args:
        model: Trained PAWN model.
        outcome_token: Outcome token ID to condition on (position 0).
        n_games: Number of games to generate.
        device: Torch device.
        mask_illegal: If True, set illegal move logits to -inf during game phase.
        prefix_moves: Optional (n_games, prefix_len) int16 array of prefix moves.
        prefix_lengths: Optional (n_games,) int array of prefix lengths.
        max_seq_len: Total sequence length.
        temperature: Sampling temperature.
        batch_size: Number of games to generate in parallel.

    Returns dict with:
        - sequences: (n_games, max_seq_len) int — full token sequences
        - actual_outcomes: (n_games,) int — termination codes from engine
        - game_lengths: (n_games,) int — ply at which game terminated
        - forfeit_ply: (n_games,) int — ply of first illegal move (-1 if none)
        - terminated_at: (n_games,) int — ply of termination
    """
    model.eval()
    all_results = []

    for batch_start in range(0, n_games, batch_size):
        batch_n = min(batch_size, n_games - batch_start)
        result = _generate_batch(
            model, outcome_token, batch_n, device, mask_illegal,
            prefix_moves[batch_start:batch_start + batch_n] if prefix_moves is not None else None,
            prefix_lengths[batch_start:batch_start + batch_n] if prefix_lengths is not None else None,
            max_seq_len, temperature,
        )
        all_results.append(result)

        # Free GPU cache between generation batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate batch results
    return {k: np.concatenate([r[k] for r in all_results], axis=0) for k in all_results[0]}


@torch.no_grad()
def _generate_batch(
    model: GenerativeModel,
    outcome_token: int,
    n_games: int,
    device: str,
    mask_illegal: bool,
    prefix_moves: np.ndarray | None,
    prefix_lengths: np.ndarray | None,
    max_seq_len: int,
    temperature: float,
) -> dict[str, np.ndarray]:
    """Generate a batch of games using batch Rust engine for state management."""
    cfg_vocab_size = getattr(model, 'cfg', CLMConfig()).vocab_size
    max_move_positions = max_seq_len - 1  # position 0 is outcome token

    # Initialize sequences with outcome token
    sequences = np.full((n_games, max_seq_len), PAD_TOKEN, dtype=np.int64)
    sequences[:, 0] = outcome_token

    # Batch game state tracking via Rust engine
    env = engine.PyBatchRLEnv(n_games, max_ply=max_move_positions, seed=0)
    env.reset()
    terminated = np.zeros(n_games, dtype=bool)
    terminated_at = np.full(n_games, -1, dtype=np.int32)
    forfeit_ply = np.full(n_games, -1, dtype=np.int32)
    term_codes = np.full(n_games, -1, dtype=np.int8)
    all_indices = np.arange(n_games, dtype=np.uint32)

    # Apply prefix moves via bulk Rust method
    prefix_end = 0
    if prefix_moves is not None and prefix_lengths is not None:
        # Pad prefix to (n_games, max_move_positions) uint16 for Rust
        padded = np.zeros((n_games, max_move_positions), dtype=np.uint16)
        clamped_pls = np.minimum(
            np.asarray(prefix_lengths, dtype=np.int32), max_move_positions
        )
        for i in range(n_games):
            pl = clamped_pls[i]
            padded[i, :pl] = prefix_moves[i, :pl]
        lengths_u32 = np.array(prefix_lengths, dtype=np.uint32)

        prefix_tc = env.load_prefixes(padded, lengths_u32)

        # Copy prefix moves into sequences and track terminations
        for i in range(n_games):
            pl = clamped_pls[i]
            sequences[i, 1:pl + 1] = prefix_moves[i, :pl]
            if prefix_tc[i] >= 0:
                terminated[i] = True
                terminated_at[i] = pl
                term_codes[i] = prefix_tc[i]
        prefix_end = int(prefix_lengths.max()) if len(prefix_lengths) > 0 else 0

    # --- Prefill: process outcome token + prefix in one forward pass ---
    prefill_ids = torch.tensor(
        sequences[:, :prefix_end + 1], dtype=torch.long, device=device
    )
    if hasattr(model, 'forward_generate'):
        logits, kv_cache = model.forward_generate(prefill_ids)
        next_logits = logits[:, 0, :].clone()
        del logits
        use_cache = True
    else:
        attn_mask = torch.ones_like(prefill_ids, dtype=torch.bool)
        logits, _ = model(prefill_ids, attn_mask, hidden_only=True)
        next_logits = logits[:, -1, :].clone()
        del logits
        kv_cache = None
        use_cache = False

    # Pre-allocate PAD-only mask row for terminated games
    pad_row = torch.zeros(1, cfg_vocab_size, dtype=torch.bool, device=device)
    pad_row[0, PAD_TOKEN] = True

    # Pre-allocate GPU buffers for legal masking to avoid per-step allocation
    if mask_illegal:
        _mask_buf = torch.empty(n_games, cfg_vocab_size, dtype=torch.bool, device=device)
        _term_buf = torch.empty(n_games, 1, dtype=torch.bool, device=device)

    # --- Autoregressive decode loop ---
    for pos in range(prefix_end + 1, max_seq_len):
        active_mask = ~terminated
        if not active_mask.any():
            break

        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Full-batch legal masking — avoids expensive boolean advanced indexing
        if mask_illegal:
            raw_mask = env.get_legal_token_masks_batch(all_indices)
            # Reuse pre-allocated GPU buffers instead of allocating each step
            _mask_buf.copy_(torch.from_numpy(np.asarray(raw_mask)))
            _term_buf[:, 0] = torch.from_numpy(terminated)
            full_mask = torch.where(_term_buf, pad_row.expand(n_games, -1), _mask_buf)
            next_logits.masked_fill_(~full_mask, float("-inf"))

        # Gumbel-max sampling (replaces softmax + multinomial)
        gumbel = -torch.log(-torch.log(
            torch.rand_like(next_logits).clamp_(min=1e-10)
        ))
        sampled = (next_logits + gumbel).argmax(dim=-1).cpu().numpy()

        # Record tokens
        sequences[:, pos] = sampled

        # Check for PAD tokens from terminated/premature games
        pad_mask = active_mask & (sampled == PAD_TOKEN)
        if pad_mask.any():
            terminated[pad_mask] = True
            terminated_at[pad_mask] = pos - 1
            term_codes[pad_mask] = -2  # premature padding

        # Apply moves in batch for active, non-PAD games
        move_mask = active_mask & ~pad_mask & ~terminated
        if move_mask.any():
            move_indices = all_indices[move_mask]
            move_tokens = sampled[move_mask].astype(np.uint16)

            legality, step_tc = env.apply_moves(move_indices, move_tokens)
            legality = np.asarray(legality)
            step_tc = np.asarray(step_tc)

            # Handle forfeits (illegal moves in unmasked mode)
            illegal = ~legality
            if illegal.any():
                forfeit_global = move_indices[illegal]
                forfeit_ply[forfeit_global] = pos - 1
                terminated[forfeit_global] = True
                terminated_at[forfeit_global] = pos - 1
                term_codes[forfeit_global] = -3

            # Handle terminations from legal moves
            termed = legality & (step_tc >= 0)
            if termed.any():
                term_global = move_indices[termed]
                terminated[term_global] = True
                terminated_at[term_global] = pos
                term_codes[term_global] = step_tc[termed]

        # Decode next token via KV-cache (skip on last position or all done)
        if pos < max_seq_len - 1 and not terminated.all():
            if use_cache:
                new_ids = torch.tensor(
                    sampled[:, None], dtype=torch.long, device=device
                )
                logits, kv_cache = model.forward_generate(
                    new_ids, kv_cache=kv_cache
                )
                next_logits = logits[:, 0, :].clone()
                del logits
            else:
                input_ids = torch.tensor(
                    sequences[:, :pos + 1], dtype=torch.long, device=device
                )
                attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
                logits, _ = model(input_ids, attn_mask, hidden_only=True)
                next_logits = logits[:, -1, :].clone()
                del logits

    # Games that never terminated within the window
    still_going = ~terminated
    terminated_at[still_going] = max_move_positions
    term_codes[still_going] = 5  # ply limit

    return {
        "sequences": sequences,
        "term_codes": term_codes,
        "game_lengths": terminated_at.astype(np.int32),
        "forfeit_ply": forfeit_ply,
    }


# ---------------------------------------------------------------------------
# Outcome signal test (§6.1-6.3)
# ---------------------------------------------------------------------------


def _map_term_code_to_outcome_name(tc: int, game_length: int) -> str:
    """Map engine termination code to CLM outcome name."""
    if tc == 0:
        # Checkmate: odd length = white delivered, even = black
        return "WHITE_CHECKMATES" if game_length % 2 == 1 else "BLACK_CHECKMATES"
    elif tc == 1:
        return "STALEMATE"
    elif tc in (2, 3, 4):
        return "DRAW_BY_RULE"
    elif tc == 5:
        return "PLY_LIMIT"
    elif tc == -2:
        return "PREMATURE_PAD"
    elif tc == -3:
        return "FORFEIT"
    return "UNKNOWN"


OUTCOME_TOKENS = {
    "WHITE_CHECKMATES": WHITE_CHECKMATES,
    "BLACK_CHECKMATES": WHITE_CHECKMATES + 1,
    "STALEMATE": WHITE_CHECKMATES + 2,
    "DRAW_BY_RULE": WHITE_CHECKMATES + 3,
    "PLY_LIMIT": PLY_LIMIT,
}


def outcome_signal_test(
    model: GenerativeModel,
    device: str,
    n_per_outcome: int = 1000,
    mask_conditions: tuple[bool, ...] = (False, True),
) -> dict:
    """Run outcome token base signal test (§6.1-6.3).

    For each outcome type, generate n_per_outcome games with that outcome
    token, both with and without illegal move masking.

    Returns nested dict: results[outcome_name][masked/unmasked] = metrics
    """
    results = {}
    for outcome_name, outcome_tok in OUTCOME_TOKENS.items():
        results[outcome_name] = {}
        for masked in mask_conditions:
            label = "masked" if masked else "unmasked"
            print(f"  {outcome_name} ({label}): generating {n_per_outcome} games...")

            gen = autoregressive_generate(
                model, outcome_tok, n_per_outcome, device,
                mask_illegal=masked,
            )
            results[outcome_name][label] = _analyze_generated_games(
                gen, outcome_name
            )

    return results


def _analyze_generated_games(gen: dict, conditioned_outcome: str) -> dict:
    """Compute metrics for a set of generated games."""
    sequences = gen["sequences"]
    term_codes = gen["term_codes"]
    game_lengths = gen["game_lengths"]
    forfeit_ply = gen["forfeit_ply"]
    n = len(sequences)
    max_seq_len = sequences.shape[1]

    # Classify actual outcomes
    outcome_dist = {}
    n_match = 0
    for i in range(n):
        actual = _map_term_code_to_outcome_name(
            int(term_codes[i]), int(game_lengths[i])
        )
        outcome_dist[actual] = outcome_dist.get(actual, 0) + 1
        if actual == conditioned_outcome:
            n_match += 1

    # Forfeit rate
    n_forfeit = int((forfeit_ply >= 0).sum())

    # Post-terminal padding rate
    n_post_terminal_tokens = 0
    n_post_terminal_pad = 0
    n_post_terminal_move = 0
    for i in range(n):
        gl = int(game_lengths[i])
        post_start = gl + 1  # positions after last move
        if post_start < max_seq_len:
            post_tokens = sequences[i, post_start:]
            n_post_terminal_tokens += len(post_tokens)
            n_post_terminal_pad += int((post_tokens == PAD_TOKEN).sum())
            n_post_terminal_move += int((post_tokens != PAD_TOKEN).sum())

    # Premature padding rate
    n_premature_pad = int((term_codes == -2).sum())

    return {
        "n_games": n,
        "outcome_match_rate": n_match / n,
        "outcome_distribution": {k: v / n for k, v in outcome_dist.items()},
        "mean_game_length": float(game_lengths.mean()),
        "forfeit_rate": n_forfeit / n,
        "post_terminal_padding_rate": (
            n_post_terminal_pad / n_post_terminal_tokens
            if n_post_terminal_tokens > 0 else 1.0
        ),
        "post_terminal_move_count": n_post_terminal_move,
        "premature_padding_rate": n_premature_pad / n,
    }


# ---------------------------------------------------------------------------
# Prefix continuation test (§6.4)
# ---------------------------------------------------------------------------


def prefix_continuation_test(
    model: GenerativeModel,
    corpus: dict,
    device: str,
    n_per_bucket: int = 200,
    prefix_pcts: tuple[float, ...] = (0.1, 0.5, 0.9),
    absolute_plies: tuple[int, ...] = (10, 50, 100, 200),
) -> dict:
    """Prefix continuation test with within-test controls (§6.4).

    For each game, generate continuations conditioned on ALL 5 outcome tokens.
    """
    import time as _time

    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    # Count total generation calls for progress tracking
    n_outcomes = len(OUTCOME_TOKENS)
    n_cond = len(OUTCOME_TOKENS)
    total_calls = 0
    done_calls = 0
    t_start = _time.perf_counter()

    # Pre-count to estimate total work
    for outcome_name in OUTCOME_TOKENS:
        mask = _outcome_mask(term_codes, game_lengths, outcome_name)
        if mask.sum() < n_per_bucket:
            continue
        total_calls += len(prefix_pcts) * n_cond
        for abs_ply in absolute_plies:
            np.random.seed(42)
            selected = np.random.choice(np.where(mask)[0], n_per_bucket, replace=False)
            if (game_lengths[selected] > abs_ply).sum() >= 10:
                total_calls += n_cond

    results = {}

    for oi, (outcome_name, outcome_tok) in enumerate(OUTCOME_TOKENS.items()):
        # Select games that ended with this outcome
        mask = _outcome_mask(term_codes, game_lengths, outcome_name)
        indices = np.where(mask)[0]
        if len(indices) < n_per_bucket:
            print(f"  {outcome_name}: only {len(indices)} games, need {n_per_bucket} — skipping")
            continue

        np.random.seed(42)
        selected = np.random.choice(indices, n_per_bucket, replace=False)

        results[outcome_name] = {}
        n_buckets = len(prefix_pcts) + len(absolute_plies)
        print(f"\n[{oi+1}/{n_outcomes}] {outcome_name} ({len(indices)} corpus games, {n_buckets} buckets)")

        for pct in prefix_pcts:
            bucket_name = f"pct_{int(pct * 100)}"
            prefix_lens = (game_lengths[selected] * pct).astype(np.int32)
            prefix_lens = np.maximum(prefix_lens, 1)  # at least 1 move
            mean_pl = prefix_lens.mean()

            bucket_results = {}
            for ci, (cond_name, cond_tok) in enumerate(OUTCOME_TOKENS.items()):
                t0 = _time.perf_counter()
                gen = autoregressive_generate(
                    model, cond_tok, n_per_bucket, device,
                    mask_illegal=True,
                    prefix_moves=move_ids[selected],
                    prefix_lengths=prefix_lens,
                )
                dt = _time.perf_counter() - t0
                analysis = _analyze_generated_games(gen, cond_name)
                bucket_results[cond_name] = analysis
                done_calls += 1
                elapsed = _time.perf_counter() - t_start
                eta = elapsed / done_calls * (total_calls - done_calls)
                match_rate = analysis["outcome_match_rate"]
                print(
                    f"  {bucket_name} | cond={cond_name:<20s} | "
                    f"match={match_rate:.1%}  len={analysis['mean_game_length']:.0f}  "
                    f"{dt:.1f}s  [{done_calls}/{total_calls}, ETA {eta:.0f}s]"
                )

            results[outcome_name][bucket_name] = bucket_results

        # Absolute ply buckets
        for abs_ply in absolute_plies:
            bucket_name = f"ply_{abs_ply}"
            long_enough = selected[game_lengths[selected] > abs_ply]
            if len(long_enough) < 10:
                print(f"  {bucket_name} | skipped (only {len(long_enough)} games long enough)")
                continue
            sub = long_enough[:n_per_bucket]
            prefix_lens = np.full(len(sub), abs_ply, dtype=np.int32)

            bucket_results = {}
            for ci, (cond_name, cond_tok) in enumerate(OUTCOME_TOKENS.items()):
                t0 = _time.perf_counter()
                gen = autoregressive_generate(
                    model, cond_tok, len(sub), device,
                    mask_illegal=True,
                    prefix_moves=move_ids[sub],
                    prefix_lengths=prefix_lens,
                )
                dt = _time.perf_counter() - t0
                analysis = _analyze_generated_games(gen, cond_name)
                bucket_results[cond_name] = analysis
                done_calls += 1
                elapsed = _time.perf_counter() - t_start
                eta = elapsed / done_calls * (total_calls - done_calls)
                match_rate = analysis["outcome_match_rate"]
                print(
                    f"  {bucket_name} | cond={cond_name:<20s} | "
                    f"match={match_rate:.1%}  len={analysis['mean_game_length']:.0f}  "
                    f"{dt:.1f}s  [{done_calls}/{total_calls}, ETA {eta:.0f}s]"
                )
            results[outcome_name][bucket_name] = bucket_results

    total_time = _time.perf_counter() - t_start
    print(f"\nPrefix continuation test complete: {done_calls} generation calls in {total_time:.0f}s")
    return results


def _outcome_mask(term_codes: np.ndarray, game_lengths: np.ndarray, outcome_name: str) -> np.ndarray:
    """Create a boolean mask for games matching the given outcome."""
    if outcome_name == "WHITE_CHECKMATES":
        return (term_codes == 0) & (game_lengths % 2 == 1)
    elif outcome_name == "BLACK_CHECKMATES":
        return (term_codes == 0) & (game_lengths % 2 == 0)
    elif outcome_name == "STALEMATE":
        return term_codes == 1
    elif outcome_name == "DRAW_BY_RULE":
        return (term_codes == 2) | (term_codes == 3) | (term_codes == 4)
    elif outcome_name == "PLY_LIMIT":
        return term_codes == 5
    return np.zeros(len(term_codes), dtype=bool)


# ---------------------------------------------------------------------------
# Poisoned prefix test (§6.5)
# ---------------------------------------------------------------------------

POISONING_PAIRS = [
    ("WHITE_CHECKMATES", "BLACK_CHECKMATES"),
    ("WHITE_CHECKMATES", "DRAW_BY_RULE"),
    ("DRAW_BY_RULE", "WHITE_CHECKMATES"),
    ("PLY_LIMIT", "WHITE_CHECKMATES"),
]


def poisoned_prefix_test(
    model: GenerativeModel,
    corpus: dict,
    device: str,
    n_per_pair: int = 500,
    prefix_pct: float = 0.5,
) -> dict:
    """Poisoned prefix test (§6.5).

    Gives the model a prefix from one outcome type but conditions on another.
    """
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    results = {}
    for actual_name, poisoned_name in POISONING_PAIRS:
        label = f"{actual_name}->{poisoned_name}"
        print(f"  Poisoned: {label}")

        # Select games with the actual outcome
        mask = _outcome_mask(term_codes, game_lengths, actual_name)
        indices = np.where(mask)[0]
        if len(indices) < n_per_pair:
            print(f"    Only {len(indices)} games, skipping")
            continue

        np.random.seed(43)
        selected = np.random.choice(indices, n_per_pair, replace=False)
        prefix_lens = (game_lengths[selected] * prefix_pct).astype(np.int32)
        prefix_lens = np.maximum(prefix_lens, 1)

        poisoned_tok = OUTCOME_TOKENS[poisoned_name]
        gen = autoregressive_generate(
            model, poisoned_tok, n_per_pair, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )

        analysis = _analyze_generated_games(gen, poisoned_name)
        # Also track match with original outcome
        original_match = 0
        for i in range(len(gen["term_codes"])):
            actual = _map_term_code_to_outcome_name(
                int(gen["term_codes"][i]), int(gen["game_lengths"][i])
            )
            if actual == actual_name:
                original_match += 1
        analysis["original_outcome_match_rate"] = original_match / n_per_pair
        analysis["actual_outcome"] = actual_name
        analysis["poisoned_outcome"] = poisoned_name

        results[label] = analysis

    return results


# ---------------------------------------------------------------------------
# Impossible task test (§6.6)
# ---------------------------------------------------------------------------


def impossible_task_test(
    model: GenerativeModel,
    corpus: dict,
    device: str,
    n_per_scenario: int = 200,
) -> dict:
    """Test model behavior under provably impossible outcome conditions (§6.6).

    Scenarios:
    - Insufficient material: K vs K, K+B vs K, K+N vs K
    - Zero remaining ply: 255-ply prefix, condition on different outcome
    """
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    results = {}

    # Scenario 1: Zero remaining ply (prefix = full game)
    # Use ply_limit games (255 ply) conditioned on checkmate
    ply_limit_mask = term_codes == 5
    ply_limit_idx = np.where(ply_limit_mask)[0]
    if len(ply_limit_idx) >= n_per_scenario:
        np.random.seed(44)
        selected = np.random.choice(ply_limit_idx, n_per_scenario, replace=False)
        # Prefix is the entire game (all 255 moves)
        prefix_lens = game_lengths[selected].copy().astype(np.int32)
        prefix_lens = np.minimum(prefix_lens, 254)  # leave at least 1 position

        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["zero_remaining_ply"] = _analyze_generated_games(
            gen, "WHITE_CHECKMATES"
        )
        print(f"  zero_remaining_ply: {results['zero_remaining_ply']['outcome_match_rate']:.2%} match")

    # Scenario 2: Insufficient material games conditioned on checkmate
    insuf_mask = term_codes == 4  # InsufficientMaterial
    insuf_idx = np.where(insuf_mask)[0]
    if len(insuf_idx) >= n_per_scenario:
        np.random.seed(45)
        selected = np.random.choice(insuf_idx, n_per_scenario, replace=False)
        # Use 90% prefix
        prefix_lens = (game_lengths[selected] * 0.9).astype(np.int32)
        prefix_lens = np.maximum(prefix_lens, 1)

        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["insufficient_material"] = _analyze_generated_games(
            gen, "WHITE_CHECKMATES"
        )
        print(f"  insufficient_material: {results['insufficient_material']['outcome_match_rate']:.2%} match")

    # Control: same prefixes with correct/most-likely outcome
    if len(ply_limit_idx) >= n_per_scenario:
        np.random.seed(44)
        selected = np.random.choice(ply_limit_idx, n_per_scenario, replace=False)
        prefix_lens = game_lengths[selected].copy().astype(np.int32)
        prefix_lens = np.minimum(prefix_lens, 254)

        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["control_ply_limit"] = _analyze_generated_games(
            gen, "PLY_LIMIT"
        )

    return results


# ---------------------------------------------------------------------------
# Improbable task test (§6.7)
# ---------------------------------------------------------------------------


def improbable_task_test(
    model: GenerativeModel,
    corpus: dict,
    device: str,
    n_per_scenario: int = 200,
) -> dict:
    """Test model behavior under highly improbable conditions (§6.7)."""
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    results = {}

    # Scenario 1: Checkmate in very few ply
    # Games with 240+ ply, no check at cutoff, conditioned on checkmate
    long_mask = game_lengths >= 240
    long_idx = np.where(long_mask)[0]
    if len(long_idx) >= n_per_scenario:
        np.random.seed(46)
        selected = np.random.choice(long_idx, n_per_scenario, replace=False)
        # Prefix: 245 ply (5-10 ply remaining in window)
        prefix_lens = np.full(n_per_scenario, 245, dtype=np.int32)
        prefix_lens = np.minimum(prefix_lens, game_lengths[selected].astype(np.int32) - 1)

        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["checkmate_few_ply"] = _analyze_generated_games(
            gen, "WHITE_CHECKMATES"
        )

        # Control with PLY_LIMIT
        gen_ctrl = autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["control_few_ply"] = _analyze_generated_games(
            gen_ctrl, "PLY_LIMIT"
        )

    # Scenario 2: Stalemate from early game (20-ply prefix)
    early_mask = game_lengths >= 40
    early_idx = np.where(early_mask)[0]
    if len(early_idx) >= n_per_scenario:
        np.random.seed(47)
        selected = np.random.choice(early_idx, n_per_scenario, replace=False)
        prefix_lens = np.full(n_per_scenario, 20, dtype=np.int32)

        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["STALEMATE"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["stalemate_early"] = _analyze_generated_games(
            gen, "STALEMATE"
        )

        # Control with most likely outcome
        gen_ctrl = autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_per_scenario, device,
            mask_illegal=True,
            prefix_moves=move_ids[selected],
            prefix_lengths=prefix_lens,
        )
        results["control_early"] = _analyze_generated_games(
            gen_ctrl, "PLY_LIMIT"
        )

    return results
