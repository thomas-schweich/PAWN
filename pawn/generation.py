"""Autoregressive generation + outcome-signal diagnostics — JAX port of
``pawn.eval_suite.generation`` from the pre-migration codebase.

Provides:

* ``autoregressive_generate(...)`` — token-by-token sampling from a PAWN
  model with optional prefix loading, optional legal-move masking, and
  Gumbel-max sampling.
* ``outcome_signal_test`` (§6.1-6.3): for each of the 5 conditioning
  outcomes (white-mate / black-mate / stalemate / draw-by-rule /
  ply-limit), generate ``n`` games, classify their actual outcomes,
  and report match rate + forfeit rate + post-terminal padding rate
  with and without legal-move masking.
* ``prefix_continuation_test`` (§6.4): generate continuations from a
  held-out corpus's game prefixes under each of the 5 conditioning
  outcomes; bucket by prefix-percentage and absolute-ply.
* ``poisoned_prefix_test`` (§6.5): mismatch the conditioning outcome
  against the prefix's true outcome and report whether the model
  follows the prefix or the conditioning signal.
* ``impossible_task_test`` (§6.6): probe behaviour on provably
  impossible conditioning (zero-remaining-ply, insufficient-material →
  checkmate request).
* ``improbable_task_test`` (§6.7): highly-improbable scenarios
  (checkmate-in-very-few-ply, early stalemate).

The port is a faithful behavioural copy of the legacy module's science.
Two implementation notes are worth flagging:

  1. **KV-cache by default.** ``autoregressive_generate(..., use_kv_cache=True)``
     (the default) runs the model's ``forward_with_cache`` over the
     prefix once, then advances per-step via ``forward_incremental``.
     The cache is shape-stable at ``cfg.max_seq_len``, so the JIT
     trace count is one prefill + one increment regardless of decode
     length. Set ``use_kv_cache=False`` to fall back to the legacy
     full-recompute path (one full forward per step); both paths are
     bitwise-equivalent and the equivalence is pinned by
     ``test_autoregressive_generate_kv_cache_matches_full_recompute``.
  2. **Legal masking uses ``engine.PyBatchRLEnv.get_legal_token_masks_batch``
     unchanged** — the engine is the single source of truth for
     legality. **Sampling is Gumbel-max** (matches the legacy;
     equivalent to softmax-then-sample for ``temperature=1``).

Prefix-length grouping: ``autoregressive_generate`` groups input rows
by ``prefix_lengths`` value before dispatching to ``_generate_batch``,
so every internal batch sees a uniform prefix length. This fixes a
correctness gap inherited from the legacy port — when prefix lengths
varied *within* a batch (the common case for percentage-based prefix
buckets in §6.4 / §6.5), the legacy decode loop started at the batch-
wide maximum ``prefix_end``, leaving PAD gaps in the attended context
of rows with shorter prefixes. The engine state machine was still
correct, but the model was conditioned on a PAD-padded context rather
than the true mid-game context. Grouping eliminates that — every row
in an internal batch shares the same ``prefix_end``, so no row sees a
PAD gap.

The grouping permutes input rows by prefix length and unpermutes
results back to the caller's original order; ``GenerationResult``
remains row-aligned with the inputs. The no-prefix path is
short-circuited and stays identical to the legacy ungrouped loop, so
non-prefix tests are bit-stable across this change.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import chess_engine as engine
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.config import OUTCOME_TOKEN_BASE, PAD_TOKEN
from pawn.model import KVCache, PAWNModel

# Outcome offsets — these are 0-indexed within the outcome band; the
# absolute token id is ``OUTCOME_TOKEN_BASE + offset`` (1969 + offset).
# Same convention as ``pawn.corpus._map_term_to_outcome_offset``.
_OFFSET_WHITE_CHECKMATES = 0
_OFFSET_BLACK_CHECKMATES = 1
_OFFSET_STALEMATE = 2
_OFFSET_DRAW_BY_RULE = 3
_OFFSET_PLY_LIMIT = 4


OUTCOME_TOKENS: dict[str, int] = {
    "WHITE_CHECKMATES": OUTCOME_TOKEN_BASE + _OFFSET_WHITE_CHECKMATES,
    "BLACK_CHECKMATES": OUTCOME_TOKEN_BASE + _OFFSET_BLACK_CHECKMATES,
    "STALEMATE": OUTCOME_TOKEN_BASE + _OFFSET_STALEMATE,
    "DRAW_BY_RULE": OUTCOME_TOKEN_BASE + _OFFSET_DRAW_BY_RULE,
    "PLY_LIMIT": OUTCOME_TOKEN_BASE + _OFFSET_PLY_LIMIT,
}


# Sentinel termination codes used by the per-game tracker. These mirror
# the engine's positive termination codes (0..5) but add two negative
# sentinels for things the engine can't produce on its own.
_TC_PREMATURE_PAD = -2   # model sampled PAD before terminating naturally
_TC_FORFEIT = -3         # model sampled an illegal move (only when legal
                         # masking is disabled)


@dataclass
class GenerationResult:
    """Output of ``autoregressive_generate``.

    All arrays are NumPy on host. ``sequences`` includes the conditioning
    outcome token at position 0 and the move tokens at positions
    ``1..N``. Under ``mask_illegal=True``, positions after
    ``game_lengths[i]`` are filled with ``PAD_TOKEN``. Under
    ``mask_illegal=False`` (used to measure premature-padding /
    post-terminal continuation rates), terminated rows keep being
    sampled and the tail carries the model's continued samples — see
    ``_analyze_generated_games`` for the metrics that depend on this.
    """

    sequences: np.ndarray        # (n_games, max_seq_len) int32
    term_codes: np.ndarray       # (n_games,) int8 — engine codes or sentinels
    game_lengths: np.ndarray     # (n_games,) int32 — ply at which game terminated
    forfeit_ply: np.ndarray      # (n_games,) int32 — first illegal-move ply, or -1

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "sequences": self.sequences,
            "term_codes": self.term_codes,
            "game_lengths": self.game_lengths,
            "forfeit_ply": self.forfeit_ply,
        }


# ---------------------------------------------------------------------------
# Core autoregressive generation
# ---------------------------------------------------------------------------


def _forward(
    model: PAWNModel, ids: jax.Array, attn: jax.Array,
) -> jax.Array:
    """Typed wrapper so ``filter_jit`` preserves the call signature
    (the previous bare lambda was inferred as ``Unknown`` by pyright,
    poisoning every downstream expression). (Type-correctness)"""
    return model(ids, attn)


def _forward_prefill(
    model: PAWNModel, ids: jax.Array, attn: jax.Array,
) -> tuple[jax.Array, KVCache]:
    """Typed wrapper around ``forward_with_cache`` for ``filter_jit``."""
    return model.forward_with_cache(ids, attn)


def _forward_incremental(
    model: PAWNModel, token: jax.Array, pos: jax.Array, cache: KVCache,
) -> tuple[jax.Array, KVCache]:
    """Typed wrapper around ``forward_incremental`` for ``filter_jit``."""
    return model.forward_incremental(token, pos, cache)


_forward_jit = eqx.filter_jit(_forward, donate="none")
_forward_prefill_jit = eqx.filter_jit(_forward_prefill, donate="none")
_forward_incremental_jit = eqx.filter_jit(_forward_incremental, donate="none")


def _subseed(seed: int, *parts: object) -> int:
    """Deterministic process-stable per-call seed derived from
    ``seed`` and the variadic key parts. Used by §6.4-6.7 tests to
    give every inner ``autoregressive_generate`` call independent
    Gumbel noise without relying on Python's process-randomised
    ``hash()``. The derivation is SHA-256 over the encoded parts;
    the high 32 bits are XOR'd onto ``seed``.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (seed ^ int.from_bytes(digest[:4], "little")) & 0x7FFFFFFF


def _gumbel_max_argmax(
    logits: np.ndarray, key: jax.Array
) -> tuple[np.ndarray, jax.Array]:
    """Sample one token per row via Gumbel-max. Returns ``(samples, next_key)``.

    ``logits: (B, V)``. Gumbel-max is equivalent to softmax-then-sample
    for ``temperature=1`` and faster than a categorical sample under
    JAX (one elementwise add + argmax). Matches the legacy torch
    implementation.
    """
    key, sub = jax.random.split(key)
    g = jax.random.gumbel(sub, logits.shape)
    return np.asarray(jnp.argmax(jnp.asarray(logits) + g, axis=-1)), key


def autoregressive_generate(
    model: PAWNModel,
    outcome_token: int,
    n_games: int,
    *,
    mask_illegal: bool = False,
    prefix_moves: np.ndarray | None = None,
    prefix_lengths: np.ndarray | None = None,
    max_seq_len: int | None = None,
    temperature: float = 1.0,
    batch_size: int = 64,
    seed: int = 0,
    use_kv_cache: bool = True,
) -> GenerationResult:
    """Generate ``n_games`` games from a PAWN model.

    Args:
        outcome_token: full token id (e.g. ``OUTCOME_TOKENS["WHITE_CHECKMATES"]``)
            written at sequence position 0.
        n_games: total games to generate; split into ``batch_size`` chunks.
        mask_illegal: if True, set illegal-move logits to ``-inf`` before
            sampling so the model can never play an illegal move
            (no forfeits possible). If False, illegal samples terminate
            the game with ``term_code = -3``.
        prefix_moves: ``(n_games, prefix_len)`` int — optional pre-loaded
            moves applied before sampling starts.
        prefix_lengths: ``(n_games,)`` int — per-game length of the
            ``prefix_moves`` window.
        max_seq_len: total CLM sequence length, including the outcome
            prefix at position 0. Defaults to the model's
            ``cfg.max_seq_len``.
        temperature: dividing factor on logits before sampling. ``1.0``
            = unmodified Gumbel-max sampling.
        batch_size: per-batch concurrency on the engine + model.
        seed: PRNG seed for sampling.
        use_kv_cache: if ``True`` (default) use the KV-cached incremental
            decoder; if ``False`` use the legacy full-recompute decoder
            (one full forward per step). Both produce numerically
            equivalent samples — the cache path is the default for
            speed.
    """
    if max_seq_len is None:
        max_seq_len = model.cfg.max_seq_len
    if n_games <= 0:
        raise ValueError(f"n_games must be > 0, got {n_games}")
    # Asymmetric prefix args silently dropped both — surface the
    # misuse so a downstream test that forgets one half doesn't
    # quietly generate from-scratch. (Test-risk)
    if (prefix_moves is None) != (prefix_lengths is None):
        raise ValueError(
            "prefix_moves and prefix_lengths must be passed together "
            "(both or neither)"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    key = jax.random.key(seed)

    # No-prefix fast path: no grouping needed and the loop is bit-stable
    # against pre-grouping behaviour so existing prefixless tests keep
    # their existing sample values.
    if prefix_moves is None or prefix_lengths is None:
        batches: list[GenerationResult] = []
        for s in range(0, n_games, batch_size):
            e = min(s + batch_size, n_games)
            key, sub = jax.random.split(key)
            result = _generate_batch(
                model, outcome_token, e - s, mask_illegal,
                None, None, max_seq_len, temperature, sub,
                use_kv_cache=use_kv_cache,
            )
            batches.append(result)
        return GenerationResult(
            sequences=np.concatenate([b.sequences for b in batches], axis=0),
            term_codes=np.concatenate([b.term_codes for b in batches], axis=0),
            game_lengths=np.concatenate([b.game_lengths for b in batches], axis=0),
            forfeit_ply=np.concatenate([b.forfeit_ply for b in batches], axis=0),
        )

    # Variable-prefix path: group rows by prefix length so each
    # ``_generate_batch`` call sees a uniform ``prefix_end``. See the
    # module docstring for why this matters.
    pls_np = np.asarray(prefix_lengths, dtype=np.int32)
    if pls_np.shape[0] != n_games:
        raise ValueError(
            f"prefix_lengths has {pls_np.shape[0]} rows but n_games={n_games}"
        )
    unique_lens, inverse_idx = np.unique(pls_np, return_inverse=True)
    group_results: list[GenerationResult] = []
    processing_order: list[np.ndarray] = []
    for g in range(len(unique_lens)):
        group_orig_idx = np.where(inverse_idx == g)[0]
        group_moves = prefix_moves[group_orig_idx]
        group_lens = pls_np[group_orig_idx]
        for s in range(0, len(group_orig_idx), batch_size):
            e = min(s + batch_size, len(group_orig_idx))
            key, sub = jax.random.split(key)
            result = _generate_batch(
                model, outcome_token, e - s, mask_illegal,
                group_moves[s:e], group_lens[s:e],
                max_seq_len, temperature, sub,
                use_kv_cache=use_kv_cache,
            )
            group_results.append(result)
            processing_order.append(group_orig_idx[s:e])

    all_orig_indices = np.concatenate(processing_order)
    sequences_cat = np.concatenate(
        [r.sequences for r in group_results], axis=0,
    )
    term_codes_cat = np.concatenate(
        [r.term_codes for r in group_results], axis=0,
    )
    game_lengths_cat = np.concatenate(
        [r.game_lengths for r in group_results], axis=0,
    )
    forfeit_ply_cat = np.concatenate(
        [r.forfeit_ply for r in group_results], axis=0,
    )
    # Unpermute: ``np.argsort(all_orig_indices)`` builds the inverse
    # permutation that puts processed row ``k`` back at original
    # position ``all_orig_indices[k]``.
    inverse = np.argsort(all_orig_indices, kind="stable")
    return GenerationResult(
        sequences=sequences_cat[inverse],
        term_codes=term_codes_cat[inverse],
        game_lengths=game_lengths_cat[inverse],
        forfeit_ply=forfeit_ply_cat[inverse],
    )


def _generate_batch(
    model: PAWNModel,
    outcome_token: int,
    n_games: int,
    mask_illegal: bool,
    prefix_moves: np.ndarray | None,
    prefix_lengths: np.ndarray | None,
    max_seq_len: int,
    temperature: float,
    key: jax.Array,
    *,
    use_kv_cache: bool = True,
) -> GenerationResult:
    """Single-batch core.

    If ``use_kv_cache`` (default) is True, runs a prefill via
    ``forward_with_cache`` then decodes each new position via
    ``forward_incremental`` — O(T_max) per step, O(T_max²) total. If
    False, runs a fresh full forward per step (legacy O(T_max²) per
    step, O(T_max³) total). Both paths share the engine state machine
    and the legal-mask + termination logic; the only difference is
    *how* the per-step logits are computed.

    Within a single batch ``prefix_lengths`` is uniform (the caller
    ``autoregressive_generate`` groups rows by prefix length before
    dispatching here), so the decode loop starts at a single
    ``prefix_end`` and no row sees a PAD gap in its attended context.
    """
    max_move_positions = max_seq_len - 1
    vocab_size = model.cfg.vocab_size

    # Sequences start with the outcome token at position 0; all
    # subsequent positions are PAD until the model fills them in.
    sequences = np.full((n_games, max_seq_len), PAD_TOKEN, dtype=np.int32)
    sequences[:, 0] = outcome_token

    env = engine.PyBatchRLEnv(n_games, max_ply=max_move_positions, seed=0)
    env.reset()
    terminated = np.zeros(n_games, dtype=bool)
    terminated_at = np.full(n_games, -1, dtype=np.int32)
    forfeit_ply = np.full(n_games, -1, dtype=np.int32)
    term_codes = np.full(n_games, -1, dtype=np.int8)
    all_indices = np.arange(n_games, dtype=np.uint32)

    prefix_end = 0
    if prefix_moves is not None and prefix_lengths is not None:
        # Engine wants ``(n_games, max_ply) uint16`` + ``(n_games,) uint32``
        # lengths.
        padded = np.zeros((n_games, max_move_positions), dtype=np.uint16)
        clamped_pls = np.minimum(
            np.asarray(prefix_lengths, dtype=np.int32),
            min(max_move_positions, int(prefix_moves.shape[1])),
        )
        for i in range(n_games):
            pl = int(clamped_pls[i])
            padded[i, :pl] = prefix_moves[i, :pl].astype(np.uint16)
        lens_u32 = clamped_pls.astype(np.uint32)
        prefix_tc = np.asarray(env.load_prefixes(padded, lens_u32))
        # Copy prefix moves into sequences. Position 0 is the outcome
        # token; prefix moves go at positions 1..pl.
        for i in range(n_games):
            pl = int(clamped_pls[i])
            sequences[i, 1:pl + 1] = prefix_moves[i, :pl].astype(np.int32)
            if prefix_tc[i] >= 0:
                terminated[i] = True
                terminated_at[i] = pl
                term_codes[i] = prefix_tc[i]
        prefix_end = int(clamped_pls.max()) if n_games > 0 else 0

    if use_kv_cache:
        next_logits, cache = _cached_prefill(
            model, sequences, prefix_end, max_seq_len, n_games,
        )
    else:
        next_logits = None  # filled lazily inside the per-step branch
        cache = None
    full_attn_template = np.zeros((n_games, max_seq_len), dtype=bool)
    for pos in range(prefix_end + 1, max_seq_len):
        if terminated.all():
            break
        if use_kv_cache:
            assert next_logits is not None
            step_logits = next_logits
        else:
            # PAD-fill ``ctx`` to ``max_seq_len``; the model's RoPE +
            # causal mask + attn_mask together handle the unused tail.
            # The full-recompute path is kept behind ``use_kv_cache=False``
            # so the KV-cache path can be regression-tested against it.
            ctx = jnp.asarray(sequences, dtype=jnp.int32)
            full_attn_template[:, :pos] = True
            full_attn_template[:, pos:] = False
            attn = jnp.asarray(full_attn_template)
            logits = _forward_jit(model, ctx, attn)
            step_logits = np.asarray(logits[:, pos - 1, :])
        if temperature != 1.0:
            step_logits = step_logits / float(temperature)

        if mask_illegal:
            raw_mask = np.asarray(env.get_legal_token_masks_batch(
                all_indices, vocab_size,
            ))
            # Terminated games stay padded — clamp them to "only PAD is
            # legal" so sampling yields PAD on those rows.
            pad_only = np.zeros((1, vocab_size), dtype=bool)
            pad_only[0, PAD_TOKEN] = True
            full_mask = np.where(terminated[:, None], pad_only, raw_mask)
            step_logits = np.where(full_mask, step_logits, -np.inf)

        sampled, key = _gumbel_max_argmax(step_logits, key)
        sequences[:, pos] = sampled

        # Premature padding: an active game sampled PAD before a real
        # termination. Marked with sentinel ``_TC_PREMATURE_PAD``.
        pad_mask = ~terminated & (sampled == PAD_TOKEN)
        if pad_mask.any():
            terminated[pad_mask] = True
            terminated_at[pad_mask] = pos - 1
            term_codes[pad_mask] = _TC_PREMATURE_PAD

        # Apply moves for still-active, non-PAD games via the engine.
        move_mask = ~terminated & (sampled != PAD_TOKEN)
        if move_mask.any():
            move_indices = all_indices[move_mask]
            move_tokens = sampled[move_mask].astype(np.uint16)
            legality, step_tc = env.apply_moves(
                move_indices.tolist(), move_tokens.tolist(),
            )
            legality = np.asarray(legality)
            step_tc = np.asarray(step_tc)

            # Illegal samples (only possible when mask_illegal is False)
            # → forfeit sentinel.
            illegal = ~legality
            if illegal.any():
                idx = move_indices[illegal]
                forfeit_ply[idx] = pos - 1
                terminated[idx] = True
                terminated_at[idx] = pos - 1
                term_codes[idx] = _TC_FORFEIT

            # Legal moves that produce a termination (engine code >= 0).
            termed = legality & (step_tc >= 0)
            if termed.any():
                idx = move_indices[termed]
                terminated[idx] = True
                terminated_at[idx] = pos
                term_codes[idx] = step_tc[termed]

        # Advance the cache for the next iteration: the next step's
        # query needs K, V at the position we just sampled into. Skip
        # the increment on the final iteration — its logits would be
        # unused (the loop is about to exit).
        if use_kv_cache and pos + 1 < max_seq_len and not terminated.all():
            assert cache is not None
            sampled_for_inc = jnp.asarray(
                sequences[:, pos:pos + 1], dtype=jnp.int32,
            )
            inc_logits, cache = _forward_incremental_jit(
                model, sampled_for_inc, jnp.int32(pos), cache,
            )
            next_logits = np.asarray(inc_logits[:, 0, :])

    # Games that never terminated within the window are PLY_LIMIT.
    still_going = ~terminated
    terminated_at[still_going] = max_move_positions
    term_codes[still_going] = 5  # engine PLY_LIMIT code

    return GenerationResult(
        sequences=sequences,
        term_codes=term_codes,
        game_lengths=terminated_at.astype(np.int32),
        forfeit_ply=forfeit_ply,
    )


def _cached_prefill(
    model: PAWNModel,
    sequences: np.ndarray,
    prefix_end: int,
    max_seq_len: int,
    n_games: int,
) -> tuple[np.ndarray, KVCache]:
    """Run ``model.forward_with_cache`` over the prefix-filled portion
    of ``sequences`` and return ``(logits_at_prefix_end, populated_cache)``.

    The input shape is always ``(n_games, max_seq_len)`` — the JIT
    cache contains exactly one trace per ``(B, max_seq_len)`` pair.
    The attention mask flags only positions ``[0, prefix_end]`` as
    real so that PAD-positions beyond the prefix don't influence the
    cache's "real" K, V entries. Subsequent ``forward_incremental``
    calls overwrite the cache at each new position.
    """
    ctx = jnp.asarray(sequences, dtype=jnp.int32)
    attn = np.zeros((n_games, max_seq_len), dtype=bool)
    attn[:, :prefix_end + 1] = True
    attn_j = jnp.asarray(attn)
    logits, cache = _forward_prefill_jit(model, ctx, attn_j)
    return np.asarray(logits[:, prefix_end, :]), cache


# ---------------------------------------------------------------------------
# Outcome classification + analysis
# ---------------------------------------------------------------------------


def _term_code_to_outcome_name(tc: int, game_length: int) -> str:
    """Engine termination code → outcome bucket name.

    Mirrors the legacy ``_map_term_code_to_outcome_name`` plus the two
    generator sentinels.
    """
    if tc == 0:
        return "WHITE_CHECKMATES" if game_length % 2 == 1 else "BLACK_CHECKMATES"
    if tc == 1:
        return "STALEMATE"
    if tc in (2, 3, 4):
        return "DRAW_BY_RULE"
    if tc == 5:
        return "PLY_LIMIT"
    if tc == _TC_PREMATURE_PAD:
        return "PREMATURE_PAD"
    if tc == _TC_FORFEIT:
        return "FORFEIT"
    return "UNKNOWN"


def _outcome_mask(
    term_codes: np.ndarray, game_lengths: np.ndarray, outcome_name: str,
) -> np.ndarray:
    """Boolean mask over a corpus selecting games of a given outcome."""
    if outcome_name == "WHITE_CHECKMATES":
        return (term_codes == 0) & (game_lengths % 2 == 1)
    if outcome_name == "BLACK_CHECKMATES":
        return (term_codes == 0) & (game_lengths % 2 == 0)
    if outcome_name == "STALEMATE":
        return term_codes == 1
    if outcome_name == "DRAW_BY_RULE":
        return (term_codes == 2) | (term_codes == 3) | (term_codes == 4)
    if outcome_name == "PLY_LIMIT":
        return term_codes == 5
    return np.zeros(len(term_codes), dtype=bool)


def _analyze_generated_games(
    gen: GenerationResult, conditioned_outcome: str,
) -> dict[str, Any]:
    """Compute the §6 metrics from a generation result.

    Returns:
        outcome_match_rate: fraction of games whose actual outcome
            matches ``conditioned_outcome``.
        outcome_distribution: ``{name: fraction}``.
        mean_game_length: float.
        forfeit_rate: fraction with forfeit_ply >= 0.
        post_terminal_padding_rate: among positions *after* game end,
            the fraction that are PAD. 1.0 if no post-terminal slots.
        post_terminal_move_count: total count of non-PAD tokens after
            game end (sum over all games).
        premature_padding_rate: fraction of games where the model
            sampled PAD before a real terminal.
    """
    sequences = gen.sequences
    term_codes = gen.term_codes
    game_lengths = gen.game_lengths
    forfeit_ply = gen.forfeit_ply
    n = len(sequences)
    max_seq_len = sequences.shape[1]

    outcome_dist: dict[str, int] = {}
    n_match = 0
    for i in range(n):
        actual = _term_code_to_outcome_name(
            int(term_codes[i]), int(game_lengths[i]),
        )
        outcome_dist[actual] = outcome_dist.get(actual, 0) + 1
        if actual == conditioned_outcome:
            n_match += 1

    n_forfeit = int((forfeit_ply >= 0).sum())

    n_post_terminal_tokens = 0
    n_post_terminal_pad = 0
    n_post_terminal_move = 0
    for i in range(n):
        gl = int(game_lengths[i])
        post_start = gl + 1
        if post_start < max_seq_len:
            post = sequences[i, post_start:]
            n_post_terminal_tokens += int(post.shape[0])
            n_post_terminal_pad += int((post == PAD_TOKEN).sum())
            n_post_terminal_move += int((post != PAD_TOKEN).sum())

    n_premature_pad = int((term_codes == _TC_PREMATURE_PAD).sum())

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
# §6.1-6.3 outcome signal test
# ---------------------------------------------------------------------------


def outcome_signal_test(
    model: PAWNModel,
    *,
    n_per_outcome: int = 1000,
    mask_conditions: tuple[bool, ...] = (False, True),
    batch_size: int = 64,
    seed: int = 0,
    verbose: bool = True,
    use_kv_cache: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run the §6.1-6.3 outcome-token signal test.

    For each outcome name + masked / unmasked condition, generate
    ``n_per_outcome`` games and report the §6 metrics.

    Returns ``results[outcome_name][masked|unmasked] = metrics``.
    """
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for oname, otok in OUTCOME_TOKENS.items():
        results[oname] = {}
        for masked in mask_conditions:
            label = "masked" if masked else "unmasked"
            if verbose:
                print(
                    f"[gen] outcome={oname} masked={masked} "
                    f"generating {n_per_outcome} games"
                )
            t0 = time.perf_counter()
            # Process-stable per-(outcome, mask) seed offset. Python's
            # built-in ``hash`` of a string is randomised per process
            # (PYTHONHASHSEED), which would silently make
            # ``outcome_signal_test`` non-reproducible across runs —
            # SHA-256 over the encoded key gives a deterministic
            # 32-bit offset instead. (Test-risk)
            digest = hashlib.sha256(
                f"{oname}:{masked}".encode("utf-8")
            ).digest()
            offset = int.from_bytes(digest[:4], "little") % 10_000
            gen = autoregressive_generate(
                model, otok, n_per_outcome,
                mask_illegal=masked, batch_size=batch_size,
                seed=seed + offset, use_kv_cache=use_kv_cache,
            )
            results[oname][label] = _analyze_generated_games(gen, oname)
            results[oname][label]["elapsed_s"] = time.perf_counter() - t0
            if verbose:
                rate = results[oname][label]["outcome_match_rate"]
                ln = results[oname][label]["mean_game_length"]
                print(
                    f"[gen]   match={rate:.1%}  mean_len={ln:.0f}  "
                    f"forfeit={results[oname][label]['forfeit_rate']:.1%}  "
                    f"({results[oname][label]['elapsed_s']:.1f}s)"
                )
    return results


# ---------------------------------------------------------------------------
# §6.4 prefix continuation
# ---------------------------------------------------------------------------


def prefix_continuation_test(
    model: PAWNModel,
    corpus: dict[str, np.ndarray],
    *,
    n_per_bucket: int = 200,
    prefix_pcts: tuple[float, ...] = (0.1, 0.5, 0.9),
    absolute_plies: tuple[int, ...] = (10, 50, 100, 200),
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True,
    use_kv_cache: bool = True,
) -> dict[str, Any]:
    """§6.4 prefix continuation test with within-test outcome controls.

    ``corpus`` must contain ``"move_ids"`` (int16 ``[N, max_ply]``),
    ``"game_lengths"`` (int16 ``[N]``), and ``"termination_codes"``
    (uint8 ``[N]``) — the standard tuple from ``engine.generate_clm_batch``.
    """
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    results: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for oname in OUTCOME_TOKENS:
        mask = _outcome_mask(term_codes, game_lengths, oname)
        indices = np.where(mask)[0]
        if len(indices) < n_per_bucket:
            if verbose:
                print(
                    f"[gen] {oname}: only {len(indices)} corpus games, "
                    f"need {n_per_bucket} — skipping"
                )
            continue
        selected = rng.choice(indices, n_per_bucket, replace=False)
        results[oname] = {}

        for pct in prefix_pcts:
            bucket = f"pct_{int(pct * 100)}"
            pls = np.maximum((game_lengths[selected] * pct).astype(np.int32), 1)
            bucket_results: dict[str, Any] = {}
            for cname, ctok in OUTCOME_TOKENS.items():
                gen = autoregressive_generate(
                    model, ctok, n_per_bucket,
                    mask_illegal=True,
                    prefix_moves=move_ids[selected], prefix_lengths=pls,
                    batch_size=batch_size,
                    # Per-(oname, bucket, cname) subseed so every inner
                    # generation draws independent Gumbel noise — passing
                    # ``seed=seed`` to every inner call would collapse
                    # cross-condition variance (every condition draws the
                    # same noise; only the logit difference distinguishes
                    # them). The legacy torch port relied on the global
                    # RNG to advance state across calls; we make that
                    # explicit. (Bug-detector)
                    seed=_subseed(seed, oname, bucket, cname),
                    use_kv_cache=use_kv_cache,
                )
                bucket_results[cname] = _analyze_generated_games(gen, cname)
            results[oname][bucket] = bucket_results

        for abs_ply in absolute_plies:
            bucket = f"ply_{abs_ply}"
            long_enough = selected[game_lengths[selected] > abs_ply]
            if len(long_enough) < 10:
                continue
            sub = long_enough[:n_per_bucket]
            pls = np.full(len(sub), abs_ply, dtype=np.int32)
            bucket_results: dict[str, Any] = {}
            for cname, ctok in OUTCOME_TOKENS.items():
                gen = autoregressive_generate(
                    model, ctok, len(sub),
                    mask_illegal=True,
                    prefix_moves=move_ids[sub], prefix_lengths=pls,
                    batch_size=batch_size,
                    seed=_subseed(seed, oname, bucket, cname),
                    use_kv_cache=use_kv_cache,
                )
                bucket_results[cname] = _analyze_generated_games(gen, cname)
            results[oname][bucket] = bucket_results

    return results


# ---------------------------------------------------------------------------
# §6.5 poisoned prefix
# ---------------------------------------------------------------------------


POISONING_PAIRS: tuple[tuple[str, str], ...] = (
    ("WHITE_CHECKMATES", "BLACK_CHECKMATES"),
    ("WHITE_CHECKMATES", "DRAW_BY_RULE"),
    ("DRAW_BY_RULE", "WHITE_CHECKMATES"),
    ("PLY_LIMIT", "WHITE_CHECKMATES"),
)


def poisoned_prefix_test(
    model: PAWNModel,
    corpus: dict[str, np.ndarray],
    *,
    n_per_pair: int = 500,
    prefix_pct: float = 0.5,
    batch_size: int = 64,
    seed: int = 43,
    use_kv_cache: bool = True,
) -> dict[str, Any]:
    """§6.5 poisoned prefix test.

    For each ``(actual, poisoned)`` pair: take ``n_per_pair`` games that
    *actually* ended with ``actual``, take a ``prefix_pct`` prefix, and
    condition the continuation on ``poisoned`` instead. Report the
    outcome distribution + match rate against both conditioning and
    actual outcomes.
    """
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    for actual, poisoned in POISONING_PAIRS:
        label = f"{actual}->{poisoned}"
        mask = _outcome_mask(term_codes, game_lengths, actual)
        indices = np.where(mask)[0]
        if len(indices) < n_per_pair:
            continue
        selected = rng.choice(indices, n_per_pair, replace=False)
        pls = np.maximum(
            (game_lengths[selected] * prefix_pct).astype(np.int32), 1,
        )
        gen = autoregressive_generate(
            model, OUTCOME_TOKENS[poisoned], n_per_pair,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "poisoned", actual, poisoned),
            use_kv_cache=use_kv_cache,
        )
        analysis = _analyze_generated_games(gen, poisoned)
        # Additionally, track how often the continuation lands on the
        # *actual* outcome (i.e. the prefix's true outcome).
        original_match = sum(
            _term_code_to_outcome_name(
                int(gen.term_codes[i]), int(gen.game_lengths[i]),
            ) == actual
            for i in range(n_per_pair)
        )
        analysis["original_outcome_match_rate"] = original_match / n_per_pair
        analysis["actual_outcome"] = actual
        analysis["poisoned_outcome"] = poisoned
        results[label] = analysis
    return results


# ---------------------------------------------------------------------------
# §6.6 impossible task
# ---------------------------------------------------------------------------


def impossible_task_test(
    model: PAWNModel,
    corpus: dict[str, np.ndarray],
    *,
    outcome_prefix_trained: bool,
    n_per_scenario: int = 200,
    batch_size: int = 64,
    seed: int = 44,
    use_kv_cache: bool = True,
) -> dict[str, Any]:
    """§6.6 provably-impossible scenarios.

    Every scenario conditions the model on an outcome token at sequence
    position 0 and asks whether the model abandons its prefix-grounded
    distribution in response. This only has interpretable meaning when
    the model was actually trained to condition on the outcome token —
    i.e., trained with ``prepend_outcome=True`` data packing. On a
    model trained without the outcome prefix, position 0's outcome
    token is just an out-of-distribution input and the resulting
    metrics measure nothing. Pass ``outcome_prefix_trained=True`` to
    opt in; the function returns a structured ``{"_skipped": ...}``
    sentinel otherwise.

    Scenarios:

    * ``zero_remaining_ply``: prefixes filled to within one move of
      the model's context window. The model has no room to produce a
      checkmate yet is conditioned on WHITE_CHECKMATES. Requires the
      corpus to carry games at least ``model.cfg.max_seq_len - 1``
      plies long — generate the diagnostic corpus with
      ``corpus_max_ply >= model.cfg.max_seq_len - 1`` for this to be
      meaningful. The scenario is silently skipped if no game in the
      corpus is that long, so an under-sized corpus produces no
      ``zero_remaining_ply`` entry rather than misleading numbers.
    * ``insufficient_material``: games that ended in
      insufficient-material draws, conditioned on WHITE_CHECKMATES at
      90% prefix.
    * ``control_ply_limit``: same prefixes as ``zero_remaining_ply``
      but conditioned on PLY_LIMIT (the natural / most-likely outcome).
    """
    if not outcome_prefix_trained:
        return {
            "_skipped": (
                "impossible_task_test only has interpretable meaning on "
                "models trained with prepend_outcome=True (outcome-token "
                "conditioning at sequence position 0). The current model "
                "wasn't, so the conditioning signal at position 0 is "
                "out-of-distribution and the scenarios would measure "
                "nothing. Re-run with --outcome-prefix-trained on a "
                "compatible checkpoint."
            )
        }

    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    term_codes = corpus["termination_codes"]

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}

    # Scenario 1: zero remaining ply. Target prefix = max_seq_len - 1
    # (with the outcome prepended at position 0, this fills positions
    # 0..max_seq_len-1, leaving the decode loop's
    # ``range(prefix_end + 1, max_seq_len)`` empty — zero plies of
    # generation room). The candidate set is filtered to games that
    # are actually long enough to fill that prefix; under-sized
    # corpora silently skip the scenario rather than emit prefixes
    # that don't realise the "zero remaining" semantics.
    target_prefix = model.cfg.max_seq_len - 1
    corpus_can_fill = int(move_ids.shape[1]) >= target_prefix
    ply_limit_idx = np.where(
        (term_codes == 5) & (game_lengths >= target_prefix)
    )[0]
    if corpus_can_fill and len(ply_limit_idx) >= n_per_scenario:
        selected = rng.choice(ply_limit_idx, n_per_scenario, replace=False)
        pls = np.full(n_per_scenario, target_prefix, dtype=np.int32)
        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "impossible", "zero_remaining_ply"),
            use_kv_cache=use_kv_cache,
        )
        results["zero_remaining_ply"] = _analyze_generated_games(
            gen, "WHITE_CHECKMATES",
        )

        # Control: same prefixes, condition on PLY_LIMIT.
        gen_ctrl = autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "impossible", "control_ply_limit"),
            use_kv_cache=use_kv_cache,
        )
        results["control_ply_limit"] = _analyze_generated_games(
            gen_ctrl, "PLY_LIMIT",
        )

    # Scenario 2: insufficient material → checkmate request
    insuf_idx = np.where(term_codes == 4)[0]
    if len(insuf_idx) >= n_per_scenario:
        selected = rng.choice(insuf_idx, n_per_scenario, replace=False)
        pls = np.maximum((game_lengths[selected] * 0.9).astype(np.int32), 1)
        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "impossible", "insufficient_material"),
            use_kv_cache=use_kv_cache,
        )
        results["insufficient_material"] = _analyze_generated_games(
            gen, "WHITE_CHECKMATES",
        )

    return results


# ---------------------------------------------------------------------------
# §6.7 improbable task
# ---------------------------------------------------------------------------


def improbable_task_test(
    model: PAWNModel,
    corpus: dict[str, np.ndarray],
    *,
    outcome_prefix_trained: bool,
    n_per_scenario: int = 200,
    batch_size: int = 64,
    seed: int = 46,
    use_kv_cache: bool = True,
) -> dict[str, Any]:
    """§6.7 highly-improbable scenarios.

    Same outcome-conditioning dependency as ``impossible_task_test``:
    the scenarios condition on improbable outcomes at sequence
    position 0 and only have interpretable meaning when the model
    was trained with ``prepend_outcome=True`` data packing. Pass
    ``outcome_prefix_trained=True`` to opt in; the function returns
    ``{"_skipped": ...}`` otherwise.

    Scenarios:

    * ``checkmate_few_ply``: 245-ply prefixes from games of length
      >= 240, conditioned on WHITE_CHECKMATES (very few ply
      remaining).
    * ``stalemate_early``: 20-ply prefixes conditioned on STALEMATE.
    * Controls condition on PLY_LIMIT.
    """
    if not outcome_prefix_trained:
        return {
            "_skipped": (
                "improbable_task_test only has interpretable meaning on "
                "models trained with prepend_outcome=True (outcome-token "
                "conditioning at sequence position 0). The current model "
                "wasn't, so the conditioning signal at position 0 is "
                "out-of-distribution and the scenarios would measure "
                "nothing. Re-run with --outcome-prefix-trained on a "
                "compatible checkpoint."
            )
        }

    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}

    # Scenario 1: checkmate-in-very-few-ply
    long_idx = np.where(game_lengths >= 240)[0]
    if len(long_idx) >= n_per_scenario:
        selected = rng.choice(long_idx, n_per_scenario, replace=False)
        pls = np.minimum(
            np.full(n_per_scenario, 245, dtype=np.int32),
            game_lengths[selected].astype(np.int32) - 1,
        )
        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "improbable", "checkmate_few_ply"),
            use_kv_cache=use_kv_cache,
        )
        results["checkmate_few_ply"] = _analyze_generated_games(
            gen, "WHITE_CHECKMATES",
        )
        gen_ctrl = autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "improbable", "control_few_ply"),
            use_kv_cache=use_kv_cache,
        )
        results["control_few_ply"] = _analyze_generated_games(
            gen_ctrl, "PLY_LIMIT",
        )

    # Scenario 2: early stalemate request
    early_idx = np.where(game_lengths >= 40)[0]
    if len(early_idx) >= n_per_scenario:
        selected = rng.choice(early_idx, n_per_scenario, replace=False)
        pls = np.full(n_per_scenario, 20, dtype=np.int32)
        gen = autoregressive_generate(
            model, OUTCOME_TOKENS["STALEMATE"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "improbable", "stalemate_early"),
            use_kv_cache=use_kv_cache,
        )
        results["stalemate_early"] = _analyze_generated_games(
            gen, "STALEMATE",
        )
        gen_ctrl = autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_per_scenario,
            mask_illegal=True,
            prefix_moves=move_ids[selected], prefix_lengths=pls,
            batch_size=batch_size,
            seed=_subseed(seed, "improbable", "control_early"),
            use_kv_cache=use_kv_cache,
        )
        results["control_early"] = _analyze_generated_games(
            gen_ctrl, "PLY_LIMIT",
        )

    return results
