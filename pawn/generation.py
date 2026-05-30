"""Generation diagnostics — autoregressive sampling + 5 outcome tests.

All five diagnostics condition on the outcome token at position 0, so
all five return a ``{"_skipped": ...}`` sentinel when
``outcome_prefix_trained=False`` (plan §10 S8, plan §11 "easy to drop
and shouldn't be").

The diagnostics:

- ``outcome_signal_test`` — does the model produce a different move
  distribution when given different outcomes at position 0?
  Generates ``n_per_outcome`` real games per outcome token,
  optionally with legal-move masking, and reports the per-outcome
  outcome-match rate, forfeit rate, mean game length, and the
  post-terminal padding-vs-move ratio.
- ``prefix_continuation_test`` — given an outcome + prefix of N moves,
  does the model continue plausibly? Reports the next-move argmax
  *and* — when `--ar-continuation` is set — the AR-decoded
  continuation analysis (outcome-match, forfeit, mean game length).
- ``poisoned_prefix_test`` — given an outcome that contradicts the
  game's actual result, does the model still respect the outcome
  prefix?
- ``impossible_task_test`` — given an outcome that's geometrically
  impossible (mate before move 2), how does the model behave?
- ``improbable_task_test`` — given an outcome with implausibly low
  prior, does the model still emit consistent moves?

Real autoregressive generation lives in :func:`autoregressive_generate`,
which iterates one token at a time, applies the engine's legal-mask
constraint when ``mask_illegal=True``, and tracks per-game termination
state via :class:`chess_engine.PyBatchRLEnv`. The default path uses
the KV-cached decoder on :class:`pawn.model.PAWNModel` (and on
:class:`pawn.adapters.bottleneck.BottleneckEffective`) — O(T) per
decode step instead of O(T²) per step, so production-scale
``n_per_outcome=1000`` runs stay tractable. Pass
``use_kv_cache=False`` for the legacy full-forward path (used by the
KV-cache parity test).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

import chess_engine as engine
from pawn.config import (
    BLACK_CHECKMATES,
    DRAW_BY_AGREEMENT,
    DRAW_BY_RULE,
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)
from pawn.corpus import build_prefix, conditioning_to_C
from pawn.model import (
    EffectiveCallable,
    KVCache,
    KVCacheCallable,
    PAWNModel,
    init_kv_cache,
)

# Default conditioning for the generation diagnostics: a single
# ``"outcome"`` slot (prefix width ``C = 2`` → ``[BOS][outcome]``). Every
# diagnostic conditions on the outcome token, so this is the layout they
# were written for; callers wanting a different prefix pass their own
# ``conditioning`` list (resolved against the checkpoint's persisted run
# block by the eval driver).
_OUTCOME_CONDITIONING: Sequence[str] = ("outcome",)

__all__ = [
    "DIAGNOSTIC_NAMES",
    "OUTCOME_TOKENS",
    "autoregressive_generate",
    "analyze_generated_games",
    "outcome_signal_test",
    "prefix_continuation_test",
    "poisoned_prefix_test",
    "impossible_task_test",
    "improbable_task_test",
    "run_all_diagnostics",
]


DIAGNOSTIC_NAMES = (
    "outcome_signal_test",
    "prefix_continuation_test",
    "poisoned_prefix_test",
    "impossible_task_test",
    "improbable_task_test",
)


# v1 parity: the 5 outcome tokens used as conditioning inputs in
# autoregressive generation tests. (DRAW_BY_AGREEMENT is exercised
# separately by `improbable_task_test`.)
OUTCOME_TOKENS: dict[str, int] = {
    "WHITE_CHECKMATES": WHITE_CHECKMATES,
    "BLACK_CHECKMATES": BLACK_CHECKMATES,
    "STALEMATE": STALEMATE,
    "DRAW_BY_RULE": DRAW_BY_RULE,
    "PLY_LIMIT": PLY_LIMIT,
}


_SKIP_REASON = (
    "model was not trained with an outcome conditioning slot "
    "(conditioning=[\"outcome\"]); this diagnostic conditions on the "
    "outcome token in the prefix and is uninterpretable without it"
)


def _skipped(name: str) -> dict[str, Any]:
    return {"_skipped": _SKIP_REASON, "diagnostic": name}


def _argmax_action(logits: Float[Array, "B T V"]) -> Int[Array, "B T"]:
    """Argmax restricted to ``[0, NUM_ACTIONS)`` so PAD / outcome tokens
    can't be sampled."""
    return jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1)


def _single_row_prefixed(
    outcome_token: int,
    seq_len: int,
    *,
    moves: np.ndarray | None = None,
    conditioning: Sequence[str] = _OUTCOME_CONDITIONING,
) -> tuple[Int[Array, "1 T"], Int[Array, "1 T"], int]:
    """Build a one-row ``(tokens, attn, C)`` buffer with the C-wide prefix.

    Lays ``[BOS][cond…]`` into slots ``[0, C)`` via the shared
    :func:`pawn.corpus.build_prefix` (so the diagnostics use the exact
    layout the model trained under), optionally appends ``moves`` at slots
    ``[C, C+len(moves))``, and sets the attention mask True over the real
    span. Returns the prefix width ``C`` so the caller reads logits /
    argmax at the right slot (the next move after ``m`` prefix moves is
    predicted at slot ``C - 1 + m``).
    """
    C = conditioning_to_C(conditioning)  # noqa: N806
    prefix = build_prefix(conditioning, np.full(1, outcome_token, dtype=np.int32), 1)
    tokens = jnp.full((1, seq_len), PAD_TOKEN, dtype=jnp.int32)
    tokens = tokens.at[0, :C].set(jnp.asarray(prefix[0]))
    real = C
    if moves is not None:
        m = int(np.asarray(moves).shape[0])
        tokens = tokens.at[0, C : C + m].set(jnp.asarray(moves, dtype=jnp.int32))
        real = C + m
    attn = jnp.zeros((1, seq_len), dtype=jnp.bool_).at[0, :real].set(True)
    return tokens, attn, C


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------


def _map_term_code_to_outcome_name(tc: int, game_length: int) -> str:
    """v1 parity: map the engine's termination code → outcome token name.

    Termination codes from :class:`chess_engine.PyBatchRLEnv`:
      0  checkmate (odd length ⇒ white delivered, even ⇒ black)
      1  stalemate
      2-4 draw by 75-move / fivefold rep / insufficient material
      5  ply limit
      -2 premature PAD emitted by the model
      -3 forfeit (illegal move while ``mask_illegal=False``)
    """
    if tc == 0:
        return "WHITE_CHECKMATES" if game_length % 2 == 1 else "BLACK_CHECKMATES"
    if tc == 1:
        return "STALEMATE"
    if tc in (2, 3, 4):
        return "DRAW_BY_RULE"
    if tc == 5:
        return "PLY_LIMIT"
    if tc == -2:
        return "PREMATURE_PAD"
    if tc == -3:
        return "FORFEIT"
    return "UNKNOWN"


def autoregressive_generate(
    model: "PAWNModel | EffectiveCallable",
    outcome_token: int,
    n_games: int,
    *,
    mask_illegal: bool = False,
    prefix_moves: np.ndarray | None = None,
    prefix_lengths: np.ndarray | None = None,
    max_seq_len: int | None = None,
    temperature: float = 1.0,
    seed: int = 0,
    use_kv_cache: bool | None = None,
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
    conditioning: Sequence[str] = _OUTCOME_CONDITIONING,
) -> dict[str, np.ndarray]:
    """Generate ``n_games`` games autoregressively from ``model``.

    Mirrors the v1 :func:`pawn.eval_suite.generation.autoregressive_generate`
    contract — same input shapes, same output dict keys. Game state is
    tracked by :class:`chess_engine.PyBatchRLEnv` so legal-move masking
    and termination detection match the v1 engine.

    ``conditioning`` (default ``("outcome",)``) is the ordered control-kind
    list assembled into the fixed-width prefix the model was trained under
    (Phase-A Chunk 4). The sequence is laid out as ``[BOS][cond…][ply…]``:
    slot 0 is BOS, slots ``1..C-1`` carry the resolved control token per
    kind (the ``"outcome"`` kind resolves to ``outcome_token``; other kinds
    NULL-fill since the diagnostics only vary the outcome), and the first
    move lands at slot ``C = 1 + len(conditioning)``. Decoding starts from
    ``pos_start = C-1`` — the last prefix slot, whose logits predict
    ``ply_1`` — replacing the v1 hardcoded outcome-at-slot-0 layout.

    ``use_kv_cache`` (default: auto-detect): when the ``model`` exposes
    :meth:`PAWNModel.forward_with_cache`, use the cached decode path —
    O(T) per step instead of O(T²) per step. Auto-detection covers
    bare :class:`PAWNModel` and :class:`BottleneckEffective`; force
    ``False`` for the legacy full-forward path (mostly for parity
    testing). Force ``True`` to assert the cached path is available
    and fail loudly otherwise.

    ``cache_dtype`` (default: ``jnp.float32``): the dtype the KV cache
    is allocated in. Bf16 cuts cache memory in half, which is the
    binding constraint at production-scale ``n_games=1000`` (round-2
    perf review). Must be paired with ``compute_dtype`` — bf16 cache
    + fp32 forward would still downcast K/V on write and lose
    precision (round-3 bug-detector). A ``ValueError`` is raised if
    ``cache_dtype`` is bf16 / fp16 but ``compute_dtype`` is not.

    ``compute_dtype`` (default: ``None`` → fp32): the forward-pass
    activation dtype passed to ``forward_with_cache``. Used together
    with ``cache_dtype`` so the cache stores at full precision of the
    surrounding forward.

    Returns a dict with:
        sequences:       (n_games, max_seq_len) int32 — full token stream
        term_codes:      (n_games,) int8 — engine termination code
        game_lengths:    (n_games,) int32 — ply at which the game stopped
        forfeit_ply:     (n_games,) int32 — ply of first illegal move (-1
                                            if none); only meaningful in
                                            ``mask_illegal=False`` mode
        conditioning_offset: 0-d int32 — the prefix width ``C`` (move
                                            ``g`` lives at slot
                                            ``C + g - 1``), read by
                                            :func:`analyze_generated_games`

    ``mask_illegal=True`` forces every sampled token to be a legal move
    in the current position; ``False`` permits the model to sample an
    illegal move (recorded as a forfeit termination, code -3).
    """
    if max_seq_len is None:
        max_seq_len = model.cfg.max_seq_len

    # Cache dtype default: track ``compute_dtype`` rather than forcing
    # fp32. Production decode at ``n_per_outcome=1000`` is memory-
    # bound on the KV cache (~2.6 GB / game-batch at fp32, ~1.3 GB at
    # bf16), so the default fp32 was leaving 1.3 GB of headroom on the
    # table for callers running a bf16 forward. The user can still
    # force fp32 (or any other dtype) explicitly; only the default
    # tracks compute_dtype. Round-3 perf review (Opus conv).
    if cache_dtype is None:
        cache_dtype = compute_dtype  # may still be None → fp32 inside init_kv_cache
    # Cache dtype / compute dtype pairing: reject any configuration
    # that would silently mis-precision cache writes. A bf16 / fp16
    # cache requires the forward to run in *exactly the same* dtype so
    # K/V being written aren't downcast (fp32→bf16) or unsafely
    # rebound (bf16↔fp16, where bf16's larger exponent range can
    # overflow fp16). fp32 cache with any compute_dtype is always
    # safe (compute writes are widening or no-op). Round-3 + round-4
    # bug-detector + codex P2.
    _low_precision = (jnp.bfloat16, jnp.float16)
    if cache_dtype in _low_precision and cache_dtype != compute_dtype:
        raise ValueError(
            f"cache_dtype={cache_dtype} requires compute_dtype to be "
            f"the same (got compute_dtype={compute_dtype}). Mismatched "
            "low-precision pairs (e.g. bf16 cache + fp16 compute, or "
            "bf16 cache + fp32 compute) lose precision or overflow on "
            "cache writes."
        )

    # ---- Conditioning prefix (Phase-A Chunk 4) ----------------------------
    # Lay the full ``C``-wide ``[BOS][cond…]`` prefix into slots ``[0, C)``
    # via the shared assembler so generation uses the *same* layout the
    # trainer / corpus packer use. The ``"outcome"`` kind resolves to
    # ``outcome_token`` for every game; any other kind NULL-fills (the
    # diagnostics only vary the outcome). Moves then start at slot ``C``.
    C = conditioning_to_C(conditioning)  # noqa: N806
    if max_seq_len <= C:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must exceed the conditioning prefix "
            f"width C={C} so at least one move slot remains"
        )
    outcome_tokens = np.full(n_games, outcome_token, dtype=np.int32)
    prefix = build_prefix(conditioning, outcome_tokens, n_games)  # (n, C)

    sequences = np.full((n_games, max_seq_len), PAD_TOKEN, dtype=np.int32)
    sequences[:, :C] = prefix

    # `max_ply` for the engine is the move-positions budget (the slots
    # after the C-wide prefix), so subtract C from the total seq budget.
    max_move_positions = max_seq_len - C
    env = engine.PyBatchRLEnv(n_games, max_ply=max_move_positions, seed=seed)
    env.reset()
    terminated = np.zeros(n_games, dtype=bool)
    terminated_at = np.full(n_games, -1, dtype=np.int32)
    forfeit_ply = np.full(n_games, -1, dtype=np.int32)
    term_codes = np.full(n_games, -1, dtype=np.int8)
    all_indices = np.arange(n_games, dtype=np.uint32)

    # ---- Prefix-move application (v1 parity) ------------------------------
    prefix_end = 0
    if prefix_moves is not None and prefix_lengths is not None:
        padded = np.zeros((n_games, max_move_positions), dtype=np.uint16)
        clamped_pls = np.minimum(
            np.asarray(prefix_lengths, dtype=np.int32), max_move_positions
        )
        for i in range(n_games):
            pl = int(clamped_pls[i])
            padded[i, :pl] = prefix_moves[i, :pl].astype(np.uint16)
        lengths_u32 = np.asarray(clamped_pls, dtype=np.uint32)
        prefix_tc = env.load_prefixes(padded, lengths_u32)
        for i in range(n_games):
            pl = int(clamped_pls[i])
            sequences[i, C : C + pl] = prefix_moves[i, :pl]
            if prefix_tc[i] >= 0:
                terminated[i] = True
                terminated_at[i] = pl
                term_codes[i] = int(prefix_tc[i])
        prefix_end = int(clamped_pls.max()) if n_games > 0 else 0

    # ---- Decode loop ------------------------------------------------------
    rng = np.random.default_rng(seed)
    import equinox as eqx

    # Auto-detect cache support: both PAWNModel and BottleneckEffective
    # expose `forward_with_cache`; legacy wrappers (e.g. FiLM/LoRA
    # apply_fn returns a fresh PAWNModel — same attribute) do too. The
    # `getattr` form keeps EffectiveCallable Protocol-typed callers
    # from having to declare the optional method.
    has_cache_method = hasattr(model, "forward_with_cache")
    if use_kv_cache is None:
        use_kv_cache = has_cache_method
    if use_kv_cache and not has_cache_method:
        raise ValueError(
            f"use_kv_cache=True but {type(model).__name__} has no "
            "forward_with_cache method"
        )

    # Common sampling/env-update step factored out so the cached and
    # non-cached paths share one body (round-1 simplification flagged
    # the prior copy-paste). Returns the sampled token vector for
    # the caller's next-forward scheduling.
    def _sample_and_step_env(
        pos: int, next_logits_arr: np.ndarray,
    ) -> np.ndarray:
        active = ~terminated
        if temperature != 1.0:
            next_logits_arr = next_logits_arr / temperature
        if mask_illegal:
            # Engine returns a default-1980 mask; pass the model's actual
            # output width so engine truncates to match. (A.2: lm_head
            # output is NUM_ACTIONS + 1 = 1969 — the PAD column is kept
            # so termination sampling still works.)
            mask_vocab = int(next_logits_arr.shape[1])
            raw = np.asarray(
                env.get_legal_token_masks_batch(all_indices, mask_vocab)
            )
            pad_row = np.zeros((1, next_logits_arr.shape[1]), dtype=bool)
            pad_row[0, PAD_TOKEN] = True
            term_mat = terminated[:, None]
            full_mask = np.where(term_mat, pad_row, raw)
            next_logits_arr = np.where(full_mask, next_logits_arr, -np.inf)
        gumbel = -np.log(
            -np.log(rng.uniform(1e-10, 1.0, size=next_logits_arr.shape))
        )
        sampled = (next_logits_arr + gumbel).argmax(axis=-1).astype(np.int32)
        sequences[:, pos] = sampled
        # Slot ``pos`` holds the move at env-ply ``pos - C`` (the prefix
        # occupies slots ``[0, C)``). A PAD emitted at ``pos`` means the
        # game produced ``pos - C`` real moves before stopping.
        ply = pos - C
        pad_mask = active & (sampled == PAD_TOKEN)
        if pad_mask.any():
            terminated[pad_mask] = True
            terminated_at[pad_mask] = ply
            term_codes[pad_mask] = -2
        move_mask = active & ~pad_mask & ~terminated
        if move_mask.any():
            mv_idx = all_indices[move_mask]
            mv_tok = sampled[move_mask].astype(np.uint16)
            legality, step_tc = env.apply_moves(mv_idx, mv_tok)
            legality = np.asarray(legality)
            step_tc = np.asarray(step_tc)
            illegal = ~legality
            if illegal.any():
                forfeit_global = mv_idx[illegal]
                forfeit_ply[forfeit_global] = ply
                terminated[forfeit_global] = True
                terminated_at[forfeit_global] = ply
                term_codes[forfeit_global] = -3
            termed = legality & (step_tc >= 0)
            if termed.any():
                tg = mv_idx[termed]
                terminated[tg] = True
                # A terminating *move* at slot ``pos`` is the
                # ``(pos - C + 1)``-th move, so the game length counts it
                # in (one more than the PAD case ``pos - C``). With the v1
                # C=1 layout this reduces to ``pos``.
                terminated_at[tg] = ply + 1
                term_codes[tg] = step_tc[termed]
        return sampled

    if use_kv_cache:
        # Cache capacity = max_seq_len exactly (one slot per token in
        # the generated sequence). The cache lives outside the jit'd
        # step function so we can functionally update it across calls.
        # `pos_start` is a Python int — `lax.dynamic_update_slice`
        # accepts Python ints for offsets, so we avoid the host→device
        # scalar transfer per decode step (round-1 perf review).
        # `eqx.filter_jit` will treat `pos_start` as a traced argument
        # via `jnp.int32(pos)` though, since a fresh Python int per
        # call would re-trace; the alternative is `jax.jit` with
        # `static_argnums=(2,)` which would recompile per pos. Sharing
        # one compiled program across positions is the right trade.
        # The cache-aware narrowing: `has_cache_method` was checked
        # above. Cast through KVCacheCallable so pyright sees the
        # forward_with_cache attribute. (Runtime isinstance via the
        # Protocol's @runtime_checkable does the same check.)
        cached_model = cast(KVCacheCallable, model)
        cache: KVCache = init_kv_cache(
            cached_model.cfg,
            batch_size=n_games,
            max_seq_len=max_seq_len,
            dtype=cache_dtype,
        )

        @eqx.filter_jit
        def _forward_cached(
            tokens: Int[Array, "B T_new"],
            cache_in: KVCache,
            pos_start: Int[Array, ""],
        ) -> tuple[Float[Array, "B T_new V"], KVCache]:
            return cached_model.forward_with_cache(
                tokens, cache_in, pos_start, compute_dtype=compute_dtype,
            )

        # Prefill: the full C-wide prefix + any prefix moves in one call.
        # ``logits[:, -1]`` is the prediction at the last prefilled slot
        # (``prefill_len - 1``); with no prefix moves that is slot ``C-1``,
        # i.e. decoding starts from ``pos_start = C-1``.
        prefill_len = prefix_end + C
        tokens_jax = jnp.asarray(sequences[:, :prefill_len])
        logits_jax, cache = _forward_cached(
            tokens_jax, cache, jnp.int32(0),
        )
        next_logits = np.asarray(logits_jax[:, -1, :])

        for pos in range(prefill_len, max_seq_len):
            if not (~terminated).any():
                break
            _sample_and_step_env(pos, next_logits)
            if pos < max_seq_len - 1 and not terminated.all():
                tok_jax = jnp.asarray(sequences[:, pos : pos + 1])
                logits_jax, cache = _forward_cached(
                    tok_jax, cache, jnp.int32(pos),
                )
                next_logits = np.asarray(logits_jax[:, -1, :])
    else:
        @eqx.filter_jit
        def _forward(
            t: Int[Array, "B T"], a: Int[Array, "B T"]
        ) -> Float[Array, "B T V"]:
            return model(t, a, compute_dtype=compute_dtype)

        prefill_len = prefix_end + C
        tokens_jax = jnp.asarray(sequences[:, :prefill_len])
        attn_jax = jnp.ones_like(tokens_jax, dtype=jnp.bool_)
        logits_jax = _forward(tokens_jax, attn_jax)
        next_logits = np.asarray(logits_jax[:, -1, :])

        for pos in range(prefill_len, max_seq_len):
            if not (~terminated).any():
                break
            _sample_and_step_env(pos, next_logits)
            if pos < max_seq_len - 1 and not terminated.all():
                tokens_jax = jnp.asarray(sequences[:, : pos + 1])
                attn_jax = jnp.ones_like(tokens_jax, dtype=jnp.bool_)
                logits_jax = _forward(tokens_jax, attn_jax)
                next_logits = np.asarray(logits_jax[:, -1, :])

    # Games that never terminated within the window.
    still_going = ~terminated
    terminated_at[still_going] = max_move_positions
    term_codes[still_going] = 5
    return {
        "sequences": sequences,
        "term_codes": term_codes,
        "game_lengths": terminated_at.astype(np.int32),
        "forfeit_ply": forfeit_ply,
        # Prefix width C, so downstream analysis can map a move-count game
        # length back to its absolute slot (move ``g`` lives at slot
        # ``C + g - 1``). Stored as a 0-d int array for dict-shape parity.
        "conditioning_offset": np.asarray(C, dtype=np.int32),
    }


def analyze_generated_games(
    gen: dict[str, np.ndarray], conditioned_outcome: str
) -> dict[str, Any]:
    """Compute the v1-parity metrics dict for a batch of generated games.

    Outcome-match rate, forfeit rate, mean game length, post-terminal
    padding rate, premature-padding rate, outcome distribution.
    """
    sequences = gen["sequences"]
    term_codes = gen["term_codes"]
    game_lengths = gen["game_lengths"]
    forfeit_ply = gen["forfeit_ply"]
    # Prefix width C (default 1 for back-compat with pre-Chunk-4 dicts).
    # A move-count game length ``gl`` ends at absolute slot ``C + gl - 1``
    # (move ``g`` lives at slot ``C + g - 1``), so the post-terminal region
    # starts at ``C + gl``.
    C = int(gen.get("conditioning_offset", np.int32(1)))  # noqa: N806
    n = len(sequences)
    max_seq_len = sequences.shape[1]

    outcome_dist: dict[str, int] = {}
    n_match = 0
    for i in range(n):
        actual = _map_term_code_to_outcome_name(
            int(term_codes[i]), int(game_lengths[i])
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
        post_start = C + gl
        if post_start < max_seq_len:
            post = sequences[i, post_start:]
            n_post_terminal_tokens += len(post)
            n_post_terminal_pad += int((post == PAD_TOKEN).sum())
            n_post_terminal_move += int((post != PAD_TOKEN).sum())

    n_premature_pad = int((term_codes == -2).sum())

    return {
        "n_games": n,
        "outcome_match_rate": n_match / max(1, n),
        "outcome_distribution": {k: v / max(1, n) for k, v in outcome_dist.items()},
        "mean_game_length": float(game_lengths.mean()) if n else 0.0,
        "forfeit_rate": n_forfeit / max(1, n),
        "post_terminal_padding_rate": (
            n_post_terminal_pad / n_post_terminal_tokens
            if n_post_terminal_tokens > 0 else 1.0
        ),
        "post_terminal_move_count": n_post_terminal_move,
        "premature_padding_rate": n_premature_pad / max(1, n),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def outcome_signal_test(
    model: "PAWNModel | EffectiveCallable",
    *,
    outcome_prefix_trained: bool,
    n_per_outcome: int = 32,
    max_seq_len: int = 64,
    mask_conditions: tuple[bool, ...] = (False, True),
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, Any]:
    """v1 §6.1-6.3 outcome signal test (autoregressive).

    Generates ``n_per_outcome`` real games per outcome token, both with
    and without legal-move masking, and returns the per-outcome metrics
    dict. ``n_per_outcome`` defaults to 32 (down from v1's 1000) so the
    diagnostic stays under a minute without KV cache; production runs
    should crank it up.
    """
    if not outcome_prefix_trained:
        return _skipped("outcome_signal_test")
    out: dict[str, Any] = {"diagnostic": "outcome_signal_test"}
    for name, tok in OUTCOME_TOKENS.items():
        per_outcome: dict[str, Any] = {}
        for masked in mask_conditions:
            label = "masked" if masked else "unmasked"
            gen = autoregressive_generate(
                model, tok, n_per_outcome,
                mask_illegal=masked, max_seq_len=max_seq_len,
                cache_dtype=cache_dtype, compute_dtype=compute_dtype,
            )
            per_outcome[label] = analyze_generated_games(gen, name)
        out[name] = per_outcome
    return out


def prefix_continuation_test(
    model: "PAWNModel | EffectiveCallable",
    prefix: Int[Array, "P"] | np.ndarray,
    outcome_token: int,
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 32,
    n_continuations: int = 8,
    ar: bool = True,
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, Any]:
    """Given an outcome + the first P moves, does the model continue?

    With ``ar=True`` (default) this AR-decodes ``n_continuations`` real
    games from the prefix and returns the analysis metrics. The old
    single-shot next-move argmax is kept under ``next_move_argmax`` so
    callers that just want the first prediction don't pay for AR
    decode.
    """
    if not outcome_prefix_trained:
        return _skipped("prefix_continuation_test")
    prefix_np = np.asarray(prefix, dtype=np.int32)
    p = int(prefix_np.shape[0])

    # Cheap single-shot next-move argmax (the previous behaviour). The
    # next move after ``p`` prefix moves is predicted at slot ``C-1+p``.
    tokens, attn, C = _single_row_prefixed(  # noqa: N806
        outcome_token, seq_len, moves=prefix_np,
    )
    logits = model(tokens, attn, compute_dtype=compute_dtype)
    next_argmax = int(_argmax_action(logits)[0, C - 1 + p])

    result: dict[str, Any] = {
        "diagnostic": "prefix_continuation_test",
        "next_move_argmax": next_argmax,
        "prefix_length": p,
    }
    if ar:
        prefix_batch = np.tile(prefix_np[None, :], (n_continuations, 1))
        prefix_lens = np.full(n_continuations, p, dtype=np.int32)
        gen = autoregressive_generate(
            model, outcome_token, n_continuations,
            mask_illegal=True,
            prefix_moves=prefix_batch, prefix_lengths=prefix_lens,
            max_seq_len=seq_len,
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        )
        outcome_name = next(
            (k for k, v in OUTCOME_TOKENS.items() if v == outcome_token),
            "UNKNOWN",
        )
        result["ar_continuation"] = analyze_generated_games(gen, outcome_name)
    return result


def poisoned_prefix_test(
    model: "PAWNModel | EffectiveCallable",
    true_prefix: Int[Array, "P"] | np.ndarray,
    poisoned_outcome: int,
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 32,
    n_continuations: int = 8,
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, Any]:
    """Prefix continuation with an outcome token that contradicts the
    actual game (e.g. white-checkmates moves paired with a
    black-checkmates outcome). Tests whether the model "capitulates"
    to the poisoned outcome."""
    if not outcome_prefix_trained:
        return _skipped("poisoned_prefix_test")
    inner = prefix_continuation_test(
        model, true_prefix, poisoned_outcome,
        outcome_prefix_trained=True, seq_len=seq_len,
        n_continuations=n_continuations,
        cache_dtype=cache_dtype, compute_dtype=compute_dtype,
    )
    inner["diagnostic"] = "poisoned_prefix_test"
    return inner


def impossible_task_test(
    model: "PAWNModel | EffectiveCallable",
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 16,
    n_games: int = 16,
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, Any]:
    """Outcome conditioning = 'white checkmates' but no opening move can
    deliver mate-in-1. Reports the top-1 prob + entropy of the model's
    first-move distribution AND, when ``n_games > 0``, the
    autoregressive analysis (forfeit rate is the headline)."""
    if not outcome_prefix_trained:
        return _skipped("impossible_task_test")
    # First-move prediction lives at the last prefix slot ``C-1``.
    tokens, attn, C = _single_row_prefixed(WHITE_CHECKMATES, seq_len)  # noqa: N806
    logits = model(tokens, attn, compute_dtype=compute_dtype)
    probs = jax.nn.softmax(logits[0, C - 1, :NUM_ACTIONS], axis=-1)
    result: dict[str, Any] = {
        "diagnostic": "impossible_task_test",
        "top1_prob": float(probs.max()),
        "entropy": float(-(probs * jnp.log(probs + 1e-12)).sum()),
    }
    if n_games > 0:
        gen = autoregressive_generate(
            model, WHITE_CHECKMATES, n_games,
            mask_illegal=True, max_seq_len=seq_len,
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        )
        result["ar_analysis"] = analyze_generated_games(gen, "WHITE_CHECKMATES")
    return result


def improbable_task_test(
    model: "PAWNModel | EffectiveCallable",
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 16,
    n_games: int = 16,
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, Any]:
    """Outcome conditioning = DRAW_BY_AGREEMENT (very low prior). Reports
    top-1 prob + entropy + AR analysis."""
    if not outcome_prefix_trained:
        return _skipped("improbable_task_test")
    # First-move prediction lives at the last prefix slot ``C-1``.
    tokens, attn, C = _single_row_prefixed(DRAW_BY_AGREEMENT, seq_len)  # noqa: N806
    logits = model(tokens, attn, compute_dtype=compute_dtype)
    probs = jax.nn.softmax(logits[0, C - 1, :NUM_ACTIONS], axis=-1)
    result: dict[str, Any] = {
        "diagnostic": "improbable_task_test",
        "top1_prob": float(probs.max()),
        "entropy": float(-(probs * jnp.log(probs + 1e-12)).sum()),
    }
    if n_games > 0:
        gen = autoregressive_generate(
            model, DRAW_BY_AGREEMENT, n_games,
            mask_illegal=True, max_seq_len=seq_len,
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        )
        result["ar_analysis"] = analyze_generated_games(gen, "DRAW_BY_AGREEMENT")
    return result


def run_all_diagnostics(
    model: "PAWNModel | EffectiveCallable",
    *,
    outcome_prefix_trained: bool,
    n_per_outcome: int = 16,
    max_seq_len: int = 32,
    cache_dtype: jnp.dtype | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all 5 diagnostics, return a dict keyed by name.

    Each entry is either a result dict or a ``{"_skipped": ...}``
    sentinel when ``outcome_prefix_trained=False``. The headline
    metrics (outcome match rate, forfeit rate, etc.) come from real
    autoregressive generation per v1 parity; defaults are conservative
    so the full suite stays under a minute on a small backbone — pass
    larger ``n_per_outcome`` / ``max_seq_len`` for the full v1 numbers.

    ``cache_dtype`` / ``compute_dtype`` (default None → fp32) are
    threaded to every internal :func:`autoregressive_generate` call so
    a single ``--cache-dtype bfloat16 --compute-dtype bfloat16``
    invocation on the CLI halves the production-scale cache budget.
    The pairing constraint is enforced by ``autoregressive_generate``.
    """
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    return {
        "outcome_signal_test": outcome_signal_test(
            model, outcome_prefix_trained=outcome_prefix_trained,
            n_per_outcome=n_per_outcome, max_seq_len=max_seq_len,
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        ),
        "prefix_continuation_test": prefix_continuation_test(
            model, prefix, WHITE_CHECKMATES,
            outcome_prefix_trained=outcome_prefix_trained,
            n_continuations=min(8, n_per_outcome),
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        ),
        "poisoned_prefix_test": poisoned_prefix_test(
            model, prefix, BLACK_CHECKMATES,
            outcome_prefix_trained=outcome_prefix_trained,
            n_continuations=min(8, n_per_outcome),
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        ),
        "impossible_task_test": impossible_task_test(
            model, outcome_prefix_trained=outcome_prefix_trained,
            n_games=min(16, n_per_outcome),
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        ),
        "improbable_task_test": improbable_task_test(
            model, outcome_prefix_trained=outcome_prefix_trained,
            n_games=min(16, n_per_outcome),
            cache_dtype=cache_dtype, compute_dtype=compute_dtype,
        ),
    }
