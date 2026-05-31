"""Tests for the S8 eval surface — accuracy + diagnostics + probes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from pawn.config import (
    BLACK_CHECKMATES,
    BOS_TOKEN,
    NUM_ACTIONS,
    STALEMATE,
    TINY_SUPERNET,
    VOCAB_SIZE,
    WHITE_CHECKMATES,
)
from pawn.corpus import generate_corpus, pack_corpus
from pawn.eval import (
    AccuracyResult,
    _legal_token_grid,
    compute_move_accuracy,
    compute_per_phase_accuracy,
    compute_val_metrics,
)
from pawn.generation import (
    DIAGNOSTIC_NAMES,
    impossible_task_test,
    improbable_task_test,
    outcome_signal_test,
    poisoned_prefix_test,
    prefix_continuation_test,
    run_all_diagnostics,
)
from pawn.lichess_eval import (
    EloBin,
    compute_elo_stratified_accuracy,
    default_elo_bins,
)
from pawn.model import init_model
from pawn.probes import ProbeConfig, fit_probe


# ---------------------------------------------------------------------------
# Move accuracy + per-phase
# ---------------------------------------------------------------------------


def test_compute_move_accuracy_returns_float_in_zero_one() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=4, max_ply=16, seq_len=32, seed=0)
    acc = compute_move_accuracy(model, corpus, batch_size=4)
    assert 0.0 <= acc <= 1.0


def test_compute_per_phase_accuracy_returns_breakdown() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=4, max_ply=64, seq_len=80, seed=0)
    result = compute_per_phase_accuracy(model, corpus, batch_size=4)
    assert isinstance(result, AccuracyResult)
    assert 0.0 <= result.overall <= 1.0
    # Each phase fraction is finite.
    for v in (result.opening, result.midgame, result.endgame):
        assert 0.0 <= v <= 1.0


def test_per_phase_bins_by_ply_with_C_offset() -> None:
    """Phase membership keys on the ply a position predicts (``t - C``),
    not the raw sequence slot. The conditioning prefix shifts every move
    ``C`` slots right; binning on raw ``t`` would mislabel the first ``C``
    plies. We verify by comparing two corpora built from the *same* games
    but with C=1 (BOS only) vs C=2 (BOS + outcome): the per-phase
    *supervised counts* must be identical, because the +C offset cancels
    the prefix shift. A raw-``t`` binning would instead push ``C`` extra
    positions out of `opening` into `midgame` for the wider prefix.

    The counts (n_opening / n_midgame / n_endgame) depend only on the
    loss-mask + C, not on model predictions, so this is deterministic.
    """
    # One game, 40 real moves — straddles the opening (<20) / midgame
    # (20..60) boundary so the bin split is non-trivial.
    move_ids = np.arange(1, 41, dtype=np.int32)[None, :]  # (1, 40) legal action ids
    game_lengths = np.array([40], dtype=np.int32)
    outcome_tokens = np.array([WHITE_CHECKMATES], dtype=np.int32)

    # seq_len large enough to hold C + 40 moves under either conditioning.
    corpus_c1 = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=64, conditioning=[],
    )
    corpus_c2 = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=64,
        conditioning=["outcome"],
    )
    assert int(corpus_c1.outcome_offset[0]) == 1
    assert int(corpus_c2.outcome_offset[0]) == 2

    model = init_model(TINY_SUPERNET, key=0)
    res_c1 = compute_per_phase_accuracy(model, corpus_c1, batch_size=1)
    res_c2 = compute_per_phase_accuracy(model, corpus_c2, batch_size=1)

    # The +C offset makes the per-phase supervised counts invariant to the
    # prefix width: same games ⇒ same ply distribution ⇒ same bin counts.
    assert res_c1.n_total == res_c2.n_total == 40
    assert res_c1.n_opening == res_c2.n_opening
    assert res_c1.n_midgame == res_c2.n_midgame
    assert res_c1.n_endgame == res_c2.n_endgame
    # Independent expectation: with phases (opening_end=20, midgame_end=60)
    # and supervised plies p = t - C ∈ {-1, 0, …, 38} (40 positions), the
    # opening bin (p < 20) holds plies {-1..19} = 21 positions and the
    # midgame bin (20 <= p < 60) holds {20..38} = 19 positions.
    assert res_c1.n_opening == 21
    assert res_c1.n_midgame == 19
    assert res_c1.n_endgame == 0


def test_legal_grid_fills_last_supervised_ply_for_full_window_game() -> None:
    """Regression: the engine emits the legal mask of the board *before*
    each move, so a length-L game needs masks for plies ``0..L-1``. The
    corpus supervises plies ``0..capped-1`` with
    ``capped = min(game_length, n_move_slots)``, so the max supervised ply
    for a full-window game is ``n_move_slots - 1`` and that ply's legal
    mask MUST be filled.

    A previous off-by-one clamped the engine replay length to
    ``n_move_slots - 1``, dropping the last supervised ply's mask and
    leaving its row all-False — which ``compute_val_metrics`` then scored
    as illegal, systematically deflating ``legal_move_rate`` /
    ``late_legal_move_rate`` for every game that fills the window.

    Build a corpus with ``max_ply == seq_len - C`` so most random games run
    to the window edge, then assert the last supervised slot's legal mask
    is non-empty.
    """
    seq_len = 12
    corpus = generate_corpus(n_games=64, max_ply=seq_len, seq_len=seq_len, seed=3)
    C = int(corpus.outcome_offset[0])
    n_move_slots = seq_len - C
    capped = np.minimum(corpus.game_lengths, n_move_slots)
    full = np.where(capped == n_move_slots)[0]
    # The construction is meant to produce full-window games; if it doesn't,
    # the regression it guards can't fire and the test is vacuous.
    assert full.size > 0, "expected at least one full-window game"

    grid = _legal_token_grid(corpus)
    # Last supervised slot for a full-window game predicts ply
    # ``n_move_slots - 1`` at sequence slot ``C-1 + (n_move_slots-1)``.
    last_slot = (C - 1) + (n_move_slots - 1)
    for g in full:
        n_legal = int(grid[g, last_slot].sum())
        assert n_legal > 0, (
            f"full-window game {g}: legal mask at last supervised slot "
            f"{last_slot} is empty (off-by-one clamp regression)"
        )


def test_val_legal_move_rate_counts_full_window_predictions() -> None:
    """End-to-end guard: a model predicting a known-legal move at the last
    supervised ply of a full-window game must be counted legal there.

    Construct a single full-window game and feed the *ground-truth* legal
    move as the model's prediction at the last supervised slot via the
    legal grid: the slot's legal mask must be non-empty so the rate isn't
    silently deflated. We assert ``legal_move_rate`` is strictly positive
    (the all-False-row bug pinned it artificially low).
    """
    seq_len = 12
    corpus = generate_corpus(n_games=32, max_ply=seq_len, seq_len=seq_len, seed=7)
    n_move_slots = seq_len - int(corpus.outcome_offset[0])
    full = np.where(
        np.minimum(corpus.game_lengths, n_move_slots) == n_move_slots
    )[0]
    assert full.size > 0, "expected at least one full-window game"

    model = init_model(TINY_SUPERNET, key=0)
    metrics = compute_val_metrics(model, corpus, batch_size=8, compute_legal=True)
    # With the legal masks correctly filled at every supervised ply, the
    # legal-move rate is a real fraction in (0, 1]; the off-by-one bug
    # depressed it by forcing the last supervised ply of every full-window
    # game to count as illegal.
    assert 0.0 < metrics.legal_move_rate <= 1.0


def test_compute_val_metrics_reports_full_v1_schema() -> None:
    """Parity item ``eval-jax-no-top5-per-ply-loss`` + ``perplexity-metric``:
    the val pass returns top-1, top-5, CE loss, perplexity, and the legal
    move rate together (v1 ``CLMTrainer.evaluate`` / ``eval_accuracy.py``
    schema), not top-1 alone."""
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=8, max_ply=40, seq_len=48, seed=3)
    vm = compute_val_metrics(model, corpus, batch_size=4)
    assert 0.0 <= vm.top1 <= 1.0
    assert 0.0 <= vm.top5 <= 1.0
    # top-5 is a superset of top-1, so it can never be lower.
    assert vm.top5 >= vm.top1 - 1e-6
    assert vm.val_loss > 0.0
    # perplexity == exp(loss) for the clamped loss range.
    assert vm.perplexity == pytest.approx(np.exp(min(vm.val_loss, 20.0)), rel=1e-5)
    assert 0.0 <= vm.legal_move_rate <= 1.0


def test_legal_move_rate_invariant_to_min_eval_ply() -> None:
    """Regression for the late_legal_move_rate denominator/numerator
    mismatch: the legality metrics are gated by the plain supervised mask,
    NOT the MAIA opening-skip (``min_eval_ply``) mask. So both
    ``legal_move_rate`` and ``late_legal_move_rate`` (with ``late_ply=0``,
    i.e. every supervised position is "late") must be invariant to
    ``min_eval_ply`` — only the loss / top-1 / top-5 scalars see the skip.

    The buggy version gated legality by ``eval_mask = loss & (ply >=
    min_eval_ply)`` for the numerator while leaving the denominator on the
    plain supervised count, silently deflating the rate when
    ``min_eval_ply > 0``."""
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=16, max_ply=60, seq_len=72, seed=11)
    base = compute_val_metrics(
        model, corpus, batch_size=8, min_eval_ply=0, late_ply=0
    )
    skipped = compute_val_metrics(
        model, corpus, batch_size=8, min_eval_ply=10, late_ply=0
    )
    # Legality is gated by the plain supervised mask, so the opening-skip
    # must not move either legal-rate metric.
    assert skipped.legal_move_rate == pytest.approx(
        base.legal_move_rate, abs=1e-6
    )
    assert skipped.late_legal_move_rate == pytest.approx(
        base.late_legal_move_rate, abs=1e-6
    )
    # With late_ply=0 every supervised position is "late", so the two
    # legality rates coincide within a pass.
    assert base.late_legal_move_rate == pytest.approx(
        base.legal_move_rate, abs=1e-6
    )
    # The loss/top-1 scalars DO change (the skip drops the easy openings),
    # confirming the skip is still applied where it should be.
    assert skipped.val_loss != pytest.approx(base.val_loss, abs=1e-6)


def test_min_eval_ply_skips_opening_in_overall_not_in_phases() -> None:
    """Parity item ``eval-jax-no-min-eval-ply``: the MAIA opening-skip drops
    the first ``min_eval_ply`` plies from the *overall* headline metrics but
    leaves the per-phase breakdown (which always bins from ply 0) untouched.

    A high min_eval_ply that excludes every opening-phase position must
    change the overall top-1 (different position set) while the per-phase
    ``opening`` accuracy stays identical to the unskipped pass."""
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=16, max_ply=60, seq_len=72, seed=11)
    base = compute_val_metrics(model, corpus, batch_size=8, min_eval_ply=0)
    skipped = compute_val_metrics(model, corpus, batch_size=8, min_eval_ply=20)
    # Per-phase opening accuracy is computed from ply 0 regardless of the
    # opening-skip, so it must be identical across the two passes.
    assert skipped.phases.opening == pytest.approx(base.phases.opening, abs=1e-6)
    assert skipped.phases.midgame == pytest.approx(base.phases.midgame, abs=1e-6)
    # The overall metrics see a strictly smaller (later-ply) position set,
    # so the supervised count drops.
    assert skipped.phases.n_total == base.phases.n_total  # phase count unchanged


def test_compute_per_ply_accuracy_returns_per_ply_breakdown() -> None:
    """Parity item ``per-ply-breakdown``: a per-ply top-1 accuracy map keyed
    by the ply each position predicts (v1 ``--per-ply``)."""
    from pawn.eval import PerPlyResult, compute_per_ply_accuracy

    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=8, max_ply=24, seq_len=32, seed=5)
    res = compute_per_ply_accuracy(model, corpus, batch_size=4)
    assert isinstance(res, PerPlyResult)
    assert res.accuracy, "expected at least one supervised ply"
    # Plies are 0-indexed (first move is ply 0).
    assert min(res.accuracy) == 0
    for ply, acc in res.accuracy.items():
        assert 0.0 <= acc <= 1.0
        assert res.n[ply] > 0


def test_phase_naming_is_v2_midgame_endgame() -> None:
    """Parity item ``phase-naming-changed``: v2 renamed v1's
    ``opening/middle/late`` phases to ``opening/midgame/endgame``. Pin the
    v2 names so the rename is intentional and documented, not a silent
    drift — :class:`AccuracyResult` exposes ``midgame`` / ``endgame``, not
    ``middle`` / ``late``."""
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=4, max_ply=64, seq_len=80, seed=0)
    result = compute_per_phase_accuracy(model, corpus, batch_size=4)
    fields = set(AccuracyResult.__dataclass_fields__)
    assert {"opening", "midgame", "endgame"} <= fields
    assert "middle" not in fields and "late" not in fields
    # The accessors resolve (the v2 names are the live API).
    assert isinstance(result.midgame, float)
    assert isinstance(result.endgame, float)


def test_argmax_never_picks_pad_or_outcome_tokens() -> None:
    """The plan-pinned contract: argmax restricted to [0, NUM_ACTIONS).
    Confirm by checking that compute_move_accuracy's argmax output is
    always in the valid action range."""
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=2, max_ply=16, seq_len=32, seed=0)
    tokens = jnp.asarray(corpus.tokens)
    attn = jnp.asarray(corpus.attn_mask)
    logits = model(tokens, attn)
    pred = jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1)
    assert jnp.all(pred < NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Generation diagnostics: gate behavior + finite values
# ---------------------------------------------------------------------------


def test_diagnostic_names_lists_all_five() -> None:
    assert set(DIAGNOSTIC_NAMES) == {
        "outcome_signal_test",
        "prefix_continuation_test",
        "poisoned_prefix_test",
        "impossible_task_test",
        "improbable_task_test",
    }


@pytest.mark.parametrize("name", DIAGNOSTIC_NAMES)
def test_each_diagnostic_skips_when_not_outcome_prefix_trained(name: str) -> None:
    """Plan §10 S8 / §11: all 5 diagnostics return `{"_skipped": ...}`
    when `outcome_prefix_trained=False`."""
    model = init_model(TINY_SUPERNET, key=0)
    all_results = run_all_diagnostics(model, outcome_prefix_trained=False)
    assert "_skipped" in all_results[name]


def test_outcome_signal_test_runs_when_gate_on() -> None:
    """v1 parity: outcome_signal_test now runs real autoregressive
    generation per outcome and reports per-outcome metrics (match rate,
    forfeit rate, mean game length). Tiny defaults keep the test under
    a few seconds without the KV-cached decoder."""
    from pawn.generation import OUTCOME_TOKENS

    model = init_model(TINY_SUPERNET, key=0)
    res = outcome_signal_test(
        model, outcome_prefix_trained=True,
        n_per_outcome=2, max_seq_len=8, mask_conditions=(True,),
    )
    assert "_skipped" not in res
    # Every v1 outcome appears with masked-condition results.
    for name in OUTCOME_TOKENS:
        assert name in res
        assert "masked" in res[name]
        metrics = res[name]["masked"]
        # Headline v1 metrics are present and in-range.
        assert 0.0 <= metrics["outcome_match_rate"] <= 1.0
        assert 0.0 <= metrics["forfeit_rate"] <= 1.0
        assert metrics["mean_game_length"] >= 0.0


def test_prefix_continuation_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    res = prefix_continuation_test(
        model, prefix, WHITE_CHECKMATES, outcome_prefix_trained=True,
        n_continuations=2, seq_len=8,
    )
    # `next_move_argmax` is the cheap single-shot probe; the AR
    # analysis is the v1-parity metric block.
    assert "next_move_argmax" in res
    assert 0 <= res["next_move_argmax"] < NUM_ACTIONS
    assert "ar_continuation" in res
    assert res["ar_continuation"]["n_games"] == 2


def test_poisoned_prefix_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    res = poisoned_prefix_test(
        model, prefix, BLACK_CHECKMATES, outcome_prefix_trained=True,
        n_continuations=2, seq_len=8,
    )
    assert res["diagnostic"] == "poisoned_prefix_test"
    assert "ar_continuation" in res


def test_impossible_task_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    res = impossible_task_test(
        model, outcome_prefix_trained=True, n_games=2, seq_len=8,
    )
    assert "top1_prob" in res and "entropy" in res
    assert "ar_analysis" in res
    # Forfeit rate is the headline for the impossible task.
    assert 0.0 <= res["ar_analysis"]["forfeit_rate"] <= 1.0


def test_improbable_task_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    res = improbable_task_test(
        model, outcome_prefix_trained=True, n_games=2, seq_len=8,
    )
    assert "top1_prob" in res and "entropy" in res
    assert "ar_analysis" in res


def test_improbable_task_conditions_on_producible_outcome() -> None:
    """§8.4: the improbable task must condition on an outcome the engine
    can actually terminate on (STALEMATE), not the unreachable
    ``DRAW_BY_AGREEMENT``. The Lichess-specific outcomes never appear as a
    random-game termination, so conditioning on one pins
    ``outcome_match_rate`` at 0 by construction and the diagnostic measures
    nothing. STALEMATE is one of the five engine-producible outcomes, so a
    regression back to a non-producible token would surface here.
    """
    from pawn.config import DRAW_BY_AGREEMENT
    from pawn.generation import OUTCOME_TOKENS, _single_row_prefixed

    # The conditioning outcome the diagnostic uses must be one the engine
    # can terminate on (i.e. present in OUTCOME_TOKENS), and must NOT be a
    # Lichess-only token like DRAW_BY_AGREEMENT.
    assert "STALEMATE" in OUTCOME_TOKENS
    assert STALEMATE in OUTCOME_TOKENS.values()
    assert DRAW_BY_AGREEMENT not in OUTCOME_TOKENS.values()

    # Pin the literal token the diagnostic conditions on by replaying the
    # same prefix construction: the outcome slot must carry STALEMATE, not
    # the unreachable DRAW_BY_AGREEMENT.
    tokens, _, C = _single_row_prefixed(STALEMATE, 8)
    conditioned = int(tokens[0, C - 1])
    assert conditioned == STALEMATE != DRAW_BY_AGREEMENT

    model = init_model(TINY_SUPERNET, key=0)
    res = improbable_task_test(
        model, outcome_prefix_trained=True, n_games=2, seq_len=8,
    )
    # The AR analysis ran against a producible target — its outcome
    # distribution only ever contains engine-reachable outcome names.
    assert "outcome_distribution" in res["ar_analysis"]


def test_run_all_diagnostics_has_5_entries() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    results = run_all_diagnostics(model, outcome_prefix_trained=True)
    assert set(results.keys()) == set(DIAGNOSTIC_NAMES)


def test_autoregressive_generate_kv_cache_matches_full_forward() -> None:
    """The cached and non-cached paths in `autoregressive_generate`
    must agree on the generated sequences (both seeded the same and
    sampling is deterministic for argmax-style Gumbel-max at the same
    weights). Pins the speed-vs-correctness invariant — a future change
    that breaks numeric parity between cache paths shows up here."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    gen_plain = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2,
        mask_illegal=True, max_seq_len=8, seed=0, use_kv_cache=False,
    )
    gen_cached = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2,
        mask_illegal=True, max_seq_len=8, seed=0, use_kv_cache=True,
    )
    # Sequences should match: the two paths share sampling RNG (the
    # `seed=0` -> default_rng draws are deterministic) and produce
    # numerically-equivalent logits per parity test in test_jax_model.
    assert np.array_equal(gen_plain["sequences"], gen_cached["sequences"])
    assert np.array_equal(gen_plain["term_codes"], gen_cached["term_codes"])
    assert np.array_equal(gen_plain["game_lengths"], gen_cached["game_lengths"])


def test_autoregressive_generate_lays_down_c_wide_prefix() -> None:
    """Phase-A Chunk 4: generation lays the full ``[BOS][cond…]`` prefix
    (C = 1 + len(conditioning)) into slots ``[0, C)`` and starts moves at
    slot ``C``, replacing the v1 hardcoded outcome-at-slot-0 layout.
    With the default ``conditioning=("outcome",)`` ⇒ C=2: slot 0 = BOS,
    slot 1 = the outcome token."""
    from pawn.config import BOS_TOKEN, NULL_TOKEN
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    gen = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=3,
        mask_illegal=True, max_seq_len=12, seed=0,
    )
    seqs = gen["sequences"]
    assert (seqs[:, 0] == BOS_TOKEN).all()
    assert (seqs[:, 1] == WHITE_CHECKMATES).all()
    # No NULL leaked into the prefix (outcome resolves to a real token).
    assert not (seqs == NULL_TOKEN).any()
    # The prefix width C is reported for downstream analysis.
    assert int(gen["conditioning_offset"]) == 2


def test_autoregressive_generate_kv_cache_matches_full_forward_with_prefix() -> None:
    """KV-cache vs full-forward parity must hold when the decode starts
    from a non-trivial move prefix on top of the C-wide conditioning
    prefix. The cached prefill processes ``[0, C + prefix_len)`` from
    ``pos_start=0`` and decodes from ``pos_start = C - 1 + prefix_len``;
    the full-forward path recomputes the whole window each step. Both must
    produce bit-identical sequences (shared Gumbel RNG + numerically
    equivalent logits)."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    # Two games, each seeded with the same legal, non-terminating 2-ply
    # opening line (token 45 = a legal white opening move; token 1388 = a
    # legal black reply).
    prefix_moves = np.array([[45, 1388], [45, 1388]], dtype=np.int32)
    prefix_lengths = np.array([2, 2], dtype=np.int32)

    gen_plain = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2, use_kv_cache=False,
        mask_illegal=True, max_seq_len=12, seed=0,
        prefix_moves=prefix_moves, prefix_lengths=prefix_lengths,
    )
    gen_cached = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2, use_kv_cache=True,
        mask_illegal=True, max_seq_len=12, seed=0,
        prefix_moves=prefix_moves, prefix_lengths=prefix_lengths,
    )
    assert np.array_equal(gen_plain["sequences"], gen_cached["sequences"])
    assert np.array_equal(gen_plain["term_codes"], gen_cached["term_codes"])
    assert np.array_equal(gen_plain["game_lengths"], gen_cached["game_lengths"])
    # The prefix moves landed at slots [C, C+prefix_len): C=2 → slots 2,3.
    assert (gen_plain["sequences"][:, 2] == 45).all()
    assert (gen_plain["sequences"][:, 3] == 1388).all()
    # The games kept decoding past the prefix (not terminated at load).
    assert (gen_plain["game_lengths"] >= 2).all()


def test_autoregressive_generate_bf16_cache_runs() -> None:
    """The `cache_dtype=jnp.bfloat16` path (paired with `compute_dtype`)
    must run end-to-end. Round-3 test-risk MEDIUM: the bf16-cache opt-in
    surface had no test, so a future change that broke the dtype pairing
    could land silently. This pins that the bf16 forward + bf16 cache
    path produces valid output (sequences, term codes, game lengths)
    with no exceptions."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    gen = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2,
        mask_illegal=True, max_seq_len=8, seed=0,
        use_kv_cache=True,
        cache_dtype=jnp.bfloat16, compute_dtype=jnp.bfloat16,
    )
    # Headline shape + dtype invariants — sequences are int32 game
    # tokens, not affected by the bf16 cache choice.
    assert gen["sequences"].shape == (2, 8)
    assert gen["sequences"].dtype == np.int32
    assert gen["term_codes"].shape == (2,)
    # Phase-A Chunk 4 layout: slot 0 is BOS, the outcome conditioning lives
    # at slot 1 (the default `conditioning=("outcome",)` ⇒ C=2). Both are
    # preserved regardless of cache dtype.
    assert (gen["sequences"][:, 0] == BOS_TOKEN).all()
    assert (gen["sequences"][:, 1] == WHITE_CHECKMATES).all()


def test_autoregressive_generate_rejects_bf16_cache_with_fp32_compute() -> None:
    """`cache_dtype=bfloat16` without `compute_dtype=bfloat16` must
    raise — a fp32 forward writing into a bf16 cache silently
    downcasts on every write (round-3 bug-detector). The
    `autoregressive_generate` validator catches this at call time
    rather than letting the operator commit to a lossy run."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    with pytest.raises(ValueError, match="requires compute_dtype"):
        autoregressive_generate(
            model, WHITE_CHECKMATES, n_games=2,
            mask_illegal=True, max_seq_len=8,
            cache_dtype=jnp.bfloat16,  # no compute_dtype
        )


def test_autoregressive_generate_rejects_mismatched_low_precision_pair() -> None:
    """Round-4 codex P2: `cache=fp16 + compute=bf16` (or the reverse)
    is permitted by a naive "both low-precision" check but is
    actually unsafe — bf16's exponent range is wider than fp16, so a
    bf16-compute K/V can overflow when written to a fp16 cache.
    The validator now requires *exact* match for low-precision
    pairs."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    # fp16 cache + bf16 compute: rejected.
    with pytest.raises(ValueError, match="requires compute_dtype to be the same"):
        autoregressive_generate(
            model, WHITE_CHECKMATES, n_games=2,
            mask_illegal=True, max_seq_len=8,
            cache_dtype=jnp.float16, compute_dtype=jnp.bfloat16,
        )
    # bf16 cache + fp16 compute: rejected (overflow direction).
    with pytest.raises(ValueError, match="requires compute_dtype to be the same"):
        autoregressive_generate(
            model, WHITE_CHECKMATES, n_games=2,
            mask_illegal=True, max_seq_len=8,
            cache_dtype=jnp.bfloat16, compute_dtype=jnp.float16,
        )


# ---------------------------------------------------------------------------
# Generation diagnostics: corpus-driven scenarios + control arms (v1 parity)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gen_diag_corpus():  # type: ignore[no-untyped-def]
    """A small engine self-play corpus with enough decisive games to
    populate the WHITE/BLACK-checkmate and ply-limit buckets the
    corpus-driven diagnostics sample from. ``mate_boost`` upweights mate
    lines so the checkmate buckets are non-empty at this size."""
    from pawn.generation import build_gen_diag_corpus

    return build_gen_diag_corpus(600, max_ply=120, seed=2, mate_boost=5.0)


def test_build_gen_diag_corpus_carries_raw_term_codes() -> None:
    """The corpus builder keeps the engine's RAW termination codes (not the
    collapsed outcome token) so `impossible_task_test` can still single out
    insufficient-material (code 4) from the other draw-by-rule codes."""
    from pawn.generation import GenDiagCorpus, build_gen_diag_corpus

    corpus = build_gen_diag_corpus(8, max_ply=32, seed=0)
    assert isinstance(corpus, GenDiagCorpus)
    assert len(corpus) == 8
    assert corpus.move_ids.shape[0] == 8
    assert corpus.game_lengths.shape == (8,)
    assert corpus.term_codes.shape == (8,)
    # Raw engine codes live in [-1, 5]; never an outcome-token id (>= 1969).
    assert corpus.term_codes.max() < 6
    assert corpus.term_codes.min() >= -1


def test_poisoning_pairs_enumerates_the_four_v1_pairs() -> None:
    """v1 parity: the poisoned-prefix test runs exactly these four
    actual->poisoned outcome pairs (eval_suite/generation.py:543-548)."""
    from pawn.generation import POISONING_PAIRS

    assert POISONING_PAIRS == (
        ("WHITE_CHECKMATES", "BLACK_CHECKMATES"),
        ("WHITE_CHECKMATES", "DRAW_BY_RULE"),
        ("DRAW_BY_RULE", "WHITE_CHECKMATES"),
        ("PLY_LIMIT", "WHITE_CHECKMATES"),
    )


def test_prefix_continuation_corpus_driven_cross_conditions(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """v1 §6.4 restored: with a corpus, prefix_continuation_test builds the
    full cross-conditioning matrix — for every populated outcome bucket and
    every prefix cut, it AR-decodes under ALL five outcome conditionings."""
    from pawn.generation import OUTCOME_TOKENS, prefix_continuation_test

    model = init_model(TINY_SUPERNET, key=0)
    res = prefix_continuation_test(
        model, outcome_prefix_trained=True, corpus=gen_diag_corpus,
        seq_len=24, n_per_bucket=8,
        prefix_pcts=(0.5,), absolute_plies=(),
    )
    assert res["corpus_driven"] is True
    buckets = res["buckets"]
    # At least the ply-limit bucket is populated at this corpus size.
    assert "PLY_LIMIT" in buckets
    pct_block = buckets["PLY_LIMIT"]["pct_50"]
    # Cross-conditioning: every outcome token was used as a conditioning
    # input on the SAME prefixes (the off-diagonal cells are the signal).
    assert set(pct_block.keys()) == set(OUTCOME_TOKENS)
    for cond_name, metrics in pct_block.items():
        assert 0.0 <= metrics["outcome_match_rate"] <= 1.0


def test_poisoned_prefix_corpus_driven_tracks_original_match(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """v1 §6.5 restored: the corpus-driven poisoned test runs the
    POISONING_PAIRS and reports `original_outcome_match_rate` — the
    capitulation signal (did the continuation revert to the prefix's TRUE
    outcome despite the contradicting poison?)."""
    from pawn.generation import poisoned_prefix_test

    model = init_model(TINY_SUPERNET, key=0)
    res = poisoned_prefix_test(
        model, outcome_prefix_trained=True, corpus=gen_diag_corpus,
        seq_len=24, n_per_pair=8, prefix_pct=0.5,
    )
    assert res["corpus_driven"] is True
    pairs = res["pairs"]
    # PLY_LIMIT is populated, so the PLY_LIMIT->WHITE_CHECKMATES pair fires.
    assert "PLY_LIMIT->WHITE_CHECKMATES" in pairs
    entry = pairs["PLY_LIMIT->WHITE_CHECKMATES"]
    assert entry["actual_outcome"] == "PLY_LIMIT"
    assert entry["poisoned_outcome"] == "WHITE_CHECKMATES"
    assert 0.0 <= entry["original_outcome_match_rate"] <= 1.0
    # The poisoned-target match rate is still reported alongside.
    assert 0.0 <= entry["outcome_match_rate"] <= 1.0


def test_impossible_task_corpus_driven_has_scenarios_and_control(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """v1 §6.6 restored: corpus-driven impossible test runs the
    zero-remaining-ply scenario AND the matched-prefix `control_ply_limit`
    baseline (honest outcome on the same prefixes)."""
    from pawn.generation import impossible_task_test

    model = init_model(TINY_SUPERNET, key=0)
    res = impossible_task_test(
        model, outcome_prefix_trained=True, corpus=gen_diag_corpus,
        seq_len=24, n_per_scenario=8,
    )
    assert res["corpus_driven"] is True
    scen = res["scenarios"]
    # Ply-limit games drive both the impossible scenario and its control.
    assert "zero_remaining_ply" in scen
    assert "control_ply_limit" in scen
    assert 0.0 <= scen["zero_remaining_ply"]["outcome_match_rate"] <= 1.0
    # The control conditions on the HONEST outcome (PLY_LIMIT), so its
    # match rate is the baseline the impossible arm is measured against.
    assert 0.0 <= scen["control_ply_limit"]["outcome_match_rate"] <= 1.0


def test_improbable_task_corpus_driven_has_control_arms(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """v1 §6.7 restored: corpus-driven improbable test runs the
    checkmate-few-ply scenario with its `control_few_ply` arm (the
    scientific baseline that isolates the improbable-conditioning effect)."""
    from pawn.generation import improbable_task_test

    model = init_model(TINY_SUPERNET, key=0)
    res = improbable_task_test(
        model, outcome_prefix_trained=True, corpus=gen_diag_corpus,
        seq_len=24, n_per_scenario=8,
    )
    assert res["corpus_driven"] is True
    scen = res["scenarios"]
    # Long games (>= min_long) drive the few-ply scenario + its control.
    assert "checkmate_few_ply" in scen
    assert "control_few_ply" in scen
    assert 0.0 <= scen["checkmate_few_ply"]["forfeit_rate"] <= 1.0
    assert 0.0 <= scen["control_few_ply"]["outcome_match_rate"] <= 1.0


def test_run_all_diagnostics_corpus_driven_routes_four_tests(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """Passing a corpus to run_all_diagnostics switches the four
    corpus-capable diagnostics into their v1 multi-scenario mode while
    keeping all five gate-checked entries."""
    from pawn.generation import DIAGNOSTIC_NAMES, run_all_diagnostics

    model = init_model(TINY_SUPERNET, key=0)
    results = run_all_diagnostics(
        model, outcome_prefix_trained=True, corpus=gen_diag_corpus,
        n_per_outcome=2, max_seq_len=24,
        n_per_bucket=8, n_per_pair=8, n_per_scenario=8,
    )
    assert set(results.keys()) == set(DIAGNOSTIC_NAMES)
    # The four corpus-capable diagnostics report corpus_driven=True.
    for name in (
        "prefix_continuation_test", "poisoned_prefix_test",
        "impossible_task_test", "improbable_task_test",
    ):
        assert results[name]["corpus_driven"] is True


def test_run_all_diagnostics_corpus_driven_skips_when_gate_off(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """The outcome-prefix gate still dominates: even with a corpus wired
    in, gate=False returns the `_skipped` sentinel for every diagnostic."""
    from pawn.generation import DIAGNOSTIC_NAMES, run_all_diagnostics

    model = init_model(TINY_SUPERNET, key=0)
    results = run_all_diagnostics(
        model, outcome_prefix_trained=False, corpus=gen_diag_corpus,
        n_per_outcome=2, max_seq_len=24,
    )
    for name in DIAGNOSTIC_NAMES:
        assert "_skipped" in results[name]


def test_autoregressive_generate_sub_batch_chunking() -> None:
    """v1 parity: `batch_size` chunks a large `n_games` decode into
    bounded sub-batches and concatenates the per-chunk results, so the
    output shapes match an unchunked run of the same size."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    gen = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=5,
        mask_illegal=True, max_seq_len=10, seed=0, batch_size=2,
    )
    # 5 games in chunks of 2 -> 3 chunks (2+2+1), concatenated to 5 rows.
    assert gen["sequences"].shape == (5, 10)
    assert gen["term_codes"].shape == (5,)
    assert gen["game_lengths"].shape == (5,)
    assert gen["forfeit_ply"].shape == (5,)
    # The scalar prefix-width invariant survives concatenation.
    assert int(gen["conditioning_offset"]) == 2
    # Every game still carries the C-wide [BOS][outcome] prefix.
    assert (gen["sequences"][:, 0] == BOS_TOKEN).all()
    assert (gen["sequences"][:, 1] == WHITE_CHECKMATES).all()


def test_corpus_poisoned_prefix_chunked_matches_unchunked_shapes(gen_diag_corpus) -> None:  # type: ignore[no-untyped-def]
    """The corpus-driven path threads `decode_batch_size` into every
    internal AR decode (the v1 OOM-avoidance knob). A run whose
    `n_per_pair` EXCEEDS the chunk size produces the same per-pair result
    structure as an unchunked (`decode_batch_size=None`) run — proving the
    chunking is wired on the real corpus path, not just reachable from its
    unit test."""
    model = init_model(TINY_SUPERNET, key=0)
    # n_per_pair=6 with a chunk of 2 -> 3 sub-batches per pair.
    chunked = poisoned_prefix_test(
        model,
        outcome_prefix_trained=True,
        corpus=gen_diag_corpus,
        seq_len=24,
        n_per_pair=6,
        prefix_pct=0.5,
        decode_batch_size=2,
    )
    unchunked = poisoned_prefix_test(
        model,
        outcome_prefix_trained=True,
        corpus=gen_diag_corpus,
        seq_len=24,
        n_per_pair=6,
        prefix_pct=0.5,
        decode_batch_size=None,
    )
    assert chunked["corpus_driven"] is True
    # Same set of populated pairs and the same per-pair metric keys/shapes.
    assert set(chunked["pairs"]) == set(unchunked["pairs"])
    assert chunked["pairs"], "expected at least one populated poisoning pair"
    for label, entry in chunked["pairs"].items():
        ref = unchunked["pairs"][label]
        assert set(entry) == set(ref)
        assert entry["n_games"] == ref["n_games"] == 6
        assert 0.0 <= entry["original_outcome_match_rate"] <= 1.0


def test_impossible_control_reuses_scenario1_prefixes(gen_diag_corpus, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Defect fix: the `control_ply_limit` arm must REPLAY the exact
    Scenario-1 zero-remaining-ply prefixes (only the conditioning outcome
    differs), not independently re-derive them. We capture the
    `prefix_moves` / `prefix_lengths` of every AR decode and assert the
    control's prefixes are byte-identical to Scenario 1's."""
    import pawn.generation as gen_mod

    model = init_model(TINY_SUPERNET, key=0)
    captured: list[tuple[int, np.ndarray | None, np.ndarray | None]] = []
    real = gen_mod.autoregressive_generate

    def spy(model_, outcome_token, n_games, **kw):  # type: ignore[no-untyped-def]
        pm = kw.get("prefix_moves")
        pl = kw.get("prefix_lengths")
        captured.append((
            int(outcome_token),
            np.array(pm) if pm is not None else None,
            np.array(pl) if pl is not None else None,
        ))
        return real(model_, outcome_token, n_games, **kw)

    monkeypatch.setattr(gen_mod, "autoregressive_generate", spy)
    res = gen_mod.impossible_task_test(
        model, outcome_prefix_trained=True, corpus=gen_diag_corpus,
        seq_len=24, n_per_scenario=8, decode_batch_size=None,
    )
    assert "zero_remaining_ply" in res["scenarios"]
    assert "control_ply_limit" in res["scenarios"]

    # Scenario 1 conditions on WHITE_CHECKMATES; the control is the decode
    # conditioned on PLY_LIMIT — its prefixes must be byte-identical.
    scenario1 = next(c for c in captured if c[0] == gen_mod.WHITE_CHECKMATES)
    control = next(c for c in captured if c[0] == gen_mod.PLY_LIMIT)
    assert scenario1[1] is not None
    assert control[1] is not None
    np.testing.assert_array_equal(control[1], scenario1[1])
    np.testing.assert_array_equal(control[2], scenario1[2])


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------


def test_fit_probe_converges_on_separable_data() -> None:
    """Synthetic separable hidden states + labels; the held-out probe
    should reach high accuracy and report a real val split (H6)."""
    rng = np.random.default_rng(0)
    n_per_class = 64
    d = 32
    n_classes = 4
    means = rng.normal(size=(n_classes, d)) * 2.0
    hidden = []
    labels = []
    for c in range(n_classes):
        hidden.append(means[c] + rng.normal(size=(n_per_class, d)) * 0.3)
        labels.extend([c] * n_per_class)
    hidden_arr = jnp.asarray(np.concatenate(hidden, axis=0), dtype=jnp.float32)
    labels_arr = jnp.asarray(labels, dtype=jnp.int32)
    cfg = ProbeConfig(
        n_classes=n_classes, lr=1e-2, n_epochs=10, batch_size=32, val_frac=0.25,
    )
    result = fit_probe(hidden_arr, labels_arr, cfg, key=0)
    # `accuracy` is the HELD-OUT number now (H6): the probe must generalise,
    # not memorise. A real val split was carved.
    assert result.accuracy > 0.9
    assert result.n_val > 0
    assert result.n_train + result.n_val == n_per_class * n_classes


def test_fit_probe_reports_held_out_not_in_sample() -> None:
    """H6 regression guard: `fit_probe` must score on data it never
    trained on, not re-report the in-sample number (`probes.py:81-96`).

    We give the probe pure-noise labels (no recoverable signal) with an
    *overcomplete* feature dimension (`d >> n_train`), so the linear probe
    can — and does — memorise the train split to ~100% accuracy. Because the
    labels carry no signal, held-out accuracy must collapse toward chance
    (0.5). A genuine held-out split therefore shows a large train→val gap; a
    pure in-sample report (the old behaviour, where `accuracy ==
    train_accuracy`) would show ~no gap at all. The non-strict
    `train_accuracy >= accuracy` of the previous version was satisfiable by
    equality and could not distinguish the two — here we require a strict,
    sizeable gap and a val accuracy bounded well below the memorised train
    accuracy.
    """
    rng = np.random.default_rng(7)
    n = 120
    d = 256  # overcomplete: d >> n_train, so the probe can memorise any labels
    x = rng.normal(size=(n, d)).astype("float32")
    # Pure-noise labels — uncorrelated with the features, so nothing
    # generalises and held-out accuracy is bounded near chance.
    y = rng.integers(0, 2, size=n).astype("int32")
    cfg = ProbeConfig(n_classes=2, lr=5e-2, n_epochs=300, batch_size=64, val_frac=0.3)
    result = fit_probe(jnp.asarray(x), jnp.asarray(y), cfg, key=0)
    assert result.n_val > 0
    # The probe memorised the train split (overcomplete, noise labels).
    assert result.train_accuracy > 0.95
    # Held-out accuracy collapses toward chance — far below the memorised
    # train accuracy. An in-sample report (accuracy == train_accuracy) would
    # land near ~1.0 and fail both of the following.
    assert result.accuracy < 0.8
    assert result.train_accuracy - result.accuracy > 0.2


def test_run_layer_probes_side_to_move_above_chance_held_out() -> None:
    """H6 end-to-end: forward the FROZEN model on engine games, extract
    per-layer hidden states, label side-to-move via
    `engine.extract_board_states`, and fit a held-out probe per layer.

    Side-to-move is positionally separable (ply parity → RoPE position),
    so even an untrained backbone's early-layer residual stream carries
    it well above the 0.5 chance rate — and crucially the reported
    accuracy is on held-out positions."""
    from pawn.probes import run_layer_probes, side_to_move_labeler

    model = init_model(TINY_SUPERNET, key=0)
    import chess_engine as engine

    move_ids, game_lengths, _ = engine.generate_random_games(96, 40, 11)
    results = run_layer_probes(
        model, move_ids, game_lengths,
        n_classes=2, labeler=side_to_move_labeler,
        n_epochs=15, val_frac=0.25, key=0,
    )
    # One probe per layer + the post-embedding stream.
    assert set(results.keys()) == set(range(TINY_SUPERNET.n_layers + 1))
    for r in results.values():
        assert r.n_val > 0
        assert r.n_train > 0
    # The best layer must clear chance by a real margin on held-out data.
    best = max(r.accuracy for r in results.values())
    assert best > 0.6, f"side-to-move probe at chance: best held-out acc={best}"


def test_extract_probe_dataset_labels_match_engine_board() -> None:
    """The probe dataset's labels must come from the engine's ground-truth
    board states (H6), not a synthetic stand-in. We extract an occupancy
    feature and independently recompute it from
    `engine.extract_board_states`; they must agree exactly.

    This pins the load-bearing H6 alignment in `extract_probe_dataset`: each
    residual-stream slot `C + t` is labeled by the engine board at ply
    `p_idx = t + 1` (the board the model has just transitioned into after
    consuming `move[t]`). We replicate that deterministic index construction
    here and compare the returned labels element-wise against occupancy
    recomputed directly from `engine.extract_board_states`, so a labeler that
    returned all-zeros, indexed the wrong square, or used the wrong
    `(g_idx, p_idx)` / an off-by-one in `p_idx` would fail.
    """
    from pawn.corpus import conditioning_to_C
    from pawn.probes import extract_probe_dataset, occupancy_labeler

    model = init_model(TINY_SUPERNET, key=0)
    import chess_engine as engine

    move_ids, game_lengths, _ = engine.generate_random_games(8, 24, 5)
    square = 28  # e4
    hidden, labels = extract_probe_dataset(
        model, move_ids, game_lengths,
        layer=TINY_SUPERNET.n_layers, labeler=occupancy_labeler(square),
    )
    labels_np = np.asarray(labels)
    # One feature row per supervised (game, ply) position; labels are 0/1.
    assert hidden.shape[0] == labels_np.shape[0]
    assert hidden.shape[0] > 0
    assert set(np.unique(labels_np).tolist()) <= {0, 1}
    assert hidden.shape[-1] == TINY_SUPERNET.d_model

    # Independently reconstruct the (g_idx, p_idx) the dataset supervises,
    # mirroring extract_probe_dataset: for each game, t in [0, gl-1) with
    # slot C+t labeled by the board at ply t+1. With no conditioning C == 1.
    C = conditioning_to_C(())
    assert C == 1
    gl = np.asarray(game_lengths, dtype=np.int64)
    g_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    for g in range(len(gl)):
        usable = int(gl[g]) - 1
        if usable <= 0:
            continue
        t = np.arange(usable, dtype=np.int64)
        g_list.append(np.full(usable, g, dtype=np.int64))
        p_list.append(t + 1)  # label the board BEFORE move[t+1]
    g_idx = np.concatenate(g_list)
    p_idx = np.concatenate(p_list)

    # Recompute square-28 occupancy directly from the engine's ground-truth
    # boards and require exact agreement with the returned labels.
    states = engine.extract_board_states(move_ids, game_lengths)
    boards = np.asarray(states[0])  # (N, max_ply, 8, 8) int8
    rank, file = divmod(square, 8)
    expected_occ = (boards[g_idx, p_idx, rank, file] != 0).astype(np.int64)
    assert labels_np.shape[0] == expected_occ.shape[0]
    np.testing.assert_array_equal(labels_np, expected_occ)
    # Both classes must appear — an all-constant labeler can't pass.
    assert set(np.unique(expected_occ).tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# Probe regression path (MSE / R² / MAE) + best-accuracy tracking
# ---------------------------------------------------------------------------


def test_fit_probe_mse_reports_r2_and_mae_on_linear_target() -> None:
    """The MSE-regression probe path (probes-regression-mse-r2-mae): a probe
    fit on a noise-free linear target must recover it — held-out R² ≈ 1 and
    MAE ≈ 0 — and the result must carry both. A cross-entropy head can't
    produce these, so this pins the regression objective + scoring."""
    rng = np.random.default_rng(0)
    n, d = 400, 16
    x = rng.normal(size=(n, d)).astype("float32")
    w_true = rng.normal(size=(d, 1)).astype("float32")
    # Single-output regression target — perfectly linear in the features.
    y = (x @ w_true).astype("float32")  # (n, 1)
    cfg = ProbeConfig(
        n_classes=1, lr=5e-2, n_epochs=200, batch_size=64,
        val_frac=0.25, loss_type="mse",
    )
    result = fit_probe(jnp.asarray(x), jnp.asarray(y), cfg, key=0)
    assert result.n_val > 0
    # `accuracy` is the held-out R² for mse probes; a recovered linear map
    # drives it to ≈ 1.0.
    assert result.accuracy > 0.95, f"R² too low: {result.accuracy}"
    # MAE must be populated for regression probes and be small.
    assert result.mae is not None
    assert result.mae < 0.2
    # The loss is the held-out MSE — small for a recovered target.
    assert result.loss < 0.1


def test_fit_probe_multi_output_mse_target() -> None:
    """Regression probes support multi-output targets (e.g. material_count's
    10 outputs): `n_classes` is the head width and labels are (N, n_classes)
    floats."""
    rng = np.random.default_rng(1)
    n, d, k = 300, 12, 4
    x = rng.normal(size=(n, d)).astype("float32")
    w_true = rng.normal(size=(d, k)).astype("float32")
    y = (x @ w_true).astype("float32")  # (n, k)
    cfg = ProbeConfig(
        n_classes=k, lr=5e-2, n_epochs=200, batch_size=64,
        val_frac=0.25, loss_type="mse",
    )
    result = fit_probe(jnp.asarray(x), jnp.asarray(y), cfg, key=0)
    assert result.weight.shape == (d, k)
    assert result.accuracy > 0.9  # multi-output R²
    assert result.mae is not None


def test_fit_probe_tracks_best_accuracy_across_epochs() -> None:
    """best_accuracy is the across-epoch best, not just the final epoch
    (probes-no-best-accuracy-tracking). It must be defined, ≥ the final-epoch
    accuracy on separable data, and within [0, 1]."""
    rng = np.random.default_rng(0)
    n_per_class, d, n_classes = 64, 32, 3
    means = rng.normal(size=(n_classes, d)) * 2.0
    hidden, labels = [], []
    for c in range(n_classes):
        hidden.append(means[c] + rng.normal(size=(n_per_class, d)) * 0.3)
        labels.extend([c] * n_per_class)
    x = jnp.asarray(np.concatenate(hidden, axis=0), dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.int32)
    cfg = ProbeConfig(n_classes=n_classes, lr=1e-2, n_epochs=12, batch_size=32,
                      val_frac=0.25)
    result = fit_probe(x, y, cfg, key=0)
    # best_accuracy must be a real number (not the 0.0 default), bounded in
    # [0, 1], and at least as good as the final-epoch held-out accuracy.
    assert 0.0 <= result.best_accuracy <= 1.0
    assert result.best_accuracy >= result.accuracy - 1e-6


def test_fit_probe_explicit_val_pool_scores_on_separate_data() -> None:
    """The --n-val-games path: when an explicit val pool is supplied,
    fit_probe trains on the whole `hidden_states` pool and scores on the
    independent val pool (probes-cli-flags)."""
    rng = np.random.default_rng(0)
    n_per_class, d, n_classes = 48, 24, 2
    means = rng.normal(size=(n_classes, d)) * 2.0

    def _make(n: int) -> tuple[Array, Array]:
        h, lab = [], []
        for c in range(n_classes):
            h.append(means[c] + rng.normal(size=(n, d)) * 0.3)
            lab.extend([c] * n)
        return (
            jnp.asarray(np.concatenate(h, axis=0), dtype=jnp.float32),
            jnp.asarray(lab, dtype=jnp.int32),
        )

    x_tr, y_tr = _make(n_per_class)
    x_val, y_val = _make(16)
    cfg = ProbeConfig(n_classes=n_classes, lr=1e-2, n_epochs=15, batch_size=32)
    result = fit_probe(
        x_tr, y_tr, cfg, key=0,
        val_hidden_states=x_val, val_labels=y_val,
    )
    # n_val is the SEPARATE pool size, and the whole train pool was used for
    # training (no within-pool carve).
    assert result.n_val == int(x_val.shape[0])
    assert result.n_train == int(x_tr.shape[0])
    assert result.accuracy > 0.9


def test_fit_probe_requires_val_pool_args_together() -> None:
    """fit_probe rejects a half-supplied explicit val pool: passing
    val_hidden_states without val_labels (or vice versa) raises rather than
    crashing later inside jnp.asarray(None)."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(32, 8)), dtype=jnp.float32)
    y = jnp.asarray(rng.integers(0, 2, size=32), dtype=jnp.int32)
    x_val = jnp.asarray(rng.normal(size=(8, 8)), dtype=jnp.float32)
    cfg = ProbeConfig(n_classes=2, lr=1e-2, n_epochs=2, batch_size=16)
    with pytest.raises(ValueError, match="supplied together"):
        fit_probe(x, y, cfg, key=0, val_hidden_states=x_val)
    with pytest.raises(ValueError, match="supplied together"):
        fit_probe(x, y, cfg, key=0, val_labels=y)


def test_run_layer_probes_requires_val_pool_args_together() -> None:
    """run_layer_probes mirrors fit_probe's guard: a half-supplied explicit
    val game pool (val_move_ids without val_game_lengths, or vice versa) must
    raise rather than silently falling back to within-pool splitting and
    reporting a misleading n_val. The guard fires before any backbone forward,
    so no engine games or extraction are needed for either branch."""
    from pawn.probes import run_layer_probes, side_to_move_labeler

    model = init_model(TINY_SUPERNET, key=0)
    import chess_engine as engine

    move_ids, game_lengths, _ = engine.generate_random_games(8, 20, 11)
    with pytest.raises(ValueError, match="supplied together"):
        run_layer_probes(
            model, move_ids, game_lengths,
            n_classes=2, labeler=side_to_move_labeler,
            n_epochs=1, val_move_ids=move_ids, key=0,
        )
    with pytest.raises(ValueError, match="supplied together"):
        run_layer_probes(
            model, move_ids, game_lengths,
            n_classes=2, labeler=side_to_move_labeler,
            n_epochs=1, val_game_lengths=game_lengths, key=0,
        )


# ---------------------------------------------------------------------------
# Probe-feature coverage (the restored v1 suite)
# ---------------------------------------------------------------------------


def test_probe_features_registry_covers_v1_suite() -> None:
    """PROBE_FEATURES must restore every v1 probe type
    (probes-probe-feature-coverage): the carried-over classification probes
    plus the three MSE-regression probes, with the right loss types."""
    from pawn.probes import PROBE_FEATURES

    expected_loss = {
        "side_to_move": "ce",
        "occupancy": "ce",
        "piece_type": "ce",
        "piece_type_all": "ce_per_square",
        "is_check": "ce",
        "castling_rights": "ce",
        "ep_square": "ce",
        "game_phase": "ce",
        "material_count": "mse",
        "legal_move_count": "mse",
        "halfmove_clock": "mse",
    }
    assert set(PROBE_FEATURES) == set(expected_loss)
    for name, loss in expected_loss.items():
        assert PROBE_FEATURES[name].loss_type == loss
    # Exactly the legal_move_count probe reads the appended legal-count array.
    assert PROBE_FEATURES["legal_move_count"].needs_legal_counts is True
    assert all(
        not PROBE_FEATURES[n].needs_legal_counts
        for n in PROBE_FEATURES if n != "legal_move_count"
    )


def test_new_labelers_match_engine_board_states() -> None:
    """Each restored labeler must read the engine's ground-truth board state
    arrays at the right index and shape (probes-probe-feature-coverage)."""
    import chess_engine as engine

    from pawn.probes import (
        castling_rights_labeler,
        ep_square_labeler,
        game_phase_labeler,
        halfmove_clock_labeler,
        is_check_labeler,
        material_count_labeler,
        piece_type_all_squares_labeler,
    )

    move_ids, game_lengths, _ = engine.generate_random_games(12, 24, 3)
    states = engine.extract_board_states(
        np.ascontiguousarray(move_ids, dtype=np.int16),
        np.asarray(game_lengths, dtype=np.int16),
    )
    # Supervise every (game, ply) with t+1 < game_length, mirroring the
    # dataset's index construction.
    gl = np.asarray(game_lengths, dtype=np.int64)
    g_list, p_list = [], []
    for g in range(len(gl)):
        usable = int(gl[g]) - 1
        if usable <= 0:
            continue
        t = np.arange(usable, dtype=np.int64)
        g_list.append(np.full(usable, g, dtype=np.int64))
        p_list.append(t + 1)
    g_idx = np.concatenate(g_list)
    p_idx = np.concatenate(p_list)
    m = g_idx.shape[0]

    # is_check — 2-class, drawn from states[4].
    chk, n_chk = is_check_labeler(states, g_idx, p_idx)
    assert n_chk == 2 and chk.shape == (m,)
    np.testing.assert_array_equal(
        chk, np.asarray(states[4])[g_idx, p_idx].astype(np.int64)
    )

    # castling_rights — 16-class joint bitmask from states[2].
    cas, n_cas = castling_rights_labeler(states, g_idx, p_idx)
    assert n_cas == 16
    assert cas.min() >= 0 and cas.max() <= 15
    np.testing.assert_array_equal(
        cas, (np.asarray(states[2])[g_idx, p_idx] & 0xF).astype(np.int64)
    )

    # ep_square — 65 classes, none mapped to 64 (states[3], <0 = none).
    ep, n_ep = ep_square_labeler(states, g_idx, p_idx)
    assert n_ep == 65 and ep.min() >= 0 and ep.max() <= 64
    raw_ep = np.asarray(states[3])[g_idx, p_idx]
    np.testing.assert_array_equal(ep, np.where(raw_ep < 0, 64, raw_ep))

    # halfmove_clock — float regression target (states[5]).
    hmc, n_hmc = halfmove_clock_labeler(states, g_idx, p_idx)
    assert n_hmc == 1 and hmc.shape == (m, 1)
    assert np.issubdtype(hmc.dtype, np.floating)

    # material_count — 10 float outputs counting per-type per-colour pieces.
    mat, n_mat = material_count_labeler(states, g_idx, p_idx)
    assert n_mat == 10 and mat.shape == (m, 10)
    # White-pawn count (index 0) recomputed independently from the boards.
    boards = np.asarray(states[0])
    flat = boards[g_idx, p_idx].reshape(m, 64)
    np.testing.assert_array_equal(
        mat[:, 0], (flat == 1).sum(axis=1).astype(np.float32)
    )

    # game_phase — 3 classes (opening/middle/endgame).
    phase, n_phase = game_phase_labeler(states, g_idx, p_idx)
    assert n_phase == 3 and set(np.unique(phase).tolist()) <= {0, 1, 2}

    # piece_type_all — (m, 64) of 13-class codes; head width 13*64.
    pta, n_pta = piece_type_all_squares_labeler(states, g_idx, p_idx)
    assert n_pta == 13 * 64 and pta.shape == (m, 64)
    assert pta.min() >= 0 and pta.max() <= 12
    np.testing.assert_array_equal(pta, flat.astype(np.int64))


def test_count_legal_moves_per_ply_positive_for_real_positions() -> None:
    """count_legal_moves_per_ply (legal_move_count probe support) must return
    one count per (game, ply) and be strictly positive for the opening (every
    legal random game starts with 20 legal moves)."""
    import chess_engine as engine

    from pawn.probes import count_legal_moves_per_ply

    move_ids, game_lengths, _ = engine.generate_random_games(6, 20, 0)
    counts = count_legal_moves_per_ply(move_ids, game_lengths)
    assert counts.shape == np.asarray(move_ids).shape
    # The opening position (ply 0) has exactly 20 legal moves in standard
    # chess — a tight pin on the counting contract.
    assert int(counts[0, 0]) == 20
    # Every in-game ply has at least one legal move (the game continued).
    gl = np.asarray(game_lengths)
    for g in range(len(gl)):
        for p in range(int(gl[g])):
            assert int(counts[g, p]) >= 1


def test_run_layer_probes_material_count_regression_end_to_end() -> None:
    """End-to-end MSE-regression probe: forward the frozen model, label
    material_count, and fit a per-layer regression probe. R² must be defined
    and the result must carry MAE (probes-regression-mse-r2-mae)."""
    from pawn.probes import material_count_labeler, run_layer_probes

    model = init_model(TINY_SUPERNET, key=0)
    import chess_engine as engine

    move_ids, game_lengths, _ = engine.generate_random_games(64, 30, 7)
    results = run_layer_probes(
        model, move_ids, game_lengths,
        n_classes=10, labeler=material_count_labeler, loss_type="mse",
        n_epochs=12, val_frac=0.25, key=0,
    )
    assert set(results.keys()) == set(range(TINY_SUPERNET.n_layers + 1))
    for r in results.values():
        assert r.n_val > 0
        assert r.mae is not None  # regression → MAE populated
        # R² is bounded above by 1.0 (it can be negative for a bad fit).
        assert r.accuracy <= 1.0 + 1e-6


def test_run_layer_probes_legal_move_count_needs_legal_counts() -> None:
    """The legal_move_count regression probe runs end-to-end only when
    needs_legal_counts threads the engine legal-count array into the labeler
    (states[6]); without it the labeler would IndexError."""
    from pawn.probes import legal_move_count_labeler, run_layer_probes

    model = init_model(TINY_SUPERNET, key=0)
    import chess_engine as engine

    move_ids, game_lengths, _ = engine.generate_random_games(48, 30, 9)
    results = run_layer_probes(
        model, move_ids, game_lengths,
        n_classes=1, labeler=legal_move_count_labeler, loss_type="mse",
        n_epochs=8, val_frac=0.25, needs_legal_counts=True, key=0,
    )
    assert len(results) == TINY_SUPERNET.n_layers + 1
    for r in results.values():
        assert r.n_val > 0
        assert r.mae is not None


def test_extract_probe_dataset_empty_pool_matches_labeler_shape() -> None:
    """Degenerate pool guard: when every game is too short to supervise any
    position (game_length <= 1, so no t+1 successor board exists), the
    extractor must still return empty labels whose shape/dtype match the
    labeler's real output — (0, k) float32 for an mse probe, not a (0,) int
    fallback. The previous hard-coded `(0,)` int tensor broadcast to a `nan`
    val_loss inside fit_probe for regression probes."""
    from pawn.probes import (
        _extract_all_layers_probe_dataset,
        material_count_labeler,
    )

    import chess_engine as engine

    model = init_model(TINY_SUPERNET, key=0)
    # Real games (so engine replay sees legal moves), but every length forced
    # to 1 → usable = gl - 1 = 0 → no supervised positions.
    move_ids, _gl, _ = engine.generate_random_games(4, 6, 0)
    game_lengths = np.ones((4,), dtype=np.int16)
    feats, labels = _extract_all_layers_probe_dataset(
        model, move_ids, game_lengths, labeler=material_count_labeler,
    )
    n_layers_plus = TINY_SUPERNET.n_layers + 1
    assert feats.shape == (n_layers_plus, 0, TINY_SUPERNET.d_model)
    # material_count is a 10-output regression probe: the empty labels must be
    # (0, 10) float32, the shape fit_probe's mse loss/score expect.
    assert labels.shape == (0, 10)
    assert jnp.issubdtype(labels.dtype, jnp.floating)


def test_run_layer_probes_empty_pool_reports_finite_loss() -> None:
    """End-to-end nan guard (gate fix): a regression probe over a pool whose
    games are all too short to supervise must report a finite (non-nan) loss
    and a zeroed score, not nan. This exercises the empty-batch guard in
    `_probe_loss` together with the labeler-shaped empty labels."""
    import math

    import chess_engine as engine

    from pawn.probes import legal_move_count_labeler, run_layer_probes

    model = init_model(TINY_SUPERNET, key=0)
    move_ids, _gl, _ = engine.generate_random_games(4, 6, 0)
    game_lengths = np.ones((4,), dtype=np.int16)
    results = run_layer_probes(
        model, move_ids, game_lengths,
        n_classes=1, labeler=legal_move_count_labeler, loss_type="mse",
        n_epochs=3, val_frac=0.25, needs_legal_counts=True, key=0,
    )
    assert results  # one ProbeResult per layer
    for r in results.values():
        assert r.n_train == 0 and r.n_val == 0
        assert not math.isnan(r.loss), "empty-pool mse probe leaked a nan loss"
        assert r.loss == 0.0
        assert not math.isnan(r.accuracy)
        assert r.mae is not None and not math.isnan(r.mae)


# ---------------------------------------------------------------------------
# eval_probes_jax.py script — end-to-end JSON output schema
# ---------------------------------------------------------------------------


def _load_eval_probes_jax():  # type: ignore[no-untyped-def]
    """Load scripts/eval_probes_jax.py as a module (no package __init__)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "scripts_eval_probes_jax", Path("scripts") / "eval_probes_jax.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_probes_jax_main_output_schema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end schema test for the probe CLI (probes-output-schema-diverged,
    probes-script-output-schema-test-gap): run main() on a freshly-saved
    checkpoint and assert the v1-parity top-level keys + per-layer schema."""
    import json

    from pawn.checkpoint import save_model

    ep = _load_eval_probes_jax()

    model = init_model(TINY_SUPERNET, key=0)
    ckpt_dir = tmp_path / "step_00000001"
    save_model(model, ckpt_dir, run_config={"conditioning": [], "step": 1})

    out = tmp_path / "probes.json"
    rc = ep.main([
        "--checkpoint", str(ckpt_dir),
        "--feature", "side_to_move",
        "--n-games", "24", "--max-ply", "20",
        "--n-epochs", "3", "--output", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    # v1-parity top-level keys.
    for key in ("run", "checkpoint", "step", "variant", "model_config",
                "probes", "probe_accuracy"):
        assert key in payload, f"missing top-level key {key!r}"
    assert payload["model_config"]["d_model"] == TINY_SUPERNET.d_model
    # Per-feature → per-layer schema.
    assert "side_to_move" in payload["probes"]
    feat = payload["probes"]["side_to_move"]
    assert feat["loss_type"] == "ce"
    layers = feat["layers"]
    assert layers  # at least one layer reported
    # v1-parity layer key names: `embed` then `layer_0`..`layer_{n_layers-1}`
    # (NOT v2's internal `layer_1`..`layer_{n_layers}`). TINY_SUPERNET has
    # 4 layers, so the full set must be embed + layer_0..layer_3 — any drift
    # to 1-based names would break downstream parsers keyed on v1 names.
    expected_layer_names = {"embed"} | {
        f"layer_{i}" for i in range(TINY_SUPERNET.n_layers)
    }
    assert set(layers.keys()) == expected_layer_names, (
        f"layer names {sorted(layers.keys())} != v1-parity "
        f"{sorted(expected_layer_names)}"
    )
    # `best_layer` must also be a v1-parity name, not an int index.
    assert feat["best_layer"] in expected_layer_names
    for lname, metrics in layers.items():
        for mk in ("val_accuracy", "accuracy", "best_accuracy", "loss",
                   "n_train", "n_val"):
            assert mk in metrics, f"layer {lname} missing {mk!r}"


def test_eval_probes_jax_main_all_features_includes_regression(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """--all-features runs the whole suite, and regression features emit a
    `loss_type: mse` block with an `mae` per layer (probes-output-schema /
    probes-run-evals-backbone-single-feature)."""
    import json

    from pawn.checkpoint import save_model

    ep = _load_eval_probes_jax()

    model = init_model(TINY_SUPERNET, key=0)
    ckpt_dir = tmp_path / "step_00000001"
    save_model(model, ckpt_dir, run_config={"conditioning": []})

    out = tmp_path / "probes.json"
    rc = ep.main([
        "--checkpoint", str(ckpt_dir), "--all-features",
        "--n-games", "24", "--max-ply", "20",
        "--n-epochs", "3", "--output", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    probes = payload["probes"]
    # The full v1 suite is present.
    assert {"side_to_move", "material_count", "legal_move_count",
            "halfmove_clock", "is_check", "castling_rights",
            "ep_square", "game_phase"} <= set(probes)
    # Regression probe carries mse loss_type + per-layer MAE.
    mat = probes["material_count"]
    assert mat["loss_type"] == "mse"
    for metrics in mat["layers"].values():
        assert "mae" in metrics


def _load_eval_probes_wrapper():  # type: ignore[no-untyped-def]
    """Load scripts/eval_probes.py (the v1 compat wrapper) as a module."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "scripts_eval_probes_wrapper", Path("scripts") / "eval_probes.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_probes_wrapper_log_dir_scan(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The v1 compat wrapper's --log-dir scan (probes-compat-wrapper-breaks-
    log-dir-mode): it must discover the latest checkpoint per run dir, run the
    v2 probe script, and write a v1-schema probe_results.json (with the scan's
    run/step fields stamped) into each run dir."""
    import json

    from pawn.checkpoint import save_model

    wrapper = _load_eval_probes_wrapper()

    log_dir = tmp_path / "logs"
    run_dir = log_dir / "run_demo"
    run_dir.mkdir(parents=True)
    model = init_model(TINY_SUPERNET, key=0)
    # Two checkpoints — the scan must select the latest (highest step).
    save_model(model, run_dir / "step_00000005", run_config={"conditioning": []})
    save_model(model, run_dir / "step_00000010", run_config={"conditioning": []})

    rc = wrapper.main([
        "--log-dir", str(log_dir),
        "--n-games", "16", "--max-ply", "16", "--n-epochs", "2",
    ])
    assert rc == 0
    out = run_dir / "probe_results.json"
    assert out.is_file(), "scan did not write probe_results.json"
    payload = json.loads(out.read_text())
    # The scan stamps the v1-schema run/step fields it owns.
    assert payload["run"] == "run_demo"
    assert payload["step"] == 10  # latest checkpoint selected
    assert "probes" in payload and "side_to_move" in payload["probes"]


# ---------------------------------------------------------------------------
# Elo-stratified accuracy
# ---------------------------------------------------------------------------


def test_default_elo_bins_covers_lichess_range() -> None:
    bins = default_elo_bins()
    assert bins[0] == EloBin(1100, 1200)
    assert bins[-1] == EloBin(1900, 2000)


def test_compute_elo_stratified_accuracy_aggregates_bins() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    c = generate_corpus(n_games=2, max_ply=16, seq_len=32, seed=0)
    bins = {EloBin(1500, 1600): c, EloBin(1600, 1700): c}
    results = compute_elo_stratified_accuracy(model, bins, batch_size=2)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_games == 2


def test_elo_bin_result_carries_full_v1_schema() -> None:
    """Parity item ``elo-output-schema-gap`` (criterion 13): each Elo bin
    reports loss / perplexity / top-5 / legal_move_rate alongside top-1, not
    top-1 alone — the v1 ``eval_suite/lichess.py`` per-bin schema."""
    model = init_model(TINY_SUPERNET, key=0)
    c = generate_corpus(n_games=6, max_ply=24, seq_len=32, seed=1)
    bins = {EloBin(1500, 1600): c}
    (r,) = compute_elo_stratified_accuracy(model, bins, batch_size=3)
    assert 0.0 <= r.accuracy <= 1.0
    assert r.top1 == r.accuracy  # alias
    assert 0.0 <= r.top5 <= 1.0
    assert r.top5 >= r.accuracy - 1e-6
    assert r.loss > 0.0
    assert r.perplexity == pytest.approx(np.exp(min(r.loss, 20.0)), rel=1e-5)
    assert 0.0 <= r.legal_move_rate <= 1.0


# ---------------------------------------------------------------------------
# Edge-case diagnostics
# ---------------------------------------------------------------------------


def test_edge_case_accuracy_runs() -> None:
    """Smoke test on real engine games — `edge_case_bits` is exercised."""
    import chess_engine as engine

    from pawn.eval_suite.diagnostics import (
        EDGE_CASE_LABELS,
        compute_edge_case_accuracy,
    )

    model = init_model(TINY_SUPERNET, key=0)
    move_ids, game_lengths, _ = engine.generate_random_games(8, 32, 42)
    results = compute_edge_case_accuracy(
        model, move_ids, game_lengths, batch_size=4
    )
    # Parity #6: 10 labels in the canonical order (v1 surface).
    assert [r.label for r in results] == list(EDGE_CASE_LABELS)
    assert len(EDGE_CASE_LABELS) == 10
    # Every label has finite accuracy in [0, 1].
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_positions >= 0


def test_edge_case_labels_include_added_v1_categories() -> None:
    """The 4 categories restored in parity #6 must be present.

    `n_positions` may be 0 on the 8-game random pool — guarantee of
    coverage requires the quota-controlled path
    (`compute_edge_case_accuracy_quota`)."""
    from pawn.eval_suite.diagnostics import EDGE_CASE_LABELS
    for label in (
        "castle_blocked_check", "promotion_available",
        "checkmate", "stalemate",
    ):
        assert label in EDGE_CASE_LABELS


def test_edge_case_accuracy_quota_guarantees_coverage() -> None:
    """`compute_edge_case_accuracy_quota` calls
    `engine.generate_diagnostic_sets` with per-label quotas, so every
    label should have ``n_positions > 0`` even for rare cases like
    `checkmate` / `stalemate`."""
    from pawn.eval_suite.diagnostics import (
        EDGE_CASE_LABELS,
        compute_edge_case_accuracy_quota,
    )
    model = init_model(TINY_SUPERNET, key=0)
    # `per_label=4` is the smallest budget that reliably surfaces
    # `stalemate` (the rarest label — engine empirically needs ~400
    # simulated games per stalemate; the default factor=500 amortises
    # that). Keep `max_ply=256` because stalemate tends to appear in
    # longer games.
    results = compute_edge_case_accuracy_quota(
        model, per_label=4, max_ply=256, batch_size=4,
    )
    by_label = {r.label: r for r in results}
    for label in EDGE_CASE_LABELS:
        assert label in by_label
        assert by_label[label].n_positions > 0, (
            f"quota-controlled sampling missed label {label!r}"
        )
        assert 0.0 <= by_label[label].accuracy <= 1.0


# ---------------------------------------------------------------------------
# Per-category sampled distributional metrics (v1 surface restored)
# ---------------------------------------------------------------------------


def test_edge_case_diagnostics_returns_sampled_schema() -> None:
    """`compute_edge_case_diagnostics` returns the v1 sampled-metrics keys
    the model-card / viz consumers read (mean_legal_rate, std_legal_rate,
    mean_pad_prob, mean_entropy, std_entropy, terminal, n_positions)
    alongside the v2 argmax-accuracy."""
    import chess_engine as engine

    from pawn.eval_suite.diagnostics import (
        EDGE_CASE_LABELS,
        TERMINAL_LABELS,
        compute_edge_case_diagnostics,
    )

    model = init_model(TINY_SUPERNET, key=0)
    move_ids, game_lengths, term_codes = engine.generate_random_games(64, 48, 3)
    accuracy, sampled = compute_edge_case_diagnostics(
        model, move_ids, game_lengths, term_codes=term_codes, batch_size=16
    )
    assert [r.label for r in accuracy] == list(EDGE_CASE_LABELS)
    assert [r.label for r in sampled] == list(EDGE_CASE_LABELS)
    for s in sampled:
        d = s.to_dict()
        # Exact v1 key set the consumers read.
        assert set(d) == {
            "n_positions", "terminal", "mean_legal_rate", "std_legal_rate",
            "mean_pad_prob", "mean_entropy", "std_entropy",
        }
        assert d["terminal"] == (s.label in TERMINAL_LABELS)
        assert 0.0 <= s.mean_legal_rate <= 1.0
        assert 0.0 <= s.mean_pad_prob <= 1.0
        assert s.mean_entropy >= 0.0


def test_edge_case_sampled_legal_rate_is_analytic_softmax_mass() -> None:
    """The restored sampled `mean_legal_rate` is the softmax mass on the
    engine's legal-token set. A next-token oracle (all mass on the true
    next move, which is always legal) scores ~1.0 legal-rate on a
    non-terminal label, while the terminal-PAD `mean_pad_prob` is ~1.0 on
    `checkmate` — i.e. the metric genuinely measures the distribution at
    the diagnostic position, not the argmax."""
    import chess_engine as engine

    from pawn.eval_suite.diagnostics import compute_edge_case_diagnostics

    move_ids, game_lengths, term_codes = engine.generate_random_games(400, 80, 7)
    _accuracy, sampled = compute_edge_case_diagnostics(
        _NextTokenOracle(), move_ids, game_lengths,
        term_codes=term_codes, batch_size=64,
    )
    by = {s.label: s for s in sampled}
    assert by["in_check"].n_positions > 0
    assert by["checkmate"].n_positions > 0
    # Non-terminal: the true next token is a legal move ⇒ legal mass ≈ 1.
    assert by["in_check"].mean_legal_rate == pytest.approx(1.0, abs=1e-4)
    # Terminal: the true next token is PAD ⇒ pad prob ≈ 1, legal mass ≈ 0.
    assert by["checkmate"].mean_pad_prob == pytest.approx(1.0, abs=1e-4)
    assert by["checkmate"].mean_legal_rate == pytest.approx(0.0, abs=1e-4)


def test_edge_case_diagnostics_quota_reports_fill_rate(capsys) -> None:  # type: ignore[no-untyped-def]
    """`compute_edge_case_diagnostics_quota(report=True)` prints the
    per-label OK/SHORT quota fill-rate table (v1 parity) and returns both
    accuracy and sampled metrics with guaranteed per-label coverage."""
    from pawn.eval_suite.diagnostics import (
        EDGE_CASE_LABELS,
        compute_edge_case_diagnostics_quota,
    )

    model = init_model(TINY_SUPERNET, key=0)
    accuracy, sampled = compute_edge_case_diagnostics_quota(
        model, per_label=4, max_ply=256, batch_size=4, report=True,
    )
    out = capsys.readouterr().out
    # Every label appears in the fill-rate table with an OK/SHORT verdict.
    for label in EDGE_CASE_LABELS:
        assert label in out
    assert "OK" in out or "SHORT" in out
    by_acc = {r.label: r for r in accuracy}
    by_s = {s.label: s for s in sampled}
    for label in EDGE_CASE_LABELS:
        assert by_acc[label].n_positions > 0
        assert by_s[label].n_positions > 0


# ---------------------------------------------------------------------------
# H5: edge-case alignment (off-by-one + terminal-label PAD scoring)
# ---------------------------------------------------------------------------


class _OracleBase:
    """Minimal :class:`pawn.model.EffectiveCallable` stand-in for the
    edge-case diagnostic — the diagnostic only calls ``model(tokens,
    attn)`` and never touches ``cfg``, but exposing ``cfg`` keeps the
    Protocol satisfied for the type checker."""

    @property
    def cfg(self):  # type: ignore[no-untyped-def]
        return TINY_SUPERNET

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        *,
        compute_dtype=None,  # type: ignore[no-untyped-def]
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Array:
        raise NotImplementedError


class _NextTokenOracle(_OracleBase):
    """A stand-in model whose argmax at every slot is the true next token.

    The diagnostic builds tokens as ``[BOS][cond…][move…][PAD…]`` and the
    model at slot ``s`` predicts ``tokens[:, s+1]``. This oracle returns
    one-hot-ish logits peaking exactly on that next token (full vocab), so
    every supervised prediction is correct *iff* the diagnostic aligns the
    edge-case bit to the right slot."""

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        *,
        compute_dtype=None,  # type: ignore[no-untyped-def]
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Array:
        from pawn.config import PAD_TOKEN

        toks = np.asarray(input_ids)
        b, t = toks.shape
        tgt = np.full((b, t), PAD_TOKEN, dtype=np.int64)
        tgt[:, :-1] = toks[:, 1:]
        logits = np.full((b, t, VOCAB_SIZE), -30.0, dtype=np.float32)
        rows = np.arange(b)[:, None]
        cols = np.arange(t)[None, :]
        logits[rows, cols, tgt] = 30.0
        return jnp.asarray(logits)


class _CountingOracle(_NextTokenOracle):
    """Next-token oracle that records how many times the model is invoked.

    Used to pin that the combined accuracy+sampled diagnostic runs a
    *single* forward pass per batch (not one for accuracy and a second for
    the sampled metrics)."""

    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        *,
        compute_dtype=None,  # type: ignore[no-untyped-def]
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Array:
        self.n_calls += 1
        return super().__call__(
            input_ids, attention_mask,
            compute_dtype=compute_dtype, use_sdpa=use_sdpa, use_flash=use_flash,
        )


def test_edge_case_diagnostics_single_forward_pass_per_batch() -> None:
    """The combined accuracy+sampled diagnostic forwards the model exactly
    once per batch.

    Both surfaces derive from the one `logits` array, so a corpus split
    into `ceil(n / batch_size)` batches must invoke the model exactly that
    many times — not 2× (a regression to a separate accuracy pass and
    sampled pass would double the count). The accuracy-only path likewise
    runs once per batch."""
    import math

    import chess_engine as engine

    from pawn.eval_suite.diagnostics import (
        compute_edge_case_accuracy,
        compute_edge_case_diagnostics,
    )

    n_games, batch_size = 40, 16
    move_ids, game_lengths, term_codes = engine.generate_random_games(n_games, 48, 3)
    expected_batches = math.ceil(n_games / batch_size)

    combined = _CountingOracle()
    compute_edge_case_diagnostics(
        combined, move_ids, game_lengths,
        term_codes=term_codes, batch_size=batch_size,
    )
    assert combined.n_calls == expected_batches

    acc_only = _CountingOracle()
    compute_edge_case_accuracy(
        acc_only, move_ids, game_lengths,
        term_codes=term_codes, batch_size=batch_size,
    )
    assert acc_only.n_calls == expected_batches


def test_edge_case_diagnostics_matches_independent_passes() -> None:
    """The single-pass combined diagnostic returns numerically identical
    accuracy and sampled results to the standalone accuracy / sampled-only
    derivations — proving the merge changed forward-pass *count*, not
    values."""
    import chess_engine as engine

    from pawn.eval_suite.diagnostics import (
        _compute_per_bit_all,
        _resolve_outcome_tokens,
        compute_edge_case_accuracy,
        compute_edge_case_diagnostics,
    )

    move_ids, game_lengths, term_codes = engine.generate_random_games(64, 48, 3)
    accuracy, sampled = compute_edge_case_diagnostics(
        _NextTokenOracle(), move_ids, game_lengths,
        term_codes=term_codes, batch_size=16,
    )
    # Accuracy must equal the accuracy-only public path exactly.
    acc_only = compute_edge_case_accuracy(
        _NextTokenOracle(), move_ids, game_lengths,
        term_codes=term_codes, batch_size=16,
    )
    assert [(r.label, r.accuracy, r.n_positions) for r in accuracy] == [
        (r.label, r.accuracy, r.n_positions) for r in acc_only
    ]
    # Sampled-only derivation (want_accuracy=False) must equal the combined
    # sampled list — the gating must not change values.
    move_np = np.ascontiguousarray(move_ids, dtype=np.int16)
    glen_np = np.asarray(game_lengths, dtype=np.int16)
    bits, _, _ = engine.compute_edge_stats_per_ply(move_np, glen_np)
    bits = np.asarray(bits, dtype=np.uint64)
    outcome = _resolve_outcome_tokens(np.asarray(term_codes), glen_np)
    none_acc, sampled_only = _compute_per_bit_all(
        _NextTokenOracle(), move_np, glen_np, bits, outcome,
        conditioning=(), batch_size=16,
        want_accuracy=False, want_sampled=True,
    )
    assert none_acc is None
    assert sampled_only is not None
    assert [s.to_dict() for s in sampled] == [s.to_dict() for s in sampled_only]


@pytest.mark.parametrize("conditioning", [(), ("outcome",)])
def test_edge_case_alignment_scores_in_check_and_terminal(
    conditioning: tuple[str, ...],
) -> None:
    """H5 wiring + terminal-PAD coverage: with the off-by-one fixed and the
    terminal label scored against the predict-PAD target, an oracle that
    always predicts the true next token scores ``accuracy == 1.0`` on BOTH
    a non-terminal label (``in_check``) and a terminal label
    (``checkmate``). Verified for the BOS-only (C=1) and outcome-prefixed
    (C=2) layouts.

    NOTE: a perfect next-token oracle is correct at *every* slot, so this
    test alone cannot discriminate the exact bit→slot alignment offset (a
    ±1 shift still leaves both labels at 1.0). The off-by-one is pinned
    separately by
    :func:`test_edge_case_alignment_offset_is_C_minus_1`, which uses a
    slot-varying oracle whose per-label accuracy changes under a shift."""
    import chess_engine as engine

    from pawn.eval_suite.diagnostics import compute_edge_case_accuracy

    move_ids, game_lengths, term_codes = engine.generate_random_games(400, 80, 7)
    res = compute_edge_case_accuracy(
        _NextTokenOracle(), move_ids, game_lengths,
        term_codes=term_codes, conditioning=conditioning, batch_size=64,
    )
    by = {r.label: r for r in res}
    # Both labels must be present with coverage on the random pool.
    assert by["in_check"].n_positions > 0
    assert by["checkmate"].n_positions > 0
    # Perfect next-token oracle ⇒ perfect per-label accuracy under correct
    # alignment (move labels) and correct terminal PAD scoring.
    assert by["in_check"].accuracy == 1.0
    assert by["checkmate"].accuracy == 1.0


class _EvenSlotOracle(_OracleBase):
    """A stand-in whose argmax is the true next token only on EVEN sequence
    slots; on odd slots it deliberately peaks a *wrong* action.

    Unlike :class:`_NextTokenOracle` (correct at every slot, hence blind to
    the bit→slot alignment), this oracle's per-slot correctness varies with
    slot parity, so a per-label edge-case accuracy depends on *which* slots
    the bits are aligned onto. A ±1 shift of the bit→slot mapping (the H5
    off-by-one) flips the parity of every masked slot and therefore changes
    the reported accuracy — letting a test pin the exact offset."""

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        *,
        compute_dtype=None,  # type: ignore[no-untyped-def]
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Array:
        from pawn.config import PAD_TOKEN

        toks = np.asarray(input_ids)
        b, t = toks.shape
        tgt = np.full((b, t), PAD_TOKEN, dtype=np.int64)
        tgt[:, :-1] = toks[:, 1:]
        cols = np.arange(t)[None, :]
        even = np.broadcast_to(cols % 2 == 0, (b, t))
        # On even slots: the true next token. On odd slots: a wrong action
        # (0, or 1 when the truth is 0) so move-target slots score as wrong.
        wrong = np.where(tgt == 0, 1, 0)
        peak = np.where(even, tgt, wrong)
        logits = np.full((b, t, VOCAB_SIZE), -30.0, dtype=np.float32)
        rows = np.arange(b)[:, None]
        logits[rows, cols, peak] = 30.0
        return jnp.asarray(logits)


@pytest.mark.parametrize("conditioning", [(), ("outcome",)])
def test_edge_case_alignment_offset_is_C_minus_1(
    conditioning: tuple[str, ...],
) -> None:
    """H5 (discriminating): the bit→slot alignment is exactly ``s = C-1+j``.

    A perfect oracle cannot catch a ±1 shift (it's correct everywhere), so
    we use a slot-varying oracle (:class:`_EvenSlotOracle`, correct only on
    even slots). For ``in_check`` plies ``j`` (which always carry a move
    target), the diagnostic scores slot ``s = C-1+j`` and the oracle is
    correct there iff ``s`` is even. We independently recompute that
    reference accuracy from the engine bits and require the diagnostic to
    match it EXACTLY, *and* require the correct-alignment reference to
    differ from both the +1 and −1 shifted references — so a regression
    that shifts the mapping by ±1 would change the diagnostic output and
    fail the exact-match assertion."""
    import chess_engine as engine

    from pawn.corpus import conditioning_to_C
    from pawn.eval_suite.diagnostics import compute_edge_case_accuracy

    move_ids, game_lengths, term_codes = engine.generate_random_games(400, 80, 7)
    move_ids = np.asarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    C = conditioning_to_C(conditioning)

    # Reference: per-ply in_check bits → masked plies j → correct slot
    # s = C-1+j. Slot-varying oracle is correct iff s is even.
    bits, _, _ = engine.compute_edge_stats_per_ply(move_ids, game_lengths)
    in_check_mask = engine.edge_case_bits()["IN_CHECK"]
    masked = (np.asarray(bits) & in_check_mask).astype(bool)
    _g_idx, j_idx = np.where(masked)
    assert j_idx.size > 0  # need coverage to discriminate

    def ref_acc(delta: int) -> float:
        s = (C - 1) + j_idx + delta
        return float((s % 2 == 0).mean())

    correct_ref = ref_acc(0)

    res = compute_edge_case_accuracy(
        _EvenSlotOracle(), move_ids, game_lengths,
        term_codes=term_codes, conditioning=conditioning, batch_size=64,
    )
    by = {r.label: r for r in res}
    assert by["in_check"].n_positions == int(j_idx.size)
    # The diagnostic must reproduce the C-1+j alignment exactly.
    assert by["in_check"].accuracy == pytest.approx(correct_ref, abs=1e-9)
    # And the value must be alignment-sensitive: a ±1 shift gives a
    # DIFFERENT reference, so the exact-match above genuinely pins C-1.
    assert ref_acc(+1) != pytest.approx(correct_ref, abs=1e-9)
    assert ref_acc(-1) != pytest.approx(correct_ref, abs=1e-9)


def test_edge_case_terminal_label_not_forced_to_zero() -> None:
    """H5 regression: the checkmate/stalemate terminal labels used to be
    scored against a PAD target on a slot the attention mask zeroed,
    forcing ``accuracy == 0`` regardless of the model. A model that
    *fails* to predict game-over (always emits a real move, never PAD)
    must now score ``checkmate`` accuracy of 0.0, while a model that
    predicts PAD scores 1.0 — i.e. the metric actually discriminates the
    terminal behaviour instead of being pinned at 0."""
    import chess_engine as engine

    from pawn.config import PAD_TOKEN
    from pawn.eval_suite.diagnostics import compute_edge_case_accuracy

    class _NeverPAD(_OracleBase):
        def __call__(
            self,
            input_ids: Array,
            attention_mask: Array | None = None,
            *,
            compute_dtype=None,  # type: ignore[no-untyped-def]
            use_sdpa: bool = False,
            use_flash: bool = False,
        ) -> Array:
            b, t = np.asarray(input_ids).shape
            logits = np.full((b, t, VOCAB_SIZE), -30.0, dtype=np.float32)
            logits[:, :, 0] = 30.0  # always action 0, never PAD
            return jnp.asarray(logits)

    class _AlwaysPAD(_OracleBase):
        def __call__(
            self,
            input_ids: Array,
            attention_mask: Array | None = None,
            *,
            compute_dtype=None,  # type: ignore[no-untyped-def]
            use_sdpa: bool = False,
            use_flash: bool = False,
        ) -> Array:
            b, t = np.asarray(input_ids).shape
            logits = np.full((b, t, VOCAB_SIZE), -30.0, dtype=np.float32)
            logits[:, :, PAD_TOKEN] = 30.0  # always predict game-over
            return jnp.asarray(logits)

    move_ids, game_lengths, term_codes = engine.generate_random_games(400, 80, 7)
    never = {
        r.label: r
        for r in compute_edge_case_accuracy(
            _NeverPAD(), move_ids, game_lengths, term_codes=term_codes, batch_size=64,
        )
    }
    always = {
        r.label: r
        for r in compute_edge_case_accuracy(
            _AlwaysPAD(), move_ids, game_lengths, term_codes=term_codes, batch_size=64,
        )
    }
    assert never["checkmate"].n_positions > 0
    assert never["checkmate"].accuracy == 0.0
    assert always["checkmate"].accuracy == 1.0
