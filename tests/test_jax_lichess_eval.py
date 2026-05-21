"""Tests for ``pawn.lichess_eval`` — Lichess Elo-stratified eval (chunk 4.4).

Most of these tests construct a ``LichessCorpus`` directly (skipping
the Rust PGN parse) so they're hermetic + don't depend on a real PGN
fixture. The full parse-from-PGN path is covered by a small inline
PGN string fed to the engine + tmp-path cache verification.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from pawn.config import PAD_TOKEN, TINY_SUPERNET
from pawn.lichess_eval import (
    LichessCorpus,
    LichessElo,
    evaluate_elo_accuracy,
    filter_elo_slice,
    load_lichess_corpus,
)
from pawn.model import init_model


def _make_lichess_corpus(
    n: int = 4, seq_len: int = 16, ply: int = 6,
) -> LichessCorpus:
    """Synthesise a tiny LichessCorpus with deterministic move tokens
    and a handful of known Elos so the Elo-filter tests can pin exact
    selection behaviour."""
    rng = np.random.default_rng(0)
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :ply] = rng.integers(0, 1968, size=(n, ply), dtype=np.int32)
    targets = np.full_like(tokens, PAD_TOKEN, dtype=np.int32)
    targets[:, :-1] = tokens[:, 1:]
    attn = tokens != PAD_TOKEN
    pos = np.arange(seq_len, dtype=np.int32)[None, :]
    # Broadcast to ``(n, seq_len)``; the constant ``ply`` on the RHS
    # would otherwise leave loss_mask at shape ``(1, seq_len)``.
    loss_mask = np.broadcast_to(pos < ply, (n, seq_len)).copy()
    # Elos chosen to land in distinct buckets:
    #   game 0: 1200 / 1300 — 1200 bucket
    #   game 1: 1450 / 1500 — 1500 bucket
    #   game 2: 1800 / 1700 — 1700-1800 bucket
    #   game 3: 2200 / 2100 — 2100-2200 bucket
    white_elo = np.array([1200, 1450, 1800, 2200], dtype=np.int32)[:n]
    black_elo = np.array([1300, 1500, 1700, 2100], dtype=np.int32)[:n]
    result = np.array([1, 0, -1, 0], dtype=np.int8)[:n]
    return LichessCorpus(
        tokens=tokens, attn_mask=attn, targets=targets, loss_mask=loss_mask,
        game_lengths=np.full(n, ply, dtype=np.int32),
        white_elo=white_elo, black_elo=black_elo, result=result,
        seq_len=seq_len, prepend_outcome=False,
    )


# ---------------------------------------------------------------------------
# Elo filter
# ---------------------------------------------------------------------------


def test_filter_elo_slice_both_keeps_when_either_matches() -> None:
    corpus = _make_lichess_corpus(n=4)
    # 1400-1600 → game 1 (1450 white) matches via either side; game 0
    # has 1300 black, 1200 white — neither inside; game 2 has 1700
    # black which is outside but 1800 white also outside.
    sliced = filter_elo_slice(corpus, elo_min=1400, elo_max=1600)
    assert sliced.tokens.shape[0] == 1
    np.testing.assert_array_equal(
        sliced.white_elo, np.array([1450], dtype=np.int32),
    )


def test_filter_elo_slice_side_white_only() -> None:
    corpus = _make_lichess_corpus(n=4)
    sliced = filter_elo_slice(
        corpus, elo_min=1700, elo_max=1900, side="white",
    )
    # Only game 2 (white=1800) qualifies.
    assert sliced.tokens.shape[0] == 1
    assert int(sliced.white_elo[0]) == 1800


def test_filter_elo_slice_rejects_bad_range() -> None:
    corpus = _make_lichess_corpus(n=4)
    with pytest.raises(ValueError, match="elo_min"):
        filter_elo_slice(corpus, elo_min=1600, elo_max=1500)
    with pytest.raises(ValueError, match="side"):
        filter_elo_slice(corpus, elo_min=1400, elo_max=1600, side="invalid")


def test_filter_elo_slice_empty_range_returns_empty_corpus() -> None:
    corpus = _make_lichess_corpus(n=4)
    sliced = filter_elo_slice(corpus, elo_min=3000, elo_max=3500)
    assert sliced.tokens.shape[0] == 0
    assert sliced.seq_len == corpus.seq_len


def test_filter_elo_slice_side_white_masks_to_white_plies() -> None:
    """``side='white'`` keeps games where white's Elo is in band and
    scores only white's plies. Pinned because the existing
    ``test_filter_elo_slice_side_white_only`` only checked row count
    (test-risk round 7 MEDIUM).
    """
    corpus = _make_lichess_corpus(n=4, ply=6)
    sliced = filter_elo_slice(
        corpus, elo_min=1700, elo_max=1900, side="white",
    )
    # Game 2 (white=1800) is the only one in band.
    assert sliced.tokens.shape[0] == 1
    pos = np.arange(sliced.seq_len, dtype=np.int32)
    # prepend_outcome=False → odd positions predict white's moves.
    white_pos = pos % 2 == 1
    expected = corpus.loss_mask[2] & white_pos
    np.testing.assert_array_equal(sliced.loss_mask[0], expected)
    # Black plies must NOT be scored.
    assert not bool((sliced.loss_mask[0] & (pos % 2 == 0)).any())


def test_filter_elo_slice_side_black_masks_to_black_plies() -> None:
    """``side='black'`` is otherwise entirely untested (test-risk round 7
    MEDIUM). Mirrors the white-side check above; pins both game
    selection and per-ply masking.
    """
    corpus = _make_lichess_corpus(n=4, ply=6)
    # Pick a band that captures exactly one black Elo (game 2's
    # black=1700; game 1 is 1500 below, game 3 is 2100 above).
    sliced = filter_elo_slice(
        corpus, elo_min=1700, elo_max=1900, side="black",
    )
    assert sliced.tokens.shape[0] == 1
    assert int(sliced.black_elo[0]) == 1700
    pos = np.arange(sliced.seq_len, dtype=np.int32)
    # prepend_outcome=False → even positions predict black's moves.
    black_pos = pos % 2 == 0
    expected = corpus.loss_mask[2] & black_pos
    np.testing.assert_array_equal(sliced.loss_mask[0], expected)
    assert not bool((sliced.loss_mask[0] & (pos % 2 == 1)).any())


def test_filter_elo_slice_prepend_outcome_true_inverts_parity() -> None:
    """``filter_elo_slice`` must read ``corpus.prepend_outcome`` to
    pick the right mover parity. Under ``[outcome, m1, m2, ..., mN]``
    layout, white's moves land at *even* positions (i=0 predicts m1).
    A bug that hard-coded the prepend_outcome=False parity would
    silently flip the per-side attribution for every outcome-
    conditioned eval. Test-risk round 7 MEDIUM.
    """
    base = _make_lichess_corpus(n=4, ply=6)
    # Build an otherwise-identical corpus with prepend_outcome=True.
    corpus = LichessCorpus(
        tokens=base.tokens, attn_mask=base.attn_mask,
        targets=base.targets, loss_mask=base.loss_mask,
        game_lengths=base.game_lengths,
        white_elo=base.white_elo, black_elo=base.black_elo,
        result=base.result, seq_len=base.seq_len,
        prepend_outcome=True,
    )
    # side='white' with a band that captures only game 2's white=1800.
    sliced = filter_elo_slice(
        corpus, elo_min=1700, elo_max=1900, side="white",
    )
    assert sliced.tokens.shape[0] == 1
    pos = np.arange(sliced.seq_len, dtype=np.int32)
    # prepend_outcome=True → even positions predict white's moves.
    white_pos_under_prepend = pos % 2 == 0
    expected = corpus.loss_mask[2] & white_pos_under_prepend
    np.testing.assert_array_equal(sliced.loss_mask[0], expected)
    # And the prepend=False parity (odd=white) would give the
    # complement — assert it's NOT the result, so a parity-inversion
    # regression surfaces.
    wrong_parity = corpus.loss_mask[2] & (pos % 2 == 1)
    assert not np.array_equal(sliced.loss_mask[0], wrong_parity), (
        "filter_elo_slice ignored prepend_outcome and used the "
        "non-prepend parity"
    )


def test_filter_elo_slice_masks_out_of_band_mover_plies() -> None:
    """Per-ply masking by mover-Elo (Codex P2 rounds 3/4/6). A game
    with one side in band + one out of band should contribute only
    the in-band side's plies to the resulting ``loss_mask``, not all
    plies. The whole-game-only filter the earlier code did was
    contaminating reported per-Elo accuracy with the out-of-band
    opponent's moves.

    Fixture: white_elo=1450 (in band [1400, 1500)), black_elo=1500
    (out of band — slice is half-open ``[lo, hi)``). The mover for
    even positions is black (prepend_outcome=False layout: i % 2 == 1
    → white). So white plies are at odd positions; the resulting
    loss_mask should be True only on odd positions for this game.
    """
    # Single-game corpus with deliberately asymmetric Elos. ply=6 →
    # positions 0..5 supervised in the pre-filter loss_mask.
    corpus = _make_lichess_corpus(n=4, ply=6)
    # game 1: white=1450, black=1500 — only white in [1400, 1500).
    # Filter to that band; verify only odd positions remain True for
    # game 1.
    sliced = filter_elo_slice(corpus, elo_min=1400, elo_max=1500)
    assert sliced.tokens.shape[0] == 1
    assert int(sliced.white_elo[0]) == 1450
    assert int(sliced.black_elo[0]) == 1500
    # Under prepend_outcome=False, target[i] = m_{i+2}; even i →
    # black's move, odd i → white's. White-only → keep odd positions.
    pos = np.arange(sliced.seq_len, dtype=np.int32)
    white_pos = pos % 2 == 1
    pre_filter_loss_mask = corpus.loss_mask[1]   # game index 1 in original
    expected = pre_filter_loss_mask & white_pos
    np.testing.assert_array_equal(sliced.loss_mask[0], expected)
    # And a regression to the old whole-game behavior would yield
    # ``sliced.loss_mask[0] == pre_filter_loss_mask`` — strictly more
    # True positions than expected. Verify the new mask is a proper
    # subset.
    assert int(sliced.loss_mask[0].sum()) < int(pre_filter_loss_mask.sum()), (
        f"masked loss has {int(sliced.loss_mask[0].sum())} positions; "
        f"unfiltered had {int(pre_filter_loss_mask.sum())} — the "
        f"out-of-band black plies should have been dropped"
    )


# ---------------------------------------------------------------------------
# Accuracy eval
# ---------------------------------------------------------------------------


def test_evaluate_elo_accuracy_empty_slice_returns_nan() -> None:
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    corpus = _make_lichess_corpus(n=4)
    result = evaluate_elo_accuracy(
        model, corpus, elo_min=3000, elo_max=3500,
    )
    assert isinstance(result, LichessElo)
    assert result.n_games == 0
    assert np.isnan(result.accuracy.overall)
    assert result.accuracy.n_supervised == 0


def test_evaluate_elo_accuracy_returns_finite_overall() -> None:
    """End-to-end: filter + run model + collect AccuracyResult. The
    fresh-init TINY model will land near chance; the structural
    contract is what's pinned."""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    corpus = _make_lichess_corpus(n=4)
    result = evaluate_elo_accuracy(
        model, corpus, elo_min=1100, elo_max=2300,
    )
    assert result.n_games == 4
    assert 0.0 <= result.accuracy.overall <= 1.0
    assert result.accuracy.n_supervised > 0
    # Phase keys match pawn.eval.AccuracyResult convention.
    assert set(result.accuracy.phase.keys()) == {"opening", "midgame", "endgame"}


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def _make_lichess_pgn(games: list[dict]) -> str:
    """Build a Lichess-format PGN string with the headers + per-move
    clock annotations the Rust parser requires. Without ``Site``,
    ``UTCDate``, ``UTCTime``, ``TimeControl``, ``Termination``, and
    ``[%clk ...]`` move comments, the parser silently drops the game,
    making any cache-layout test inadvertently skip.
    """
    parts: list[str] = []
    for i, g in enumerate(games):
        parts.append(f'[Event "test{i}"]\n')
        parts.append(f'[Site "https://lichess.org/abc{i}1"]\n')
        parts.append(f'[White "{g.get("white", "alice")}"]\n')
        parts.append(f'[Black "{g.get("black", "bob")}"]\n')
        parts.append(f'[Result "{g["result"]}"]\n')
        parts.append('[UTCDate "2023.01.01"]\n')
        parts.append('[UTCTime "00:00:01"]\n')
        parts.append(f'[WhiteElo "{g["white_elo"]}"]\n')
        parts.append(f'[BlackElo "{g["black_elo"]}"]\n')
        parts.append('[WhiteRatingDiff "+5"]\n')
        parts.append('[BlackRatingDiff "-5"]\n')
        parts.append('[ECO "C60"]\n')
        parts.append('[Opening "X"]\n')
        parts.append('[TimeControl "120+1"]\n')
        parts.append('[Termination "Normal"]\n\n')
        # Each move carries a clock annotation in a comment. Format as
        # full H:MM:SS so the helper survives longer fixtures and
        # longer starting clocks — the earlier ``0:0{m}:...`` form
        # silently produced malformed strings (``0:010:00`` for ≥ 10
        # min, ``0:0-1:59`` for clocks that decremented past zero).
        moves = g["moves"]
        clock_s = max(120, len(moves) + 10)
        line = []
        for ply, mv in enumerate(moves):
            h, rem = divmod(clock_s, 3600)
            m, s = divmod(rem, 60)
            clk = f"{h}:{m:02d}:{s:02d}"
            if ply % 2 == 0:
                line.append(f"{ply // 2 + 1}. {mv} {{ [%clk {clk}] }}")
            else:
                line.append(f"{ply // 2 + 1}... {mv} {{ [%clk {clk}] }}")
            clock_s -= 1
        parts.append(" ".join(line) + f" {g['result']}\n\n")
    return "".join(parts)


def test_materialise_prepend_outcome_true_includes_position_0() -> None:
    """``_materialise(prepend_outcome=True)`` must include position 0
    (outcome→m1) in the loss_mask. Layout is
    ``[outcome, m1, ..., mN, PAD, ...]`` and ``targets[0] = m1`` is a
    legitimate supervised step — the whole point of outcome
    conditioning. A regression that reverted to ``pos >= 1`` would
    silently drop the first-move supervision from every
    outcome-prefixed eval; both bug-detector and test-risk in round 5
    flagged this fix as untested.
    """
    from pawn.config import OUTCOME_TOKEN_BASE
    from pawn.lichess_eval import _materialise

    seq_len = 8
    n_moves = 3
    # Simulated Rust-parser output for one prepend_outcome=True game.
    tokens = np.full((1, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[0, 0] = OUTCOME_TOKEN_BASE     # any outcome token
    tokens[0, 1:1 + n_moves] = [10, 11, 12]
    parsed = {
        "tokens": tokens.astype(np.int16),
        "game_lengths": np.array([n_moves], dtype=np.int32),
        "white_elo": [1500],
        "black_elo": [1500],
        "result": ["1-0"],
    }
    corpus = _materialise(parsed, seq_len, prepend_outcome=True)

    # Positions 0..n_moves inclusive → n_moves + 1 supervised rows.
    # The pre-fix `pos >= 1` cut would give n_moves rows; the
    # current `pos <= game_lengths` form gives n_moves + 1.
    assert int(corpus.loss_mask[0].sum()) == n_moves + 1, (
        f"loss_mask sum={int(corpus.loss_mask[0].sum())}, expected "
        f"{n_moves + 1} (regression would drop position 0 → "
        f"{n_moves})"
    )
    # Position 0 specifically: outcome→m1 supervision must be active.
    assert bool(corpus.loss_mask[0, 0]), (
        "loss_mask[0, 0] is False — the outcome→m1 supervision step "
        "is being dropped"
    )
    # Position n_moves (the terminal m_N→PAD slot) must also be
    # supervised at the loss_mask level. (The eval-time mask in
    # ``pawn.eval`` further drops it via the ``target < NUM_ACTIONS``
    # AND — pinned separately.)
    assert bool(corpus.loss_mask[0, n_moves]), (
        f"loss_mask[0, {n_moves}] is False — the terminal "
        "supervision step is missing"
    )


def test_load_lichess_corpus_round_trips_through_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny inline PGN is parsed, cached, then re-loaded from cache
    without re-invoking the Rust parser. Verifies the cache layout +
    round-trip equivalence."""
    monkeypatch.setenv("PAWN_DATA_CACHE", str(tmp_path / "cache"))

    pgn = _make_lichess_pgn([
        {
            "white": "alice", "black": "bob", "result": "1-0",
            "white_elo": 1800, "black_elo": 1750,
            "moves": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
        },
        {
            "white": "carol", "black": "dan", "result": "0-1",
            "white_elo": 1500, "black_elo": 1550,
            "moves": ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"],
        },
    ])
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(pgn)

    corpus1 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    assert corpus1.tokens.shape[0] >= 1, (
        "Lichess-format PGN should parse at least one game; if this "
        "fails the fixture is wrong, not the cache code"
    )
    # Re-load: comes from cache (path exists). Result is identical.
    corpus2 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    np.testing.assert_array_equal(corpus1.tokens, corpus2.tokens)
    np.testing.assert_array_equal(corpus1.white_elo, corpus2.white_elo)
    np.testing.assert_array_equal(corpus1.black_elo, corpus2.black_elo)
    # Bypass cache → parses again, returns the same content.
    corpus3 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
        use_cache=False,
    )
    np.testing.assert_array_equal(corpus1.tokens, corpus3.tokens)


def test_load_lichess_corpus_reparses_when_metadata_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-side cache hit requires *both* ``games.npz`` and
    ``metadata.json``. Pins the ``and`` guard against accidental
    weakening to ``or`` — a refactor that read a partial write would
    silently load an .npz next to stale or missing metadata.

    Setup: prime the cache, then delete only ``metadata.json``. The
    next call must re-parse (not load the orphan .npz).
    """
    monkeypatch.setenv("PAWN_DATA_CACHE", str(tmp_path / "cache"))
    pgn = _make_lichess_pgn([
        {
            "white": "alice", "black": "bob", "result": "1-0",
            "white_elo": 1800, "black_elo": 1750,
            "moves": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
        },
    ])
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(pgn)

    corpus1 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    assert corpus1.tokens.shape[0] >= 1, "fixture should parse 1 game"

    # Locate the cache dir from the public root + the same key the
    # loader would derive. There's exactly one subdir under the
    # cache root after the first load.
    cache_root = tmp_path / "cache"
    cache_dirs = list(cache_root.iterdir())
    assert len(cache_dirs) == 1
    cache_dir = cache_dirs[0]
    npz = cache_dir / "games.npz"
    meta = cache_dir / "metadata.json"
    assert npz.exists() and meta.exists(), (
        "expected both cache files after first load"
    )
    # Orphan: delete metadata only. games.npz remains.
    meta.unlink()

    # Second load must re-parse (and rewrite both files), not crash
    # on the orphan .npz nor silently load it.
    corpus2 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    np.testing.assert_array_equal(corpus1.tokens, corpus2.tokens)
    # The metadata file should have been rewritten by the re-parse.
    assert meta.exists()
