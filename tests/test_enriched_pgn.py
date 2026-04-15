"""Tests for enriched PGN parsing and dataset extraction pipeline."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import chess_engine


def _load_extract_module():
    """Load scripts/extract_lichess_parquet.py as a standalone module.

    The scripts/ directory isn't a package, so we use importlib.util to
    load the file directly by path. Registering in sys.modules before
    exec_module is required for dataclasses defined at module scope under
    Python 3.10 (dataclass decorator looks up ``sys.modules[cls.__module__]``
    when checking for KW_ONLY sentinels).
    """
    path = Path(__file__).resolve().parent.parent / "scripts" / "extract_lichess_parquet.py"
    spec = importlib.util.spec_from_file_location("extract_lichess_parquet", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extract_lichess_parquet"] = mod
    spec.loader.exec_module(mod)
    return mod


batch_to_dataframe = _load_extract_module().batch_to_dataframe


# ---------------------------------------------------------------------------
# Test PGN data — each game has distinct moves, metadata, and annotations
# ---------------------------------------------------------------------------

PGNS = {
    "alice_v_bob": """\
[Event "Rated Rapid game"]
[Site "https://lichess.org/game001"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1873"]
[BlackElo "1844"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-26"]
[ECO "C20"]
[Opening "King's Pawn Game"]
[TimeControl "600+0"]
[Termination "Normal"]
[UTCDate "2025.01.10"]
[UTCTime "10:00:00"]

1. e4 { [%clk 0:10:00] [%eval 0.23] } 1... e5 { [%clk 0:09:58] [%eval 0.31] } 2. Nf3 { [%clk 0:09:50] [%eval 0.25] } 2... Nc6 { [%clk 0:09:45] [%eval 0.30] } 1-0
""",
    "bob_v_alice": """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/game002"]
[White "bob"]
[Black "alice"]
[Result "0-1"]
[WhiteElo "1850"]
[BlackElo "1880"]
[WhiteRatingDiff "-5"]
[BlackRatingDiff "+5"]
[ECO "B20"]
[Opening "Sicilian Defense"]
[TimeControl "300+3"]
[Termination "Time forfeit"]
[UTCDate "2025.02.14"]
[UTCTime "20:00:00"]

1. e4 { [%clk 0:05:00] [%eval 0.20] } 1... c5 { [%clk 0:04:55] [%eval 0.25] } 2. d4 { [%clk 0:04:48] [%eval 0.40] } 0-1
""",
    "alice_v_xavier": """\
[Event "Rated Classical game"]
[Site "https://lichess.org/game003"]
[White "alice"]
[Black "xavier"]
[Result "1/2-1/2"]
[WhiteElo "1900"]
[BlackElo "2100"]
[WhiteRatingDiff "+3"]
[BlackRatingDiff "-1"]
[ECO "D30"]
[Opening "Queen's Gambit Declined"]
[TimeControl "1800+30"]
[Termination "Normal"]
[UTCDate "2025.03.01"]
[UTCTime "15:30:00"]

1. d4 { [%clk 0:30:00] [%eval 0.10] } 1... d5 { [%clk 0:29:50] [%eval 0.15] } 2. c4 { [%clk 0:29:40] [%eval 0.20] } 2... e6 { [%clk 0:29:30] [%eval 0.18] } 3. Nf3 { [%clk 0:29:20] [%eval 0.22] } 1/2-1/2
""",
    "xavier_v_alice": """\
[Event "Rated Rapid game"]
[Site "https://lichess.org/game004"]
[White "xavier"]
[Black "alice"]
[Result "1-0"]
[WhiteElo "2105"]
[BlackElo "1895"]
[WhiteRatingDiff "+2"]
[BlackRatingDiff "-4"]
[ECO "A45"]
[Opening "Trompowsky Attack"]
[TimeControl "900+10"]
[Termination "Normal"]
[UTCDate "2025.01.20"]
[UTCTime "18:00:00"]

1. d4 { [%clk 0:15:00] } 1... Nf6 { [%clk 0:14:55] } 2. Bg5 { [%clk 0:14:48] } 1-0
""",
    "bob_v_xavier": """\
[Event "Rated Bullet game"]
[Site "https://lichess.org/game005"]
[White "bob"]
[Black "xavier"]
[Result "0-1"]
[WhiteElo "1840"]
[BlackElo "2110"]
[WhiteRatingDiff "-8"]
[BlackRatingDiff "+3"]
[ECO "C50"]
[Opening "Italian Game"]
[TimeControl "60+0"]
[Termination "Normal"]
[UTCDate "2025.02.28"]
[UTCTime "23:59:00"]

1. e4 { [%clk 0:01:00] [%eval 0.20] } 1... e5 { [%clk 0:00:59] [%eval 0.25] } 2. Nf3 { [%clk 0:00:55] [%eval 0.30] } 2... Nc6 { [%clk 0:00:53] [%eval 0.28] } 3. Bc4 { [%clk 0:00:50] [%eval 0.35] } 3... Bc5 { [%clk 0:00:48] [%eval 0.30] } 0-1
""",
    "xavier_v_bob": """\
[Event "Rated Rapid game"]
[Site "https://lichess.org/game006"]
[White "xavier"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "2115"]
[BlackElo "1835"]
[WhiteRatingDiff "+1"]
[BlackRatingDiff "-7"]
[ECO "E00"]
[Opening "Queen's Pawn Game"]
[TimeControl "600+5"]
[Termination "Normal"]
[UTCDate "2025.03.15"]
[UTCTime "09:00:00"]

1. d4 { [%clk 0:10:00] [%eval 0.15] } 1... Nf6 { [%clk 0:09:55] [%eval 0.20] } 2. c4 { [%clk 0:09:48] [%eval 0.25] } 2... e6 { [%clk 0:09:40] [%eval 0.22] } 3. Nc3 { [%clk 0:09:35] [%eval 0.28] } 3... Bb4 { [%clk 0:09:28] [%eval 0.30] } 4. Qc2 { [%clk 0:09:20] [%eval 0.32] } 1-0
""",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnrichedParsing:
    """Test the Rust parse_pgn_enriched function."""

    def test_basic_parsing(self):
        pgn = PGNS["alice_v_bob"]
        r = chess_engine.parse_pgn_enriched(pgn)
        assert r["tokens"].shape == (1, 255)
        assert r["clocks"].shape == (1, 255)
        assert r["evals"].shape == (1, 255)
        assert r["game_lengths"].shape == (1,)
        assert r["game_lengths"][0] == 4

    def test_return_types(self):
        pgn = PGNS["alice_v_bob"]
        r = chess_engine.parse_pgn_enriched(pgn)
        for key in ("tokens", "clocks", "evals"):
            assert isinstance(r[key], np.ndarray), f"{key} should be ndarray"
        for key in ("game_lengths", "white_elo", "black_elo",
                     "white_rating_diff", "black_rating_diff"):
            assert isinstance(r[key], np.ndarray), f"{key} should be ndarray"
        for key in ("result", "white", "black", "eco", "opening",
                     "time_control", "termination", "date_time", "site"):
            assert isinstance(r[key], list), f"{key} should be list"

    def test_clock_extraction(self):
        pgn = PGNS["alice_v_bob"]
        r = chess_engine.parse_pgn_enriched(pgn)
        clocks = r["clocks"][0, :4]
        assert list(clocks) == [600, 598, 590, 585]

    def test_eval_extraction(self):
        pgn = PGNS["alice_v_bob"]
        r = chess_engine.parse_pgn_enriched(pgn)
        evals = r["evals"][0, :4]
        assert list(evals) == [23, 31, 25, 30]

    def test_missing_eval_sentinel(self):
        pgn = PGNS["xavier_v_alice"]  # no eval annotations
        r = chess_engine.parse_pgn_enriched(pgn)
        length = r["game_lengths"][0]
        evals = r["evals"][0, :length]
        # Rust uses i16::MIN (-32768) as the "no eval" sentinel
        assert all(e == -32768 for e in evals), (
            f"Missing evals should be -32768 (i16::MIN), got {list(evals)}"
        )

    def test_padding_is_zero(self):
        pgn = PGNS["alice_v_bob"]
        r = chess_engine.parse_pgn_enriched(pgn)
        length = r["game_lengths"][0]
        assert np.all(r["tokens"][0, length:] == 0)
        assert np.all(r["clocks"][0, length:] == 0)
        assert np.all(r["evals"][0, length:] == 0)

    def test_headers_extracted(self):
        pgn = PGNS["alice_v_bob"]
        r = chess_engine.parse_pgn_enriched(pgn)
        assert r["white"][0] == "alice"
        assert r["black"][0] == "bob"
        assert r["result"][0] == "1-0"
        assert r["white_elo"][0] == 1873
        assert r["black_elo"][0] == 1844
        assert r["white_rating_diff"][0] == 6
        assert r["black_rating_diff"][0] == -26
        assert r["eco"][0] == "C20"
        assert r["time_control"][0] == "600+0"
        assert r["site"][0] == "https://lichess.org/game001"

    def test_different_games_produce_different_tokens(self):
        """Each test PGN has distinct moves — tokens must differ."""
        all_tokens = {}
        for name, pgn in PGNS.items():
            r = chess_engine.parse_pgn_enriched(pgn)
            length = r["game_lengths"][0]
            all_tokens[name] = tuple(r["tokens"][0, :length])

        names = list(all_tokens.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert all_tokens[names[i]] != all_tokens[names[j]], (
                    f"{names[i]} and {names[j]} should have different token sequences"
                )


class TestLichessParserOutput:
    """Direct assertions on parse_pgn_lichess (pure-moves layout)."""

    def test_pure_moves_default_shape(self):
        """Default `prepend_outcome=False` → tokens shape is (N, max_ply)
        and tokens[0] is the first move, not the outcome."""
        r = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"])
        # Default max_ply is 512 (matches the 512-token context of current models)
        assert r["tokens"].shape[1] == 512
        # The first move is white's e4 → UCI e2e4 → token id from vocab
        m2t = chess_engine.export_move_vocabulary()["move_to_token"]
        e2e4 = m2t["e2e4"]
        assert r["tokens"][0, 0] == e2e4
        # Outcome lives in its own column
        assert r["outcome_tokens"].shape == (1,)
        assert 1969 <= int(r["outcome_tokens"][0]) <= 1979

    def test_prepend_outcome_opt_in_shape(self):
        """`prepend_outcome=True` still fits in `max_ply` total slots — the
        outcome takes slot 0 and the move cap is reduced by one. The tensor
        width matches `max_ply` in both modes, matching the convention used
        by ``generate_clm_batch``'s ``seq_len`` parameter."""
        r = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"], prepend_outcome=True)
        assert r["tokens"].shape[1] == 512  # same width as pure-moves default
        assert int(r["tokens"][0, 0]) == int(r["outcome_tokens"][0])

    def test_san_and_uci_columns_present(self):
        r = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"])
        assert isinstance(r["san"], list)
        assert isinstance(r["uci"], list)
        assert r["san"][0] == ["e4", "e5", "Nf3", "Nc6"]
        assert r["uci"][0] == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_san_raw_from_pgn(self):
        """SAN strings are what the PGN actually wrote, including short-SAN
        disambiguation (e.g. `Bg5` not `B1g5`)."""
        r = chess_engine.parse_pgn_lichess(PGNS["xavier_v_alice"])
        assert r["san"][0] == ["d4", "Nf6", "Bg5"]


class TestSanUciTokenConsistency:
    """The three move representations must agree row by row.

    tokens are derived from SAN by the Rust parser; UCI is derived from
    tokens via `vocab::token_to_uci`. These tests prove both derivations
    are internally consistent by round-tripping through independent
    engine entry points.
    """

    @pytest.fixture(scope="class")
    def combined_output(self):
        pgns = "\n".join(PGNS.values())
        return chess_engine.parse_pgn_lichess(pgns)

    def test_lengths_agree(self, combined_output):
        n = combined_output["tokens"].shape[0]
        for i in range(n):
            gl = int(combined_output["game_lengths"][i])
            assert len(combined_output["san"][i]) == gl
            assert len(combined_output["uci"][i]) == gl
            # tokens row is fixed-width; the first `gl` entries are the moves
            row = combined_output["tokens"][i]
            assert gl <= row.shape[0]

    def test_uci_matches_vocab_token_lookup(self, combined_output):
        """For every (token, uci) pair, vocab['token_to_move'][token] == uci."""
        t2m = chess_engine.export_move_vocabulary()["token_to_move"]
        n = combined_output["tokens"].shape[0]
        for i in range(n):
            gl = int(combined_output["game_lengths"][i])
            tokens = combined_output["tokens"][i, :gl].tolist()
            ucis = combined_output["uci"][i]
            for t, u in zip(tokens, ucis):
                assert t2m[int(t)] == u, f"game {i}: token {t} → {t2m[int(t)]!r}, uci column says {u!r}"

    def test_san_converted_to_uci_matches_uci_column(self, combined_output):
        """pgn_to_uci(san) must reproduce the uci column exactly."""
        san_rows: list[list[str]] = combined_output["san"]
        recovered = chess_engine.pgn_to_uci(san_rows)
        assert recovered == combined_output["uci"]

    def test_uci_back_to_tokens_matches_token_column(self, combined_output):
        """Feeding the UCI strings back through `uci_to_tokens` reproduces
        the token column exactly (and lengths agree)."""
        uci_rows: list[list[str]] = combined_output["uci"]
        # uci_to_tokens returns (move_ids: (N, max_ply) i16, lengths: (N,) i16)
        recovered_tokens, recovered_lengths = chess_engine.uci_to_tokens(uci_rows, max_ply=512)
        n = combined_output["tokens"].shape[0]
        for i in range(n):
            gl = int(combined_output["game_lengths"][i])
            assert int(recovered_lengths[i]) == gl
            assert recovered_tokens[i, :gl].tolist() == combined_output["tokens"][i, :gl].tolist()


class TestPlayerHashing:
    """Test that player username hashing is deterministic and independent of context.

    Each PGN has different moves, metadata, Elo, time control, dates, and
    game lengths — the only thing shared is the player name strings. If
    hashing accidentally depended on row context, these would diverge.
    """

    @pytest.fixture
    def player_hashes(self):
        """Parse all 6 PGNs separately and collect per-player hash values."""
        hashes = {}  # name -> list of observed uint64 hashes
        for name, pgn in PGNS.items():
            r = chess_engine.parse_pgn_lichess(pgn)
            df = batch_to_dataframe(r)
            w_name = r["white"][0]
            b_name = r["black"][0]
            w_hash = df["white_player"][0]
            b_hash = df["black_player"][0]
            hashes.setdefault(w_name, []).append(w_hash)
            hashes.setdefault(b_name, []).append(b_hash)
        return hashes

    def test_same_name_always_same_hash(self, player_hashes):
        """A player name must always produce the same hash regardless of
        which game it appears in, whether as white or black, and what
        the surrounding metadata looks like."""
        for name, vals in player_hashes.items():
            unique = set(vals)
            assert len(unique) == 1, (
                f"'{name}' produced {len(unique)} distinct hashes across "
                f"{len(vals)} appearances: {unique}"
            )

    def test_different_names_different_hashes(self, player_hashes):
        """alice, bob, and xavier must all have distinct hashes."""
        canonical = {name: vals[0] for name, vals in player_hashes.items()}
        hash_vals = list(canonical.values())
        assert len(set(hash_vals)) == len(hash_vals), (
            f"Hash collision among players: {canonical}"
        )

    def test_hash_dtype_is_uint64(self, player_hashes):
        # Parse one game and check the Polars column dtype
        r = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"])
        df = batch_to_dataframe(r)
        assert df["white_player"].dtype == pl.UInt64
        assert df["black_player"].dtype == pl.UInt64

    def test_hash_appears_in_both_columns(self, player_hashes):
        """alice appears as both white and black — hash must match in both columns."""
        # alice is white in alice_v_bob and black in bob_v_alice
        r1 = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"])
        df1 = batch_to_dataframe(r1)
        r2 = chess_engine.parse_pgn_lichess(PGNS["bob_v_alice"])
        df2 = batch_to_dataframe(r2)

        alice_as_white = df1["white_player"][0]
        alice_as_black = df2["black_player"][0]
        assert alice_as_white == alice_as_black, (
            f"alice hash differs: white={alice_as_white}, black={alice_as_black}"
        )


class TestPlayerHashRegression:
    """Snapshot test: catch if a Polars update changes the hash algorithm.

    These exact values were recorded with Polars 1.39.3 using the default
    hash() seed (xxHash64). If this test fails after a Polars upgrade, the
    dataset must be regenerated to stay consistent (or the old Polars
    version must be pinned).
    """

    EXPECTED_HASHES = {
        "alice": 573680751236103438,
        "bob": 11376496890720967193,
        "xavier": 2453512920044318708,
    }

    def test_hash_values_match_snapshot(self):
        """Verify that pl.Series.hash() produces the exact same uint64
        values that were recorded when the dataset was built."""
        for name, expected in self.EXPECTED_HASHES.items():
            actual = pl.Series([name]).hash()[0]
            assert actual == expected, (
                f"Hash regression for '{name}': expected {expected}, got {actual}. "
                f"Polars hash algorithm may have changed — dataset must be regenerated."
            )

    def test_snapshot_matches_pipeline(self):
        """The snapshot values must agree with what batch_to_dataframe produces."""
        combined = "\n".join(PGNS.values())
        r = chess_engine.parse_pgn_lichess(combined)
        df = batch_to_dataframe(r)

        for name, expected in self.EXPECTED_HASHES.items():
            # Find rows where this player appears as white
            white_rows = [
                i for i, w in enumerate(r["white"]) if w == name
            ]
            for i in white_rows:
                actual = df["white_player"][i]
                assert actual == expected, (
                    f"Pipeline hash for '{name}' (white, row {i}): "
                    f"expected {expected}, got {actual}"
                )

            # Find rows where this player appears as black
            black_rows = [
                i for i, b in enumerate(r["black"]) if b == name
            ]
            for i in black_rows:
                actual = df["black_player"][i]
                assert actual == expected, (
                    f"Pipeline hash for '{name}' (black, row {i}): "
                    f"expected {expected}, got {actual}"
                )


class TestBatchToDataframe:
    """Test the full batch_to_dataframe pipeline (pure-moves layout)."""

    def test_schema(self):
        r = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"])
        df = batch_to_dataframe(r)
        assert df["tokens"].dtype == pl.List(pl.Int16)
        assert df["san"].dtype == pl.List(pl.Utf8)
        assert df["uci"].dtype == pl.List(pl.Utf8)
        assert df["clock"].dtype == pl.List(pl.UInt16)
        assert df["game_length"].dtype == pl.UInt16
        assert df["outcome_token"].dtype == pl.UInt16
        assert df["white_elo"].dtype == pl.UInt16
        assert df["black_elo"].dtype == pl.UInt16
        assert df["white_rating_diff"].dtype == pl.Int16
        assert df["black_rating_diff"].dtype == pl.Int16
        assert df["white_player"].dtype == pl.UInt64
        assert df["black_player"].dtype == pl.UInt64

    def test_list_columns_trimmed_to_game_length(self):
        """All per-ply list columns (tokens, san, uci, clock) have length
        exactly == game_length — no outcome prefix, no padding."""
        r = chess_engine.parse_pgn_lichess(PGNS["bob_v_xavier"])
        df = batch_to_dataframe(r)
        gl = int(df["game_length"][0])
        assert len(df["tokens"][0]) == gl
        assert len(df["san"][0]) == gl
        assert len(df["uci"][0]) == gl
        assert len(df["clock"][0]) == gl

    def test_outcome_token_column_is_separate(self):
        """The outcome token lives only in its own column; the tokens list
        is pure moves with no outcome prefix."""
        r = chess_engine.parse_pgn_lichess(PGNS["alice_v_bob"])
        df = batch_to_dataframe(r)
        outcome = int(df["outcome_token"][0])
        assert 1969 <= outcome <= 1979
        # The tokens column must not contain the outcome value anywhere.
        for tok in df["tokens"][0]:
            assert tok != outcome
            assert 0 <= int(tok) <= 1967

    def test_san_uci_tokens_consistent_in_dataframe(self):
        """End-to-end: pull san/uci/tokens out of the DataFrame and verify
        they agree with each other via the engine's vocab tables."""
        combined = "\n".join(PGNS.values())
        r = chess_engine.parse_pgn_lichess(combined)
        df = batch_to_dataframe(r)

        t2m = chess_engine.export_move_vocabulary()["token_to_move"]
        san_rows = df["san"].to_list()
        uci_rows = df["uci"].to_list()
        token_rows = df["tokens"].to_list()

        # SAN → UCI via engine matches the uci column.
        assert chess_engine.pgn_to_uci(san_rows) == uci_rows
        # token → UCI via vocab matches the uci column.
        for toks, ucis in zip(token_rows, uci_rows):
            assert [t2m[int(t)] for t in toks] == ucis

    def test_parquet_roundtrip(self, tmp_path):
        """Write to Parquet and read back — all values must survive."""
        r = chess_engine.parse_pgn_lichess(PGNS["xavier_v_bob"])
        df = batch_to_dataframe(r)
        path = tmp_path / "test.parquet"
        df.write_parquet(path, compression="zstd")
        df2 = pl.read_parquet(path)
        assert df.shape == df2.shape
        assert df["tokens"].to_list() == df2["tokens"].to_list()
        assert df["san"].to_list() == df2["san"].to_list()
        assert df["uci"].to_list() == df2["uci"].to_list()
        assert df["clock"].to_list() == df2["clock"].to_list()
        assert df["outcome_token"].to_list() == df2["outcome_token"].to_list()
        assert df["white_player"].to_list() == df2["white_player"].to_list()

    def test_multi_game_batch(self):
        """Parse all 6 games in a single PGN string."""
        combined = "\n".join(PGNS.values())
        r = chess_engine.parse_pgn_lichess(combined)
        df = batch_to_dataframe(r)
        assert len(df) == 6
        # Each game should have a different game length
        lengths = df["game_length"].to_list()
        assert len(set(lengths)) > 1, "Games should have different lengths"
