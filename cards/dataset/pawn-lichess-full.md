---
license: apache-2.0
task_categories:
  - other
tags:
  - chess
  - lichess
  - pawn
  - pre-tokenized
size_categories:
  - 100M<n<1B
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
      - split: test
        path: data/test-*.parquet
---

# PAWN Lichess Full

Rated Lichess games from Q1 2025 plus a January 2026 holdout, pre-tokenized in the [PAWN](https://github.com/thomas-schweich/PAWN) v1.0.0 training format. Designed for finetuning the [PAWN](https://huggingface.co/thomas-schweich/pawn-base) backbones on real human play, but also usable as a general-purpose pre-parsed Lichess feed since it includes raw SAN/UCI move strings, clock annotations, and full game metadata.

## Splits

| Split | Source | Games | Shards |
|-------|--------|------:|-------:|
| train | January, February, March 2025 | 286,010,319 | 287 |
| validation | January 1–3, 2026 | 9,295,654 | 10 |
| test | January 15–17, 2026 | 8,966,793 | 9 |


## Schema

Pre-tokenized PAWN format. Each row is one rated Lichess game.

| Column | Type | Description |
|--------|------|-------------|
| `tokens` | `list[int16]` | PAWN v1.0.0 token IDs, one per ply (variable length, up to 512). Pure moves — no outcome prefix. The vocabulary is the [searchless_chess action set](https://github.com/google-deepmind/searchless_chess): 1,968 reachable (src, dst[, promo]) tuples. |
| `san` | `list[str]` | Standard algebraic notation per ply (e.g. `Nf3`, `O-O`, `exd5`, `e8=Q+`). Same length as `tokens`. |
| `uci` | `list[str]` | UCI move strings per ply (e.g. `b1c3`, `e7e8q`). Same length as `tokens`. |
| `clock` | `list[uint16]` | Seconds remaining per ply, decoded from `[%clk]` annotations. `0` = no annotation. |
| `game_length` | `uint16` | Number of plies in the game. Capped at 512; longer games are truncated and tagged `PLY_LIMIT`. |
| `outcome_token` | `uint16` | Game outcome as a PAWN outcome token (one of 11; see [Outcome tokens](#outcome-tokens) below). Lives in its own column rather than being prepended to `tokens`. |
| `result` | `string` | Raw Lichess `Result` header: `1-0`, `0-1`, `1/2-1/2`. |
| `white_player` | `uint64` | xxHash64 of white's username (Polars `pl.Series.hash()`). Same player → same hash. |
| `black_player` | `uint64` | xxHash64 of black's username. |
| `white_elo` | `uint16` | White's Lichess rating at game start. |
| `black_elo` | `uint16` | Black's Lichess rating at game start. |
| `white_rating_diff` | `int16` | White's rating change from this game. |
| `black_rating_diff` | `int16` | Black's rating change from this game. |
| `eco` | `string` | ECO opening code (e.g. `A03`). |
| `opening` | `string` | Opening name as stamped by Lichess. |
| `time_control` | `string` | Lichess time control string, e.g. `600+0`, `180+2`. |
| `termination` | `string` | Lichess termination header: `Normal`, `Time forfeit`, `Abandoned`, etc. |
| `date` | `datetime[ms]` | Game start time (UTC), parsed from `UTCDate`/`UTCTime`. |
| `site` | `string` | Lichess game URL. Useful for looking up original usernames. |

### Outcome tokens

The 11 outcome tokens span the v1.0.0 vocabulary indices 1969–1979:

| ID | Meaning |
|---:|---------|
| 1969 | White delivers checkmate |
| 1970 | Black delivers checkmate |
| 1971 | Stalemate |
| 1972 | Draw by rule (75-move, fivefold repetition, insufficient material) |
| 1973 | Ply limit reached (game exceeded 512 plies and was truncated) |
| 1974 | White wins by resignation |
| 1975 | Black wins by resignation |
| 1976 | Draw by agreement |
| 1977 | White wins on time |
| 1978 | Black wins on time |
| 1979 | Draw on time (insufficient mating material) |

The first five also appear in random-game pretraining; the last six are Lichess-specific.

### Sentinel values

- **Clock** `0`: no clock annotation for that ply
- **`outcome_token == 1973`** indicates that a game was truncated to 512 plies.

## Usage

The headline use is finetuning the PAWN backbones via behavioral cloning. The `tokens` column is what the PAWN training pipeline consumes directly; everything else is metadata for filtering and analysis.

### Behavioral cloning of an Elo band

The PAWN trainer accepts the dataset as a `--pgn` source and pulls predicates straight through to Polars. Filter to a 100-Elo band:

```bash
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --elo-min 1800 --elo-max 1900 \
    --bottleneck-dim 32 --lr 1e-4 --local-checkpoints
```

### Polars with predicate pushdown

You can easily filter to a specific Elo band without downloading the full dataset. Polars pushes the predicate down into each parquet file's row-group statistics, so only matching row groups get fetched:

```python
import polars as pl

df = (
    pl.scan_parquet("hf://datasets/thomas-schweich/pawn-lichess-full/data/train-*.parquet")
    .filter(
        (pl.col("white_elo").is_between(1800, 1899)) &
        (pl.col("black_elo").is_between(1800, 1899))
    )
    .head(50_000)
    .collect()
)
```

### Column projection

Polars only downloads the columns you select:

```python
import polars as pl

df = (
    pl.scan_parquet("hf://datasets/thomas-schweich/pawn-lichess-full/data/train-*.parquet")
    .select(["uci", "white_elo", "black_elo", "result", "time_control", "date"])
    .head(100_000)
    .collect()
)
```

### HuggingFace `datasets`

```python
from datasets import load_dataset

ds = load_dataset("thomas-schweich/pawn-lichess-full", split="train", streaming=True)
for game in ds.take(5):
    print(game["uci"][:5], game["result"], game["white_elo"])
```

## Other uses

The `tokens` column is PAWN-specific, but the rest of the schema (`san`, `uci`, Elo, clock, metadata) stands on its own as a parsed Lichess feed — no PAWN install required. Use column projection to download only what you need.

## Raw monthly archives

The repo includes the original `.pgn.zst` files (`lichess_2025-01.pgn.zst`, etc.) alongside the parquet data. They are the exact monthly dumps that the parser consumed, mirrored here so the parquet outputs are reproducible and because HuggingFace's CDN can hand them out faster than the upstream Lichess server.

## Generation

Extracted from [Lichess database dumps](https://database.lichess.org/) (CC0) using the PAWN Rust chess engine for SAN→UCI→token conversion. See [`scripts/extract_lichess_parquet.py`](https://github.com/thomas-schweich/PAWN/blob/main/scripts/extract_lichess_parquet.py) in the [PAWN repository](https://github.com/thomas-schweich/PAWN).

Pipeline at a glance:

```
zstd PGN dumps  →  Rust enriched PGN parser  →  Polars DataFrame  →  zstd Parquet shard  →  HuggingFace
                  (single-pass extraction
                   of moves, clocks, headers,
                   token ID conversion)
```

The extractor is streaming and resumable: shards are uploaded as they finish, sentinel files mark completed work units, and orphaned partial shards are cleaned up on rerun.

## Legacy revisions

Earlier revisions of this dataset used the legacy 4,278-token PAWN vocabulary in the `tokens` column and a different val/test layout (50K random samples instead of full days). Those revisions are still accessible via git history if you need to reproduce prior experiments — the current `main` revision uses the v1.0.0 1,980-token searchless_chess action vocabulary described above.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Derived from the [Lichess database](https://database.lichess.org/), released under [Creative Commons CC0](https://creativecommons.org/publicdomain/zero/1.0/).
