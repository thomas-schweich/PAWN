# stockfish-datagen

Standalone Stockfish self-play data generator. Produces zstd-compressed
parquet shards in the same column shape as `extract_lichess_parquet.py`'s
output (the shared columns), so the model's data pipeline doesn't need to
care which source the data came from.

## Usage

The binary takes one argument: the path to a JSON config.

```bash
# Validate a config without running anything.
cargo run --release -p stockfish-datagen -- dry-run \
  --config stockfish-datagen/examples/smoke.json

# Tiny smoke run (64 games, ~2s).
cargo run --release -p stockfish-datagen -- run \
  --config stockfish-datagen/examples/smoke.json

# Production: 20M games across 5 tiers.
cargo run --release -p stockfish-datagen -- run \
  --config stockfish-datagen/examples/stockfish_20m.json
```

## Reproducibility

Every row in the output carries a `game_seed` (i64). Combined with
`stockfish_version` and the per-row tier config (`nodes`, `multi_pv`,
`opening_multi_pv`, `opening_plies`, `sample_plies`, `temperature`),
that seed is sufficient to deterministically replay the exact game.

The version pin is enforced at startup: if the config says
`"stockfish_version": "Stockfish 18"` and the binary at
`stockfish_path` reports anything else, the run aborts before any
parquet is written. Different Stockfish releases ship different NNUE
nets and would silently produce different games from the same seed.

The seed hierarchy is:

```
master_seed (config)
  → tier_seed (per tier, derived from tier index)
    → worker_seed (per worker, derived from worker_id)
      → game_seed (per game, derived from game_index)
```

Each step is splitmix64 mixing — pure functional, no PRNG state to
thread around. Resume can recompute any worker's `game_seed[i]`
without replaying earlier games.

## Output layout

```
<output_dir>/
  <tier_name>/
    _manifest.json                    # written last, signals tier-complete
    shard-w000-c0000.parquet          # one shard per (worker, chunk)
    shard-w000-c0001.parquet
    ...
    shard-w011-c0000.parquet
    ...
```

Shards are zstd-compressed parquet (level 3). They're written
atomically via `<path>.tmp` → `<path>.parquet` rename, so a crash
mid-shard leaves an orphan `.tmp` rather than a half-valid parquet.

The manifest is the tier-complete signal. Re-running with the same
config no-ops any tier whose manifest's config fingerprint matches
the current config. Deleting the manifest causes a re-scan: any
`.parquet` shards still on disk count toward `games_done` and the
worker resumes from the next chunk index.

## Schema

| column              | type            | notes                                   |
|---------------------|-----------------|-----------------------------------------|
| `tokens`            | `List<Int16>`   | searchless_chess action tokens (0..1967) |
| `san`               | `List<String>`  | SAN with check/mate suffixes            |
| `uci`               | `List<String>`  | UCI strings as Stockfish emitted them   |
| `game_length`       | `UInt16`        | == len of tokens / san / uci            |
| `outcome_token`     | `UInt16`        | 1969..1973 (mirrors engine vocab.rs)    |
| `result`            | `String`        | "1-0", "0-1", or "1/2-1/2"              |
| `nodes`             | `Int32`         | per-row, from tier config               |
| `multi_pv`          | `Int32`         | per-row, from tier config               |
| `opening_multi_pv`  | `Int32`         | per-row, from tier config               |
| `opening_plies`     | `Int32`         | per-row, from tier config               |
| `sample_plies`      | `Int32`         | per-row, from tier config               |
| `temperature`       | `Float32`       | per-row, from tier config               |
| `worker_id`         | `Int16`         | which worker produced the row           |
| `game_seed`         | `Int64`         | per-game seed (reproduction key)        |
| `stockfish_version` | `String`        | from Stockfish's `id name` line         |

## Drop tripwire

If any tier ends with more than 0.1% of its games dropped (engine
couldn't tokenize a Stockfish move, or the game was empty), the run
fails loudly. Drops indicate either a real bug or an engine version
drift, not normal play.
