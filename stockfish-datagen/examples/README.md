# Example configs

| File | Purpose | Tiers | Total games | `master_seed` |
|---|---|---|---|---|
| `smoke.json` | Local sanity test (vanilla SF). | 1 | 64 | 42 |
| `bench_evallegal_14k.json` | Local 14k-game benchmark for the searchless / evallegal protocol. | 1 | 14,000 | 42 |
| `sf_large_distill_20M.json` | Standalone NNUE-distillation dataset: searchless + `sample_score=v` + `net_selection=large` + `store_legal_move_evals=true`. | 1 | 20M | **142** |
| `stockfish_100m.json` | Full production ladder: tier-0 distillation + four `nodes_N` search-strength tiers. | 5 | 100M | 42 |
| `tournament_cp_vs_v_T0.json` | cp-vs-v argmax match config (separate `tournament` subcommand). | n/a | 2,000 | 42 |

## Seed convention

A tier's per-game seeds come from the cascade

```
master_seed            → tier_seed = mix(master_seed, sha256(tier.name))
tier_seed + game_index → game_seed = mix(tier_seed, global_game_index)
```

A tier's identity is its `name` — not its position in the `tiers:`
list — and `n_workers` plays no part. Two tiers generate the same games
when they share a `name`, a `master_seed`, and a tier config (run
against the same `stockfish_version` and `max_ply`).

`sf_large_distill_20M.json` names its tier `large_distill` and uses
`master_seed = 142`, so it never overlaps the `stockfish_100m.json`
production ladder in either name or seed.

`smoke.json` and `bench_evallegal_14k.json` reuse production tier names
(`nodes_0001`, `tier0_evallegal`) with `master_seed = 42`. Each writes
to its own `/tmp` output directory, so neither collides with a
production run on disk. `smoke.json` also differs in content — its tier
raises `temperature` and leaves the net unpinned — whereas
`bench_evallegal_14k.json`'s tier config matches production tier 0
exactly, making its 14,000 games the leading prefix of that 20M-game
tier.

If you derive a new config and want its output to be distinct, change
the `master_seed` or the tier `name`. Both feed the per-tier
fingerprint, so changing one for a tier already underway is caught at
resume rather than silently mixing two seedings into one tier.

## Shard size + memory

`shard_size_games` is the per-worker buffer size before flushing one
parquet file. The current `ShardWriter` accumulates rows in Arrow
column builders for an entire shard before writing — fine for
search-mode tiers (a few hundred bytes per row) but heavy for tiers
that set `store_legal_move_evals: true`, where each row carries
~200 plies × ~19 legal-move evals × 6 bytes ≈ 23 KB.

Memory budget per worker = `shard_size_games × ~23 KB`. At 126 workers,
total peak buffer ≈ that × 126. The bundled production configs
(`stockfish_100m.json`, `sf_large_distill_20M.json`) use **2000** —
~46 MB / worker, ~5.8 GB total — leaving comfortable headroom on
typical 256 GB+ vast.ai pods. The trade is more shard files: 100M
games / 2000 = ~50K shards across all workers (vs 10K at the older
shard_size).

Don't push this above 5000 for `store_legal_move_evals: true` tiers
on a 126-worker pod without first checking pod RAM. Search-mode tiers
without legal-move-eval storage tolerate much larger shards but live
under the same global setting; lower wins.
