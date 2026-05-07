# Example configs

| File | Purpose | Tiers | Total games | `master_seed` |
|---|---|---|---|---|
| `smoke.json` | Local sanity test (vanilla SF). | 1 | 64 | 42 |
| `bench_evallegal_14k.json` | Local 14k-game benchmark for the searchless / evallegal protocol. | 1 | 14,000 | 42 |
| `sf_large_distill_20M.json` | Standalone NNUE-distillation dataset: searchless + `sample_score=v` + `net_selection=large` + `store_legal_move_evals=true`. | 1 | 20M | **142** |
| `stockfish_100m.json` | Full production ladder: tier-0 distillation + four `nodes_N` search-strength tiers. | 5 | 100M | 42 |
| `tournament_cp_vs_v_T0.json` | cp-vs-v argmax match config (separate `tournament` subcommand). | n/a | 2,000 | 42 |

## Seed convention

`master_seed` is **deliberately distinct** across configs whose tiers
overlap in shape. The seed cascade is

```
master_seed + tier_index → tier_seed
tier_seed   + worker_id  → worker_seed
worker_seed + game_index → game_seed
```

— so if two configs put a tier with identical parameters at the same
`tier_index` and use the same `master_seed`, every per-game seed
collides and the generated games are byte-for-byte identical (modulo
tier-name metadata). Running both configs into a shared output
directory would silently duplicate work.

The configs whose tier-0 specs are identical to each other
(`sf_large_distill_20M.json` and `stockfish_100m.json`'s tier 0) use
**different** master seeds — `142` vs `42` — so a user can run
`stockfish_100m.json` on one pod and `sf_large_distill_20M.json` on
another (or sequentially on the same pod) and get genuinely
non-overlapping data. The `master_seed` field is part of the per-tier
fingerprint, so a stray seed reset would be caught at resume time
rather than silently corrupting a tier.

If you derive a new config from one of these, **change the
`master_seed`** unless you specifically want to reproduce existing
output.

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
