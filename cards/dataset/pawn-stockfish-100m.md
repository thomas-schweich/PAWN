---
license: cc-by-4.0
pretty_name: PAWN Stockfish 100M
task_categories:
  - other
tags:
  - chess
  - stockfish
  - nnue
  - distillation
  - policy-learning
size_categories:
  - 100M<n<1B
configs:
  - config_name: tier0_evallegal
    data_files:
      - split: train
        path: train/tier0_evallegal/*.parquet
      - split: validation
        path: val/tier0_evallegal/*.parquet
      - split: test
        path: test/tier0_evallegal/*.parquet
  - config_name: nodes_0001
    data_files:
      - split: train
        path: train/nodes_0001/*.parquet
      - split: validation
        path: val/nodes_0001/*.parquet
      - split: test
        path: test/nodes_0001/*.parquet
  - config_name: nodes_0128
    data_files:
      - split: train
        path: train/nodes_0128/*.parquet
      - split: validation
        path: val/nodes_0128/*.parquet
      - split: test
        path: test/nodes_0128/*.parquet
  - config_name: nodes_0256
    data_files:
      - split: train
        path: train/nodes_0256/*.parquet
      - split: validation
        path: val/nodes_0256/*.parquet
      - split: test
        path: test/nodes_0256/*.parquet
  - config_name: nodes_1024
    data_files:
      - split: train
        path: train/nodes_1024/*.parquet
      - split: validation
        path: val/nodes_1024/*.parquet
      - split: test
        path: test/nodes_1024/*.parquet
---

# PAWN Stockfish 100M

100,000,000 self-play chess games generated with **Stockfish 18**, each
annotated with per-position, per-legal-move evaluations. Built as training
data for [PAWN](https://github.com/thomas-schweich/PAWN) — a testbed for
finetuning and augmentation methods at small scale — but useful for any
chess policy-learning or NNUE-distillation work.

> **Engine note.** Games and evaluations were produced with a *patched*
> Stockfish build:
> [**thomas-schweich/stockfish-ml-extensions**](https://github.com/thomas-schweich/stockfish-ml-extensions).
> The patch adds two things this dataset depends on: an **`evallegal`**
> UCI protocol that runs a single raw-NNUE forward pass over *every* legal
> move and reports the full set of per-move evaluations, and a
> **`net_selection`** option that pins which NNUE network is used. Without
> this fork, references below to `evallegal`, `net_selection`, the
> searchless tier, and the raw per-head score fields (`score_psqt`,
> `score_positional`) would not make sense — they are all features of the
> fork, not of vanilla Stockfish.

The dataset is split into **5 tiers of 20,000,000 games each**. The tiers
share an identical game-sampling policy (temperature-0.5 softmax move
selection) but differ in how much *search* Stockfish was allowed per move,
from zero (a pure raw-network tier) up to a 1024-node search. Each tier is
exposed as a separate dataset config; every config carries `train`,
`validation`, and `test` splits (see [Splits](#splits)).

## Generation summary

- **Engine**: Stockfish 18, patched build
  ([stockfish-ml-extensions](https://github.com/thomas-schweich/stockfish-ml-extensions)).
  The version is pinned at generation time — a mismatch aborts before any
  data is written, since different SF releases ship different NNUE nets and
  would silently produce different games from the same seed.
- **Network**: Stockfish was configured to use **only its large NNUE
  network**, never the small net (`net_selection = large` on every tier).
  Stockfish 18 normally switches to a smaller, faster net on positions with
  large material imbalance; that switching is disabled here so every
  evaluation in the dataset comes from one and the same network. This makes
  the raw-eval columns a clean, single-network distillation target.
- **Move-selection policy**: at every ply the move actually played is
  sampled from a temperature-0.5 softmax over candidate evaluations (see
  the per-tier descriptions). Openings are widened: the first 2 plies of
  each game sample from the top 20 candidates rather than the top 5, to
  diversify game starts.
- **Determinism**: every game is reproducible from
  `(master_seed, tier_name, global_game_index)`. The seed hierarchy is
  `master_seed → tier_seed = mix(master_seed, sha256(tier.name)) →
  game_seed = mix(tier_seed, global_game_index)`. `master_seed = 42`.
- **Game length cap**: 512 plies (`max_ply`). The `tokens` column is the
  played move sequence; the longest games are truncated at 512.

## Tiers

Search depth is the only thing that varies across tiers.

| Tier (config) | Search budget | `multi_pv` | Games (total) | Mean game length (plies) |
|---|---|---|---|---|
| `tier0_evallegal` | none (searchless) | — | 20,000,000 | 268.0 |
| `nodes_0001` | `nodes = 1` | 5 | 20,000,000 | 153.8 |
| `nodes_0128` | `nodes = 128` | 5 | 20,000,000 | 143.5 |
| `nodes_0256` | `nodes = 256` | 5 | 20,000,000 | 126.1 |
| `nodes_1024` | `nodes = 1024` | 5 | 20,000,000 | 129.5 |

**Total: 100,000,000 games.** Mean game lengths are exact — see
*Statistics* below.

`nodes = N` is a hard cap on the number of search nodes Stockfish expands
per move. **`nodes = 1` is equivalent to `depth = 1`**: the search completes
exactly the first iteration of iterative deepening — the root's immediate
one-ply evaluation — and stops. Higher tiers (`128 / 256 / 1024`) let the
search go progressively deeper. As search deepens, play gets sharper and
games end faster (note the mean game length dropping from ~154 plies at
`nodes=1` to ~129 at `nodes=1024`), and the dominant outcome shifts from
draws toward decisive results.

### The searchless tier (`tier0_evallegal`)

`tier0_evallegal` does **no search at all**. At each position the patched
Stockfish runs its `evallegal` protocol: it evaluates *every legal move*
with a single raw NNUE forward pass (no tree, no lookahead) and returns the
full set of per-move evaluations.

The move played is then sampled from a **temperature-scaled softmax over
those raw, centipawn-adjusted network evaluations** — i.e. the policy for
this tier is, exactly:

```
P(move_i) = softmax( eval(move_i) / T )      with  T = 0.5
```

where `eval(move_i)` is the network's evaluation of the position after
`move_i`, mover-POV. There is no search ranking anywhere in this tier — it
is a pure, move-by-move readout of what the raw network "thinks", and the
played game is a sample from that raw-network policy. This makes
`tier0_evallegal` the reference tier for studying / cloning the network's
intrinsic positional judgment without any search on top.

## Eval columns

Every row carries two per-position, per-legal-move eval columns. Both are
`List<List<Struct>>` — outer list indexed by ply, inner list indexed by
move, each `Struct` being one `LegalMoveEval`:

```
LegalMoveEval {
  move_idx:         int16     # searchless_chess action-vocab index (0..1967)
  score_cp:         int16     # normalized centipawns, mover-POV
  score_eval_v:     int16?    # post-processed Value (what SF plays with)
  score_psqt:       int16?    # raw NNUE PSQT head output, mover-POV
  score_positional: int16?    # raw NNUE positional head output, mover-POV
}
```

`move_idx` indexes the same 1,968-entry searchless_chess action vocabulary
as the `tokens` column (see [Schema](#schema)).

### `legal_move_evals` — *MultiPV / search-ranked* (policy-learning target)

- On the **search tiers** (`nodes_0001` … `nodes_1024`): the **MultiPV
  search output** — Stockfish's top `multi_pv = 5` candidate moves as ranked
  by the (depth-limited) search. Only `score_cp` is populated; the three
  raw-NNUE fields are `null` (MultiPV reports normalized centipawns, not the
  network's internal head outputs).
- On the **searchless tier** (`tier0_evallegal`): the **full evallegal
  output** — *every* legal move, with all five score fields populated.

**Use case: policy learning.** On the search tiers this column is the
search-ranked top-k — a supervised target for "which moves does a real
(depth-limited) Stockfish search prefer here." Train a policy head against
it to imitate search-quality move selection.

### `static_legal_move_evals` — *raw network evals* (distillation target)

- On the **search tiers**: a *separate* full-legal-move `evallegal` call at
  every position — every legal move, all five score fields populated,
  captured independently of how the move was actually selected.
- On the **searchless tier**: `null` (it would exactly duplicate
  `legal_move_evals`, which is already the full raw-eval set — see the
  consumer convention below).

**Use case: direct network distillation.** This is the raw NNUE's verdict
on every legal move at every position — `score_eval_v` (the post-processed
Value), `score_psqt` and `score_positional` (the un-post-processed per-head
outputs). It is the right target for hot-swap NNUE-replacement distillation:
train a student to reproduce the network's per-move evaluations directly,
independent of search.

**Consumer convention**: to read "the canonical raw per-move eval" uniformly
across all tiers, use
`static_legal_move_evals if static_legal_move_evals is not None else legal_move_evals`.
On the searchless tier this falls back to `legal_move_evals` (which *is* the
raw eval there); on the search tiers it picks the dedicated raw-eval column.

## Splits

Every tier (config) is divided into three splits. The split is a
directory-level packaging of the tier's 10,000 shards — the held-out shards
are simply the last 50 of each tier:

| Split | Shards / tier | Shard ids | Games / tier | Total games |
|---|---|---|---|---|
| `train` | 9,950 | `s000000`–`s009949` | 19,900,000 | 99,500,000 |
| `validation` | 25 | `s009950`–`s009974` | 50,000 | 250,000 |
| `test` | 25 | `s009975`–`s009999` | 50,000 | 250,000 |

Each game is seeded independently by `mix(tier_seed, global_game_index)`,
and `global_game_index` is just an enumeration order with no correlation to
game content — so the tail shards are an i.i.d. sample of each tier, not a
biased slice. Shard ids are **not** renumbered per split: a shard's
`s<NNNNNN>` id is its seed / game-index key, so `validation` and `test`
keep their original high ids (see [Shard naming](#shard-naming-scheme)).

## Statistics

All counts below are **exact** unless marked *estimated*. The exact figures
were computed from the parquet file footers alone — every shard's
per-column `num_values` / `null_count` metadata — without reading a single
data page (~370 MB of footer metadata across all 50,000 shards). `move_idx`
is non-nullable inside each `LegalMoveEval`, so a leaf column's
`num_values − null_count` is the exact entry count.

### Games and positions (exact)

A "position" is one ply — a board state from which a move was chosen and
its legal moves evaluated.

| Tier | Games | Positions | Mean game length |
|---|---|---|---|
| `tier0_evallegal` | 20,000,000 | 5,359,412,373 | 268.0 |
| `nodes_0001` | 20,000,000 | 3,075,361,857 | 153.8 |
| `nodes_0128` | 20,000,000 | 2,870,813,480 | 143.5 |
| `nodes_0256` | 20,000,000 | 2,521,727,240 | 126.1 |
| `nodes_1024` | 20,000,000 | 2,589,518,029 | 129.5 |
| **Total** | **100,000,000** | **16,416,832,979** | 164.2 |

### Evaluation entries (exact)

| Tier | Raw network evals | MultiPV evals |
|---|---|---|
| `tier0_evallegal` | 131,022,823,122 | — |
| `nodes_0001` | 69,814,451,439 | 15,219,863,450 |
| `nodes_0128` | 68,964,296,966 | 14,215,676,259 |
| `nodes_0256` | 65,045,515,778 | 12,621,204,731 |
| `nodes_1024` | 70,196,200,065 | 13,046,231,906 |
| **Total** | **405,043,287,370** | **55,102,976,346** |

- **Raw network evals — 405,043,287,370** (`static_legal_move_evals` on the
  search tiers + `legal_move_evals` on the searchless tier). The
  direct-distillation targets.
- **MultiPV / search-ranked evals — 55,102,976,346** (`legal_move_evals` on
  the search tiers). The policy-learning targets.
- **Combined: 460,146,263,716 `LegalMoveEval` entries.**

On the search tiers MultiPV averages ~5.0 entries per position (`multi_pv =
5`; the first 2 plies widen to `opening_multi_pv = 20`, and positions with
fewer than 5 legal moves report fewer). Raw eval sets average ~23–27
entries per position — the mean number of legal moves.

On-disk size: **987 GB** of zstd-compressed parquet (294 GB
`tier0_evallegal`; 163–182 GB each search tier).

### Unique positions (estimated)

Evaluated positions are the board states before each played move; `tokens`
stores the moves, not the board states, so distinct-position counts are not
recoverable from metadata. They were estimated by replaying one 2,000-game
sample shard per tier (only the small `uci` column was downloaded) and
hashing each position (piece placement, side to move, castling, en passant):

| Tier | Sample positions | Distinct | Distinct % | Collisions vanish by depth |
|---|---|---|---|---|
| `tier0_evallegal` | 533,179 | 525,367 | 98.5% | ~5 |
| `nodes_0001` | 307,872 | 292,165 | 94.9% | ~7 |
| `nodes_0128` | 289,449 | 277,238 | 95.8% | ~8 |
| `nodes_0256` | 253,546 | 243,484 | 96.0% | ~8 |
| `nodes_1024` | 256,525 | 247,169 | 96.4% | ~7 |

Duplicate positions sit entirely in the opening: depth 0 is always the
start position, depth 1 has 20 distinct positions, depth 2 ~390, and from
roughly depth 8 onward every position in the sample is distinct.

This **does not extrapolate by a flat ×10,000**: the opening positions form
a small finite universe (a few million board states) shared by all
100,000,000 games, so adding games multiplies duplicate *instances*, not
distinct positions — the dataset-wide distinct fraction is therefore lower
than the per-shard 95–98%. Positions past the opening (the large majority,
given mean game lengths of 126–268 plies) are very nearly all distinct.
Netting the two, the dataset holds on the order of **15.5 billion distinct
evaluated positions (~94–95% of the 16.4 billion total)** — dominated by
the all-but-unique middlegame and endgame, with an opening-overlap
correction of order 0.8 billion. Treat this figure as a rough estimate.

## Usage

Each tier is a separate config; pick one with the config name, then a
split. The eval columns are large nested structures, so prefer **column
projection** (polars) or **streaming** (`datasets`) over materializing a
whole tier.

### Polars with column projection

Polars only downloads the columns you select — projecting away the eval
columns turns a ~750 GB dataset into a tiny moves-only feed:

```python
import polars as pl

# Moves-only view of the nodes_1024 training split
df = (
    pl.scan_parquet(
        "hf://datasets/thomas-schweich/pawn-stockfish-100m/train/nodes_1024/*.parquet"
    )
    .select(["tokens", "uci", "result", "game_length"])
    .head(50_000)
    .collect()
)
```

Pull the per-move evaluations only when you need them — here, the held-out
`validation` split of the searchless tier:

```python
import polars as pl

df = (
    pl.scan_parquet(
        "hf://datasets/thomas-schweich/pawn-stockfish-100m/val/tier0_evallegal/*.parquet"
    )
    .select(["tokens", "legal_move_evals"])
    .head(1_000)
    .collect()
)
```

### HuggingFace `datasets`

`name` selects the tier (config); `split` is one of `train` / `validation`
/ `test`:

```python
from datasets import load_dataset

# Stream the nodes_0256 training split
ds = load_dataset(
    "thomas-schweich/pawn-stockfish-100m",
    name="nodes_0256",
    split="train",
    streaming=True,
)
for game in ds.take(3):
    print(game["tokens"][:10], game["result"], game["game_length"])

# Small held-out split — fine to load fully
val = load_dataset(
    "thomas-schweich/pawn-stockfish-100m",
    name="nodes_0256",
    split="validation",
)
print(len(val), "games")
```

## File layout

```
train/
  <tier_name>/
    shard-s<NNNNNN>-r<NNNNNN>.parquet   # s000000 .. s009949
val/
  <tier_name>/
    shard-s<NNNNNN>-r<NNNNNN>.parquet   # s009950 .. s009974
test/
  <tier_name>/
    shard-s<NNNNNN>-r<NNNNNN>.parquet   # s009975 .. s009999
_meta/
  <tier_name>/
    _manifest.json        # canonical per-tier manifest (full shard list, fingerprint)
    _tier_state.json      # canonical per-tier generation state
```

The `_meta/` manifests describe the *generation* of each tier (all 10,000
shards, the config fingerprint) and are split-agnostic — the train / val /
test split is a packaging layer applied on top.

### Shard naming scheme

`shard-s<NNNNNN>-r<NNNNNN>.parquet`

- **`s<NNNNNN>`** — the global **shard id**, zero-padded to 6 digits. Shard
  `s` owns the contiguous global-game-index range
  `[s × 2000, (s + 1) × 2000)`. A tier has shards `s000000` … `s009999`,
  split across the three split directories (`train` holds `s000000`–
  `s009949`, `validation` `s009950`–`s009974`, `test` `s009975`–`s009999`).
  Ids are **not** renumbered per split.
- **`r<NNNNNN>`** — the **row count** of the shard, zero-padded to 6 digits.
  Every shard in this dataset is `r002000` (2,000 games). The row count is
  in the filename so tooling can read it from a directory listing without
  opening parquet metadata.

Example: `train/nodes_0128/shard-s004217-r002000.parquet` is shard 4217 of
the `nodes=128` tier, holding games with `global_game_index` 8,434,000 …
8,435,999.

## Schema

Per-row columns (parquet, zstd level 19):

| Column | Type | Notes |
|---|---|---|
| `tokens` | `List<int16>` | Played move sequence, one token per ply (variable length, up to 512). The vocabulary is the [searchless_chess action set](https://github.com/google-deepmind/searchless_chess): 1,968 reachable (src, dst[, promo]) tuples. `move_idx` in the eval structs uses this same index space. |
| `san` | `List<str>` | Same moves in SAN. |
| `uci` | `List<str>` | Same moves in UCI. |
| `game_length` | `int32` | Number of plies. |
| `outcome_token` | `int16` | Granular game outcome token. |
| `result` | `str` | `1-0` / `0-1` / `1/2-1/2`. |
| `nodes`, `multi_pv`, `opening_multi_pv`, `opening_plies`, `sample_plies`, `temperature` | scalars | Per-tier search/sampling config, denormalized per row. `null` on the searchless tier where not applicable. |
| `sample_score` | `str?` | `cp` / `v` — score scale used for sampling. |
| `net_selection` | `str?` | NNUE net pin — `large` for every row in this dataset. |
| `global_game_index` | `uint64` | Canonical per-tier game index; with `tier_name` it fully determines the game seed. |
| `game_seed` | `uint64` | Per-game seed (derived). |
| `stockfish_version` | `str` | `Stockfish 18`. |
| `legal_move_evals` | `List<List<Struct>>` | See *Eval columns*. |
| `static_legal_move_evals` | `List<List<Struct>>?` | See *Eval columns*. `null` on the searchless tier. |

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The games and
evaluations are machine-generated facts. Stockfish itself (and the
[stockfish-ml-extensions](https://github.com/thomas-schweich/stockfish-ml-extensions)
fork used to generate this data) is licensed GPLv3 but is not redistributed
here.

## Citation

If you use this dataset, please cite the PAWN project:
<https://github.com/thomas-schweich/PAWN>
