---
license: cc-by-4.0
pretty_name: PAWN Stockfish 100M
task_categories:
  - other
tags:
  - chess
  - pawn
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

100,000,000 self-play chess games generated with Stockfish 18, each
annotated with per-position, per-legal-move evaluations — for chess
policy-learning and NNUE-distillation research.

## Dataset Description

- **Repository:** [PAWN](https://github.com/thomas-schweich/PAWN)
- **Generation engine:** [stockfish-ml-extensions](https://github.com/thomas-schweich/stockfish-ml-extensions) — a patched Stockfish 18
- **Point of Contact:** Thomas Schweich

### Dataset Summary

100,000,000 machine-generated self-play chess games. Every position in
every game is annotated with an evaluation of *every legal move*, not just
the move played. The dataset was built as training data for
[PAWN](https://github.com/thomas-schweich/PAWN) — a testbed for finetuning
and augmentation methods at small scale — but is useful for any chess
policy-learning or NNUE-distillation work.

Games were produced with a patched build of Stockfish 18
([stockfish-ml-extensions](https://github.com/thomas-schweich/stockfish-ml-extensions)).
The patch adds two things the dataset depends on: an `evallegal` UCI
protocol that runs a single raw-NNUE forward pass over every legal move and
reports the full per-move evaluations, and a `net_selection` option that
pins which NNUE network is used.

The dataset is divided into **5 tiers of 20,000,000 games each**, exposed
as five dataset configs. The tiers share an identical move-sampling policy
but differ in how much *search* Stockfish was allowed per move — from zero
(a pure raw-network tier) up to a 1024-node search. Each tier carries
`train` / `validation` / `test` splits.

Stockfish ran single-threaded, so its search is fully deterministic —
left to itself, self-play would replay one fixed game. Variability is
therefore injected deliberately: each move is sampled from a
temperature-`T` softmax over the candidate evaluations (`T = 0.5` for all
tiers). Because that sampler is the only source of randomness and it is
driven by a per-game seeded PRNG, every game stays exactly reproducible.

### Supported Tasks

- **Policy learning** — train a policy head to imitate search-quality move
  selection, supervised by the `legal_move_evals` (MultiPV / search-ranked)
  column. Note that on the search tiers `legal_move_evals` ranks only
  Stockfish's top `multi_pv = 5` moves, so sixth-best-or-worse moves have
  no search-ranked target there — scoring predictions of them is
  design-dependent. (`static_legal_move_evals` still covers every legal
  move on the search tiers, and the searchless tier's `legal_move_evals`
  covers every move.)
- **NNUE distillation** — train a student network to reproduce the raw
  network's per-move evaluations, supervised by the `static_legal_move_evals`
  (raw-eval) column for search tiers and/or the `legal_move_evals` column for the searchless tier. Raw evaluations are provided for every legal move in every position.

## Dataset Structure

### Data Instances

Each row is one complete game. The move-sequence columns (`tokens` / `san` /
`uci`) and the two evaluation columns are all per-ply lists of length
`game_length`. An abbreviated row:

```json
{
  "tokens": [387, 1102, 945, "..."],
  "san": ["e4", "c5", "Nf3", "..."],
  "uci": ["e2e4", "c7c5", "g1f3", "..."],
  "game_length": 142,
  "result": "1-0",
  "outcome_token": 1969,
  "nodes": 128,
  "multi_pv": 5,
  "temperature": 0.5,
  "sample_score": "cp",
  "net_selection": "large",
  "stockfish_version": "Stockfish 18",
  "global_game_index": 8434217,
  "game_seed": 14072583910256044311,
  "legal_move_evals":        [[{"move_idx": 945, "score_cp": 31}, "..."], "..."],
  "static_legal_move_evals": [[{"move_idx": 945, "score_cp": 28, "score_eval_v": 33}, "..."], "..."]
}
```

Per-position FENs and Zobrist hashes are not included as they would balloon the size of the dataset, but they are fast to compute on-the-fly if needed — see [Computing FENs and Zobrist hashes](#computing-fens-and-zobrist-hashes).

### Data Fields

Per-row columns (parquet, zstd level 19):

| Column | Type | Notes |
|---|---|---|
| `tokens` | `List<int16>` | Played move sequence, one token per ply (variable length, up to 512). The vocabulary is the [searchless_chess](https://github.com/google-deepmind/searchless_chess) action set: 1,968 reachable (src, dst[, promo]) tuples. `move_idx` in the eval structs uses this same index space. See [`searchless_chess_vocabulary.json` in the PAWN project](https://github.com/thomas-schweich/PAWN/blob/main/searchless_chess_vocabulary.json) for the full enumeration. |
| `san` | `List<str>` | Same moves in SAN. |
| `uci` | `List<str>` | Same moves in UCI. |
| `game_length` | `uint16` | Number of plies. |
| `outcome_token` | `uint16` | Granular game outcome token. |
| `result` | `str` | `1-0` / `0-1` / `1/2-1/2`. |
| `nodes`, `multi_pv`, `opening_multi_pv`, `opening_plies`, `sample_plies` | `int32?` | Per-tier search config, denormalized per row; `null` on the searchless tier. `sample_plies` is the number of leading plies that use softmax sampling before play switches to top-1 — `999` on every tier here, so sampling runs the whole game. |
| `temperature` | `float32` | Softmax sampling temperature; `0.5` on every tier, never null. |
| `sample_score` | `str?` | `cp` / `v` — score scale used for sampling. |
| `net_selection` | `str?` | NNUE net pin — `large` for every row in this dataset. |
| `global_game_index` | `uint64` | Canonical per-tier game index; together with the tier name (the directory the shard lives in) it fully determines the game seed. |
| `game_seed` | `uint64` | Per-game seed (derived). |
| `stockfish_version` | `str` | `Stockfish 18`. |
| `legal_move_evals` | `List<List<Struct>>` | MultiPV / search-ranked per-move evaluations — see [Annotation Process](#annotation-process). |
| `static_legal_move_evals` | `List<List<Struct>>?` | Raw network per-move evaluations — see [Annotation Process](#annotation-process). `null` on the searchless tier. |

The two evaluation columns are both `List<List<Struct>>` — the outer list
indexed by ply, the inner list by move — and each `Struct` is one
`LegalMoveEval`:

```
LegalMoveEval {
  move_idx:         int16     # searchless_chess action-vocab index (0..1967)
  score_cp:         int16     # normalized centipawns, mover-POV
  score_eval_v:     int16?    # post-processed Value (what SF plays with)
  score_psqt:       int16?    # raw NNUE PSQT head output, mover-POV
  score_positional: int16?    # raw NNUE positional head output, mover-POV
}
```

### Data Splits

The dataset has two axes: **5 tiers** (dataset configs) and, within each
tier, **3 splits** (`train` / `validation` / `test`).

The five tiers each hold 20,000,000 games and differ only in search budget
(see [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)):

| Tier (config) | Search budget | `multi_pv` | Games | Positions | Mean game length |
|---|---|---|---|---|---|
| `tier0_evallegal` | none (searchless) | — | 20,000,000 | 5,359,412,373 | 268.0 |
| `nodes_0001` | `nodes = 1` | 5 | 20,000,000 | 3,075,361,857 | 153.8 |
| `nodes_0128` | `nodes = 128` | 5 | 20,000,000 | 2,870,813,480 | 143.5 |
| `nodes_0256` | `nodes = 256` | 5 | 20,000,000 | 2,521,727,240 | 126.1 |
| `nodes_1024` | `nodes = 1024` | 5 | 20,000,000 | 2,589,518,029 | 129.5 |
| **Total** | | | **100,000,000** | **16,416,832,979** | 164.2 |

A "position" is one ply — a board state from which a move was chosen and
its legal moves evaluated.

Each tier's 10,000 shards (2,000 games each) are partitioned into splits by
holding out the last 50 shards:

| Split | Shards / tier | Shard ids | Games / tier | Total games |
|---|---|---|---|---|
| `train` | 9,950 | `s000000`–`s009949` | 19,900,000 | 99,500,000 |
| `validation` | 25 | `s009950`–`s009974` | 50,000 | 250,000 |
| `test` | 25 | `s009975`–`s009999` | 50,000 | 250,000 |

The split is a clean i.i.d. holdout: each game is seeded independently by
`mix(tier_seed, global_game_index)`, and `global_game_index` is an
enumeration order with no correlation to game content, so the tail shards
are a uniform sample of each tier. Shard ids are **not** renumbered per
split — a shard's `s<NNNNNN>` id is its seed / game-index key, so
`validation` and `test` keep their original high ids.

**File layout:**

```
train/<tier_name>/shard-s<NNNNNN>-r<NNNNNN>.parquet   # s000000 .. s009949
val/<tier_name>/shard-s<NNNNNN>-r<NNNNNN>.parquet     # s009950 .. s009974
test/<tier_name>/shard-s<NNNNNN>-r<NNNNNN>.parquet    # s009975 .. s009999
_meta/<tier_name>/_manifest.json                      # canonical per-tier manifest
_meta/<tier_name>/_tier_state.json                    # canonical per-tier generation state
```

In `shard-s<NNNNNN>-r<NNNNNN>.parquet`, `s<NNNNNN>` is the global shard id
(shard `s` owns global-game-index range `[s × 2000, (s + 1) × 2000)`) and
`r<NNNNNN>` is the row count (always `r002000`). The `_meta/` manifests
describe the generation of each tier (all 10,000 shards, the config
fingerprint) and are split-agnostic — the train/val/test split is a
packaging layer applied on top.

### Statistics

Counts of positions and evaluation entries are exact, computed from the
parquet footer metadata (per-column `num_values` / `null_count`). The
unique-position counts are HyperLogLog++ measurements (≈ 0.2% standard
error).

**Evaluation entries** — total `LegalMoveEval` structs:

| Tier | Raw network evals | MultiPV evals |
|---|---|---|
| `tier0_evallegal` | 131,022,823,122 | — |
| `nodes_0001` | 69,814,451,439 | 15,219,863,450 |
| `nodes_0128` | 68,964,296,966 | 14,215,676,259 |
| `nodes_0256` | 65,045,515,778 | 12,621,204,731 |
| `nodes_1024` | 70,196,200,065 | 13,046,231,906 |
| **Total** | **405,043,287,370** | **55,102,976,346** |

Combined, **460,146,263,716** `LegalMoveEval` entries — 405.0 B raw-network
evals (the direct-distillation target) and 55.1 B MultiPV evals (the
policy-learning target). On the search tiers MultiPV averages ~5.0 entries
per position; raw eval sets average ~23–27 entries per position (the mean
number of legal moves).

**On-disk size:** 987 GB of zstd-compressed parquet (294 GB
`tier0_evallegal`; 163–182 GB each search tier).

**Unique positions: 13,447,893,206** distinct board states among the
16.4 billion evaluated positions — ≈ 82%; the other ~18% are positions
revisited across games, dominated by shared openings. This is also the
count of unique raw evaluations, since a raw NNUE eval is a pure function
of the position. Of these, **8,690,806,916** distinct positions carry a
MultiPV evaluation (the four search tiers; tier 0 is searchless). Both
figures were measured by replaying every game and estimating distinct
Zobrist-hashed board states with HyperLogLog++.

## Usage

Each tier is a separate config; select one by config name, then a split.
The evaluation columns are large nested structures, so prefer **column
projection** (polars) or **streaming** (`datasets`) over materializing a
whole tier.

### Polars with column projection

Polars only downloads the columns you select — projecting away the eval
columns turns the ~987 GB dataset into a small moves-only feed:

```python
import polars as pl

# Moves-only view of the nodes_1024 training split
df = (
    pl.scan_parquet(
        "hf://datasets/thomas-schweich/pawn-stockfish-100m/train/nodes_1024/*.parquet"
    )
    .select(["tokens", "result", "game_length"])
    .head(50_000)
    .collect()
)

# Pull per-move evaluations only when needed
evals = (
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

### Computing FENs and Zobrist hashes

The dataset stores moves, not board states (see [Data Instances](#data-instances)),
so per-position FENs and Zobrist hashes are most easily reconstructed by replaying the `uci` column.

**Python** ([`python-chess`](https://python-chess.readthedocs.io/)):

```python
import chess
import chess.polyglot

def evaluated_positions(uci_moves):
    """Yield (fen, zobrist_key) for each evaluated position in a game."""
    board = chess.Board()
    for uci in uci_moves:
        yield board.fen(), chess.polyglot.zobrist_hash(board)
        board.push_uci(uci)
```

**Rust** ([`shakmaty`](https://docs.rs/shakmaty/0.30)):

```rust
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, EnPassantMode, Position};

/// (fen, zobrist_key) for each evaluated position in a game.
fn evaluated_positions(uci_moves: &[String]) -> Vec<(String, u64)> {
    let mut pos = Chess::default();
    let mut out = Vec::with_capacity(uci_moves.len());
    for uci in uci_moves {
        let fen = Fen::try_from(pos.to_setup(EnPassantMode::Legal))
            .expect("a legal position always produces a valid FEN")
            .to_string();
        let key: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
        out.push((fen, key.0));
        let mv = uci
            .parse::<UciMove>()
            .expect("dataset uci is well-formed")
            .to_move(&pos)
            .expect("dataset uci is legal in-position");
        pos.play_unchecked(mv);
    }
    out
}
```

Note: the two hashes are **not interchangeable**: `python-chess` uses Polyglot
Zobrist keys, `shakmaty` its own `Zobrist64` key set. Either is fine for
deduplicating positions within your own run, but the
[unique-position counts](#statistics) were measured with `shakmaty`'s
`Zobrist64`.

## Dataset Creation

### Curation Rationale

PAWN studies finetuning and augmentation methods on small chess models, and
needed a large, perfectly-reproducible corpus that pairs each position with
a *dense* supervision signal — every legal move scored — rather than just
the single move played. Two complementary signals were wanted from one
corpus: the raw NNUE evaluation of every move (a clean, search-free
distillation target) and the move ranking from a real depth-limited search
(a policy-learning target). The 5-tier search-budget ladder lets a consumer
study how supervision quality scales with search effort.

### Source Data

#### Initial Data Collection and Normalization

All games are Stockfish 18 self-play. At every ply the move played is
sampled from a **temperature-0.5 softmax** over candidate evaluations; the
first 2 plies of each game widen to the top 20 candidates (rather than the
top 5) to diversify openings. Games are capped at 512 plies; a truncated
game is tagged with the `PLY_LIMIT` outcome.

**Search-budget tiers.** The five tiers differ only in how much search
Stockfish runs per move. `nodes = N` is a hard cap on the number of search
nodes expanded per move. `nodes = 1` is equivalent to `depth = 1`: the
search completes exactly the first iteration of iterative deepening — the
root's immediate one-ply evaluation — and stops. Higher tiers
(`128 / 256 / 1024`) let the search go progressively deeper. As search
deepens, play gets sharper, games end faster (mean game length drops from
~154 plies at `nodes=1` to ~129 at `nodes=1024`), and the dominant outcome
shifts from draws toward decisive results.

`tier0_evallegal` does **no search at all**. At each position the patched
Stockfish runs its `evallegal` protocol: it evaluates *every legal move*
with a single raw NNUE forward pass (no tree, no lookahead). The move
played is then sampled from a temperature-scaled softmax over those raw,
centipawn-adjusted network evaluations — the policy for this tier is,
exactly:

```
P(move_i) = softmax( eval(move_i) / T )      with  T = 0.5
```

where `eval(move_i)` is the network's evaluation of the position after
`move_i`, mover-POV. There is no search ranking anywhere in this tier — it
is a pure, move-by-move readout of the raw network's judgment, and the
played game is a sample from that raw-network policy.

**Network.** Stockfish was configured to use only its large NNUE network,
never the small net (`net_selection = large` on every tier). Stockfish 18
normally switches to a smaller net on positions with large material
imbalance; that switching is disabled here so every evaluation comes from
one and the same network — making the raw-eval column a clean,
single-network distillation target.

**Determinism.** Every game is reproducible from
`(master_seed, tier_name, global_game_index)`. The seed hierarchy is
`master_seed → tier_seed = mix(master_seed, sha256(tier.name)) → game_seed
= mix(tier_seed, global_game_index)`, with `master_seed = 42`. The Stockfish
version is pinned at generation time — a mismatch aborts before any data is
written, since different releases ship different NNUE nets.

#### Who Produced the Data

The data is entirely machine-generated. Games and evaluations were produced
by self-play of the patched Stockfish 18 NNUE engine; no humans were
involved in producing or annotating the games.

### Annotations

#### Annotation Process

Each position is annotated with evaluations of its legal moves, in two
columns. Both are `List<List<Struct>>` — outer list per ply, inner list per
move, each struct a `LegalMoveEval` (see [Data Fields](#data-fields)).

**`legal_move_evals` — MultiPV / search-ranked.** On the search tiers
(`nodes_0001` … `nodes_1024`) this is the MultiPV search output: Stockfish's
top `multi_pv = 5` candidate moves as ranked by the depth-limited search.
Only `score_cp` is populated; the three raw-NNUE fields are `null` (MultiPV
reports normalized centipawns, not the network's internal head outputs). On
the searchless tier it is the full `evallegal` output — *every* legal move,
with all five struct fields populated. This column is the **policy-learning**
target: which moves does a real depth-limited search prefer.

**`static_legal_move_evals` — raw network evals.** On the search tiers this
is a *separate* full-legal-move `evallegal` call at every position — every
legal move, all five struct fields populated, captured independently of how
the move was actually selected. On the searchless tier it is `null` (it
would exactly duplicate `legal_move_evals`). This column is the
**distillation** target: the raw NNUE's verdict on every legal move —
`score_eval_v` (the post-processed Value), `score_psqt` and
`score_positional` (the un-post-processed per-head outputs).

To read "the canonical raw per-move eval" uniformly across all tiers, use
`static_legal_move_evals if static_legal_move_evals is not None else
legal_move_evals` — on the searchless tier this falls back to
`legal_move_evals` (which *is* the raw eval there).

#### Who Are the Annotators

The annotations are machine-generated by the patched Stockfish 18 NNUE
engine — the same engine that produced the games.

### Personal and Sensitive Information

None. The dataset is entirely machine-generated synthetic self-play; it
contains no personal data and no human-authored content.

## Considerations for Using the Data

### Discussion of Biases

The games are neither human play nor uniform-random: every move is sampled
from a temperature-0.5 softmax over Stockfish's own candidate set, so the
move distribution reflects Stockfish 18's NNUE preferences. The search
tiers additionally skew toward sharper, more decisive play as the node
budget grows. Opening variety is deliberately widened (top-20 sampling for
the first 2 plies), but the opening distribution is still Stockfish-shaped,
not a uniform or human-frequency opening book.

### Other Known Limitations

- **MultiPV is path-dependent.** Stockfish's transposition table carries
  over between the moves of a game (it is cleared per game, not per move),
  so the `legal_move_evals` (MultiPV) output for a position depends on the
  game prefix that reached it, not on the position alone. The raw
  `static_legal_move_evals` column is search-free and has no such
  dependence.
- **Game-length cap.** Games are truncated at 512 plies; a truncated game
  carries the `PLY_LIMIT` outcome.
- **Position overlap.** ~82% of the 16.4 billion evaluated positions are
  distinct (see [Statistics](#statistics)); the other ~18% are positions
  revisited across games — mostly shared openings, but also common
  middlegame and endgame positions.

## Additional Information

### Licensing Information

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Citation Information

If you use this dataset, please cite the PAWN project:

```
PAWN — Playstyle-Agnostic World-model Network for Chess.
https://github.com/thomas-schweich/PAWN
```

### Dataset Curators

Generated and maintained by Thomas Schweich as part of the
[PAWN](https://github.com/thomas-schweich/PAWN) project.
