# Stockfish patches

Patches we apply on top of upstream Stockfish to support `pawn`'s data
generation. Apply via `scripts/build_patched_stockfish.sh`.

## `0001-searchless-uci-extension.patch`

Adds a `searchless` bare flag to UCI's `go` command. When present
(`go ... searchless`), the engine's `qsearch()` is short-circuited to return
the raw NNUE static eval instead of resolving captures.

### Why

For "tier 0" of the dataset we want pure NNUE per-legal-move ranking — what
the static evaluator thinks of each child position, without any tactical
augmentation from quiescence search. Combine `searchless` with `depth 1
multipv N` (where N ≥ legal moves count) to get a complete softmax-able
ranking of every legal move.

### Behavior

- **`searchless` flag absent**: engine behaves as vanilla Stockfish 18 —
  bit-identical, byte for byte. All existing tiers (`go nodes K`) work
  unchanged.
- **`searchless` present**: every `qsearch()` invocation returns the raw NNUE
  static eval. The recursive capture chase is bypassed.
- **In-check / tactical positions**: NNUE has no input feature for "in check"
  and is trained only on quiet positions. Static evals on in-check (and
  generally tactical) positions are out-of-distribution. The patch still
  returns whatever NNUE says — no crash, but expect the score to be
  unreliable. This is the documented tradeoff for having a single-source
  pure-NNUE dataset; the alternative ("fall back to qsearch when in check")
  was rejected because tactical positions are *also* technically OOD, and
  the line is artificial.

### UCI grammar

`searchless` is a **bare flag**, matching `infinite` and `ponder`. Pass it
without a value:

```
go depth 1 multipv 256 searchless
```

### Patch base

Generated against Stockfish tag `sf_18` (commit
`cb3d4ee9b47d0c5aae855b12379378ea1439675c`, the SF18 release).

NNUE weights bundled at this tag: `nn-c288c895ea92.nnue` (big) and
`nn-37f18f62d772.nnue` (small) — identical to the SF18 release binary
shipped to users. The patched-no-flag behavior is bit-identical to vanilla
SF18, so existing tiers using `go nodes K` get the exact same evaluations
as before.
