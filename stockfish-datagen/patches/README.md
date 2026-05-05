# Stockfish patches

Patches we apply on top of upstream Stockfish to support `pawn`'s data
generation. Apply via `scripts/build_patched_stockfish.sh`.

## `0001-evallegal-uci-extension.patch`

Adds a new top-level UCI command `evallegal`. For each legal move at the
current position, the engine plays the move, runs `Eval::evaluate` on the
resulting position, and undoes the move — emitting one line summarizing
all of them. The standard `go` / search machinery is bypassed entirely;
this is just move-gen + per-child NNUE forward + undo.

### Why

For "tier 0" of the dataset we want the raw NNUE per-legal-move ranking —
what the static evaluator thinks of each child position, without any
tactical augmentation from quiescence search and without any of the
overhead of the search loop (thread spawn, `info ... multipv` lines,
TT/history bookkeeping, `bestmove` round-trip). `evallegal` gives that in
a single line.

### Behavior

- **Vanilla Stockfish 18 untouched.** No existing command, option, or
  data structure is modified — the patch only *adds* `evallegal` and a
  new `Engine::eval_legal()` method. All other tiers (`go nodes K`) work
  bit-identically to vanilla SF18.
- **In-check / tactical positions**: NNUE has no input feature for "in
  check" and is trained only on quiet positions. We still emit the eval
  for completeness, but the response leads with the `check` status so
  consumers can flag/discard those plies if they care.

### UCI grammar

```
evallegal
```

No arguments. Operates on whatever position was last set via `position`.

### Output format

Single line, always:

```
info string evallegal <status> [<uci> <cp>]...
```

- `status` is one of `none`, `check`, `mate`, `stalemate`.
- For `none` / `check`, the rest of the line is space-separated alternating
  `<uci>` / `<centipawn>` tokens, one pair per legal move.
- For `mate` / `stalemate`, no pairs follow (no legal moves to emit).
- Centipawns are mover-POV and use the same conversion as `info ... score
  cp N` lines from a normal search (`UCIEngine::to_cp` applied to the
  post-move evaluation, negated to flip back to the mover's perspective).

Example, startpos:

```
info string evallegal none a2a3 0 b2b3 0 c2c3 0 ... g1h3 0
```

Example, in-check position with one legal escape (Black king takes Qa7):

```
info string evallegal check a8a7 -3
```

Example, checkmate / stalemate:

```
info string evallegal mate
info string evallegal stalemate
```

### Patch base

Generated against Stockfish tag `sf_18` (commit
`cb3d4ee9b47d0c5aae855b12379378ea1439675c`, the SF18 release).

NNUE weights bundled at this tag: `nn-c288c895ea92.nnue` (big) and
`nn-37f18f62d772.nnue` (small) — identical to the SF18 release binary
shipped to users. Since the patch is purely additive, the patched binary
is bit-identical to vanilla SF18 for every command other than `evallegal`.
