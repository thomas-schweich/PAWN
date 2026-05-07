# Stockfish extensions

The patched Stockfish binary used by `pawn`'s tier-0 data generation lives
in a separate repo:

  **<https://github.com/thomas-schweich/stockfish-ml-extensions>**

That fork pins Stockfish to the SF18 release (commit `cb3d4ee9`, tag `sf_18`)
and adds a single additive UCI command — `evallegal` — that emits per-legal-
move NNUE evals (both normalized cp and raw `v`) on a single line. Splitting
the fork out keeps the GPLv3 boundary clean (it's a derivative of Stockfish,
licensed accordingly; PAWN itself stays Apache 2.0) and lets the extension be
useful to other projects without dragging the rest of PAWN along.

`scripts/build_patched_stockfish.sh` clones the fork at the pinned tag
(`v18.evallegal.0`) and builds it with PGO. The resulting binary is dropped
at `stockfish-datagen/stockfish-patched`. The runner's preflight check
(`main.rs::preflight_check_patched_binary`) probes for the `evallegal`
command at startup and aborts loudly if a tier with `searchless: true` is
configured against an unpatched binary.

## Output format (reference)

```
info string evallegal <status> [<uci> <cp> <v>]...
```

- `<status>` ∈ `none | check | mate | stalemate`. For `none` / `check`, the
  rest of the line is space-separated `<uci> <cp> <v>` triplets, one per
  legal move. For `mate` / `stalemate`, no triplets follow.
- `<cp>` = `UCIEngine::to_cp(v, pos)` — same conversion as a normal search's
  `info ... score cp N` lines (100 cp ≈ "1 pawn equivalent" regardless of
  material).
- `<v>` = raw internal `Value` the NNUE produced before normalization. The
  ratio `v / cp` varies with the position's win-rate-model `a` parameter
  (typically ~3–5×).
- Both scores are mover-POV (negated from the post-move side-to-move POV
  that `Eval::evaluate` returns).
