# Stockfish extensions

The patched Stockfish binary used by `pawn`'s tier-0 data generation lives
in a separate repo:

  **<https://github.com/thomas-schweich/stockfish-ml-extensions>**

That fork pins Stockfish to the SF18 release (commit `cb3d4ee9`, tag `sf_18`)
and adds two purely-additive extensions:

- **`evallegal`** UCI command — per-legal-move NNUE evals (both normalized
  cp and raw `v`) on a single line.
- **`NetSelection`** UCI option (`auto | small | large`, default `auto`) —
  forces uniform use of one NNUE network across all evaluation, useful for
  distillation labelling where a per-position dynamic small/large pick
  introduces eval-source heterogeneity.

Splitting the fork out keeps the GPLv3 boundary clean (it's a derivative of
Stockfish, licensed accordingly; PAWN itself stays Apache 2.0) and lets the
extensions be useful to other projects without dragging the rest of PAWN
along.

`scripts/build_patched_stockfish.sh` clones the fork at the pinned commit
SHA (`777b8807…`, the commit annotated tag `sf18-v0.2.0` currently points
at) and runs `make -j build ARCH=x86-64-avx2`. The resulting binary is
dropped at `stockfish-datagen/stockfish-patched`. The Dockerfile uses
the same SHA pin (a lightweight tag could be force-moved on the remote
and silently change the binary).

The runner's preflight check (`main.rs::preflight_check_patched_binary`)
gates startup on two distinct binary capabilities, both detected during
the UCI handshake:

- `evallegal` UCI command — required if any tier sets `searchless: true`
  or `net_selection: <X>`. Vanilla SF responds `Unknown command` to the
  post-handshake probe; the patched binary returns the expected
  `info string evallegal …` line.
- `option name NetSelection` — required if any tier sets
  `net_selection: <X>`. Older fork builds (`sf18-v0.1.0`) advertise
  `evallegal` but not this option. UCI silently ignores unknown
  setoption names, so without this gate a `setoption name NetSelection
  value large` would no-op while shard fingerprints claimed the
  requested choice. The preflight rejects this combination loudly.

The `tournament` subcommand has a parallel
`preflight_check_tournament_binary` that probes only for `evallegal`
(tournament workers always drive that protocol regardless of which
`sample_score` either side picks; no per-side `net_selection` exists),
so a `sf18-v0.1.0` fork build is sufficient there. Either way, an
unpatched `stockfish_path` aborts startup before any worker spawns.

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
