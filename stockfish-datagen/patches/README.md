# Stockfish extensions

The patched Stockfish binary used by `pawn`'s tier-0 data generation lives
in a separate repo:

  **<https://github.com/thomas-schweich/stockfish-ml-extensions>**

That fork pins Stockfish to the SF18 release (commit `cb3d4ee9`, tag `sf_18`)
and adds two purely-additive extensions:

- **`evallegal`** UCI command — per-legal-move static eval emitted on a
  single line, with four scores per move:
  - `cp`: normalized centipawn (`UCIEngine::to_cp(eval_v, pos)`).
  - `eval_v`: `Eval::evaluate`'s post-processed Value before `to_cp` —
    head-blend + complexity damp + material/optimism mix + 50-move
    shuffling damp + TB-clamp baked in. **What Stockfish plays with**;
    the right target for play-policy distillation.
  - `psqt`, `positional`: raw NNUE per-head outputs, before any
    post-processing. **The right targets for hot-swap NNUE-replacement
    distillation** — Stockfish itself applies the post-processing on top,
    so the student must not have it baked in.
- **`NetSelection`** UCI option (`auto | small | large`, default `auto`) —
  forces uniform use of one NNUE network across all evaluation, useful for
  distillation labelling where a per-position dynamic small/large pick
  introduces eval-source heterogeneity.

Splitting the fork out keeps the GPLv3 boundary clean (it's a derivative of
Stockfish, licensed accordingly; PAWN itself stays Apache 2.0) and lets the
extensions be useful to other projects without dragging the rest of PAWN
along.

`scripts/build_patched_stockfish.sh` clones the fork at the pinned commit
SHA (`14f92699…`, the commit annotated tag `sf_18-v0.3.0` currently points
at) and runs `make -j build ARCH=x86-64-avx2`. The resulting binary is
dropped at `stockfish-datagen/stockfish-patched`. The Dockerfile uses
the same SHA pin (a lightweight tag could be force-moved on the remote
and silently change the binary).

The runner's preflight checks gate startup on three distinct binary
capabilities, all detected during the UCI handshake:

- **`evallegal` UCI command + v0.3.0 output shape** — required if any
  tier sets `searchless: true` or `net_selection: <X>`. Vanilla SF
  responds `Unknown command` to the post-handshake probe. Patched binaries
  emit `info string evallegal …` lines, but earlier fork builds
  (`sf_18-v0.1.0` / `sf_18-v0.2.0`) used a 3-tuple shape (`<uci> <cp> <v>`)
  that's incompatible with the current parser's strict 5-tuple expectation
  (`<uci> <cp> <eval_v> <psqt> <positional>`). The probe verifies the
  parser yields exactly 20 candidates at startpos (the invariant legal-move
  count) — older patched binaries silently parse to 0 candidates and would
  crash a worker mid-run with `NoCandidates`. Failing fast at preflight is
  much louder. `is_patched` on `StockfishProcess` means
  "patched **and** v0.3.0+ output shape" since the upgrade.
- **`option name NetSelection`** — required if any tier sets
  `net_selection: <X>`. Older fork builds (pre-`sf_18-v0.2.0`) advertise
  `evallegal` but not this option. UCI silently ignores unknown
  setoption names, so without this gate a `setoption name NetSelection
  value large` would no-op while shard fingerprints claimed the
  requested choice. The preflight rejects this combination loudly.

The `tournament` subcommand has a parallel
`preflight_check_tournament_binary` that probes only for `evallegal`
(tournament workers always drive that protocol regardless of which
`sample_score` either side picks; no per-side `net_selection` exists),
so the v0.3.0 shape check is sufficient there too. Either way, an
unpatched or stale `stockfish_path` aborts startup before any worker
spawns.

## Output format (reference)

```
info string evallegal <status> [<uci> <cp> <eval_v> <psqt> <positional>]...
```

- `<status>` ∈ `none | check | mate | stalemate`. For `none` / `check`, the
  rest of the line is space-separated 5-tuples
  `<uci> <cp> <eval_v> <psqt> <positional>`, one per legal move. For
  `mate` / `stalemate`, no tuples follow.
- `<cp>` = `UCIEngine::to_cp(eval_v, pos)` — same conversion as a normal
  search's `info ... score cp N` lines (100 cp ≈ "1 pawn equivalent"
  regardless of material).
- `<eval_v>` = `Eval::evaluate`'s post-processed `Value` before `to_cp`.
  The right target for **play-policy distillation** (student plays chess).
  The ratio `eval_v / cp` varies with the position's win-rate-model `a`
  parameter (typically ~3–5×) plus rule50 / complexity / material adjustments.
- `<psqt>`, `<positional>` = raw NNUE per-head outputs from
  `Networks::evaluate()`, before any post-processing. The right targets
  for **hot-swap NNUE-replacement distillation** (student replaces a
  `Networks` member; Stockfish applies the post-processing on top, so
  the student must not have it baked in).
- All four scores are mover-POV (negated from the post-move side-to-move
  POV that `Eval::evaluate_with_components` returns).
