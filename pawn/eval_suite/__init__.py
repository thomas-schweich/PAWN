"""PAWN evaluation suite — v2 surface.

The v2 eval lives in :mod:`pawn.eval` (move accuracy + per-phase),
:mod:`pawn.probes` (linear probes), :mod:`pawn.generation` (the 5
gated generation diagnostics), and :mod:`pawn.lichess_eval` (Maia-style
Elo-stratified accuracy). This package retains only the bits that
genuinely belong here:

- :mod:`pawn.eval_suite.diagnostics` — edge-case diagnostics via
  ``engine.edge_case_bits()``.
- :mod:`pawn.eval_suite.bounds` — theoretical accuracy bounds
  (polars position-parquet pipeline).
- :mod:`pawn.eval_suite.viz` — plotting helpers.
- :mod:`pawn.eval_suite.corpus` — the position-parquet pipeline.

Pre-vocab-transition v1 modules (`probes.py`, `generation.py`,
`lichess.py`, `worker.py`) that lived here in v1 are replaced by
`pawn.{probes,generation,lichess_eval}` at the top level — and are
not re-exported here to keep the import graph torch-free.
"""

from pawn.eval_suite import diagnostics

__all__ = ["diagnostics"]
