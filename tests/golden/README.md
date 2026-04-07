# Golden reference files

Pinned deterministic outputs used as regression snapshots.

## Layout

Each partition owns a subdirectory:

- `rust_core/` — Partition A (first-N moves of seeded games, vocab tables)
- `rust_io/` — Partition B (PGN parse dumps, UCI round-trip fixtures)
- `core/` — Partition C (canonical checkpoint layouts, metrics schema)
- `model/` — Partition D (canonical tokenizer outputs, RoPE freq tables)
- `adapters/` — Partition F (adapter param counts per config)
- `eval/` — Partition G (probe-output shape/stat summaries via syrupy)
- `lab/` — Partition H (Trial JSON samples)

## Content rules

- **Snapshot shapes + summary stats**, not exact floating-point values.
- Prefer human-readable JSON/text formats when possible.
- Snapshots regenerated via `pytest --snapshot-update` (syrupy).
- Manual goldens should include a short comment explaining how they were generated.
