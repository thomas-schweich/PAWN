# Vocabulary Transition Plan: PAWN → searchless_chess (1968-action)

## Background

PAWN currently uses a **dense 4,284-token vocabulary**:

| Range | Count | Description |
|-------|-------|-------------|
| 0 | 1 | PAD |
| 1–4096 | 4,096 | Dense 64×64 src×dst grid (includes 2,304 impossible moves) |
| 4097–4272 | 176 | Promotions (44 pairs × 4 piece types) |
| 4273–4283 | 11 | Outcome tokens (5 pretraining + 6 Lichess) |

The [searchless_chess vocabulary](https://github.com/google-deepmind/searchless_chess) uses a **sparse 1,968-action vocabulary**: only the moves reachable by queen or knight from each square (1,792), plus the same 176 promotions. No PAD, no outcomes — just moves.

This document outlines what it would take to adopt the searchless_chess vocabulary as the engine's primary representation, with outcome tokens and padding as optional additions.

---

## Proposed New Token Layout

```
Tokens 0–1967:     searchless_chess actions (1:1 match, same indices)
Token 1968:        PAD
Tokens 1969–1979:  Outcome tokens (optional, 11 tokens)
                   ─────────────────────────────────────
Total:             1,980 (or 1,969 without outcomes)
```

The action indices 0–1967 are **identical** to the searchless_chess vocabulary — no offset, no remapping. Anyone familiar with that vocabulary gets exactly what they expect. PAD and outcome tokens live above the action range, out of the way.

---

## What Changes

### 1. Engine: `vocab.rs` — Major Rewrite

**Current state:** Token IDs are computed by formula (`src * 64 + dst + 1` for the grid). This is elegant for the dense grid but means 2,304 of the 4,096 grid tokens represent impossible moves (e.g., `a1a1`, `a1b4`).

**New state:** Token IDs come from a static lookup table loaded from the canonical vocabulary. The mapping is UCI string ↔ action index, baked into the Rust binary at compile time (e.g., via `include_str!` on the JSON, or a build-time codegen step, or a hardcoded array).

Changes:
- Replace `base_grid_token(src, dst) -> u16` (formula) with `uci_to_token(uci: &str) -> u16` (table lookup).
- Replace `promo_token(src, dst, promo_type) -> u16` with the same unified lookup.
- `decompose_token(token) -> (src, dst, promo)` becomes a table lookup instead of arithmetic. The table is small (1,968 × 3 bytes = ~6 KB).
- `PAD_TOKEN` changes from 0 to 1968.
- Outcome token constants shift from 4273–4283 to 1969–1979.
- `VOCAB_SIZE` drops from 4,284 to 1,980 (or 1,969 without outcomes).

**Implementation options for the lookup table:**
- **(a) Embed the JSON at compile time** via `include_str!` + `serde_json` in a `Lazy<>` static. Simple, but adds a runtime JSON parse on first access.
- **(b) Build-time codegen** (`build.rs`) that reads the JSON and emits a Rust source file with `const` arrays. Zero runtime cost, but adds a build step.
- **(c) Hardcoded Rust arrays.** The vocabulary is stable (it's defined by chess geometry), so a one-time code generation is fine. Simplest at runtime.

Recommendation: **(c)** — generate the arrays once with a script, commit the Rust file. The vocabulary is deterministic and will never change. A `const TOKEN_TO_UCI: [&str; 1968]` array and a `phf` or sorted-array lookup for the reverse direction.

### 2. Engine: `board.rs` — `move_to_token` / `token_to_move`

- `move_to_token(m: &Move) -> u16`: currently computes the token from src/dst indices via formula. Would instead build the UCI string from the move and look it up in the table. Slightly more work per call (string construction + lookup vs. arithmetic), but this is not on the hot path — shakmaty's `legal_moves()` dominates.
- `token_to_move(pos, token)`: currently iterates legal moves and compares token IDs. No change needed in logic, just uses the new `move_to_token`.

### 3. Engine: `board.rs` — Legal Move Representations

**Grid mask (`[u64; 64]`):** This representation is independent of the token vocabulary — it's a bitboard over (src, dst) pairs. It can stay as-is for internal use (e.g., probes, board state extraction). However, it's currently also exposed to Python for legal move masks, which couples it to the vocab.

**Token mask:** `legal_move_tokens()` returns `Vec<u16>`. These would now be searchless_chess action IDs. No structural change, just different numbers.

**Sparse legal mask (`labels.rs`):** `compute_legal_token_masks_sparse()` emits indices into a `(B, T, vocab_size)` tensor. With `vocab_size` shrinking from ~4,278 to 1,968, the mask tensor is ~2.2× smaller. The function itself just calls `move_to_token` per legal move — works unchanged with the new vocab.

### 4. Engine: `batch.rs` — CLM Batch Generation

**Current behavior:** `generate_clm_batch()` always prepends an outcome token:
```
input:  [outcome, move_1, ..., move_N, PAD, ...]   (seq_len tokens)
target: [move_1,  move_2, ..., move_N, PAD, ...]
```
This means `max_ply = seq_len - 1` — one slot is consumed by the outcome prefix.

**New default (no outcome prefix):**
```
input:  [move_1, move_2, ..., move_N, PAD, ...]     (seq_len tokens, PAD=1968)
target: [move_2, move_3, ..., move_N, PAD, ...]
```
Now `max_ply = seq_len` — every slot is a move. For `seq_len=256`, you get 256 moves instead of 255.

**Outcome prefix as opt-in:** Add a boolean parameter `prepend_outcome: bool`. When true, behaves like today (with the new token IDs). When false (the new default), sequences are pure moves.

Changes:
- `generate_clm_batch()` gains a `prepend_outcome` flag.
- `CLMBatch` struct: `max_ply` is now `seq_len` by default, or `seq_len - 1` with outcome prefix.
- Shift/alignment logic in `loss_mask` and `targets` simplifies when there's no prefix.

### 5. Engine: `labels.rs` — Legal Mask Computation

- `vocab_size` parameter changes from ~4,278 to 1,968.
- The functions call `move_to_token` internally, so they pick up the new IDs automatically.
- `compute_legal_move_masks()` returns `(grid, promo_mask)` — this grid representation is orthogonal to the vocab and could remain for backward compatibility, or be removed if nothing uses it.

### 6. Engine: `lib.rs` — Python Bindings

- `export_move_vocabulary()` returns the new 1,968-entry maps.
- `generate_clm_batch()` exposed with the new `prepend_outcome` parameter.
- New function: `convert_pawn_to_searchless(tokens: Vec<i16>) -> Vec<i16>` and the reverse, for migrating existing data/checkpoints.

### 7. Python: `config.py`

- `PAD_TOKEN` changes from 0 to 1,968.
- `vocab_size` default changes from 4,284 to 1,980 (or 1,969 without outcomes).
- Outcome token base shifts from 4,273 to 1,969.
- `max_seq_len` stays 256 but now holds 256 moves (not 255 + outcome).
- `n_outcomes` stays 11 if outcomes are kept, 0 for moves-only mode.

### 8. Python: `model.py` — Factored Embeddings

**This is the main wrinkle.** Factored embeddings decompose each token into `(src, dst, promo)` and sum three small embedding tables: `src_embed(64) + dst_embed(64) + promo_embed(5)`.

Currently, `_build_decomposition_table()` builds a `[4278, 3]` int16 lookup table mapping token → (src, dst, promo). **This approach works identically with the new vocabulary** — the table just becomes `[1968, 3]` instead. The decomposition is defined by the UCI string, not the token ID:

```python
def _build_decomposition_table() -> torch.Tensor:
    vocab = export_move_vocabulary()  # Now returns 1968 entries, 0-indexed
    table = torch.zeros(1968, 3, dtype=torch.int16)
    for action_idx, uci_str in vocab["token_to_move"].items():
        src_sq = sq_names.index(uci_str[:2])
        dst_sq = sq_names.index(uci_str[2:4])
        promo = promo_map.get(uci_str[4:], 0)
        table[action_idx] = torch.tensor([src_sq, dst_sq, promo])
    return table
```

**No fundamental change to factored embeddings.** The decomposition table is already a lookup (not formula-based) on the Python side. The only difference is that the table has 1,968 rows instead of 4,278. PAD (1968) and outcome tokens (1969+) are above the table range and handled by standalone embeddings, same as before.

One consideration: with the dense grid, `src_embed` and `dst_embed` see all 64 squares uniformly during training (every square appears as both source and destination in the grid). With the sparse vocabulary, some (src, dst) combinations are removed — but all 64 squares still appear as sources and destinations (queen covers all files/ranks, knight reaches all squares). So embedding coverage is unchanged.

### 9. Python: `data.py` — Sequence Packing

- `pack_clm_sequences()`: currently always prepends an outcome token. Needs a `prepend_outcome=False` default, which simplifies to just copying moves and padding.
- `strip_outcome_token()`: becomes unnecessary for the default path. Keep for backward compatibility when `prepend_outcome=True`.
- `CLMDataset`: passes the `prepend_outcome` flag through to the engine.

### 10. Python: `lichess_data.py`

- `LegalMaskBuilder`: `vocab_size` changes. Pre-allocated buffers shrink ~2.2×.
- `compute_legal_indices()`: `vocab_size` parameter changes.
- Parquet dataset: v2 format (outcome-prepended) detection logic needs updating for new outcome token base.

### 11. Python: `model.py` — LM Head

The output projection (`lm_head`) maps from `d_model` to `vocab_size`. With 1,968 actions instead of 4,272 moves:

- **Smaller weight matrix:** `d_model × 1980` vs `d_model × 4284`. For `d_model=512`, that's ~1.0M → ~0.5M parameters saved on the LM head alone.
- **Faster softmax:** 1,980-way vs 4,284-way. Minor but measurable.
- If outcomes are in the vocab, `lm_head` covers them too (indices 1969–1979). If not, outcome prediction becomes a separate head or is dropped entirely.

---

## Vocabulary Conversion Functions

For interoperability with existing checkpoints and data, the engine should provide:

```rust
/// Convert a PAWN token (dense 4284-vocab) to a searchless_chess action (1968-vocab).
/// Returns None for PAD, outcome tokens, and impossible moves (the 2,304 pruned entries).
pub fn pawn_to_searchless(pawn_token: u16) -> Option<u16>;

/// Convert a searchless_chess action to a PAWN token.
pub fn searchless_to_pawn(sc_action: u16) -> u16;
```

Both are simple: convert to UCI string, then look up in the other vocabulary. These can be precomputed as static arrays for O(1) conversion.

Use cases:
- **Checkpoint weight migration:** Map `lm_head` rows from old vocab to new. The 2,304 impossible-move rows are simply dropped. All 1,968 searchless actions have a corresponding PAWN token.
- **Data migration:** Convert existing Lichess Parquet datasets from PAWN tokens to searchless tokens.
- **Evaluation compatibility:** Compare models trained with different vocabularies.

---

## Pros

1. **Standard vocabulary.** Identical to DeepMind's searchless_chess, making the engine useful for other projects and enabling direct comparison with published results.
2. **Compact.** 1,968 vs 4,272 move tokens — 54% reduction. Smaller LM head, smaller legal masks, faster softmax.
3. **No wasted tokens.** Every token in the vocabulary is a move that can actually occur in a game. The current vocab has 2,304 impossible tokens (54% of the grid) that the model must learn to assign zero probability.
4. **Simpler no-outcome default.** Sequences are pure move sequences by default. Outcome conditioning becomes an explicit opt-in, not a shift-and-strip dance.
5. **Full sequence length.** 256 tokens = 256 moves, not 255 moves + 1 outcome. For the 512-token context window, this means 512 moves vs 511.

## Cons

1. **Breaking change for all existing checkpoints.** Requires retraining from scratch. Existing PAWN checkpoints (small/base/large) and all adapter checkpoints become incompatible. (Partial weight migration is possible for the transformer body — see below.)
2. **Loss of formula-based token arithmetic.** The dense grid allows `token = src * 64 + dst + 1`, which is O(1) with no lookup. The sparse vocab requires a table lookup. In practice, this doesn't matter — the Python side already uses a lookup table, and the Rust hot path (game generation) is dominated by `legal_moves()`, not tokenization.
3. **Grid mask representation decouples from vocab.** The `[u64; 64]` bitboard is currently aligned with the base grid tokens (bit `d` of `grid[s]` ↔ token `s*64+d+1`). With the sparse vocab, this bitboard is still useful internally but no longer maps trivially to token IDs. Legal masks must go through `move_to_token` conversion, which they already do for the sparse mask path.
4. **Two-vocab maintenance during transition.** Until old checkpoints/data are fully retired, conversion functions and possibly a vocab-mode flag are needed.

---

## Other Considerations

### Partial Weight Migration

The transformer body (attention, FFN, norms) is vocab-agnostic — weights transfer directly. For embeddings and the LM head:

- **Factored input embeddings** (`src_embed`, `dst_embed`, `promo_embed`): These embed squares and promo types, not token IDs. **They transfer 1:1** — same 64 source squares, 64 destination squares, 5 promo types.
- **Outcome embeddings:** Transfer directly if outcome tokens are kept (just different IDs).
- **PAD embedding:** Transfers directly.
- **LM head:** 4,284 → 1,968 output neurons. Each searchless action has a corresponding PAWN token, so rows can be copied by matching UCI strings. The 2,304 pruned rows are dropped. This gives a warm start, not a full transfer.

This means **you could warm-start a new model from existing PAWN weights** with minimal loss. Only the LM head needs surgery; everything else copies over.

### Outcome Tokens in the New World

Three options:
1. **Keep outcome tokens in the vocab** (tokens 1969–1979). Outcome-conditioned generation works as before. `vocab_size = 1980`.
2. **Separate outcome head.** The main vocab is pure moves (1,968). Outcome prediction uses a separate linear head from the same hidden states. Cleaner separation of concerns.
3. **Drop outcomes entirely.** Train on move sequences only. Outcome conditioning is added later via a separate mechanism (e.g., classifier-free guidance, probing).

Recommendation: **Option 1 for now** — it's the smallest change and preserves existing training infrastructure. Option 2 is a natural follow-up if outcome conditioning proves useful enough to warrant a dedicated head.

### Compatibility with searchless_chess

Action indices 0–1967 are identical between PAWN and searchless_chess — no offset, no remapping. PAD (1968) and outcome tokens (1969+) live above the action range and don't interfere. Anyone consuming PAWN sequences who doesn't care about PAD/outcomes can treat them as raw searchless_chess action sequences.

### Legal Mask Efficiency

With 1,968 tokens, legal masks are:
- **Dense:** `B × T × 1968` bools = ~1 MB for a typical batch (vs ~2.2 MB with 4,278)
- **Sparse:** unchanged in format, slightly smaller indices (values fit in fewer bits)
- **Scatter buffer:** pre-allocated GPU buffer shrinks proportionally

The relative benefit of sparse over dense is slightly less dramatic (1,968 is already small), but sparse is still preferred for the GPU scatter pattern.

### Data Pipeline Impact

- **Random game generation:** No change in throughput. Token assignment is a trivial lookup after `legal_moves()`.
- **Lichess Parquet:** Existing v2 datasets use PAWN tokens and would need re-tokenization. A one-time batch conversion via the `pawn_to_searchless` function.
- **DataLoader:** No structural changes. `CLMDataset` generates sequences with the new tokens.

### Test Coverage

All existing vocab tests in `vocab.rs` (20+ tests) and label tests in `labels.rs` (8+ tests) need rewriting for the new token layout. The test structure stays the same — roundtrip tests, boundary tests, exhaustive checks — just with different constants.

---

## Rough Ordering of Work

1. **Generate the Rust lookup tables** from `searchless_chess_vocabulary.json` (script → committed `.rs` file).
2. **Rewrite `vocab.rs`** with new constants, lookup tables, and conversion functions. Keep outcome token logic.
3. **Update `board.rs`** `move_to_token` / `token_to_move` to use new tables.
4. **Update `batch.rs`** with `prepend_outcome` flag (default false).
5. **Update `labels.rs`** (mostly just `vocab_size` changes).
6. **Update `lib.rs`** Python bindings.
7. **Update Python side:** `config.py`, `model.py`, `data.py`, `lichess_data.py`.
8. **Rewrite all tests** for new token layout.
9. **Add conversion functions** (PAWN ↔ searchless) for data migration.
10. **Re-train** from scratch with the new vocabulary (or warm-start from existing weights).
