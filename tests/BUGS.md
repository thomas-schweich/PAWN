# Bug Registry

Central list of bugs discovered during the TDD test suite overhaul.
Every `@pytest.mark.xfail(strict=True, reason="BUG-N: ...")` MUST have a
corresponding entry below.

**Status legend**
- `open` — bug confirmed, test is xfail, fix not yet landed
- `verified` — lead confirmed this is a real bug (not agent misunderstanding)
- `invalid` — lead reviewed and concluded the test was wrong; entry kept for history, xfail removed
- `fixed` — source fixed, xfail removed, test now green

**Format**

```
## BUG-N: <one-line summary>
- Status: open | verified | invalid | fixed
- Discoverer: <partition letter>
- Test: tests/<path>::<name>
- Source: <file>:<line>
- Expected: <correct behavior>
- Actual: <observed wrong behavior>
- Repro: <minimal invocation>
- Notes: <optional>
```

---

<!-- Workers append entries below. Lead reconciles and updates statuses in Wave 4. -->

## BUG-100: diagnostic per_ply_stats omit terminal-ply bits that are included in per-game accumulators
- Status: fixed
- Discoverer: B (Rust IO)
- Test: engine/src/diagnostic.rs::tests::test_per_ply_stats_match_accumulators
- Source: engine/src/diagnostic.rs:183-201 (compute_game_stats uses max_ply=length)
- Expected: For every accepted diagnostic game, OR-aggregating per_ply_stats by color (even t -> white, odd t -> black) should equal the per-game white/black accumulators.
- Actual: compute_game_stats calls edgestats::compute_edge_stats_per_ply with max_ply = length.max(1). The terminal ply's bits live at index `length`, but the packing loop inside compute_edge_stats_per_ply uses copy_len = min(length+1, max_ply) = length, so the terminal bits never reach the returned per_ply vector. The accumulators, however, are OR'd with the terminal bits, so accumulator and per_ply disagree for any game with nonzero terminal bits (every game that reaches a terminal position).
- Repro: cd engine && cargo test --release -- --ignored diagnostic::tests::test_per_ply_stats_match_accumulators
- Notes: Fix: pass max_ply = length + 1 (or any value strictly greater than length) into compute_edge_stats_per_ply so the terminal slot is included. Alternatively change compute_game_stats to explicitly size per_ply with length + 1.

## BUG-200: is_rocm() heuristic misses AMD Instinct MI250X
- Status: fixed
- Discoverer: C (Python Infra)
- Test: tests/core/test_gpu.py::TestIsRocm::test_is_rocm_matches_mi250x
- Source: pawn/gpu.py:21
- Expected: is_rocm() returns True for every AMD/ROCm device, including 'AMD Instinct MI250X'
- Actual: Returns False — the heuristic checks for substrings 'radeon', 'rx ', 'mi ' (with trailing space), or 'mi3'. 'amd instinct mi250x' contains none of these ('mi2' after space is not 'mi '; no 'mi3'), so MI250X cards are misdetected as NVIDIA.
- Repro: UV_CACHE_DIR="/home/tas/pawn/.uv-cache" uv run pytest tests/core/test_gpu.py::TestIsRocm::test_is_rocm_matches_mi250x -v
- Notes: Add 'mi2' to the match list, or better, check torch.version.hip is not None (the canonical ROCm detection). The MI300 series is coincidentally covered by 'mi3' but the heuristic is fragile.

## BUG-500: run_evals_local.py ignores --help and runs eagerly at import time
- Status: fixed
- Discoverer: H (Lab + Dashboard + Scripts)
- Test: tests/scripts/test_script_smoke.py::test_eager_scripts_support_help[run_evals_local.py]
- Source: scripts/run_evals_local.py:1-46 (no argparse, no __main__ guard)
- Expected: `python scripts/run_evals_local.py --help` prints a usage string and exits 0.
- Actual: The script has no argparse and no `if __name__ == "__main__":` guard. It immediately calls `configure_gpu()`, generates probe data, generates a diagnostic corpus (~10K games/category), and tries to load checkpoints from `data/eval_small/...`. `--help` is silently ignored. On a fresh checkout this also crashes with a state_dict size mismatch because the checkpoints predate the expanded vocabulary.
- Repro: cd /home/tas/pawn && UV_CACHE_DIR="/home/tas/pawn/.uv-cache" uv run pytest tests/scripts/test_script_smoke.py::test_eager_scripts_support_help -v
- Notes: Wrap everything below the imports in a `main()` function guarded by `if __name__ == "__main__":`, add an `argparse.ArgumentParser` so `--help` short-circuits before the expensive corpus generation. Also update the hardcoded paths to support `--checkpoint` overrides.

## BUG-700: _extract_elos_from_pgn double-flushes when max_games is reached
- Status: fixed
- Discoverer: G (Eval Suite)
- Test: tests/eval/test_lichess.py::TestExtractElosFromPGN::test_max_games_respected
- Source: pawn/eval_suite/lichess.py:82-113
- Expected: `_extract_elos_from_pgn(path, max_games=N)` returns at most N entries.
- Actual: Returns N+1 entries when the PGN has more than N games. On each `[Event ...]` line the previous game is flushed and, if `len(elos) >= max_games`, we `break`. The `break` exits before `white_elo/black_elo` are reset, and the post-loop `if in_headers: elos.append(...)` appends the already-flushed previous game a second time. With max_games=2 and 3 input games we get `[(1200,1250),(1600,1650),(1600,1650)]`.
- Repro: cd /home/tas/pawn && UV_CACHE_DIR="/home/tas/pawn/.uv-cache" uv run pytest tests/eval/test_lichess.py::TestExtractElosFromPGN::test_max_games_respected -v
- Notes: Fix: set `in_headers = False` before breaking, or move `break` before the `elos.append` and flush outside the loop body.

## BUG-701: autoregressive_generate mask_illegal uses mismatched vocab_size with the Rust engine
- Status: fixed
- Discoverer: G (Eval Suite)
- Test: tests/eval/test_generation.py::TestAutoregressiveGenerate::test_mask_illegal_produces_legal_moves_only
- Source: pawn/eval_suite/generation.py:189-191, engine/src/lib.rs:1356 (PyBatchRLEnv.get_legal_token_masks_batch)
- Expected: `env.get_legal_token_masks_batch(all_indices)` returns a mask with width equal to `CLMConfig.vocab_size` (4284), so it can be copied into `_mask_buf` which is allocated at `(n_games, cfg_vocab_size)`.
- Actual: The pyo3 signature defaults to `vocab_size=4278`, so the returned array has width 4278 while `_mask_buf` has width 4284. The `_mask_buf.copy_(...)` raises `RuntimeError: The size of tensor a (4284) must match the size of tensor b (4278)`. Every call path that uses `mask_illegal=True` (including outcome_signal_test, prefix_continuation_test, poisoned_prefix_test, impossible_task_test, improbable_task_test) crashes on the first decode step.
- Repro: cd /home/tas/pawn && UV_CACHE_DIR="/home/tas/pawn/.uv-cache" uv run pytest tests/eval/test_generation.py::TestAutoregressiveGenerate::test_mask_illegal_produces_legal_moves_only -v
- Notes: Fix: call `env.get_legal_token_masks_batch(all_indices, cfg_vocab_size)` in generation.py, or update the pyo3 default to the current `VOCAB_SIZE` (4284). The Rust constant in `engine/src/vocab.rs` is already 4284; the pyo3 default in `engine/src/lib.rs:1356` is stale.

## BUG-501: run_evals_toplayer.py ignores --help and runs eagerly at import time
- Status: fixed
- Discoverer: H (Lab + Dashboard + Scripts)
- Test: tests/scripts/test_script_smoke.py::test_eager_scripts_support_help[run_evals_toplayer.py]
- Source: scripts/run_evals_toplayer.py:1-50 (no argparse, no __main__ guard)
- Expected: `python scripts/run_evals_toplayer.py --help` prints a usage string and exits 0.
- Actual: Same as BUG-500 — no argparse, no `if __name__ == "__main__":` guard. The file runs its full eval pipeline (probe data generation, diagnostic corpus, and loading three checkpoints from `data/eval_*`) on import. `--help` is silently consumed and then the script crashes with a state_dict size mismatch on stale checkpoints.
- Repro: cd /home/tas/pawn && UV_CACHE_DIR="/home/tas/pawn/.uv-cache" uv run pytest tests/scripts/test_script_smoke.py::test_eager_scripts_support_help -v
- Notes: Same fix pattern as BUG-500: wrap in `main()` + argparse + `__main__` guard. These two scripts are near-duplicates and should share a common module.

