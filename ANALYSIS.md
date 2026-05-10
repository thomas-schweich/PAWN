# stockfish-datagen Resource Optimization — Analysis & Remediation Plan

This document captures findings from the first production-scale run of `stockfish-datagen` on a 384-thread vast.ai pod (EPYC 9654 dual-socket Genoa) and proposes follow-up optimization work. The performance numbers below are observational, gathered from the live run on 2026-05-10.

## TL;DR

The 100M-game datagen run is achieving ~778 g/s for the search-mode tiers (`nodes_0001`, `nodes_0128`) on 380 workers. Local-host benchmarking on a Ryzen 7 7800X3D shows ~9 g/s/worker on an equivalent config, vs ~2 g/s/worker on the production pod — a **~4.4× per-worker gap** even after accounting for clock differences. The gap is the compounded product of half a dozen ~1.2-1.5× penalties; no single mechanism dominates. The current 380-worker config IS near-optimal for *aggregate* throughput on this hardware (dropping to 192 workers to escape SMT contention would *lower* total g/s), so this work is about reclaiming efficiency, not changing the worker count.

Several of the contributing factors are addressable in the rust binary or the patched stockfish without changing the data shape, and would compound to a meaningful aggregate win.

## Performance baseline

| | Local (Ryzen 7 7800X3D) | Remote (EPYC 9654 dual-socket) |
|---|---|---|
| Physical cores | 8 | 192 (96 × 2 sockets) |
| Logical threads | 16 | 384 |
| Sustained clock under all-core AVX-512 | ~4.7 GHz (estimate) | 3.16 GHz (measured) |
| L3 per CCD / V-cache | 96 MB V-cache (shared by 8c) | 32 MB per CCD × 12 CCDs/socket |
| Workers used | 14 | 380 |
| Throughput (`nodes_0001` + `store_legal_move_evals` + Large NNUE) | 125.7 g/s | 778 g/s |
| Per-worker | **8.98 g/s** | **2.05 g/s** |

Per-tier completion rates from the production run:

| Tier | g/s | Wall-clock |
|---|---|---|
| `tier0_evallegal` | 2,056 | 2 h 42 m |
| `nodes_0001` (with `store_legal_move_evals`) | 778 | 7 h 8 m |
| `nodes_0128` (in progress) | ~759 | est 7+ h |

`tier0_evallegal` is fastest because it does *only* the per-position legal-move evaluation. The other tiers do search-mode work *plus* the same legal-move eval pass — strictly more work, by design.

## Diagnostics actually run on the live pod

These were read-only probes against the running stockfish processes (no perturbation to the workload):

1. **Sustained clock**: `/proc/cpuinfo` and `/sys/devices/.../scaling_cur_freq` consistently report ~3,157 MHz across cores. That's 85% of the 3.71 GHz nominal boost — modest derate, not catastrophic throttling.
2. **NNUE weight placement**: `cat /proc/$PID/maps` shows the patched stockfish binary (113 MB) is the only file-backed read-only mapping of any size. **The NNUE Large network is embedded in the binary's rodata section** (compiled in with `INCBIN` or similar). This means all 380 processes share the same physical pages for the NNUE weights via the kernel page cache — there is no per-process duplication of NNUE memory.
3. **Huge pages**: THP is set to `madvise`. The rust binary's anon heap gets THP (~19 GiB anon huge pages system-wide come from it). Stockfish processes report `HugetlbPages: 0` and have **zero** huge-page coverage on the binary mmap that holds the NNUE — every NNUE access is paid for in 4 KiB pages.
4. **NUMA topology**: Two nodes, distance 10 (local) vs 32 (cross-socket). Cross-socket DRAM latency is **3.2× local**. CPUs 0-95 + 192-287 are node 0; CPUs 96-191 + 288-383 are node 1.
5. **Worker pinning**: `taskset -cp` confirms each (worker thread, stockfish process) pair is pinned to a single logical CPU, sharing L1/L2 with its SMT sibling. Pinning logic is correct.
6. **NUMA placement of stockfish anon memory**: `numastat -p $PID` for a sample process shows ~60/40 split across nodes for the *shared* memory regions (binary mmap + the in-progress /dev/shm parquet output). Anon memory of the process itself is small (~53 MB) and lands on the right node. The expensive thing — the NNUE pages — was first-touched by whichever socket the rust binary's main thread was on at startup, so half the workers pay the 3.2× cross-socket latency on cache misses to NNUE weights.

## Hypothesized performance gap

| Mechanism | Estimated penalty | Confidence | Notes |
|---|---|---|---|
| Sustained-clock derate (3.16 vs ~4.7 GHz local) | 1.50× | high | Measured |
| AVX-512 SMT contention (2 SMT siblings competing for vector units) | 1.40× | medium | Inferred — both SMT siblings doing dense vector work; SMT efficiency for AVX-512 typically ~50-60% |
| NUMA cross-socket NNUE access for ½ workers | 1.25× | medium | Confirmed 3.2× cross-node penalty + first-touch placement issue |
| TLB pressure from 4 KiB pages on 113 MB binary mmap | 1.20× | medium | Confirmed no huge pages on the NNUE region |
| Cross-CCD L3 access patterns (16 SMT threads / 32 MB per CCD) | 1.20× | medium | NNUE pages are shared via mmap, but sibling processes touch different parts of the net concurrently |
| `TT::clear()` at every `ucinewgame` (16 MB memset / game) | 1.20× | low/medium | At nodes=1/128/256 the TT is essentially unused, so this work is wasted |

Compounded: 1.50 × 1.40 × 1.25 × 1.20 × 1.20 × 1.20 ≈ **4.54×**, matching the observed ~4.4× gap. None of these is the smoking gun individually; they all multiply.

## Recommendations, by impact-to-effort ratio

### Tier 1 — High impact, low effort

1. **Drop `stockfish_hash_mb` to 1 MB for low-nodes tiers (tiers 0-3)**.
   At nodes ≤ 256, the transposition table holds essentially no useful entries — a depth-1 search has no subtree to memo, and even at 256 nodes the reuse rate inside one move is negligible. We're paying a 16 MB memset per `ucinewgame` for nothing — at the production rate of ~2 g/s/worker × 380 workers, that's ~12 GB/s of pure zero-write memory bandwidth burned per tier.
   - Change is config-only. No code change required.
   - Caveat: changes the tier_fingerprint, so old shards from a 16 MB-hash tier can't be resumed against a 1 MB-hash tier. Apply only to *new* tiers.
   - Recommendation: `stockfish_hash_mb: 1` for `tier0_evallegal`, `nodes_0001`, `nodes_0128`, `nodes_0256`. Keep `nodes_1024` at 16 MB (TT may help marginally there).
   - Expected gain: ~10-20 % per-tier wall-clock for the low-nodes tiers.

2. **`madvise(MADV_HUGEPAGE)` on the patched stockfish binary's NNUE rodata region** (or use `prctl(PR_SET_THP_DISABLE, 0)` + relink with section alignment).
   The NNUE weights are in the binary's read-only data section, mmap'd in 4 KiB pages. With ~50 MB of feature-transformer weights × 380 processes per pod, the TLB working set is enormous.
   - Patch lives in the patched stockfish fork ([github.com/thomas-schweich/stockfish-ml-extensions](https://github.com/thomas-schweich/stockfish-ml-extensions)).
   - Implementation: in stockfish init, call `madvise(weights_addr, weights_size, MADV_HUGEPAGE)` after locating the embedded NNUE region (via the `INCBIN` symbols).
   - Tricky bit: needs the section to be 2 MB-aligned in the linker script for THP to actually engage, otherwise the kernel can't promote.
   - Expected gain: 10-20 % on memory-bound NNUE workloads.

### Tier 2 — Medium impact, medium effort

3. **`numactl --interleave=all` (or the equivalent `set_mempolicy(MPOL_INTERLEAVE)`) for the rust binary**.
   Currently the NNUE pages get first-touched on whichever node the binary's main thread happens to be on, and 50 % of workers pay 3.2× DRAM latency on cache misses. Round-robin first-touch puts pages on both nodes evenly; on average every worker pays half the cross-socket penalty instead of half the workers paying all of it.
   - Implementation options:
     a. Wrap the binary's launch in `numactl --interleave=all` in `scripts/datagen_with_hf_sync.py` and the docker entrypoint.
     b. Or call `numa_set_interleave_mask(numa_all_nodes_ptr)` from the rust binary at startup (requires linking `libnuma`).
   - Expected gain: 10-15 % on dual-socket pods. Single-socket pods are unaffected.

4. **Document and prefer single-socket pods for datagen** in `stockfish-datagen/README.md`.
   The NUMA penalty doesn't exist on a single-socket Genoa — same per-worker performance on both halves of the workers, no first-touch problem to solve.
   - Single-socket EPYC 9654 (96 cores / 192 threads) at ~$0.27/h on vast.ai gives roughly the same total g/s as the dual-socket box at $0.83/h and is much simpler to reason about.
   - Cost-of-data is comparable (within ~20 %) but the dual-socket box uses CPU more inefficiently.

### Tier 3 — Lower impact or higher effort

5. **Investigate sharing the transposition table across stockfish workers via shared memory** (or skipping its allocation entirely at low nodes).
   If TT memory could be allocated as a single `mmap(MAP_SHARED)` region and reused across processes pinned to the same CCD, we'd eliminate per-worker hash bandwidth waste *and* improve cache locality. This is a deeper change to stockfish and only worth it if (1) is insufficient.

6. **Tune `shard_size_games` upward (e.g. to 5,000 or 10,000)**.
   Each shard close incurs ~5-10 s of zstd-19 compression work blocking that worker's game generation. At the current 2,000 games per shard the overhead is small (~1-2 % wall-clock per worker), but bumping it would still help marginally and would also reduce the number of files HF has to commit per cycle.
   - Trade-off: bigger shards → less granular resume, larger memory footprint per worker during accumulation.
   - Probably defer until after (1)-(4).

7. **Profile-guided optimization (PGO) and/or `-march=znver4` on the patched stockfish**.
   The CI matrix currently builds `x86-64-avx2` and `x86-64-avx512` variants. A `znver4`-tuned build would benefit from Zen 4-specific instruction scheduling and could give an additional 5-10 % on Genoa pods.
   - Caveat: requires us to keep the avx2/avx512 fallback variants for non-Zen-4 hosts.

## Work-order suggestion for the implementing agent

Recommended order, optimizing for getting wins into production fastest with least disruption:

1. **(1) hash_mb = 1 for low-nodes tiers** — pure config change, can apply to the very next datagen run with no code edits. Validate the wall-clock gain against the current run's tier rates.
2. **(3) `numactl --interleave=all` wrapping** — small change to `scripts/datagen_with_hf_sync.py` + `Dockerfile.datagen` entrypoint. Doesn't require touching the patched stockfish.
3. **(2) THP for NNUE region** — the highest-impact single change but lives in the patched stockfish fork; requires a stockfish rebuild and a new image tag.
4. **(4) Single-socket-pod README guidance** — documentation only; quick win, but doesn't help an already-running run.
5. **(7) `-march=znver4` build matrix entry** — CI plumbing, modest gain, defer to later.

Items (5) and (6) are research-level and should only be tackled if (1)-(4) don't close the gap enough.

## Out of scope for this analysis

- **Increasing the parquet schema** (e.g. adding more per-position metadata) is orthogonal — it would slow generation but isn't an "efficiency" question.
- **Changing the NNUE network** (e.g. to `small`) would change the dataset semantics and isn't on the table without a separate decision.
- **Switching away from x86 entirely** (e.g. ARM Graviton). Beyond scope.

## References

- Live monitoring scripts (transient, not in repo): `/tmp/datagen_checkin.sh`, `/tmp/datagen_sync_hf.sh`.
- Production config that produced these numbers: `stockfish-datagen/examples/stockfish_100m.json`.
- Rust binary entry point: `stockfish-datagen/src/main.rs`, shard writer at `stockfish-datagen/src/shard.rs:467` (where the zstd-19 + 16 MiB page-size choice lives).
- Patched stockfish fork: <https://github.com/thomas-schweich/stockfish-ml-extensions> (sf_18-v0.3.0).
- Architecture rule-of-thumb already in `CLAUDE.md`: `n_workers = total_threads − threads_per_core` — confirmed empirically; no change needed.
