# Patches for the stockfish-ml-extensions fork

Reference copies of fork-side patches that the datagen workload depends
on, kept in this repo so the wiring story stays self-contained even
if the fork's git history rotates / rebases / re-tags.

The currently-pinned fork commit is documented in `Dockerfile.datagen`'s
`SF_FORK_COMMIT` ARG; the patch listed here is included in that pinned
state (i.e. the patch is informational at this point, not something
the build still has to `git am`).

## `02-nnue-thp.patch` — NNUE feature-transformer THP hint

**Status:** merged into the fork as PR
[#2](https://github.com/thomas-schweich/stockfish-ml-extensions/pull/2),
shipped in `SF_FORK_COMMIT` 341d1cd1+.

Adds a `FeatureTransformer::advise_huge_pages()` helper that calls
`madvise(MADV_HUGEPAGE)` over its weight/bias arrays after they've
been read in. Hooked into `Network::load(std::istream&)` so it runs
once per network at load time. Gated on `SF_NNUE_HUGEPAGE=1` so it's
easy to A/B without a rebuild. Linux-only; no-op elsewhere.

**Why:** the ~50 MiB of NNUE weight pages dominate the per-worker TLB
working set on a 4 KiB-page system. On a 380-worker pod, the kernel
coalescing those into 2 MiB pages via khugepaged shrinks TLB pressure
by ~500× for the bulk of the weight bytes (~99% coverage after the
in-patch alignment math; the tiny leading/trailing sub-page stub of
each `alignas(64)` array stays on 4 KiB pages, which is too small to
matter).

**Wiring in this repo:** `Dockerfile.datagen` sets
`ENV SF_NNUE_HUGEPAGE=1` so pods run with the hint enabled by default.
Disable per launch with `vast.sh launch ... -e SF_NNUE_HUGEPAGE=0` if
you want to A/B against the un-hinted baseline.

**Caveats:**

- Effect requires `transparent_hugepage=madvise` (the typical Linux
  default) and `khugepaged` running. Verify on a pod with
  `cat /sys/kernel/mm/transparent_hugepage/enabled`.
- A/B verification: `cat /proc/<sf-pid>/status | grep AnonHugePages`
  should show a non-zero value (expect ~50 MB per stockfish process
  on a 380-worker EPYC pod) with the env var set, roughly zero
  without. Promotion happens asynchronously via khugepaged so the
  value may take a few seconds to climb after process start.
- THP_enabled per-process must be 1; some containerization/WSL
  setups disable THP at the cgroup or per-process level. The hint is
  a no-op there but is harmless.

## Considered and rejected: `SkipTTClear` (closed, not landed)

PR [#1](https://github.com/thomas-schweich/stockfish-ml-extensions/pull/1)
drafted a UCI option to skip `tt.clear()` in `Engine::search_clear`,
motivated by the per-game memset bandwidth on low-nodes tiers. Closed
without merging on a determinism review:

The post-refactor datagen architecture promises that shard bytes are a
pure function of `(master_seed, tier.name, global_game_index)`. With
`SkipTTClear=true` the TT persists across `ucinewgame`, so a search at
fixed node budget can take a different path depending on TT residue,
which depends on which other games this worker played first, which
depends on the AtomicU64 shard-counter race — all of which is
operationally non-deterministic. The shard contents become a function
of `(master_seed, tier.name, global_game_index, worker_shard_history)`
with the last term non-reproducible. That violates the resume +
multi-pod + extension bit-identical guarantees the architecture
invests in (see the `live_*` tests in `stockfish-datagen/src/runner.rs`
plus the cross-pod manifest reconciliation contract).

The perf win after the related-but-cheap `stockfish_hash_mb: 1`
config change was estimated at <1%, vs. silently weakening the
determinism contract — a bad trade.

A determinism-safe alternative (a `Hash=0` / `NoTT` mode that
allocates no TT and treats every probe as a miss) was discussed and
also passed on: the ~1% residual after the `hash_mb=1` reduction
isn't worth another fork patch.
