# Patches for the stockfish-ml-extensions fork

Two perf changes that live in the patched stockfish fork at
[github.com/thomas-schweich/stockfish-ml-extensions](https://github.com/thomas-schweich/stockfish-ml-extensions),
prepared as `git format-patch` output ready to apply via `git am`.
Both are minimal-diff additions on top of the current pinned commit
(`14f92699`, fork tag `sf_18-v0.3.0`). Built + smoke-tested locally
on `x86-64-avx2` (avx512/vnni512 variants behave identically — no
arch-specific code paths touched).

## What they do

### `01-skip-tt-clear.patch`

Adds a `SkipTTClear` UCI option (boolean, default `false`). When
true, `Engine::search_clear()` (the single function `ucinewgame`
funnels through) skips the `tt.clear(threads)` memset. The TT is
still allocated and still consulted by search; only the per-game
zero-fill stops.

**Why**: at the production 100M-game datagen workload's low-node
tiers (`nodes_0001`, `nodes_0128`, `nodes_0256`) the TT barely
populates before the next `ucinewgame`. Memsetting 16 MB × 380
workers × 1 game/several-seconds is ~12 GB/s of wasted bandwidth.
At `stockfish_hash_mb: 1` (which the post-refactor
`stockfish_100m.json` config already uses for these tiers) it's
smaller in absolute terms but still useless. The patch lets us
turn it off explicitly.

**Wiring in this repo**: the rust binary's UCI command stream
(`stockfish-datagen/src/stockfish.rs`) issues `setoption name
SkipTTClear value true` early in spawn, before the per-game
`ucinewgame` calls. Probably gate on `effective_hash_mb <= 4` so
nodes_1024 (which uses 16 MB and might benefit from the TT)
preserves vanilla behavior. Wire this up in a follow-up PR after
the fork PR merges.

### `02-nnue-thp.patch`

Adds a `FeatureTransformer::advise_huge_pages()` helper that calls
`madvise(MADV_HUGEPAGE)` over its weight/bias arrays. Hooked into
`Network::load(std::istream&)` so it runs once per network at load
time. Gated on `SF_NNUE_HUGEPAGE=1` env var so we can A/B without a
rebuild. Linux-only; no-op elsewhere (header guards on `__linux__`).

**Why**: the ~50 MB of NNUE weight pages dominate the per-worker
TLB working set on a 4 KiB-page system. On a 380-worker pod that's
180+ million 4 KiB page-table walks per second under load. THP
coalesces aligned subregions into 2 MiB pages → roughly 512×
fewer TLB entries needed for the bulk of the weights.

**Caveats**:

- The weight arrays are inline `std::array` members of `FeatureTransformer`
  (which is itself an inline member of `Network`, heap-allocated via
  `std::make_unique<NN::Networks>`). The array memory has `alignas(CacheLineSize)`
  (64 bytes), NOT 2 MiB. So `madvise` will only promote the 2 MiB-aligned
  *subregion* inside each array — the head/tail (few KiB max per array)
  stays on 4 KiB pages. For a 50 MiB array that's >96% of the bytes covered,
  which is good enough; a follow-up could swap the allocator for a custom
  2 MiB-aligned `mmap` to get the rest.
- Effect requires `transparent_hugepage=madvise` (the typical Linux
  default) and `khugepaged` running. Verify on the pod with
  `cat /sys/kernel/mm/transparent_hugepage/enabled`.
- A/B verification: `cat /proc/<sf-pid>/status | grep AnonHugePages`
  should show a non-zero value with the env var set and roughly zero
  without it. On the production EPYC pod, expect ~50 MB of AnonHugePages
  per stockfish process.

## Apply / review / push workflow

```bash
# 1. Clone the fork (you probably already have it elsewhere; otherwise:)
git clone https://github.com/thomas-schweich/stockfish-ml-extensions.git
cd stockfish-ml-extensions
git checkout 14f92699733a3d24a9067a20b32f95f0fcf9c4ab

# 2. Apply each patch as a fresh branch.
git checkout -b skip-tt-clear
git am < /path/to/pawn/stockfish-datagen/external-patches/01-skip-tt-clear.patch

git checkout 14f92699733a3d24a9067a20b32f95f0fcf9c4ab
git checkout -b nnue-thp
git am < /path/to/pawn/stockfish-datagen/external-patches/02-nnue-thp.patch

# 3. Build + smoke-test each.
cd src && make -j$(nproc) build ARCH=x86-64-avx2
printf 'uci\nquit\n' | ./stockfish | grep -i SkipTTClear     # for 01
SF_NNUE_HUGEPAGE=1 ./stockfish bench 16 1 13                 # for 02

# 4. Push branches and open PRs against the fork's main branch.
git push origin skip-tt-clear nnue-thp
gh pr create --base main --head skip-tt-clear --title "feat(uci): SkipTTClear option"
gh pr create --base main --head nnue-thp --title "feat(nnue): madvise huge pages for FT weights"

# 5. After merging the PRs, bump `SF_FORK_COMMIT` in
#    `Dockerfile.datagen` to the new fork-main commit SHA. The next CI
#    build of `:datagen` will pick up the changes via the JIT preflight
#    build script.
```

## Post-merge wiring in this repo

After the PRs land:

1. Update `Dockerfile.datagen`'s `SF_FORK_COMMIT` and the env var
   stamp to the new commit SHA. Bump any docs that mention
   `sf_18-v0.3.0` if a new tag is cut.
2. In `stockfish-datagen/src/stockfish.rs::spawn`, after the
   `setoption name Hash value` and before the `isready` barrier,
   issue `setoption name SkipTTClear value true` whenever
   `effective_hash_mb <= 4` (i.e. tiers that have explicitly opted
   into the low-TT path). Add a small unit test that asserts the
   option is set for tier0/nodes_0001/nodes_0128/nodes_0256 in
   `stockfish_100m.json`.
3. For the NNUE THP, no rust-side wiring needed — `SF_NNUE_HUGEPAGE=1`
   is set as an env var on the pod (either in `Dockerfile.datagen`
   permanently, or in the `vast.sh launch` env). Recommend baking
   it into the runtime ENV since the trade-off is essentially free
   on any modern Linux pod.
