#!/bin/sh
#
# Preflight: build the patched-stockfish binary for the host's actual CPU.
#
# Detects the best Stockfish Makefile ARCH from /proc/cpuinfo flags, builds
# the binary in-place, and caches the result under
# /workspace/.cache/stockfish/<commit-arch-hash>/ so subsequent pod launches
# (or restarts after `vast.sh stop` / RunPod stop) skip the rebuild.
#
# Why JIT-build at startup instead of CI-baking one binary per microarch:
# stockfish's Makefile knows ~10 microarch targets (x86-64-avx2, -avx512,
# -vnni512, -modern, -ssse3, ...). Baking a matrix of CI tags forces the
# operator to pick the right one per pod (and re-baking when a new
# microarch appears like znver4 / Sapphire Rapids). JIT building at first
# launch detects the host once and is then a cache hit forever.
#
# Build wall-clock is small: stockfish is ~30 KLOC of C++, so even on
# a 16-core CPU it builds in ~30-60 s; on a 96-core EPYC pod it's ~5-10 s.
# Compared against typical 5-10 hour datagen runs, the first-build cost
# is noise.
#
# Env vars:
#   DATAGEN_BUILD_CACHE — root cache dir (default /workspace/.cache/stockfish).
#                         Falls back to /tmp/stockfish-cache if /workspace
#                         doesn't exist (warns: cache is volatile).
#   SF_FORK_COMMIT      — passed in by the Dockerfile; mixes into the
#                         cache key so a fork-commit bump invalidates.
#   SF_SRC              — source dir to build in (default /opt/datagen/sf-fork/src).
#   SF_TARGET           — symlink path the rust configs reference
#                         (default /opt/datagen/stockfish-datagen/stockfish-patched).
#
# Output:
#   /opt/datagen/stockfish-datagen/stockfish-patched -> cache file.
#   One-line summary on stdout:
#     [stockfish-build] arch=<X>, cache=<hit|miss in Ns>, commit=<short>, path=<...>
#
# Exit codes:
#   0  — binary is in place and runnable.
#   1  — build failed (make returned non-zero).
#   2  — cache dir not writable AND fallback to /tmp also failed.

set -eu

SF_SRC="${SF_SRC:-/opt/datagen/sf-fork/src}"
SF_TARGET="${SF_TARGET:-/opt/datagen/stockfish-datagen/stockfish-patched}"
SF_COMMIT="${SF_FORK_COMMIT:-unknown}"
CACHE_ROOT="${DATAGEN_BUILD_CACHE:-/workspace/.cache/stockfish}"

if [ ! -d "$SF_SRC" ]; then
    echo "[stockfish-build] ERROR: source dir $SF_SRC missing" >&2
    exit 1
fi

# ── 1. Pick the best Makefile ARCH from /proc/cpuinfo flags ────────────
# Order matters: prefer the most specific target the CPU supports.
# Stockfish's Makefile (Makefile@sf_18-v0.3.0 in the patched fork)
# accepts these among others:
#   x86-64-vnni512  — AVX-512F + AVX-512 VNNI (Zen 4, Sapphire Rapids+)
#   x86-64-avx512   — AVX-512F (Skylake-X+, Zen 4)
#   x86-64-avx2     — AVX2 (Haswell+, Zen 2+)
#   x86-64-modern   — SSE4.2 / popcnt baseline (legacy fallback)
flags=$(awk -F: '/^flags/{print $2; exit}' /proc/cpuinfo || echo "")
case " $flags " in
    *' avx512_vnni '*' avx512f '* | *' avx512f '*' avx512_vnni '*)
        arch="x86-64-vnni512"
        ;;
    *' avx512f '*)
        arch="x86-64-avx512"
        ;;
    *' avx2 '*)
        arch="x86-64-avx2"
        ;;
    *)
        arch="x86-64-modern"
        ;;
esac

# ── 2. Resolve cache dir ───────────────────────────────────────────────
# /workspace is the canonical persistent mount on RunPod (network volume)
# and vast.ai (instance disk), so cached binaries survive stop/start.
# Fall back to /tmp if /workspace isn't writable — the binary still
# works, but every restart rebuilds.
cache_root="$CACHE_ROOT"
if ! mkdir -p "$cache_root" 2>/dev/null; then
    cache_root="/tmp/stockfish-cache"
    if ! mkdir -p "$cache_root" 2>/dev/null; then
        echo "[stockfish-build] ERROR: neither $CACHE_ROOT nor $cache_root writable" >&2
        exit 2
    fi
    echo "[stockfish-build] WARN: $CACHE_ROOT not writable; using volatile $cache_root" >&2
fi

# Cache key: commit + arch. Stable across re-runs of this script with the
# same source. Hashed because some pod filesystems are funny about long
# directory names with colons / slashes.
cache_key=$(printf '%s:%s' "$SF_COMMIT" "$arch" | sha256sum | cut -c1-16)
cache_dir="$cache_root/$cache_key"
cache_bin="$cache_dir/stockfish-patched"

short_commit=$(printf '%s' "$SF_COMMIT" | cut -c1-12)

# ── 3. Build (if not cached) ───────────────────────────────────────────
if [ -x "$cache_bin" ]; then
    cache_status="hit"
    elapsed=0
else
    cache_status="miss"
    t0=$(date +%s 2>/dev/null || echo 0)
    # `make build` is the stockfish standard target (runs `make profile-build`
    # for the configured ARCH). Use -j to parallelize.
    (cd "$SF_SRC" && make -j"$(nproc)" build ARCH="$arch" >/tmp/stockfish-build.log 2>&1) || {
        echo "[stockfish-build] ERROR: make failed for ARCH=$arch" >&2
        tail -40 /tmp/stockfish-build.log >&2 || true
        exit 1
    }
    # Strip is mandatory: unstripped binary is ~30 MB vs ~7 MB stripped.
    strip "$SF_SRC/stockfish"
    mkdir -p "$cache_dir"
    # Atomic install: copy to a tempfile in the SAME directory (so `mv` is
    # a rename, not a cross-fs copy), chmod, then `mv` over the final
    # path. Without this, a pod interrupted mid-`cp` leaves a truncated
    # binary at "$cache_bin"; the next launch's `-x "$cache_bin"` test
    # passes (`-x` only checks the executable bit, not size or content),
    # so the script symlinks a corrupt binary and the rust runner crashes
    # on first UCI handshake. The operator would have to manually delete
    # the cache entry to recover.
    cache_tmp="$cache_dir/.stockfish-patched.tmp.$$"
    cp "$SF_SRC/stockfish" "$cache_tmp"
    chmod +x "$cache_tmp"
    mv "$cache_tmp" "$cache_bin"
    t1=$(date +%s 2>/dev/null || echo 0)
    elapsed=$((t1 - t0))
fi

# ── 4. Symlink into the path configs reference ─────────────────────────
mkdir -p "$(dirname "$SF_TARGET")"
ln -sfn "$cache_bin" "$SF_TARGET"

if [ "$cache_status" = "miss" ]; then
    echo "[stockfish-build] arch=$arch, cache=miss (built in ${elapsed}s), commit=$short_commit, path=$SF_TARGET"
else
    echo "[stockfish-build] arch=$arch, cache=hit, commit=$short_commit, path=$SF_TARGET"
fi
