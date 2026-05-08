#!/usr/bin/env bash
# Quick n_workers sweep on the pod hardware.
#
# Run this BEFORE launching the production fire-and-forget run to validate
# that the box's actual topology matches the affinity rule of thumb's
# prediction (n_workers = total_threads - threads_per_core, see
# stockfish-datagen/src/affinity.rs).
#
# Each point is a 5000-game nodes=1 run (~5-15s of pod time) and exercises
# the full pipeline including CPU pinning. Total sweep is ~1-3 minutes.
# Output is to the pod's local disk; nothing is uploaded to HF.
#
# Usage on a vast.ai pod:
#
#   # Auto-detect topology, sweep ±4 around the rule's prediction in steps of 2:
#   bash /opt/datagen/scripts/sweep_pod_workers.sh
#
#   # Explicit list of n_workers to test:
#   bash /opt/datagen/scripts/sweep_pod_workers.sh 122 124 126 128
#
#   # Single value (e.g. just verify the predicted optimum):
#   bash /opt/datagen/scripts/sweep_pod_workers.sh 126
#
# Override the binary location with STOCKFISH_DATAGEN_BIN if not on PATH;
# override the per-point game count with SWEEP_N_GAMES (default 5000).

set -euo pipefail

BIN="${STOCKFISH_DATAGEN_BIN:-stockfish-datagen}"
N_GAMES="${SWEEP_N_GAMES:-5000}"
OUT_BASE="/tmp/sf_sweep"

if ! command -v "$BIN" >/dev/null 2>&1 && ! [ -x "$BIN" ]; then
    echo "error: $BIN not found on PATH and not an executable file" >&2
    echo "       set STOCKFISH_DATAGEN_BIN to the binary's path" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    workers_list=("$@")
else
    total=$(nproc)
    # Threads per core from lscpu. Falls back to 2 if lscpu is missing
    # or the field isn't present (uncommon on real Linux but possible
    # in stripped containers).
    tpc=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/{gsub(/[ \t]/,"",$2); print $2}')
    : "${tpc:=2}"
    rule=$((total - tpc))
    workers_list=()
    for delta in -4 -2 0 2 4; do
        n=$((rule + delta))
        if [ "$n" -gt 0 ] && [ "$n" -le "$total" ]; then
            workers_list+=("$n")
        fi
    done
    echo "auto-detected: total_threads=$total, threads_per_core=$tpc"
    echo "rule prediction: $rule workers"
    echo "sweeping: ${workers_list[*]}"
    echo
fi

mk_config() {
    local n=$1 outdir=$2
    cat <<EOC
{
  "stockfish_path": "~/bin/stockfish",
  "stockfish_version": "Stockfish 18",
  "output_dir": "$outdir",
  "master_seed": 42,
  "n_workers": $n,
  "max_ply": 512,
  "stockfish_hash_mb": 16,
  "shard_size_games": 1000000,
  "tiers": [{"name":"sweep","nodes":1,"n_games":$N_GAMES,"multi_pv":5,"opening_multi_pv":20,"opening_plies":2,"sample_plies":12,"temperature":1.0}]
}
EOC
}

results=()
for n in "${workers_list[@]}"; do
    outdir="${OUT_BASE}_w${n}"
    rm -rf "$outdir"
    cfg=$(mktemp --suffix=.json)
    mk_config "$n" "$outdir" > "$cfg"
    # Two-arg `match()` + RSTART/RLENGTH for POSIX/mawk compatibility.
    # The three-arg form (`match(str, re, array)`) is a gawk extension
    # and crashes mawk — which is the default `awk` on Ubuntu 24.04
    # minimal containers (incl. our dev image), causing "rate=ERROR" for
    # every worker on the pod.
    rate=$("$BIN" run --config "$cfg" 2>&1 \
        | awk '/games\/s/{if(match($0, /\([0-9.]+/)){print substr($0, RSTART+1, RLENGTH-1); exit}}')
    rm -f "$cfg"
    rm -rf "$outdir"
    if [ -z "${rate:-}" ]; then
        printf "  workers=%4d  rate=ERROR (run failed; rerun manually for details)\n" "$n"
        continue
    fi
    printf "  workers=%4d  rate=%s g/s\n" "$n" "$rate"
    results+=("$n $rate")
done

echo
echo "=== summary (sorted by rate, descending) ==="
printf '%s\n' "${results[@]}" \
    | sort -k2 -n -r \
    | awk '{printf "  workers=%4d  rate=%s g/s\n", $1, $2}'
echo
echo "Set this in your run config's n_workers field for the production run."
