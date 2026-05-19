#!/usr/bin/env bash
# Status + ETA check for the in-flight stockfish-datagen run on a vast.ai pod.
#
# Pulls live state from two sources:
#   - the pod (via SSH): tmux session, worker count, /dev/shm usage, load
#     average, and the latest per-tier / per-worker progress lines from
#     /workspace/datagen.log.
#   - HuggingFace (via huggingface_hub): authoritative count of shards
#     committed per tier on the target dataset.
#
# ETA is averaged: (games_remaining) / (games_committed / elapsed_run_hours).
# That averages over all tiers seen so far. Nodes=1024 tier games are
# ~1000× slower per game than nodes=1, so once we cross tier boundaries the
# rate drops and the average ETA stretches — that's expected, not a bug.
#
# Intended for `/loop 30m bash scripts/check_datagen_pod.sh`.
set -euo pipefail

POD_NAME=datagen-md
REPO=thomas-schweich/pawn-stockfish-100m
DPH=1.4370  # Maryland EPYC 9654 verified 382d (offer 36721322)
ENVFILE="$HOME/.config/pawn/vast/$POD_NAME.env"

if [ ! -f "$ENVFILE" ]; then
    echo "[check] no instance config at $ENVFILE — pod is not running"
    exit 0
fi

# shellcheck source=/dev/null
. "$ENVFILE"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
          -o LogLevel=ERROR -p "$INSTANCE_PORT" "root@$INSTANCE_HOST")

echo "=== pod $POD_NAME ($INSTANCE_HOST:$INSTANCE_PORT) ==="
if ! ssh "${SSH_OPTS[@]}" true 2>/dev/null; then
    echo "  SSH unreachable — pod may be stopped or networking down"
    exit 0
fi

ssh "${SSH_OPTS[@]}" bash -s <<'EOSSH'
echo "  tmux:    $(tmux list-sessions 2>/dev/null || echo 'NO ACTIVE SESSION')"
echo "  workers: $(pgrep -fc stockfish-datagen/stockfish-patched 2>/dev/null || echo 0)"
echo "  shm:     $(df -h /dev/shm | awk 'NR==2 {print $3" / "$2" ("$5" used)"}')"
echo "  uptime:  $(uptime -p)"
echo "  load1:   $(awk '{print $1}' /proc/loadavg)"
echo
echo "=== last 6 per-worker checkpoints from log ==="
tail -1000 /workspace/datagen.log 2>/dev/null | grep -E '\[.*w *[0-9]+\] *[0-9]+ shard.*[0-9]+ games$' | tail -6 || echo "  (no progress lines yet)"
echo
echo "=== last per-tier completion summaries ==="
grep -E '^\[(tier0|nodes)_.*\] (starting|complete:|.* in .*games/s)' /workspace/datagen.log 2>/dev/null | tail -8 || echo "  (no tier summaries yet)"
EOSSH

echo
echo "=== HF state: $REPO ==="
uv run --with huggingface_hub python3 - <<EOF
import os, re, time
from huggingface_hub import HfApi
api = HfApi()
repo = "$REPO"
files = api.list_repo_files(repo, repo_type="dataset")
shard_re = re.compile(r"([^/]+)/shard-s(\d+)-r(\d+)\.parquet\$")

tiers_total_shards = {
    "tier0_evallegal": 10000,
    "nodes_0001":      10000,
    "nodes_0128":      10000,
    "nodes_0256":      10000,
    "nodes_1024":      10000,
}
shard_size = 2000
target_games = sum(tiers_total_shards.values()) * shard_size

by_tier = {}
for f in files:
    m = shard_re.match(f)
    if m:
        tier, sid, rows = m.group(1), int(m.group(2)), int(m.group(3))
        by_tier.setdefault(tier, []).append(rows)

games_done = 0
for tier, total in tiers_total_shards.items():
    rows_list = by_tier.get(tier, [])
    n_shards = len(rows_list)
    g = sum(rows_list)
    games_done += g
    pct = 100 * n_shards / total if total else 0
    print(f"  {tier:18s} {n_shards:6d}/{total:>5d} shards ({pct:5.1f}%)  {g:>12,} games")
remaining = target_games - games_done
pct_total = 100 * games_done / target_games
print(f"  {'TOTAL':18s} {sum(len(v) for v in by_tier.values()):6d}/{sum(tiers_total_shards.values()):>5d} shards ({pct_total:5.2f}%) {games_done:>12,}/{target_games:,} games")

# ETA: derive elapsed-since-start from the tmux session creation, fall back
# to first sentinel commit time on HF (less precise — orchestrator primer
# may run a minute or two before games start).
EOF

# Wall-clock from tmux session, ETA, cost so far
START_EPOCH=$(ssh "${SSH_OPTS[@]}" "tmux display -t datagen -p '#{session_created}'" 2>/dev/null || echo "")
if [ -n "$START_EPOCH" ] && [ "$START_EPOCH" -gt 0 ] 2>/dev/null; then
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_EPOCH))
    HOURS=$(awk "BEGIN {printf \"%.2f\", $ELAPSED/3600}")
    COST=$(awk "BEGIN {printf \"%.2f\", $HOURS * $DPH}")
    echo
    echo "=== wall clock + in-flight rate ==="
    echo "  elapsed (this run):  ${HOURS}h ($ELAPSED s)"
    echo "  cost (this run):     \$$COST  (at \$$DPH/h)"

    # In-flight games-done this run: each worker logs "[tier wN] X shard(s), Y
    # games" at every 500-game checkpoint. Y is cumulative for the worker for
    # this run, so summing the max(Y) per (tier, worker) gives total games
    # produced — including the ones already committed to HF *during this
    # run* AND the ones still buffered in the rust binary's parquet writer.
    # This is the right denominator for "actual throughput right now" since
    # the watcher's HF-upload cadence lags real production by ~1-2 shards
    # per worker.
    WORKER_GAMES_THIS_RUN=$(ssh "${SSH_OPTS[@]}" "tail -50000 /workspace/datagen.log 2>/dev/null | grep -oE '\[[a-z0-9_]+ w *[0-9]+\] *[0-9]+ shard\(s\), *[0-9]+ games\$'" 2>/dev/null | python3 -c "
import sys, re
w = {}
for line in sys.stdin:
    m = re.match(r'\[([a-z0-9_]+) w *(\d+)\] *\d+ shard\(s\), *(\d+) games', line.strip())
    if m:
        key = (m.group(1), int(m.group(2)))
        g = int(m.group(3))
        if g > w.get(key, 0):
            w[key] = g
print(sum(w.values()), len(w))
" 2>/dev/null || echo "0 0")
    THIS_RUN_GAMES=$(echo "$WORKER_GAMES_THIS_RUN" | awk '{print $1}')
    N_WORKERS_REPORTING=$(echo "$WORKER_GAMES_THIS_RUN" | awk '{print $2}')
    echo "  in-flight + done:    $THIS_RUN_GAMES games across $N_WORKERS_REPORTING workers (this run)"

    # HF count is cumulative across all runs against this repo, so it's the
    # right thing to subtract from TARGET for "remaining work" math.
    HF_GAMES=$(uv run --with huggingface_hub python3 - <<EOF2
from huggingface_hub import HfApi
import re
api = HfApi()
files = api.list_repo_files("$REPO", repo_type="dataset")
shard_re = re.compile(r"([^/]+)/shard-s(\d+)-r(\d+)\.parquet\$")
print(sum(int(m.group(3)) for f in files if (m:=shard_re.match(f))))
EOF2
    )

    if [ -n "$THIS_RUN_GAMES" ] && [ "$THIS_RUN_GAMES" -gt 0 ] 2>/dev/null && [ "$ELAPSED" -gt 60 ]; then
        RATE=$(awk "BEGIN {printf \"%.2f\", $THIS_RUN_GAMES / $ELAPSED}")
        # "Done so far" for ETA arithmetic: HF cumulative + new in-flight
        # not yet on HF. This double-counts the in-flight-already-committed
        # delta but it's a small bias and self-corrects as the watcher
        # flushes.
        EFFECTIVE_DONE=$(( HF_GAMES > THIS_RUN_GAMES ? HF_GAMES : THIS_RUN_GAMES ))
        REMAINING=$(( 100000000 - EFFECTIVE_DONE ))
        ETA_S=$(awk "BEGIN {printf \"%d\", $REMAINING / ($THIS_RUN_GAMES / $ELAPSED)}")
        ETA_H=$(awk "BEGIN {printf \"%.1f\", $ETA_S / 3600}")
        FINISH_ESTIMATED_AT=$(date -d "+$ETA_S seconds" '+%Y-%m-%d %H:%M:%S %Z')
        EST_TOTAL_COST=$(awk "BEGIN {printf \"%.2f\", ($ELAPSED + $ETA_S) / 3600 * $DPH}")
        echo "  rate:                $RATE games/s (this-run average, in-flight included)"
        echo "  ETA:                 ${ETA_H}h remaining → finish around $FINISH_ESTIMATED_AT"
        echo "  est total compute:   \$$EST_TOTAL_COST  (egress ~\$4 extra per TB)"
        # Sanity flag: ETA growing rapidly vs the simple HF-only rate
        # is expected during warm-up (in-flight outpaces watcher); flagging
        # only when in-flight rate drops to 1/2 of itself between checks
        # would need cross-check state — out of scope here.
    else
        echo "  rate:                (need >60s elapsed and >0 worker checkpoints)"
    fi
fi
