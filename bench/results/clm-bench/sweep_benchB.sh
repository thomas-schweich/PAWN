#!/usr/bin/env bash
# Bench B batch-size sweep: v1 cotrain vs v2 supernet (matched + all-in).
# Sequential (no parallel training locally). Tolerates OOMs (records them).
set -uo pipefail

V2=/home/tas/pawn
V1=/tmp/pawn-v1
OUT=$V2/bench/results/clm-bench/sweep
mkdir -p "$OUT"
STEPS=1000
BATCHES="4 8 16 32 64"
RES="$OUT/results.tsv"
echo -e "config\tbatch\tms_per_step\tgames_per_sec\tstatus" > "$RES"

oom_in() { grep -qiE "RESOURCE_EXHAUSTED|out of memory|HIP out of memory|CUDA out of memory|OutOfMemory" "$1"; }
err_in() { grep -qiE "Traceback|Error:|assert|Killed" "$1"; }
# record config batch logfile ms gs
record() {
  local cfg=$1 b=$2 log=$3 ms=$4 gs=$5 status=ok
  if oom_in "$log"; then status=OOM; ms=NA; gs=NA
  elif [ "$ms" = NA ]; then status=$(err_in "$log" && echo ERR || echo NODATA)
  fi
  echo -e "${cfg}\t${b}\t${ms}\t${gs}\t${status}" >> "$RES"
}

# v2 steady-state from cumulative step_time deltas in metrics.jsonl
v2_extract() {  # $1=logdir $2=batch
  local f
  f=$(find "$1" -name metrics.jsonl 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "NA NA"; return; }
  python3 - "$f" "$2" <<'PY'
import json,sys,statistics
f,b=sys.argv[1],float(sys.argv[2])
tr=[json.loads(l) for l in open(f) if l.strip() and json.loads(l).get("type")=="train"]
pts=[(r["step"], r["step_time"]*r["step"]) for r in tr if r.get("step_time")]
if len(pts)<3: print("NA NA"); sys.exit()
d=[(pts[i][1]-pts[i-1][1])/(pts[i][0]-pts[i-1][0]) for i in range(1,len(pts))]
ms=statistics.median(d[1:])*1000  # drop first (compile-heavy) interval
print(f"{ms:.1f} {b/(ms/1000):.0f}")
PY
}

# v1 cotrain logs "step N | X g/s | Ys" to stdout; take median of steady tail.
v1_extract() {  # $1=logfile $2=batch
  python3 - "$1" "$2" <<'PY'
import re,sys,statistics
log,b=sys.argv[1],float(sys.argv[2])
gs=[]; st=[]
for line in open(log,errors="ignore"):
    m=re.search(r"\|\s*([0-9.]+)\s*g/s\s*\|\s*([0-9.]+)s",line)
    if m: gs.append(float(m.group(1))); st.append(float(m.group(2)))
if len(gs)<4: print("NA NA"); sys.exit()
tail=slice(len(gs)//2,None)  # second half = steady
g=statistics.median(gs[tail]); s=statistics.median(st[tail])
print(f"{s*1000:.1f} {g:.0f}")
PY
}

for B in $BATCHES; do
  # ---- v1 cotrain (PyTorch, fixed-512, all 3 variants/step) ----
  log="$OUT/v1_cotrain_b${B}.log"; ld="$OUT/v1_cotrain_b${B}"
  rm -rf "$ld"
  ( cd "$V1" && uv run --extra rocm python scripts/train.py \
      --config cotrain_bench.json --batch-size "$B" --total-steps "$STEPS" \
      --log-dir "$ld" --local-checkpoints ) > "$log" 2>&1
  read ms gs <<<"$(v1_extract "$log" "$B")"; record v1-cotrain "$B" "$log" "$ms" "$gs"

  # ---- v2 supernet matched (fixed-512, all 3 variants/step) ----
  log="$OUT/v2_matched_b${B}.log"; ld="$OUT/v2_matched_b${B}"
  rm -rf "$ld"
  ( cd "$V2" && uv run --extra rocm python scripts/train_jax.py \
      --supernet production --batch-size "$B" --seq-len 512 --total-steps "$STEPS" --k 50 \
      --no-bucketing --no-stochastic-variants \
      --local-checkpoints --logs-dir "$ld" ) > "$log" 2>&1
  read ms gs <<<"$(v2_extract "$ld" "$B")"; record v2-matched "$B" "$log" "$ms" "$gs"

  # ---- v2 supernet all-in (bucketed + stochastic) ----
  log="$OUT/v2_allin_b${B}.log"; ld="$OUT/v2_allin_b${B}"
  rm -rf "$ld"
  ( cd "$V2" && uv run --extra rocm python scripts/train_jax.py \
      --supernet production --batch-size "$B" --seq-len 512 --total-steps "$STEPS" --k 50 \
      --local-checkpoints --logs-dir "$ld" ) > "$log" 2>&1
  read ms gs <<<"$(v2_extract "$ld" "$B")"; record v2-allin "$B" "$log" "$ms" "$gs"

  echo "=== batch $B done ===" >> "$OUT/progress.log"
  column -t "$RES" >> "$OUT/progress.log"
done

echo "SWEEP_COMPLETE"
column -t "$RES"
