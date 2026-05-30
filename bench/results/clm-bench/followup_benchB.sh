#!/usr/bin/env bash
# Bench B follow-up: (1) does --bucket-outer-factor change throughput
# (i.e. is the GPU datagen-starved?), and (2) does Lion unlock a bigger
# local batch than AdamW (which OOM'd v2-allin at B=32)?
# rocm-smi can't report util in WSL2, so datagen-boundedness is inferred
# from whether g/s rises with the outer factor.
set -uo pipefail
V2=/home/tas/pawn
OUT=$V2/bench/results/clm-bench/followup
mkdir -p "$OUT"
STEPS=1000
RES="$OUT/results.tsv"
echo -e "run\tbatch\touter_factor\toptimizer\tms_per_step\tgames_per_sec\tstatus" > "$RES"

oom_in() { grep -qiE "RESOURCE_EXHAUSTED|out of memory|HIP out of memory|OutOfMemory" "$1"; }
err_in() { grep -qiE "Traceback|Error:|assert|Killed" "$1"; }
extract() {  # logdir batch -> "ms gs"
  local f; f=$(find "$1" -name metrics.jsonl 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "NA NA"; return; }
  python3 - "$f" "$2" <<'PY'
import json,sys,statistics
f,b=sys.argv[1],float(sys.argv[2])
tr=[json.loads(l) for l in open(f) if l.strip() and json.loads(l).get("type")=="train"]
pts=[(r["step"], r["step_time"]*r["step"]) for r in tr if r.get("step_time")]
if len(pts)<3: print("NA NA"); sys.exit()
d=[(pts[i][1]-pts[i-1][1])/(pts[i][0]-pts[i-1][0]) for i in range(1,len(pts))]
ms=statistics.median(d[1:])*1000
print(f"{ms:.1f} {b/(ms/1000):.0f}")
PY
}
record() {  # run batch of opt log ms gs
  local run=$1 b=$2 of=$3 opt=$4 log=$5 ms=$6 gs=$7 st=ok
  if oom_in "$log"; then st=OOM; ms=NA; gs=NA
  elif [ "$ms" = NA ]; then st=$(err_in "$log" && echo ERR || echo NODATA); fi
  echo -e "${run}\t${b}\t${of}\t${opt}\t${ms}\t${gs}\t${st}" >> "$RES"
}

run_allin() {  # batch outer_factor optimizer [extra lr args]
  local b=$1 of=$2 opt=$3; shift 3
  local tag="b${b}_of${of}_${opt}"
  local log="$OUT/${tag}.log" ld="$OUT/${tag}"
  rm -rf "$ld"
  local optflag=""; [ "$opt" = lion ] && optflag="--optimizer lion --lr 1e-4"
  ( cd "$V2" && uv run --extra rocm python scripts/train_jax.py \
      --supernet production --batch-size "$b" --seq-len 512 --total-steps "$STEPS" --k 50 \
      --bucket-outer-factor "$of" $optflag \
      --local-checkpoints --logs-dir "$ld" ) > "$log" 2>&1
  read ms gs <<<"$(extract "$ld" "$b")"
  record "allin" "$b" "$of" "$opt" "$log" "$ms" "$gs"
  echo "  done: $tag -> ${gs} g/s (${ms} ms)"
}

# 1) Outer-factor sweep at the local-max all-in batch (B=16), AdamW.
for OF in 3 6 12; do run_allin 16 "$OF" adamw; done

# 2) Lion: does it fit a bigger batch than AdamW (which OOM'd at B=32)?
run_allin 16 3 lion
run_allin 32 3 lion

echo "FOLLOWUP_COMPLETE"
column -t "$RES"
