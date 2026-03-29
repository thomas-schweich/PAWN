#!/usr/bin/env bash
# check_ceiling_run.sh — check progress of ceiling computation
# Output file being written to: data/theoretical_ceiling.json
# Log file: /tmp/ceiling_run.log
set -euo pipefail

LOG="/tmp/ceiling_run.log"
OUT="/home/tas/pawn/.claude/worktrees/improve-ceiling-methodology/data/theoretical_ceiling.json"

echo "=== Ceiling Computation Status ($(date '+%H:%M:%S')) ==="
echo

# Check if the process is still running
if pgrep -f "compute_theoretical_ceiling" > /dev/null 2>&1; then
    echo "Status: RUNNING"
    # Find the actual python process (highest CPU), not the shell wrapper
    pid=$(pgrep -f "compute_theoretical_ceiling" | tail -1)
    elapsed=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$elapsed" ]; then
        mins=$((elapsed / 60))
        secs=$((elapsed % 60))
        echo "Elapsed: ${mins}m ${secs}s"
    fi
    echo
    echo "--- process stats ---"
    ps -p "$pid" -o pid,pcpu,pmem,cputime,etime,args --no-headers 2>/dev/null
else
    echo "Status: NOT RUNNING"
fi
echo

# Show latest log output
if [ -f "$LOG" ]; then
    lines=$(wc -l < "$LOG")
    echo "Log lines: $lines"
    echo "--- Last 15 lines of log ---"
    tail -15 "$LOG"
else
    echo "No log file yet."
fi
echo

# Check if output file exists and is complete
if [ -f "$OUT" ]; then
    size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    echo "Output file: $OUT (${size} bytes)"
    if python3 -c "import json; d=json.load(open('$OUT')); print(f\"  Positions: {d['n_positions']:,}\"); print(f\"  Uncond: {d['unconditional_ceiling']:.4f}\"); print(f\"  MC naive: {d['conditional_ceiling']:.4f}\"); print(f\"  MC corrected: {d['conditional_corrected_ceiling']:.4f}\"); print(f\"  Bracket: {d['bias_bracket_pp']:.3f}pp\")" 2>/dev/null; then
        echo "  -> Output is valid JSON with results!"
    fi
else
    echo "Output file not yet written."
fi
