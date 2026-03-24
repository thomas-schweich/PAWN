#!/usr/bin/env bash
# Check training progress from HuggingFace submodules and local logs.
# Usage: check_progress.sh [--sync] [LOG_DIR]
set -euo pipefail

SYNC=false
LOG_DIR=""

for arg in "$@"; do
    case "$arg" in
        --sync) SYNC=true ;;
        *) LOG_DIR="$arg" ;;
    esac
done
LOG_DIR="${LOG_DIR:-logs}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Sync submodules from HuggingFace
if $SYNC; then
    bash "$REPO/deploy/sync.sh" 2>/dev/null || true
fi

# Show progress from all metrics.jsonl files (local logs + submodules)
N=5
{
    find "$LOG_DIR" -name metrics.jsonl -printf '%T@ %p\n' 2>/dev/null
    find "$REPO/checkpoints" -name metrics.jsonl -printf '%T@ %p\n' 2>/dev/null
} | sort -rn | head -n "$N" | while read -r _ path; do
    run_name="$(basename "$(dirname "$path")")"

    python3 -c "
import json, sys
records = [json.loads(l) for l in open('$path')]
cfg = next((r for r in records if r.get('type') == 'config'), {})
train = [r for r in records if r.get('type') == 'train']
if not train:
    print(f'$run_name  (no training steps yet)')
    sys.exit(0)
last = train[-1]
model = cfg.get('model', {})
tcfg = cfg.get('training', {})
variant = f\"{model.get('d_model','?')}d/{model.get('n_layers','?')}L\"
discard = tcfg.get('discard_ply_limit', False)
total = tcfg.get('total_steps', '?')
step = last.get('step', '?')
loss = last.get('train/loss', last.get('loss', 0))
acc = last.get('train/accuracy', last.get('acc', 0))
gs = last.get('games_per_sec', 0)
print(f'{\"$run_name\":<28}  {variant}  discard_ply={str(discard):<5}  step {step}/{total}  loss {loss:.4f}  acc {acc:.3f}  {gs:.0f} g/s')
" 2>/dev/null || echo "$run_name  (parse error)"
done

# Check local training process
if pgrep -f 'train.py' > /dev/null 2>&1; then
    echo "Local training: RUNNING"
else
    echo "Local training: NOT RUNNING"
fi
