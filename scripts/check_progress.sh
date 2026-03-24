#!/usr/bin/env bash
# Check training progress, sync from pods, and auto-stop finished pods.
# Usage: check_progress.sh [--sync] [--auto-stop] [LOG_DIR]
set -euo pipefail

SYNC=false
AUTO_STOP=false
LOG_DIR=""

for arg in "$@"; do
    case "$arg" in
        --sync) SYNC=true ;;
        --auto-stop) AUTO_STOP=true ;;
        *) LOG_DIR="$arg" ;;
    esac
done
LOG_DIR="${LOG_DIR:-logs}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
POD_DIR="$HOME/.config/pawn/pods"

# Sync from all pods
if $SYNC && [ -d "$POD_DIR" ]; then
    bash "$REPO/deploy/sync.sh" 2>/dev/null || true
fi

# Show progress for top 5 most recent runs
find "$LOG_DIR" -name metrics.jsonl -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -n 5 | while read -r _ path; do
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
if pgrep -f 'train.py.*discard-ply-limit' > /dev/null 2>&1; then
    echo "Local discard-ply-limit run: RUNNING"
else
    echo "WARNING: Local discard-ply-limit run: NOT RUNNING"
fi

# Auto-stop finished pods
if $AUTO_STOP && [ -d "$POD_DIR" ]; then
    for env_file in "$POD_DIR"/*.env; do
        [ -f "$env_file" ] || continue
        pod_name="$(basename "${env_file%.env}")"
        unset POD_ID POD_HOST POD_PORT POD_GPU 2>/dev/null || true
        source "$env_file"

        # Check if process is alive on pod
        alive=$(ssh -o ConnectTimeout=5 -p "$POD_PORT" "root@$POD_HOST" \
            "pgrep -f 'train.py' > /dev/null 2>&1 && echo yes || echo no" 2>/dev/null || echo "unreachable")

        if [ "$alive" = "no" ]; then
            echo ">>> $pod_name: training finished. Final sync + stopping..."
            bash "$REPO/deploy/sync.sh" "$pod_name" 2>/dev/null || true
            runpodctl pod stop "$POD_ID" 2>/dev/null
            echo ">>> $pod_name ($POD_ID) STOPPED"
        fi
    done
fi
