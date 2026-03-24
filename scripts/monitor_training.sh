#!/usr/bin/env bash
# Monitor multi-model training: check pod log + HuggingFace checkpoints.
# Usage: bash scripts/monitor_training.sh <host> <port>
set -euo pipefail

HOST="${1:-50.145.48.110}"
PORT="${2:-13321}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $PORT root@$HOST"

echo "=== Training Log ==="
$SSH "tail -15 /workspace/logs/train_all.log" 2>/dev/null || echo "  (SSH failed)"

echo ""
echo "=== Process Status ==="
$SSH "pgrep -f train_all > /dev/null && echo RUNNING || echo STOPPED" 2>/dev/null || echo "  (SSH failed)"

echo ""
echo "=== HuggingFace Checkpoints ==="
uv run python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for variant in ['small', 'base', 'large']:
    repo = f'thomas-schweich/pawn-{variant}'
    try:
        branches = [b.name for b in api.list_repo_refs(repo, repo_type='model').branches if b.name.startswith('run/')]
        for branch in branches:
            files = [f.rfilename for f in api.list_repo_tree(repo, revision=branch, repo_type='model', recursive=True) if hasattr(f, 'rfilename') and 'checkpoints/' in f.rfilename]
            ckpts = sorted(set(f.split('/')[1] for f in files if f.startswith('checkpoints/step_')))
            print(f'  {repo}@{branch}: {len(ckpts)} checkpoints ({ckpts[-1] if ckpts else \"none\"})')
        if not branches:
            print(f'  {repo}: no run branches')
    except Exception as e:
        print(f'  {repo}: {e}')
" 2>/dev/null || echo "  (HF check failed)"

echo ""
echo "=== Metrics Sync ==="
rsync -az --include='*/' --include='metrics.jsonl' --include='config.json' --exclude='*' \
    -e "ssh -o StrictHostKeyChecking=accept-new -p $PORT" \
    "root@$HOST:/workspace/logs/" logs/ 2>/dev/null && echo "  Synced" || echo "  (Sync failed)"
