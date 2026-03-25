#!/usr/bin/env bash
# Monitor multi-model training: check pod log + HuggingFace checkpoints.
# Usage: bash scripts/monitor_training.sh [<pod-id>]
#
# If pod-id is given, resolves SSH host/port via runpodctl.
# Otherwise checks HuggingFace only (no SSH).
set -euo pipefail

POD_ID="${1:-}"
SSH=""

if [ -n "$POD_ID" ]; then
    # Resolve SSH connection from runpodctl
    ssh_info=$(runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
ssh = d.get('ssh', {})
host = ssh.get('ip', '') or ssh.get('host', '')
port = ssh.get('port', '')
status = ssh.get('status', '')
error = ssh.get('error', '')
if host and port:
    print(f'{host} {port}')
elif error:
    print(f'ERROR {error}')
else:
    print(f'ERROR status={status}')
" 2>/dev/null || echo "ERROR runpodctl-failed")

    if [[ "$ssh_info" == ERROR* ]]; then
        echo "=== Pod Status ==="
        echo "  Pod $POD_ID: ${ssh_info#ERROR }"
        echo ""
    else
        HOST=$(echo "$ssh_info" | cut -d' ' -f1)
        PORT=$(echo "$ssh_info" | cut -d' ' -f2)
        SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $PORT root@$HOST"
    fi
fi

if [ -n "$SSH" ]; then
    echo "=== Training Log ==="
    $SSH "tail -15 /opt/pawn/logs/*/metrics.jsonl 2>/dev/null | tail -15" 2>/dev/null || echo "  (SSH failed)"

    echo ""
    echo "=== Process Status ==="
    $SSH "pgrep -f train_all > /dev/null && echo RUNNING || echo STOPPED" 2>/dev/null || echo "  (SSH failed)"

    echo ""
    echo "=== Metrics Sync ==="
    rsync -az --include='*/' --include='metrics.jsonl' --include='config.json' --exclude='*' \
        -e "ssh -o StrictHostKeyChecking=accept-new -p $PORT" \
        "root@$HOST:/opt/pawn/logs/" logs/ 2>/dev/null && echo "  Synced" || echo "  (Sync failed)"
fi

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
