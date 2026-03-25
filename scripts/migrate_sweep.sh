#!/usr/bin/env bash
# Migrate an Optuna sweep from one Runpod pod to another.
#
# Usage:
#   bash scripts/migrate_sweep.sh <old-pod-id> <new-pod-id> [remote-dir]
#
# Steps:
#   1. Resolves SSH for both pods via runpodctl
#   2. Syncs sweep directory from old pod to local staging
#   3. Syncs from local staging to new pod
#   4. Reports what was transferred
#
# The remote directory defaults to /workspace/sweeps.
set -euo pipefail

OLD_POD="${1:?Usage: migrate_sweep.sh <old-pod-id> <new-pod-id> [remote-dir]}"
NEW_POD="${2:?Usage: migrate_sweep.sh <old-pod-id> <new-pod-id> [remote-dir]}"
REMOTE_DIR="${3:-/workspace/sweeps}"
LOCAL_STAGING="/tmp/pawn_sweep_migrate"

resolve_ssh() {
    local pod_id="$1"
    runpodctl pod get "$pod_id" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
ssh = d.get('ssh', {})
ip = ssh.get('ip', '') or ssh.get('host', '')
port = ssh.get('port', '')
if ip and port:
    print(f'{ip} {port}')
else:
    print('ERROR ' + ssh.get('error', 'unknown'))
" 2>/dev/null
}

echo "=== Migrate Sweep ==="
echo "Old pod: $OLD_POD"
echo "New pod: $NEW_POD"
echo "Remote dir: $REMOTE_DIR"
echo ""

# Resolve SSH for old pod
echo "Resolving old pod SSH..."
old_ssh=$(resolve_ssh "$OLD_POD")
if [[ "$old_ssh" == ERROR* ]]; then
    echo "  Old pod SSH failed: $old_ssh"
    exit 1
fi
OLD_HOST=$(echo "$old_ssh" | cut -d' ' -f1)
OLD_PORT=$(echo "$old_ssh" | cut -d' ' -f2)
echo "  $OLD_HOST:$OLD_PORT"

# Resolve SSH for new pod
echo "Resolving new pod SSH..."
new_ssh=$(resolve_ssh "$NEW_POD")
if [[ "$new_ssh" == ERROR* ]]; then
    echo "  New pod SSH failed: $new_ssh"
    exit 1
fi
NEW_HOST=$(echo "$new_ssh" | cut -d' ' -f1)
NEW_PORT=$(echo "$new_ssh" | cut -d' ' -f2)
echo "  $NEW_HOST:$NEW_PORT"

# Pull from old pod
echo ""
echo "Pulling from old pod..."
mkdir -p "$LOCAL_STAGING"
rsync -az --progress \
    -e "ssh -o StrictHostKeyChecking=accept-new -p $OLD_PORT" \
    "root@$OLD_HOST:$REMOTE_DIR/" "$LOCAL_STAGING/"
echo "  Pulled to $LOCAL_STAGING"

# Show what we got
echo ""
echo "=== Transferred ==="
du -sh "$LOCAL_STAGING"
find "$LOCAL_STAGING" -name "study.db" -exec ls -lh {} \;
echo "Trial dirs: $(find "$LOCAL_STAGING" -maxdepth 2 -name 'trial_*' -type d | wc -l)"
echo "Metrics files: $(find "$LOCAL_STAGING" -name 'metrics.jsonl' | wc -l)"

# Push to new pod
echo ""
echo "Pushing to new pod..."
ssh -o StrictHostKeyChecking=accept-new -p "$NEW_PORT" "root@$NEW_HOST" "mkdir -p $REMOTE_DIR"
rsync -az --no-owner --no-group --progress \
    -e "ssh -o StrictHostKeyChecking=accept-new -p $NEW_PORT" \
    "$LOCAL_STAGING/" "root@$NEW_HOST:$REMOTE_DIR/"
echo "  Pushed to $NEW_HOST:$REMOTE_DIR"

# Inject HF token on new pod
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo ""
    echo "Injecting HF token on new pod..."
    cat "$HOME/.cache/huggingface/token" | ssh -p "$NEW_PORT" "root@$NEW_HOST" \
        'mkdir -p /root/.cache/huggingface && cat > /root/.cache/huggingface/token'
    echo "  Done"
fi

echo ""
echo "=== Migration complete ==="
echo "Old pod ($OLD_POD) can now be stopped."
echo "Resume sweep on new pod ($NEW_POD) with:"
echo "  python scripts/sweep.py --adapter architecture --checkpoint dummy \\"
echo "    --n-trials 30 --n-jobs 1 --n-gpus 1 --total-steps 20000 \\"
echo "    --output-dir $REMOTE_DIR"
