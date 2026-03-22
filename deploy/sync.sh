#!/usr/bin/env bash
# Sync logs and checkpoints from Runpod pod(s) to local machine.
# Usage: bash deploy/sync.sh [pod-name]
#
# With no args, syncs from all pods in ~/.config/pawn/pods/
# With a pod name, syncs from that specific pod only.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
POD_DIR="$HOME/.config/pawn/pods"

sync_pod() {
    local name="$1" host="$2" port="$3" remote_root="${4:-/opt/pawn}"
    local ssh_opts="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $port"

    echo "=== Syncing from $name ($host:$port) ==="

    echo "--- Logs ---"
    rsync -avz --progress --no-owner --no-group \
        -e "ssh $ssh_opts" \
        "root@$host:$remote_root/logs/" "$REPO/logs/" || {
        echo "  Failed to sync logs from $name"
        return 1
    }

    echo "--- Checkpoints (best.pt only) ---"
    rsync -avz --progress --no-owner --no-group \
        --include='*/' --include='best.pt' --include='*.pt' --exclude='*' \
        -e "ssh $ssh_opts" \
        "root@$host:$remote_root/logs/" "$REPO/logs/" || {
        echo "  Failed to sync checkpoints from $name"
        return 1
    }

    echo "=== $name sync complete ==="
    echo ""
}

if [ ! -d "$POD_DIR" ] || [ -z "$(ls "$POD_DIR"/*.env 2>/dev/null)" ]; then
    echo "No pods configured. Add .env files to $POD_DIR/"
    exit 1
fi

if [ $# -ge 1 ]; then
    # Sync specific pod
    POD_FILE="$POD_DIR/$1.env"
    if [ ! -f "$POD_FILE" ]; then
        echo "Pod '$1' not found. Available pods:"
        ls "$POD_DIR"/*.env 2>/dev/null | xargs -I{} basename {} .env | sed 's/^/  /'
        exit 1
    fi
    source "$POD_FILE"
    sync_pod "$1" "$POD_HOST" "$POD_PORT" "${POD_REMOTE_ROOT:-/opt/pawn}"
else
    # Sync all pods
    for pod_file in "$POD_DIR"/*.env; do
        pod_name="$(basename "${pod_file%.env}")"
        unset POD_HOST POD_PORT POD_REMOTE_ROOT
        source "$pod_file"
        sync_pod "$pod_name" "$POD_HOST" "$POD_PORT" "${POD_REMOTE_ROOT:-/opt/pawn}" || true
    done
fi

echo "Local logs:"
du -sh "$REPO/logs/"
