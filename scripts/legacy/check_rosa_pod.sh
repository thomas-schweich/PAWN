#!/usr/bin/env bash
# Check rosa-sweep pod status and GPU availability.
# Usage: bash scripts/check_rosa_pod.sh
set -euo pipefail

POD_ID="dkci2nnyzbanyu"

echo "=== Pod Status ==="
runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
gpu = d.get('gpuDisplayName', 'unknown')
count = d.get('gpuCount', '?')
status = d.get('desiredStatus', d.get('status', '?'))
runtime = d.get('runtime', {}) or {}
uptime = runtime.get('uptimeInSeconds', 0)

ssh = d.get('ssh', {})
host = ssh.get('ip', '') or ssh.get('host', '')
port = ssh.get('port', '')

print(f'  ID: $POD_ID')
print(f'  GPU: {count}x {gpu}')
print(f'  Status: {status}')
print(f'  Uptime: {uptime}s')
if host and port:
    print(f'  SSH: ssh -p {port} root@{host}')
else:
    print(f'  SSH: not ready')
" 2>/dev/null || echo "  (failed to query pod)"

# Try SSH if available
ssh_info=$(runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
ssh = d.get('ssh', {})
host = ssh.get('ip', '') or ssh.get('host', '')
port = ssh.get('port', '')
if host and port:
    print(f'{host} {port}')
else:
    print('NOTREADY')
" 2>/dev/null || echo "NOTREADY")

if [ "$ssh_info" != "NOTREADY" ]; then
    HOST=$(echo "$ssh_info" | cut -d' ' -f1)
    PORT=$(echo "$ssh_info" | cut -d' ' -f2)
    SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $PORT root@$HOST"

    echo ""
    echo "=== GPU Info ==="
    $SSH "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader" 2>/dev/null || echo "  (SSH failed)"

    echo ""
    echo "=== Sweep Processes ==="
    $SSH "pgrep -fa 'sweep\|train_rosa' || echo '  (none running)'" 2>/dev/null || echo "  (SSH failed)"

    echo ""
    echo "=== Sweep DB Files ==="
    $SSH "find /workspace/sweeps -name 'study.db' -exec ls -lh {} \; 2>/dev/null || echo '  (no sweep data yet)'" 2>/dev/null || echo "  (SSH failed)"
fi
