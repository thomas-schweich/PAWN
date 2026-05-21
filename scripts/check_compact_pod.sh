#!/usr/bin/env bash
# Monitor the pawn-stockfish-100m compaction run on vast.ai pod 36977347.
# Prints a digest: run status, groups done, current group/slice/bin, errors.
set -u
SSH="ssh -p 17346 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 root@ssh8.vast.ai"

$SSH bash -s <<'REMOTE'
echo "### check $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if tmux has-session -t compact 2>/dev/null; then
  echo "STATUS: running"
else
  echo "STATUS: tmux session GONE — run finished or died"
fi
echo "--- groups done ($(ls /dev/shm/compact/.done 2>/dev/null | wc -l)/15) ---"
ls /dev/shm/compact/.done 2>/dev/null | sed 's/^/  /'
echo "--- current group / slice / latest bin ---"
grep -E "^=== " /root/compact.log 2>/dev/null | tail -1
grep -E "^  slice [0-9]+/[0-9]+" /root/compact.log 2>/dev/null | tail -1
grep -E "bin [0-9]+ compacted" /root/compact.log 2>/dev/null | tail -1
echo "--- errors / rate-limiting ---"
grep -nE "Traceback|RuntimeError|CalledProcessError|FAILED|Too Many Requests|rate limit|Bad request" \
  /root/compact.log 2>/dev/null | tail -8 || echo "  none"
echo "--- cpu / shm ---"
echo "  python procs: $(pgrep -fc compact_stockfish_dataset.py 2>/dev/null || echo 0)"
du -sh /dev/shm/compact 2>/dev/null | sed 's/^/  shm: /'
echo "--- log milestones (tail) ---"
grep -E "^=== |uploading [0-9]+ files|done — |All groups compacted|migration PR" \
  /root/compact.log 2>/dev/null | tail -12
REMOTE
