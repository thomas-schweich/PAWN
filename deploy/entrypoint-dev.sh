#!/usr/bin/env bash
set -euo pipefail

# Fix ownership on mount points — volumes arrive as root
chown -R pawn:pawn /opt/pawn 2>/dev/null || true
chown -R pawn:pawn /workspace 2>/dev/null || true

# Export the HF_TOKEN environment variable for the pawn user (set via RunPod console)
echo "export HF_TOKEN=${HF_TOKEN}" >> /home/pawn/.bashrc

exec /opt/pawn/deploy/entrypoint.sh
