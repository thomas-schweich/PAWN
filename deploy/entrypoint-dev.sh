#!/usr/bin/env bash
set -euo pipefail

# Fix ownership on mount points — volumes arrive as root
chown -R pawn:pawn /opt/pawn 2>/dev/null || true
chown -R pawn:pawn /workspace 2>/dev/null || true

exec /opt/pawn/deploy/entrypoint.sh
