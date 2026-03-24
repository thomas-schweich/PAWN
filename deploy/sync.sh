#!/usr/bin/env bash
# Pull latest checkpoints and metrics from HuggingFace submodules.
# Usage: bash deploy/sync.sh [submodule-name]
#
# With no args, pulls all checkpoint submodules.
# With a name, pulls only that submodule.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

pull_submodule() {
    local sub="$1"
    local name="$(basename "$sub")"
    echo "=== Pulling $name ==="
    git -C "$sub" fetch origin 2>/dev/null || { echo "  Failed to fetch $name"; return 1; }
    git -C "$sub" pull origin main 2>/dev/null || { echo "  Failed to pull $name (main)"; return 1; }
    echo "=== $name up to date ==="
    echo ""
}

if [ $# -ge 1 ]; then
    sub="$REPO/checkpoints/$1"
    if [ ! -d "$sub/.git" ]; then
        echo "Submodule '$1' not found. Available:"
        for s in "$REPO"/checkpoints/pawn-*/; do
            [ -d "$s/.git" ] && echo "  $(basename "$s")"
        done
        exit 1
    fi
    pull_submodule "$sub"
else
    for sub in "$REPO"/checkpoints/pawn-*/; do
        [ -d "$sub/.git" ] || continue
        pull_submodule "$sub" || true
    done
fi
