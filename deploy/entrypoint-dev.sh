#!/usr/bin/env bash
set -euo pipefail

# Fix ownership on mount points — volumes arrive as root
chown -R pawn:pawn /opt/pawn 2>/dev/null || true
chown -R pawn:pawn /workspace 2>/dev/null || true

# Persist HF token into pawn's cache so `su - pawn` + huggingface_hub find it.
# Root's copy is handled by entrypoint.sh. Both users need their own — HF's
# token file is read from $HOME/.cache/huggingface/token and `su - pawn`
# resets HOME, so a single copy in /root isn't enough.
if [ -n "${HF_TOKEN:-}" ]; then
    install -d -o pawn -g pawn -m 700 /home/pawn/.cache/huggingface
    printf '%s' "$HF_TOKEN" > /home/pawn/.cache/huggingface/token
    chown pawn:pawn /home/pawn/.cache/huggingface/token
    chmod 600 /home/pawn/.cache/huggingface/token
fi

exec /opt/pawn/deploy/entrypoint.sh
