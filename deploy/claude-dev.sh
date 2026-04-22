#!/usr/bin/env bash
# Dev-pod entrypoint. When run as root (the default on RunPod / vast.ai
# pods) it first chowns the mounted paths to the pawn user — we keep
# hitting root-owned /opt/pawn and /workspace on fresh pods regardless of
# how the Dockerfile is set up, so we just fix it every time — then
# re-execs itself as pawn. The pawn phase verifies HF / W&B auth
# (prompting if missing), ensures a per-pod private HF bucket exists
# and exports its ID as PAWN_HF_BUCKET, then drops into a claude tmux
# session.
set -euo pipefail

SESSION="claude"
SELF=$(readlink -f "$0")

if [ "$(id -u)" -eq 0 ]; then
    for path in /opt/pawn /workspace; do
        if [ -d "$path" ]; then
            chown -R pawn:pawn "$path"
        fi
    done
    exec su - pawn -c "exec '$SELF'"
fi

# `su -` resets PATH from /etc/login.defs, so /opt/pawn/.venv/bin (where
# `hf` and `wandb` live) drops off. Put it back — same value the Dockerfile
# sets via ENV PATH for the dev/dev-rocm images.
export PATH="/opt/pawn/.venv/bin:/home/pawn/.local/bin:/home/pawn/.cargo/bin:$PATH"

hf=(uvx hf)
command -v hf >/dev/null 2>&1 && hf=(hf)
if ! "${hf[@]}" auth whoami >/dev/null 2>&1; then
    echo "==> No HuggingFace credentials found — running hf auth login"
    "${hf[@]}" auth login
fi

# One private HF bucket per pod. Name keyed on hostname + date so a later
# rental of the same host doesn't collide with an old bucket. First run
# creates it and writes the full ID (`<user>/<name>`) to ~/.pawn-bucket;
# subsequent runs (including across days) reuse the file so the bucket is
# stable for the life of the pod. `--exist-ok` keeps create idempotent
# on the off chance the state file was wiped but the bucket still exists.
bucket_state="$HOME/.pawn-bucket"
if [ -r "$bucket_state" ]; then
    PAWN_HF_BUCKET=$(cat "$bucket_state")
    echo "==> Reusing HF bucket: $PAWN_HF_BUCKET"
else
    hf_user=$("${hf[@]}" auth whoami --format json 2>/dev/null | jq -r '.name // empty')
    if [ -z "$hf_user" ]; then
        echo "==> Could not determine HF user; skipping bucket setup" >&2
    else
        bucket_host=$(hostname -s | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed 's/^-*//;s/-*$//')
        bucket_date=$(date +%Y%m%d)
        PAWN_HF_BUCKET="${hf_user}/pawn-${bucket_host}-${bucket_date}"
        echo "==> Creating private HF bucket: $PAWN_HF_BUCKET"
        if "${hf[@]}" buckets create "$PAWN_HF_BUCKET" --private --exist-ok; then
            echo "$PAWN_HF_BUCKET" > "$bucket_state"
        else
            echo "==> Bucket creation failed; continuing without PAWN_HF_BUCKET" >&2
            unset PAWN_HF_BUCKET
        fi
    fi
fi
if [ -n "${PAWN_HF_BUCKET:-}" ]; then
    export PAWN_HF_BUCKET
    bashrc_marker='# pawn: load active HF bucket'
    if ! grep -qF "$bashrc_marker" "$HOME/.bashrc" 2>/dev/null; then
        {
            echo ""
            echo "$bashrc_marker"
            echo '[ -r "$HOME/.pawn-bucket" ] && export PAWN_HF_BUCKET=$(cat "$HOME/.pawn-bucket")'
        } >> "$HOME/.bashrc"
    fi
fi

wandb=(uvx wandb)
command -v wandb >/dev/null 2>&1 && wandb=(wandb)
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q 'api\.wandb\.ai' "$HOME/.netrc" 2>/dev/null; then
    echo "==> No W&B credentials found — running wandb login"
    "${wandb[@]}" login
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    exec tmux attach -t "$SESSION"
fi
tmux new-session -d -s "$SESSION" -c /opt/pawn
tmux send-keys -t "$SESSION" 'cd /opt/pawn && claude --dangerously-skip-permissions' Enter
exec tmux attach -t "$SESSION"
