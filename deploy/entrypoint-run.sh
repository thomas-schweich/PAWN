#!/bin/bash
# Entrypoint for auto-stop runner target.
# Starts Runpod's SSH/Jupyter in the background, runs the command, then exits.
set -e

cd /opt/pawn
export PYTHONPATH=/opt/pawn

# Persist HF_TOKEN to huggingface-hub's token cache so it survives env changes
if [ -n "$HF_TOKEN" ]; then
    mkdir -p /root/.cache/huggingface
    echo -n "$HF_TOKEN" > /root/.cache/huggingface/token
fi

# Let Runpod's own start script handle SSH, Jupyter, key injection, etc.
if [ -f /start.sh ]; then
    /start.sh &
    sleep 2  # give sshd a moment to bind
fi

# Pull checkpoint from HuggingFace if PAWN_MODEL is set
if [ -n "$PAWN_MODEL" ]; then
    echo "Pulling model: $PAWN_MODEL"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$PAWN_MODEL', local_dir='checkpoints/model')
print('Model downloaded to checkpoints/model/')
"
fi

# Run the command: Docker CMD args, or PAWN_CMD env var
# When it finishes, the container exits and the pod auto-stops.
if [ $# -gt 0 ]; then
    exec "$@"
elif [ -n "$PAWN_CMD" ]; then
    exec bash -c "$PAWN_CMD"
else
    echo "ERROR: No command provided. Set PAWN_CMD env var or pass Docker CMD args."
    exit 1
fi
