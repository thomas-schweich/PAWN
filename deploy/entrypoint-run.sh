#!/bin/bash
# Entrypoint for auto-stop runner target.
# Optionally pulls a checkpoint from HuggingFace before running the command.
set -e

cd /opt/pawn
export PYTHONPATH=/opt/pawn

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
if [ $# -gt 0 ]; then
    exec "$@"
elif [ -n "$PAWN_CMD" ]; then
    exec bash -c "$PAWN_CMD"
else
    echo "ERROR: No command provided. Set PAWN_CMD env var or pass Docker CMD args."
    exit 1
fi
