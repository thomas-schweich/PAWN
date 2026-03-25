#!/bin/bash
# Entrypoint for auto-stop runner target.
# Starts sshd for monitoring, runs the command, then exits (pod auto-stops).
set -e

cd /opt/pawn
export PYTHONPATH=/opt/pawn

# Start sshd for monitoring/debugging (Runpod base image has it installed)
if [ -f /usr/sbin/sshd ]; then
    mkdir -p /run/sshd
    # Generate host keys if missing
    ssh-keygen -A 2>/dev/null || true
    # Allow root login
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config 2>/dev/null || true
    # Runpod injects PUBLIC_KEY env var for SSH access
    if [ -n "$PUBLIC_KEY" ]; then
        mkdir -p /root/.ssh
        echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
        chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
    fi
    /usr/sbin/sshd 2>/dev/null && echo "sshd started" || echo "sshd failed to start"
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
