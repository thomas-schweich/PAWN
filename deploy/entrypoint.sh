#!/bin/bash
set -e

# Runpod injects the SSH public key via this env var
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

service ssh start

exec "$@"
