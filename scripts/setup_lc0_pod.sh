#!/usr/bin/env bash
# Setup Lc0 data generation on a GPU pod.
# Run from inside the pod SSH session:
#   bash <(curl -sL https://raw.githubusercontent.com/thomas-schweich/PAWN/main/scripts/setup_lc0_pod.sh)
set -euo pipefail

echo "=== Installing deps ==="
apt-get update -qq && apt-get install -y -qq zstd git > /dev/null 2>&1
pip install --break-system-packages -q pyarrow huggingface-hub 2>&1 | tail -1

echo "=== Installing Lc0 ==="
# Lc0 release binary (CUDA backend)
cd /tmp
curl -sL https://github.com/LeelaChessZero/lc0/releases/download/v0.31.2/lc0-v0.31.2-linux-gpu-nvidia-cuda12-2.tar.gz -o lc0.tar.gz
tar xzf lc0.tar.gz
cp lc0-v0.31.2-linux-gpu-nvidia-cuda12-2/lc0 /usr/local/bin/lc0
chmod +x /usr/local/bin/lc0
rm -rf lc0.tar.gz lc0-v0.31.2-linux-gpu-nvidia-cuda12-2/

echo "=== Downloading Lc0 weights ==="
# Large T2 network (best policy quality)
mkdir -p /usr/local/share/lc0
cd /usr/local/share/lc0
curl -sL https://storage.lczero.org/files/networks-contrib/t2-768x15x24h-swa-7464000.pb.gz -o weights.pb.gz
gunzip weights.pb.gz
echo "Weights: $(ls -lh weights.pb)"

echo "=== Cloning repo ==="
cd /dev/shm
git clone --depth 1 https://github.com/thomas-schweich/PAWN.git pawn

echo "=== Persisting HF token ==="
if [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p /root/.cache/huggingface
    echo -n "$HF_TOKEN" > /root/.cache/huggingface/token
    echo "HF token saved"
else
    echo "WARNING: HF_TOKEN not set — upload will fail"
fi

echo "=== Creating Lc0 wrapper ==="
# Wrapper that passes weights arg — our generate script expects a plain binary path
cat > /usr/local/bin/lc0-wrap << 'WRAPPER'
#!/bin/bash
exec /usr/local/bin/lc0 --weights=/usr/local/share/lc0/weights.pb "$@"
WRAPPER
chmod +x /usr/local/bin/lc0-wrap

echo "=== Verifying Lc0 ==="
echo -e "uci\nquit" | lc0-wrap 2>/dev/null | head -2

echo ""
echo "=== Ready ==="
echo "Generate with:"
echo "  cd /dev/shm/pawn"
echo "  mkdir -p /dev/shm/lc0"
echo "  nohup python3 scripts/generate_stockfish_data.py \\"
echo "    --stockfish /usr/local/bin/lc0-wrap \\"
echo "    --output /dev/shm/lc0 \\"
echo "    --tier nodes_0001 \\"
echo "    --workers 1 \\"
echo "    --games 1000000 \\"
echo "    > /dev/shm/lc0_gen.log 2>&1 &"
