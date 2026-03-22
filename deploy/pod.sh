#!/usr/bin/env bash
# Manage Runpod GPU pods for PAWN experiments.
#
# Usage:
#   pod.sh create <name> [--gpu <type>] [--disk <gb>] [--volume <gb>] [--image <name>]
#   pod.sh start <name>
#   pod.sh stop <name>
#   pod.sh delete <name>
#   pod.sh ssh <name>
#   pod.sh list
#   pod.sh gpus
#   pod.sh status <name>
#   pod.sh setup <name>          # Run setup.sh on the pod
#   pod.sh deploy <name>         # Build, transfer, and setup in one step
#   pod.sh launch <name> <cmd>   # Run a training command via nohup
#
# Pod configs are cached in ~/.config/pawn/pods/<name>.env
# Requires: runpodctl (wget -qO- cli.runpod.net | sudo bash)
#
# GPU type shortcuts:
#   a5000  -> "NVIDIA RTX A5000"
#   a40    -> "NVIDIA A40"
#   a6000  -> "NVIDIA RTX A6000"
#   4090   -> "NVIDIA GeForce RTX 4090"
#   5090   -> "NVIDIA GeForce RTX 5090"
#   l40s   -> "NVIDIA L40S"
#   h100   -> "NVIDIA H100 80GB HBM3"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
POD_DIR="$HOME/.config/pawn/pods"
mkdir -p "$POD_DIR"

# Default pod settings
DEFAULT_GPU="NVIDIA RTX A5000"
DEFAULT_CONTAINER_DISK=20
DEFAULT_VOLUME_DISK=75
DEFAULT_IMAGE="runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn9.8.0-runtime"

# --- Helpers ---

gpu_shortcut() {
    case "${1,,}" in
        a5000)  echo "NVIDIA RTX A5000" ;;
        a40)    echo "NVIDIA A40" ;;
        a6000)  echo "NVIDIA RTX A6000" ;;
        4090)   echo "NVIDIA GeForce RTX 4090" ;;
        5090)   echo "NVIDIA GeForce RTX 5090" ;;
        l40s)   echo "NVIDIA L40S" ;;
        h100)   echo "NVIDIA H100 80GB HBM3" ;;
        *)      echo "$1" ;;
    esac
}

save_pod_config() {
    local name="$1" pod_id="$2" host="$3" port="$4" gpu="$5"
    cat > "$POD_DIR/$name.env" << EOF
POD_ID=$pod_id
POD_HOST=$host
POD_PORT=$port
POD_GPU=$gpu
EOF
    echo "Saved pod config to $POD_DIR/$name.env"
}

load_pod_config() {
    local name="$1"
    local pod_file="$POD_DIR/$name.env"
    if [ ! -f "$pod_file" ]; then
        echo "Error: Pod '$name' not found. Available pods:" >&2
        list_local_pods >&2
        exit 1
    fi
    source "$pod_file"
}

list_local_pods() {
    local files=("$POD_DIR"/*.env)
    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        local n="$(basename "${f%.env}")"
        source "$f"
        echo "  $n  (id=$POD_ID, gpu=${POD_GPU:-unknown})"
    done
}

wait_for_pod_running() {
    local pod_id="$1" name="$2"
    echo -n "Waiting for pod to be ready"
    for i in $(seq 1 60); do
        if echo "$(runpodctl pod get "$pod_id" 2>/dev/null)" | grep -q '"gpuDisplayName"'; then
            local ssh_info
            ssh_info=$(runpodctl pod get "$pod_id" 2>/dev/null)
            local host port
            host=$(echo "$ssh_info" | grep -oP '"ip"\s*:\s*"\K[^"]+' || true)
            port=$(echo "$ssh_info" | grep -oP '"publicPort"\s*:\s*\K\d+' || true)

            if [ -n "$host" ] && [ -n "$port" ]; then
                if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
                       -p "$port" "root@$host" "echo ok" &>/dev/null; then
                    echo " ready!"
                    local gpu
                    gpu=$(echo "$ssh_info" | grep -oP '"gpuDisplayName"\s*:\s*"\K[^"]+' || echo "unknown")
                    save_pod_config "$name" "$pod_id" "$host" "$port" "$gpu"
                    return 0
                fi
            fi
        fi
        echo -n "."
        sleep 5
    done
    echo " timeout!"
    echo "Pod may still be starting. Check: runpodctl pod get $pod_id"
    return 1
}

ssh_opts() {
    echo "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $POD_PORT"
}

# --- Commands ---

cmd_create() {
    local name="" gpu="$DEFAULT_GPU" container_disk="$DEFAULT_CONTAINER_DISK"
    local volume_disk="$DEFAULT_VOLUME_DISK" image="$DEFAULT_IMAGE"

    name="${1:-}"
    shift || true
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu)    gpu="$(gpu_shortcut "$2")"; shift 2 ;;
            --disk)   container_disk="$2"; shift 2 ;;
            --volume) volume_disk="$2"; shift 2 ;;
            --image)  image="$2"; shift 2 ;;
            *)        echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    if [ -z "$name" ]; then
        echo "Usage: $0 create <name> [--gpu <type>] [--disk <gb>] [--volume <gb>]"
        exit 1
    fi

    echo "Creating pod '$name'..."
    echo "  GPU: $gpu"
    echo "  Container disk: ${container_disk}GB"
    echo "  Volume disk: ${volume_disk}GB"
    echo "  Image: $image"
    echo ""

    local output
    output=$(runpodctl pod create \
        --name "pawn-$name" \
        --gpuType "$gpu" \
        --gpuCount 1 \
        --imageName "$image" \
        --containerDiskSize "$container_disk" \
        --volumeSize "$volume_disk" \
        2>&1)

    echo "$output"

    local pod_id
    pod_id=$(echo "$output" | grep -oP '[a-z0-9]{20,}' | head -1 || true)

    if [ -z "$pod_id" ]; then
        echo "Error: Could not extract pod ID from output"
        exit 1
    fi

    echo "Pod ID: $pod_id"
    wait_for_pod_running "$pod_id" "$name"
}

cmd_start() {
    local name="${1:?Usage: $0 start <name>}"
    load_pod_config "$name"
    echo "Starting pod '$name' ($POD_ID)..."
    runpodctl pod start "$POD_ID"
    wait_for_pod_running "$POD_ID" "$name"
}

cmd_stop() {
    local name="${1:?Usage: $0 stop <name>}"
    load_pod_config "$name"
    echo "Stopping pod '$name' ($POD_ID)..."
    runpodctl pod stop "$POD_ID"
    echo "Pod stopped. Volume data preserved. Resume with: $0 start $name"
}

cmd_delete() {
    local name="${1:?Usage: $0 delete <name>}"
    load_pod_config "$name"
    read -p "Delete pod '$name' ($POD_ID)? This destroys all data. [y/N] " confirm
    if [ "${confirm,,}" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
    runpodctl pod delete "$POD_ID"
    rm -f "$POD_DIR/$name.env"
    echo "Pod deleted and config removed."
}

cmd_ssh() {
    local name="${1:?Usage: $0 ssh <name>}"
    load_pod_config "$name"
    ssh $(ssh_opts) "root@$POD_HOST"
}

cmd_list() {
    echo "=== Remote pods (runpodctl) ==="
    runpodctl pod list 2>/dev/null || echo "  (runpodctl not configured or no pods)"
    echo ""
    echo "=== Local pod configs ==="
    list_local_pods
}

cmd_gpus() {
    runpodctl gpu list
}

cmd_status() {
    local name="${1:?Usage: $0 status <name>}"
    load_pod_config "$name"
    runpodctl pod get "$POD_ID"
}

cmd_setup() {
    local name="${1:?Usage: $0 setup <name>}"
    load_pod_config "$name"
    echo "Running setup on pod '$name'..."
    ssh $(ssh_opts) "root@$POD_HOST" "cd /workspace/pawn && bash deploy/setup.sh"
}

cmd_deploy() {
    local name="${1:?Usage: $0 deploy <name>}"
    load_pod_config "$name"

    echo "=== Full deploy to '$name' ==="
    echo ""

    # Step 1: Build
    echo "--- Step 1: Build deploy package ---"
    bash "$REPO/deploy/build.sh"
    echo ""

    # Step 2: Transfer
    echo "--- Step 2: Transfer to pod ---"
    # Install rsync on pod if needed
    ssh $(ssh_opts) "root@$POD_HOST" "command -v rsync &>/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null
    rsync -avz --progress -e "ssh $(ssh_opts)" \
        "$REPO/deploy/pawn-deploy/" "root@$POD_HOST:/workspace/pawn/"
    echo ""

    # Step 3: Setup
    echo "--- Step 3: Run setup ---"
    ssh $(ssh_opts) "root@$POD_HOST" "cd /workspace/pawn && bash deploy/setup.sh"
    echo ""

    echo "=== Deploy complete ==="
}

cmd_launch() {
    local name="${1:?Usage: $0 launch <name> <command...>}"
    shift
    local cmd="$*"

    if [ -z "$cmd" ]; then
        echo "Usage: $0 launch <name> <command...>"
        echo ""
        echo "Examples:"
        echo "  $0 launch exp1 scripts/train.py --variant base"
        echo "  $0 launch exp1 scripts/train_bottleneck.py --checkpoint checkpoints/pawn-base.pt \\"
        echo "      --pgn data/lichess_1800_1900.pgn --bottleneck-dim 32"
        exit 1
    fi

    load_pod_config "$name"

    local script_name
    script_name=$(echo "$cmd" | grep -oP 'scripts/\K[^ ]+' | sed 's/\.py//' || echo "train")

    echo "Launching on '$name': $cmd"
    ssh $(ssh_opts) "root@$POD_HOST" "cd /workspace/pawn && \
        nohup uv run python $cmd \
            --log-dir logs \
            > logs/${script_name}.log 2>&1 & \
        sleep 2 && \
        echo 'PID: '\$(pgrep -f '$script_name' | head -1) && \
        echo 'Log: logs/${script_name}.log'"
}

# --- Main ---

case "${1:-}" in
    create)  shift; cmd_create "$@" ;;
    start)   shift; cmd_start "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    delete)  shift; cmd_delete "$@" ;;
    ssh)     shift; cmd_ssh "$@" ;;
    list)    shift; cmd_list "$@" ;;
    gpus)    shift; cmd_gpus "$@" ;;
    status)  shift; cmd_status "$@" ;;
    setup)   shift; cmd_setup "$@" ;;
    deploy)  shift; cmd_deploy "$@" ;;
    launch)  shift; cmd_launch "$@" ;;
    sync)    shift; bash "$REPO/deploy/sync.sh" "$@" ;;
    *)
        echo "PAWN Pod Manager"
        echo ""
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  create <name> [--gpu <type>] [--disk <gb>] [--volume <gb>]"
        echo "                           Create a new pod"
        echo "  start  <name>            Resume a stopped pod"
        echo "  stop   <name>            Pause a pod (preserves volume, stops billing)"
        echo "  delete <name>            Destroy a pod and its data"
        echo "  ssh    <name>            SSH into a pod"
        echo "  list                     List all pods"
        echo "  gpus                     List available GPU types"
        echo "  status <name>            Get pod details"
        echo "  setup  <name>            Run setup.sh on the pod"
        echo "  deploy <name>            Build + transfer + setup (full deploy)"
        echo "  launch <name> <cmd>      Run a training command via nohup"
        echo "  sync   [name]            Sync logs/checkpoints from pod(s)"
        echo ""
        echo "GPU shortcuts: a5000, a40, a6000, 4090, 5090, l40s, h100"
        echo ""
        echo "Examples:"
        echo "  $0 create exp1 --gpu a5000"
        echo "  $0 deploy exp1"
        echo "  $0 launch exp1 scripts/train.py --variant base"
        echo "  $0 stop exp1"
        ;;
esac
