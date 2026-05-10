#!/usr/bin/env bash
# Manage Runpod GPU pods for PAWN experiments.
#
# Usage:
#   pod.sh create <name> [--gpu <type>] [--count <n>] [--disk <gb>] [--volume <gb>] [--community]
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
# Requires: runpodctl (curl -sSL https://cli.runpod.net | bash) and ripgrep (rg).
#
# GPU type shortcuts (mapped to runpodctl --gpu-id values):
#   a5000      -> "NVIDIA RTX A5000"
#   a40        -> "NVIDIA A40"
#   a6000      -> "NVIDIA RTX 6000 Ada Generation"
#   4090       -> "NVIDIA GeForce RTX 4090"
#   5090       -> "NVIDIA GeForce RTX 5090"
#   l40s       -> "NVIDIA L40S"
#   a100-pcie  -> "NVIDIA A100 80GB PCIe"
#   a100-sxm   -> "NVIDIA A100-SXM4-80GB"
#   h100       -> "NVIDIA H100 80GB HBM3"
#   h200       -> "NVIDIA H200"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
POD_DIR="$HOME/.config/pawn/pods"
mkdir -p "$POD_DIR"

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' is required but not installed." >&2
        case "$1" in
            runpodctl) echo "  Install: curl -sSL https://cli.runpod.net | bash" >&2 ;;
            rg)        echo "  Install: apt-get install ripgrep  (or brew install ripgrep)" >&2 ;;
        esac
        exit 1
    fi
}
require_tool rg

# Default pod settings
DEFAULT_GPU="NVIDIA RTX A5000"
DEFAULT_CONTAINER_DISK=20
DEFAULT_VOLUME_DISK=75
DEFAULT_IMAGE="thomasschweich/pawn:latest"

# --- Helpers ---

gpu_shortcut() {
    case "${1,,}" in
        a5000)     echo "NVIDIA RTX A5000" ;;
        a40)       echo "NVIDIA A40" ;;
        a6000)     echo "NVIDIA RTX 6000 Ada Generation" ;;
        4090)      echo "NVIDIA GeForce RTX 4090" ;;
        5090)      echo "NVIDIA GeForce RTX 5090" ;;
        l40s)      echo "NVIDIA L40S" ;;
        a100-pcie) echo "NVIDIA A100 80GB PCIe" ;;
        a100-sxm)  echo "NVIDIA A100-SXM4-80GB" ;;
        a100)      echo "NVIDIA A100 80GB PCIe" ;;
        h100)      echo "NVIDIA H100 80GB HBM3" ;;
        h200)      echo "NVIDIA H200" ;;
        *)         echo "$1" ;;
    esac
}

save_pod_config() {
    local name="$1" pod_id="$2" host="$3" port="$4" gpu="$5"
    # Single-quote each value so values containing spaces (e.g.
    # POD_GPU="NVIDIA RTX A5000" from runpodctl's gpuDisplayName) parse
    # correctly when the .env is sourced. printf %q escapes embedded
    # single quotes.
    cat > "$POD_DIR/$name.env" << EOF
POD_ID=$(printf %q "$pod_id")
POD_HOST=$(printf %q "$host")
POD_PORT=$(printf %q "$port")
POD_GPU=$(printf %q "$gpu")
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
        local ssh_info host port gpu
        ssh_info=$(runpodctl pod get "$pod_id" 2>/dev/null || true)
        if echo "$ssh_info" | rg -q '"gpuDisplayName"'; then
            host=$(echo "$ssh_info" | rg -o '"ip"\s*:\s*"([^"]+)"' -r '$1' | head -1 || true)
            port=$(echo "$ssh_info" | rg -o '"publicPort"\s*:\s*([0-9]+)' -r '$1' | head -1 || true)

            if [ -n "$host" ] && [ -n "$port" ]; then
                if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
                       -p "$port" "root@$host" "echo ok" &>/dev/null; then
                    echo " ready!"
                    gpu=$(echo "$ssh_info" | rg -o '"gpuDisplayName"\s*:\s*"([^"]+)"' -r '$1' | head -1 || echo "unknown")
                    save_pod_config "$name" "$pod_id" "$host" "$port" "$gpu"
                    return 0
                fi
            fi
        fi
        echo -n "."
        sleep 5
    done
    echo " timeout!"
    echo "Pod $pod_id may still be starting. Check: runpodctl pod get $pod_id"
    return 1
}

ssh_opts() {
    echo "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $POD_PORT"
}

# --- Commands ---

cmd_create() {
    local name="" gpu="$DEFAULT_GPU" gpu_count=1
    local container_disk="$DEFAULT_CONTAINER_DISK"
    local volume_disk="$DEFAULT_VOLUME_DISK" image="$DEFAULT_IMAGE"
    local cloud_type="SECURE"

    name="${1:-}"
    shift || true
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu)       gpu="$(gpu_shortcut "$2")"; shift 2 ;;
            --count)     gpu_count="$2"; shift 2 ;;
            --disk)      container_disk="$2"; shift 2 ;;
            --volume)    volume_disk="$2"; shift 2 ;;
            --image)     image="$2"; shift 2 ;;
            --community) cloud_type="COMMUNITY"; shift ;;
            *)           echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    if [ -z "$name" ]; then
        echo "Usage: $0 create <name> [--gpu <type>] [--count <n>] [--disk <gb>] [--volume <gb>] [--community]"
        exit 1
    fi

    echo "Creating pod '$name'..."
    echo "  GPU: ${gpu_count}x $gpu"
    echo "  Cloud: $cloud_type"
    echo "  Container disk: ${container_disk}GB"
    echo "  Volume disk: ${volume_disk}GB"
    echo "  Image: $image"
    echo ""

    local output
    output=$(runpodctl pod create \
        --name "pawn-$name" \
        --gpu-id "$gpu" \
        --gpu-count "$gpu_count" \
        --image "$image" \
        --container-disk-in-gb "$container_disk" \
        --volume-in-gb "$volume_disk" \
        --cloud-type "$cloud_type" \
        2>&1)

    echo "$output"

    local pod_id
    pod_id=$(echo "$output" | rg -o '[a-z0-9]{20,}' | head -1 || true)

    if [ -z "$pod_id" ]; then
        echo "Error: Could not extract pod ID from output"
        exit 1
    fi

    echo "Pod ID: $pod_id"

    # Persist a partial config NOW so a wait-loop timeout doesn't orphan a
    # billable pod with no local handle. wait_for_pod_running overwrites
    # this with the full config (host/port/gpu) on success.
    save_pod_config "$name" "$pod_id" "" "" "$gpu"

    if ! wait_for_pod_running "$pod_id" "$name"; then
        echo "" >&2
        echo "Pod $pod_id was created but never became reachable over SSH." >&2
        echo "It is still running and accruing charges. Manage it with:" >&2
        echo "  $0 status $name      # check runpod state" >&2
        echo "  $0 stop $name        # pause (preserves volume)" >&2
        echo "  $0 delete $name      # destroy" >&2
        exit 1
    fi
}

cmd_start() {
    local name="${1:?Usage: $0 start <name>}"
    load_pod_config "$name"
    echo "Starting pod '$name' ($POD_ID)..."
    if ! runpodctl pod start "$POD_ID"; then
        echo "Error: runpodctl pod start $POD_ID failed." >&2
        echo "The pod may have been deleted externally — check: $0 status $name" >&2
        exit 1
    fi
    # Clear cached host/port: a resumed pod can come up on a different
    # host/port. wait_for_pod_running rewrites the .env on success; on
    # timeout the empty cache prevents stale endpoints from being reused.
    save_pod_config "$name" "$POD_ID" "" "" "${POD_GPU:-unknown}"
    wait_for_pod_running "$POD_ID" "$name" || exit 1
}

cmd_stop() {
    local name="${1:?Usage: $0 stop <name>}"
    load_pod_config "$name"
    echo "Stopping pod '$name' ($POD_ID)..."
    if ! runpodctl pod stop "$POD_ID"; then
        echo "Error: runpodctl pod stop $POD_ID failed." >&2
        echo "Check current state: $0 status $name" >&2
        exit 1
    fi
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
    if ! runpodctl pod delete "$POD_ID"; then
        echo "Error: runpodctl pod delete $POD_ID failed." >&2
        echo "Local config preserved at $POD_DIR/$name.env. Investigate before retrying." >&2
        exit 1
    fi
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
        echo "  $0 launch exp1 scripts/train.py --run-type adapter --strategy bottleneck \\"
        echo "      --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \\"
        echo "      --elo-min 1800 --elo-max 1900 --bottleneck-dim 32"
        exit 1
    fi

    load_pod_config "$name"

    local script_name
    script_name=$(echo "$cmd" | rg -o 'scripts/([^ ]+)' -r '$1' | sed 's/\.py//' || echo "train")

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
    sync)    echo "sync command removed — checkpoints load directly from HuggingFace"; exit 1 ;;
    *)
        echo "PAWN Pod Manager"
        echo ""
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  create <name> [--gpu <type>] [--count <n>] [--disk <gb>] [--volume <gb>] [--community]"
        echo "                           Create a new pod (default: 1 GPU, secure cloud)"
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
        echo "  sync                     (removed — checkpoints load from HuggingFace)"
        echo ""
        echo "GPU shortcuts: a5000, a40, a6000, 4090, 5090, l40s, a100, a100-pcie, a100-sxm, h100, h200"
        echo ""
        echo "Examples:"
        echo "  $0 create exp1 --gpu a5000"
        echo "  $0 create sweep1 --gpu a100-pcie --count 2 --community"
        echo "  $0 deploy exp1"
        echo "  $0 launch exp1 scripts/train.py --variant base"
        echo "  $0 stop exp1"
        ;;
esac
