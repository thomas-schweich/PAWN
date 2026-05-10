#!/usr/bin/env bash
# Manage vast.ai GPU instances for PAWN experiments.
#
# Usage:
#   vast.sh create <name> [--gpu <type>] [--count <n>] [--disk <gb>] [--max-price <$/hr>] [--interruptible]
#   vast.sh start <name>
#   vast.sh stop <name>
#   vast.sh delete <name>
#   vast.sh ssh <name>
#   vast.sh list
#   vast.sh search [--gpu <type>] [--count <n>] [--max-price <$/hr>]
#   vast.sh status <name>
#   vast.sh setup <name>          # Run setup.sh on the instance
#   vast.sh deploy <name>         # Build, transfer, and setup in one step
#   vast.sh launch <name> <cmd>   # Run a training command via nohup
#
# Instance configs are cached in ~/.config/pawn/vast/<name>.env
# Requires: vastai (`pip install --user vastai`) and jq.
# Auth: `vastai set api-key <KEY>` once, key from https://vast.ai/console/account
#
# GPU type shortcuts (mapped to vast.ai gpu_name values):
#   a5000      -> "RTX_A5000"
#   a40        -> "A40"
#   a6000      -> "RTX_6000Ada"
#   4090       -> "RTX_4090"
#   5090       -> "RTX_5090"
#   l40s       -> "L40S"
#   a100-pcie  -> "A100_PCIE"
#   a100-sxm   -> "A100_SXM4"
#   h100       -> "H100_PCIE"
#   h100-sxm   -> "H100_SXM"
#   h200       -> "H200"
#   3090       -> "RTX_3090"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VAST_DIR="$HOME/.config/pawn/vast"
mkdir -p "$VAST_DIR"

# Default instance settings
DEFAULT_GPU="RTX_A5000"
DEFAULT_DISK=100                       # vast.ai uses one disk (no separate volume)
DEFAULT_IMAGE="thomasschweich/pawn:latest"
DEFAULT_MAX_PRICE=""                   # empty = no cap

# --- Helpers ---

require_tools() {
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "Error: '$tool' is required but not installed." >&2
            case "$tool" in
                vastai) echo "  Install: uv tool install vastai  (or pip install --user vastai)" >&2 ;;
                jq)     echo "  Install: apt-get install jq  (or brew install jq)" >&2 ;;
                rg)     echo "  Install: apt-get install ripgrep  (or brew install ripgrep)" >&2 ;;
            esac
            exit 1
        fi
    done
}

# Tools every entry point needs (commands that hit no external CLI add their own
# extras via require_tools as well).
require_tools rg

gpu_shortcut() {
    case "${1,,}" in
        a5000)     echo "RTX_A5000" ;;
        a40)       echo "A40" ;;
        a6000)     echo "RTX_6000Ada" ;;
        4090)      echo "RTX_4090" ;;
        5090)      echo "RTX_5090" ;;
        3090)      echo "RTX_3090" ;;
        l40s)      echo "L40S" ;;
        a100-pcie) echo "A100_PCIE" ;;
        a100-sxm)  echo "A100_SXM4" ;;
        a100)      echo "A100_PCIE" ;;
        h100)      echo "H100_PCIE" ;;
        h100-sxm)  echo "H100_SXM" ;;
        h200)      echo "H200" ;;
        *)         echo "$1" ;;       # passthrough for unknown values
    esac
}

save_instance_config() {
    local name="$1" instance_id="$2" host="$3" port="$4" gpu="$5"
    # Single-quote each value so values containing spaces (e.g.
    # INSTANCE_GPU="RTX 3090" from vastai's display field) parse correctly
    # when the .env is sourced. printf %q escapes embedded single quotes.
    cat > "$VAST_DIR/$name.env" << EOF
INSTANCE_ID=$(printf %q "$instance_id")
INSTANCE_HOST=$(printf %q "$host")
INSTANCE_PORT=$(printf %q "$port")
INSTANCE_GPU=$(printf %q "$gpu")
EOF
    echo "Saved instance config to $VAST_DIR/$name.env"
}

load_instance_config() {
    local name="$1"
    local cfg="$VAST_DIR/$name.env"
    if [ ! -f "$cfg" ]; then
        echo "Error: Instance '$name' not found. Available:" >&2
        list_local_instances >&2
        exit 1
    fi
    source "$cfg"
}

# Re-query vast.ai for an instance's SSH endpoint and persist it to the .env.
# Used when the cached host/port are empty (partial config from a wait-loop
# timeout) — lets a slow-to-boot instance still be reachable later without
# manual config editing or stop/start cycling.
refresh_instance_endpoint() {
    local name="$1"
    require_tools vastai jq
    local json endpoint host port gpu
    json=$(instance_json "$INSTANCE_ID")
    endpoint=$(extract_ssh_endpoint "$json")
    if [ -z "$endpoint" ]; then
        echo "Error: instance $INSTANCE_ID has no SSH endpoint yet." >&2
        echo "Check vast.ai state: $0 status $name" >&2
        return 1
    fi
    host="${endpoint% *}"
    port="${endpoint#* }"
    gpu=$(echo "$json" | jq -r '.gpu_name // "unknown"')
    save_instance_config "$name" "$INSTANCE_ID" "$host" "$port" "$gpu"
    INSTANCE_HOST="$host"
    INSTANCE_PORT="$port"
    INSTANCE_GPU="$gpu"
}

# Load config and ensure SSH endpoint is populated. Use for any command that
# needs to ssh into the instance.
load_instance_config_with_endpoint() {
    local name="$1"
    load_instance_config "$name"
    if [ -z "${INSTANCE_HOST:-}" ] || [ -z "${INSTANCE_PORT:-}" ]; then
        echo "SSH endpoint not cached for '$name' (likely a previous create/start timeout). Refreshing..."
        refresh_instance_endpoint "$name" || exit 1
    fi
}

list_local_instances() {
    local files=("$VAST_DIR"/*.env)
    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        local n="$(basename "${f%.env}")"
        source "$f"
        echo "  $n  (id=$INSTANCE_ID, gpu=${INSTANCE_GPU:-unknown})"
    done
}

# Fetch raw JSON for a single instance.
instance_json() {
    local id="$1"
    vastai show instance "$id" --raw 2>/dev/null
}

# Pull SSH host/port from instance JSON. Echoes "host port" or empty on miss.
# Prefers direct SSH (public_ipaddr + ports["22/tcp"][0].HostPort) over the
# vast.ai proxy (ssh_host:ssh_port). The proxy reliably terminates connections
# but doesn't always execute non-interactive commands sent over stdin —
# stdin gets swallowed and only the welcome banner runs, so `vast.sh launch`
# silently does nothing. Direct SSH avoids the proxy entirely.
extract_ssh_endpoint() {
    local json="$1"
    local row host port
    row=$(echo "$json" | jq -r '
        [(.public_ipaddr // .ssh_host // ""),
         (.ports["22/tcp"][0].HostPort // .ssh_port // "")]
        | @tsv' 2>/dev/null)
    host="${row%%$'\t'*}"
    port="${row#*$'\t'}"
    if [ -n "$host" ] && [ -n "$port" ] && [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "$host $port"
    fi
}

wait_for_instance_running() {
    local instance_id="$1" name="$2"
    echo -n "Waiting for instance to be ready"
    for i in $(seq 1 90); do
        local json status
        json=$(instance_json "$instance_id" || true)
        status=$(echo "$json" | jq -r '.actual_status // empty' 2>/dev/null)

        if [ "$status" = "running" ]; then
            local endpoint host port
            endpoint=$(extract_ssh_endpoint "$json")
            if [ -n "$endpoint" ]; then
                host="${endpoint% *}"
                port="${endpoint#* }"
                if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
                       -p "$port" "root@$host" "echo ok" &>/dev/null; then
                    echo " ready!"
                    local gpu
                    gpu=$(echo "$json" | jq -r '.gpu_name // "unknown"')
                    save_instance_config "$name" "$instance_id" "$host" "$port" "$gpu"
                    return 0
                fi
            fi
        fi
        echo -n "."
        sleep 5
    done
    echo " timeout!"
    echo "Instance may still be starting. Check: vastai show instance $instance_id"
    return 1
}

ssh_opts() {
    echo "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $INSTANCE_PORT"
}

# Build a vastai search-offers query string from flag values.
# Vast.ai filter syntax: `gpu_name=RTX_4090 num_gpus=1 disk_space>=100 dph_total<=1.0 reliability>0.98`
build_search_query() {
    local gpu="$1" count="$2" disk="$3" max_price="$4" interruptible="$5"
    # No CUDA-version filter: the PyTorch cu128 wheels bundle their own
    # runtime, so only the host driver matters (any modern verified host
    # has it). vast.ai's `cuda_vers` field is null on many listings and
    # `cuda_max_good` filtering misbehaves (>=12 returns nothing while
    # >=13 returns dozens), so any threshold here just silently excludes
    # otherwise-fine hosts.
    local q="gpu_name=$gpu num_gpus=$count disk_space>=$disk reliability>0.98 inet_down>=200"
    if [ -n "$max_price" ]; then
        q="$q dph_total<=$max_price"
    fi
    if [ "$interruptible" = "1" ]; then
        # Filter for actual spot/preemptible capacity, not just on-demand offers
        # from unverified hosts. The `interruptible=true` field is the offer
        # marker for spot capacity on vast.ai.
        q="$q rentable=true interruptible=true"
    else
        q="$q rentable=true verified=true"
    fi
    echo "$q"
}

# --- Commands ---

cmd_create() {
    require_tools vastai jq

    local name="" gpu="$DEFAULT_GPU" count=1
    local disk="$DEFAULT_DISK" image="$DEFAULT_IMAGE"
    local max_price="$DEFAULT_MAX_PRICE" interruptible=0

    name="${1:-}"
    shift || true
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu)           gpu="$(gpu_shortcut "$2")"; shift 2 ;;
            --count)         count="$2"; shift 2 ;;
            --disk)          disk="$2"; shift 2 ;;
            --image)         image="$2"; shift 2 ;;
            --max-price)     max_price="$2"; shift 2 ;;
            --interruptible) interruptible=1; shift ;;
            *)               echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    if [ -z "$name" ]; then
        echo "Usage: $0 create <name> [--gpu <type>] [--count <n>] [--disk <gb>] [--max-price <\$/hr>] [--interruptible]"
        exit 1
    fi

    if [ -f "$VAST_DIR/$name.env" ]; then
        # Stale-config recovery: if the saved instance no longer exists on
        # vast.ai, drop the stale .env and proceed. Otherwise refuse.
        # `|| true` because rg exits 1 on no match — set-e would otherwise abort
        # silently before the explicit "missing or malformed" check below.
        local stale_id=""
        stale_id=$(rg -o '^INSTANCE_ID=(.*)$' -r '$1' "$VAST_DIR/$name.env" 2>/dev/null | tr -d '"'\''' || true)
        # Refuse to proceed unless we can prove the saved instance is gone.
        # Empty / non-numeric stale_id means a corrupted .env we shouldn't silently overwrite.
        if ! [[ "$stale_id" =~ ^[0-9]+$ ]]; then
            echo "Error: Instance config '$name' exists at $VAST_DIR/$name.env but INSTANCE_ID is missing or malformed."
            echo "Inspect or delete it manually before retrying."
            exit 1
        fi
        # Probe vast.ai for the instance. We must distinguish three states:
        #   (a) instance still exists      → refuse to overwrite
        #   (b) instance confirmed gone    → safe to remove .env and proceed
        #   (c) API unreachable / unknown  → REFUSE; assuming "gone" on a transient
        #       outage or expired API key would orphan the original instance.
        local probe_out probe_rc=0
        probe_out=$(vastai show instance "$stale_id" --raw 2>&1) || probe_rc=$?
        if [ "$probe_rc" -eq 0 ] && echo "$probe_out" | jq -e '.id // empty' >/dev/null 2>&1; then
            echo "Error: Instance config '$name' already exists at $VAST_DIR/$name.env (id=$stale_id)"
            echo "Delete it first ($0 delete $name) or pick a different name."
            exit 1
        fi
        # "Confirmed gone" requires both: vastai itself returned an error (rc!=0),
        # AND the error text matches a known not-found marker. Without the rc
        # gate, a clean `vastai show` response that happens to contain the
        # string "not found" anywhere (e.g. in a description field) would
        # incorrectly remove the .env.
        if [ "$probe_rc" -ne 0 ] && echo "$probe_out" | rg -qi 'not\s*found|no\s*such|does\s*not\s*exist|unknown\s*instance|404'; then
            echo "Found stale config for '$name' (instance $stale_id confirmed gone on vast.ai). Removing and proceeding..."
            rm -f "$VAST_DIR/$name.env"
        else
            echo "Error: cannot determine state of instance $stale_id (vastai probe failed):" >&2
            echo "$probe_out" | sed 's/^/  /' >&2
            echo "Refusing to remove $VAST_DIR/$name.env on a transient/API failure." >&2
            echo "Check `vastai user` / network, or remove the .env manually after verifying the instance is gone." >&2
            exit 1
        fi
    fi

    # Validate max_price as a positive decimal so it can't smuggle filter syntax.
    if [ -n "$max_price" ] && ! [[ "$max_price" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: --max-price must be a decimal number (e.g. 0.5), got: $max_price" >&2
        exit 1
    fi

    # Resolve PUBLIC_KEY before creating an instance — readiness depends on SSH
    # logging in, so a missing key would create a billable orphan that can never
    # signal "ready". Fail fast instead.
    local public_key="${PUBLIC_KEY:-}"
    # Use `cat … || true` so an unreadable ed25519 file doesn't abort the
    # rsa fallback under `set -e`.
    if [ -z "$public_key" ] && [ -r "$HOME/.ssh/id_ed25519.pub" ]; then
        public_key="$(cat "$HOME/.ssh/id_ed25519.pub" || true)"
    fi
    if [ -z "$public_key" ] && [ -r "$HOME/.ssh/id_rsa.pub" ]; then
        public_key="$(cat "$HOME/.ssh/id_rsa.pub" || true)"
    fi
    if [ -z "$public_key" ]; then
        echo "Error: No SSH public key available." >&2
        echo "Set PUBLIC_KEY in env, or place a key at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub." >&2
        echo "(Without it, the instance can never reach 'ready' and would orphan as a billed machine.)" >&2
        exit 1
    fi

    # Find a cheap offer that matches.
    local query
    query=$(build_search_query "$gpu" "$count" "$disk" "$max_price" "$interruptible")
    echo "Searching offers: $query"
    local offers offer_id offer_dph offer_dc search_rc=0
    # `set -e` would abort the script on a non-zero exit (missing API key,
    # network down, rejected query) before we get a chance to print a useful
    # error. Capture the rc explicitly.
    offers=$(vastai search offers "$query" -o "dph_total" --raw 2>&1) || search_rc=$?
    if [ "$search_rc" -ne 0 ]; then
        echo "Error: vastai search offers failed (exit $search_rc):" >&2
        echo "$offers" | sed 's/^/  /' >&2
        echo "Check that 'vastai set api-key <KEY>' has been run and your network is up." >&2
        exit 1
    fi
    if ! echo "$offers" | jq -e 'type == "array" and length > 0' >/dev/null 2>&1; then
        echo "Error: no matching offers found."
        echo "Try relaxing constraints (--gpu, --max-price) or run: $0 search --gpu $gpu"
        exit 1
    fi
    offer_id=$(echo "$offers" | jq -r '.[0].id // empty')
    offer_dph=$(echo "$offers" | jq -r '.[0].dph_total // "?"')
    offer_dc=$(echo "$offers" | jq -r '.[0].geolocation // .[0].datacenter // "?"')
    if [ -z "$offer_id" ]; then
        echo "Error: top offer has no .id field — vastai response shape may have changed."
        exit 1
    fi

    echo "Creating instance '$name' from offer $offer_id..."
    echo "  GPU: ${count}x $gpu"
    echo "  Disk: ${disk}GB"
    echo "  Image: $image"
    echo "  Price: \$$offer_dph/hr  (${offer_dc})"
    echo ""

    # vast.ai's --env arg is a single string parsed shlex-style (docker-compatible).
    # Public keys contain spaces and HF tokens may contain shell metachars, so each
    # value MUST be shell-quoted before concatenation. printf %q produces output
    # that round-trips through shell parsing.
    local env_str="-p 22:22 -p 8888:8888"
    # Resolve HF token: env var takes precedence; fall back to the
    # huggingface CLI's saved token at ~/.cache/huggingface/token.
    local hf_token="${HF_TOKEN:-}"
    if [ -z "$hf_token" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
        hf_token="$(cat "$HOME/.cache/huggingface/token" || true)"
    fi
    if [ -n "$hf_token" ]; then
        env_str+=" -e HF_TOKEN=$(printf '%q' "$hf_token")"
    fi
    if [ -n "$public_key" ]; then
        env_str+=" -e PUBLIC_KEY=$(printf '%q' "$public_key")"
    fi

    local create_args=(
        create instance "$offer_id"
        --image "$image"
        --disk "$disk"
        --label "pawn-$name"
        --ssh
        # `--direct` selects vast.ai's ssh_direct runtype rather than the
        # default ssh_proxy. The proxy runs every connection through a
        # relay shell that prints the motd and exits without executing
        # commands sent over stdin or as ssh args — which silently breaks
        # `vast.sh launch`, `setup`, and `deploy`. Hosts that don't
        # support direct SSH simply fail to create here, which is much
        # better than a "successful" create whose ssh doesn't work.
        --direct
        # Disable vast.ai's auto-tmux wrapper. Every ssh session is
        # otherwise launched inside `tmux new-session`, which fails with
        # "tmux: command not found" on the runtime image (no tmux baked
        # in) and kills the session before any user command runs —
        # silently breaking non-interactive ssh / launch / setup /
        # deploy. The documented opt-out is to touch
        # ~/.no_auto_tmux on the pod. Belt-and-suspenders: do it for
        # both /root and /home/pawn so the dev image also benefits.
        --onstart-cmd 'touch /root/.no_auto_tmux 2>/dev/null; touch /home/pawn/.no_auto_tmux 2>/dev/null; chown pawn:pawn /home/pawn/.no_auto_tmux 2>/dev/null; true'
        --env "$env_str"
    )

    # Capture stdout and stderr separately: vastai echoes the --env we just sent
    # back into stdout, which contains the just-shell-quoted secrets. We never
    # print stdout on failure and only show the (usually-clean) stderr.
    local output stderr_file rc=0
    stderr_file=$(mktemp)
    # Embed the literal path in the trap (double-quoted) so it's expanded at
    # trap-set time. A single-quoted trap would defer expansion until script
    # exit, by which point the local variable is gone — under `set -u` that
    # crashes the cleanup with "unbound variable" and turns exit code into 1.
    trap "rm -f '$stderr_file'" EXIT
    output=$(vastai "${create_args[@]}" --raw 2>"$stderr_file") || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "Error: vastai create instance failed (exit $rc)." >&2
        sed 's/^/  /' "$stderr_file" >&2
        echo "(stdout suppressed because vastai echoes --env, which contains secrets)" >&2
        exit 1
    fi

    local instance_id
    instance_id=$(echo "$output" | jq -r '.new_contract // empty' 2>/dev/null)
    if [ -z "$instance_id" ]; then
        # Fall back to text parsing — accept both "new_contract" and 'new_contract'.
        instance_id=$(echo "$output" | rg -o '["'"'"']new_contract["'"'"']\s*[:=]\s*([0-9]+)' -r '$1' | head -1 || true)
    fi
    if [ -z "$instance_id" ]; then
        echo "Error: Could not extract instance ID from vastai output." >&2
        exit 1
    fi
    echo "Instance ID: $instance_id"

    # Persist a partial config NOW so a wait-loop timeout doesn't orphan a
    # billable instance with no local handle. wait_for_instance_running will
    # overwrite this with the full config (host/port/gpu) on success.
    save_instance_config "$name" "$instance_id" "" "" "$gpu"

    if ! wait_for_instance_running "$instance_id" "$name"; then
        echo "" >&2
        echo "Instance $instance_id was created but never became reachable over SSH." >&2
        echo "It is still running and accruing charges. Manage it with:" >&2
        echo "  $0 status $name      # check vast.ai state" >&2
        echo "  $0 stop $name        # pause (preserves disk, charged at storage rate)" >&2
        echo "  $0 delete $name      # destroy (frees disk)" >&2
        exit 1
    fi
}

cmd_search() {
    require_tools vastai jq
    local gpu="$DEFAULT_GPU" count=1 disk="$DEFAULT_DISK"
    local max_price="$DEFAULT_MAX_PRICE" interruptible=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu)           gpu="$(gpu_shortcut "$2")"; shift 2 ;;
            --count)         count="$2"; shift 2 ;;
            --disk)          disk="$2"; shift 2 ;;
            --max-price)     max_price="$2"; shift 2 ;;
            --interruptible) interruptible=1; shift ;;
            *)               echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    local query
    query=$(build_search_query "$gpu" "$count" "$disk" "$max_price" "$interruptible")
    echo "Query: $query"
    echo ""
    # Capture rc explicitly: under set -e, a bare pipeline failure (auth /
    # network / rejected query) would abort silently before any diagnostic.
    local out search_rc=0
    out=$(vastai search offers "$query" -o "dph_total" 2>&1) || search_rc=$?
    if [ "$search_rc" -ne 0 ]; then
        echo "Error: vastai search offers failed (exit $search_rc):" >&2
        echo "$out" | sed 's/^/  /' >&2
        echo "Check that 'vastai set api-key <KEY>' has been run and your network is up." >&2
        exit 1
    fi
    # awk (rather than head) reads all input so pipefail doesn't trip on SIGPIPE.
    echo "$out" | awk 'NR<=25'
}

cmd_start() {
    require_tools vastai jq
    local name="${1:?Usage: $0 start <name>}"
    load_instance_config "$name"
    echo "Starting instance '$name' ($INSTANCE_ID)..."
    if ! vastai start instance "$INSTANCE_ID"; then
        echo "Error: vastai start instance $INSTANCE_ID failed." >&2
        echo "The instance may have been deleted externally — check: $0 status $name" >&2
        exit 1
    fi
    # Clear cached host/port: a resumed instance can come up on a different
    # host/port. wait_for_instance_running rewrites the .env on success; on
    # timeout the empty cache forces load_instance_config_with_endpoint to
    # re-query later instead of using stale values.
    save_instance_config "$name" "$INSTANCE_ID" "" "" "${INSTANCE_GPU:-unknown}"
    wait_for_instance_running "$INSTANCE_ID" "$name" || exit 1
}

cmd_stop() {
    require_tools vastai
    local name="${1:?Usage: $0 stop <name>}"
    load_instance_config "$name"
    echo "Stopping instance '$name' ($INSTANCE_ID)..."
    if ! vastai stop instance "$INSTANCE_ID"; then
        echo "Error: vastai stop instance $INSTANCE_ID failed." >&2
        echo "Check current state: $0 status $name" >&2
        exit 1
    fi
    echo "Instance stopped. Disk preserved (still billed at storage rate)."
    echo "Resume with: $0 start $name   |   Free with: $0 delete $name"
}

cmd_delete() {
    require_tools vastai
    local name="${1:?Usage: $0 delete <name>}"
    load_instance_config "$name"
    read -p "Destroy instance '$name' ($INSTANCE_ID)? This wipes the disk. [y/N] " confirm
    if [ "${confirm,,}" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
    # Pass -y so vastai doesn't prompt again (its own confirmation would
    # silently abort under our non-interactive stdin and we'd remove the
    # local .env while leaving a billable instance alive).
    if ! vastai destroy instance -y "$INSTANCE_ID"; then
        echo "Error: vastai destroy instance $INSTANCE_ID failed." >&2
        echo "Local config preserved at $VAST_DIR/$name.env. Investigate before retrying." >&2
        exit 1
    fi
    rm -f "$VAST_DIR/$name.env"
    echo "Instance destroyed and config removed."
}

cmd_ssh() {
    local name="${1:?Usage: $0 ssh <name>}"
    load_instance_config_with_endpoint "$name"
    ssh $(ssh_opts) "root@$INSTANCE_HOST"
}

cmd_list() {
    require_tools vastai
    echo "=== Remote instances (vastai) ==="
    vastai show instances 2>/dev/null || echo "  (vastai not configured or no instances)"
    echo ""
    echo "=== Local instance configs ==="
    list_local_instances
}

cmd_status() {
    require_tools vastai
    local name="${1:?Usage: $0 status <name>}"
    load_instance_config "$name"
    vastai show instance "$INSTANCE_ID"
}

cmd_setup() {
    local name="${1:?Usage: $0 setup <name>}"
    load_instance_config_with_endpoint "$name"
    echo "Running setup on instance '$name'..."
    ssh $(ssh_opts) "root@$INSTANCE_HOST" "cd /workspace/pawn && bash deploy/setup.sh"
}

cmd_deploy() {
    local name="${1:?Usage: $0 deploy <name>}"
    load_instance_config_with_endpoint "$name"

    echo "=== Full deploy to '$name' ==="

    echo "--- Step 1: Build deploy package ---"
    bash "$REPO/deploy/build.sh"
    echo ""

    echo "--- Step 2: Transfer to instance ---"
    ssh $(ssh_opts) "root@$INSTANCE_HOST" "command -v rsync &>/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null
    rsync -avz --progress -e "ssh $(ssh_opts)" \
        "$REPO/deploy/pawn-deploy/" "root@$INSTANCE_HOST:/workspace/pawn/"
    echo ""

    echo "--- Step 3: Run setup ---"
    ssh $(ssh_opts) "root@$INSTANCE_HOST" "cd /workspace/pawn && bash deploy/setup.sh"
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

    load_instance_config_with_endpoint "$name"

    local script_name
    script_name=$(echo "$cmd" | rg -o 'scripts/([^ ]+)' -r '$1' | sed 's/\.py//' || echo "train")

    echo "Launching on '$name': $cmd"
    ssh $(ssh_opts) "root@$INSTANCE_HOST" "cd /workspace/pawn && \
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
    search)  shift; cmd_search "$@" ;;
    start)   shift; cmd_start "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    delete)  shift; cmd_delete "$@" ;;
    ssh)     shift; cmd_ssh "$@" ;;
    list)    shift; cmd_list "$@" ;;
    status)  shift; cmd_status "$@" ;;
    setup)   shift; cmd_setup "$@" ;;
    deploy)  shift; cmd_deploy "$@" ;;
    launch)  shift; cmd_launch "$@" ;;
    *)
        echo "PAWN vast.ai Instance Manager"
        echo ""
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  create <name> [--gpu <type>] [--count <n>] [--disk <gb>] [--max-price <\$/hr>] [--interruptible]"
        echo "                            Create a new instance from the cheapest matching offer"
        echo "  search [--gpu <type>] [--count <n>] [--max-price <\$/hr>]"
        echo "                            List matching offers without creating anything"
        echo "  start  <name>             Resume a stopped instance"
        echo "  stop   <name>             Pause an instance (disk preserved, storage billed)"
        echo "  delete <name>             Destroy an instance and its disk"
        echo "  ssh    <name>             SSH into an instance"
        echo "  list                      List all instances"
        echo "  status <name>             Get instance details"
        echo "  setup  <name>             Run setup.sh on the instance"
        echo "  deploy <name>             Build + transfer + setup (full deploy)"
        echo "  launch <name> <cmd>       Run a training command via nohup"
        echo ""
        echo "GPU shortcuts: a5000, a40, a6000, 4090, 5090, 3090, l40s, a100, a100-pcie, a100-sxm, h100, h100-sxm, h200"
        echo ""
        echo "Examples:"
        echo "  $0 search --gpu 4090 --max-price 0.5"
        echo "  $0 create exp1 --gpu 4090 --max-price 0.5"
        echo "  $0 create cheap1 --gpu 3090 --interruptible"
        echo "  $0 deploy exp1"
        echo "  $0 launch exp1 scripts/train.py --variant base"
        echo "  $0 stop exp1"
        echo ""
        echo "Setup:"
        echo "  pip install --user vastai"
        echo "  vastai set api-key <KEY>   # from https://vast.ai/console/account"
        ;;
esac
