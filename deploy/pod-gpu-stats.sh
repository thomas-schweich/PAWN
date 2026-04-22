#!/usr/bin/env bash
# Compact GPU stats for the tmux status bar — "util% vram-used/total".
# Tries nvidia-smi first, then rocm-smi. Prints nothing if neither tool
# is installed or no device is present, so the status bar stays clean
# on CPU-only pods.
set -uo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
    read -r util mem_used mem_total < <(
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ','
    )
    if [ -n "${util:-}" ]; then
        printf '%d%% %dG/%dG\n' "$util" "$((mem_used / 1024))" "$((mem_total / 1024))"
        exit 0
    fi
fi

if command -v rocm-smi >/dev/null 2>&1; then
    json=$(rocm-smi -u --showmeminfo vram --json 2>/dev/null || true)
    if [ -n "$json" ]; then
        util=$(jq -r 'to_entries[0].value["GPU use (%)"] // empty' <<<"$json")
        vu=$(jq -r 'to_entries[0].value["VRAM Total Used Memory (B)"] // empty' <<<"$json")
        vt=$(jq -r 'to_entries[0].value["VRAM Total Memory (B)"] // empty' <<<"$json")
        if [ -n "$util" ] && [ -n "$vu" ] && [ -n "$vt" ]; then
            # rocm-smi may return a float ("45.0") for GPU use; strip any
            # decimal before printf '%d' to avoid a warning / zero on some
            # locales. nvidia-smi --nounits always returns integers so the
            # first branch doesn't need the strip.
            printf '%d%% %dG/%dG\n' "${util%%.*}" "$((vu / 1024 / 1024 / 1024))" "$((vt / 1024 / 1024 / 1024))"
            exit 0
        fi
    fi
fi
