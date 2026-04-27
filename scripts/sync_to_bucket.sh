#!/usr/bin/env bash
# Sync a workspace to an HF bucket and exit non-zero if the API returned
# auth/quota errors that ``hf sync`` would otherwise swallow.
#
# Usage:
#   sync_to_bucket.sh <local_dir> <bucket_url> [<log_path>]
#
#   <bucket_url>  e.g. hf://buckets/thomas-schweich/pawn-c8e7a1e79781-20260423
#   <log_path>    optional cumulative log; the per-run capture used for the
#                 auth/quota grep is always isolated to *this* run so an
#                 old 403 in history can't poison subsequent successes.
#
# Suitable for a cron entry; the non-zero exit lets downstream wrappers
# (or the user's monitoring) treat repeated failures as a real signal
# instead of "silence = success".
set -u
set -o pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <local_dir> <bucket_url> [<log_path>]" >&2
    exit 2
fi

local_dir=$1
bucket=$2
log_path=${3:-/tmp/sync_to_bucket.log}

if [[ ! -d "$local_dir" ]]; then
    echo "ERROR: $local_dir is not a directory" >&2
    exit 2
fi
case "$bucket" in
    hf://buckets/*) ;;
    *)
        echo "ERROR: bucket must be an hf://buckets/<ns>/<name> URL, got: $bucket" >&2
        exit 2
        ;;
esac

# Capture this run's combined output to a private temp file. The
# auth/quota grep below scans only that file so old failures in the
# cumulative log don't make every subsequent successful run exit
# non-zero. The cumulative log (if requested) is appended to after the
# grep so it preserves history without affecting the current decision.
this_run_log=$(mktemp)
trap 'rm -f "$this_run_log"' EXIT

{
    date -u +"=== sync_to_bucket.sh: %Y-%m-%dT%H:%M:%SZ"
    hf sync "$local_dir" "$bucket" 2>&1
    sync_status=$?
    echo "=== hf sync exit: $sync_status"
    exit "$sync_status"
} | tee "$this_run_log"
sync_status=${PIPESTATUS[0]}

# Append this run's output to the cumulative log for history.
cat "$this_run_log" >> "$log_path" 2>/dev/null || true

# ``hf sync`` may exit 0 even when individual blob uploads return
# 403/401/429 (the user's session postmortem caught this with the
# private-repo storage cap). Re-check *this run's* output regardless
# of exit code.
if grep -iEq '\b(403|401|429)\b|HTTP[Ee]rror|Forbidden|Unauthorized|RateLimit' "$this_run_log"; then
    echo "ERROR: sync output contains auth/quota/rate-limit signals; treating as failure" >&2
    grep -iE '\b(403|401|429)\b|HTTP[Ee]rror|Forbidden|Unauthorized|RateLimit' "$this_run_log" \
        | tail -10 >&2
    # Preserve the original sync exit code if it was non-zero; otherwise
    # surface the auth/quota signal as exit 3 so downstream cron
    # wrappers see a distinguishable failure mode.
    if [[ "$sync_status" -eq 0 ]]; then
        exit 3
    fi
fi

exit "$sync_status"
