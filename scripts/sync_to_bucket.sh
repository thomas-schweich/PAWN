#!/usr/bin/env bash
# Sync a workspace to an HF bucket and exit non-zero if the API returned
# auth/quota errors that ``hf sync`` would otherwise swallow.
#
# Usage:
#   sync_to_bucket.sh <local_dir> <bucket_url> [<log_path>]
#
#   <bucket_url>  e.g. hf://buckets/thomas-schweich/pawn-c8e7a1e79781-20260423
#   <log_path>    defaults to /tmp/sync_to_bucket.log
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

# Run the sync, teeing combined output to the log so the grep below can
# inspect it whether or not stdout/stderr were already redirected by
# the caller.
{
    date -u +"=== sync_to_bucket.sh: %Y-%m-%dT%H:%M:%SZ"
    hf sync "$local_dir" "$bucket" 2>&1
    sync_status=$?
    echo "=== hf sync exit: $sync_status"
    exit "$sync_status"
} | tee -a "$log_path"
sync_status=${PIPESTATUS[0]}

# ``hf sync`` may exit 0 even when individual blob uploads return
# 403/401/429 (the user's session postmortem caught this with the
# private-repo storage cap). Re-check the log for those patterns
# regardless of exit code.
if grep -iEq '\b(403|401|429)\b|HTTP[Ee]rror|Forbidden|Unauthorized|RateLimit' "$log_path"; then
    echo "ERROR: sync log contains auth/quota/rate-limit signals; treating as failure" >&2
    grep -iE '\b(403|401|429)\b|HTTP[Ee]rror|Forbidden|Unauthorized|RateLimit' "$log_path" \
        | tail -10 >&2
    # Preserve the original sync exit code if it was non-zero; otherwise
    # surface the auth/quota signal as exit 3 so downstream cron
    # wrappers see a distinguishable failure mode.
    if [[ "$sync_status" -eq 0 ]]; then
        exit 3
    fi
fi

exit "$sync_status"
