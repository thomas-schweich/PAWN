#!/usr/bin/env bash
# Build + push thomasschweich/pawn:jax from the current branch.
#
# H.4 housekeeping: every perf bench round so far has spent 5-10 min on
# apt-get install + uv sync + image pull on a cold pod. This pre-bakes the
# JAX-targeted runtime image so subsequent vast.ai launches start in
# seconds instead of minutes.
#
# Usage:
#   bash deploy/build-jax-image.sh                    # build :jax, push
#   bash deploy/build-jax-image.sh --no-push          # local only
#   bash deploy/build-jax-image.sh --target dev-rocm  # override target
#   bash deploy/build-jax-image.sh --tag jax-pre-A4   # alt tag
#
# The Dockerfile already supports both ROCm and CUDA runtime targets;
# this script picks the CUDA target by default (most vast.ai bench pods
# are NVIDIA). For ROCm, pass --target runtime-rocm.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_REPO="thomasschweich/pawn"
TAG="jax"
TARGET="runtime"
PUSH=1

while [ $# -gt 0 ]; do
    case "$1" in
        --no-push) PUSH=0; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --repo) IMAGE_REPO="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

IMAGE="$IMAGE_REPO:$TAG"
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
SHA_TAG="$IMAGE_REPO:$TAG-$SHA"

echo "=== Building $IMAGE (target=$TARGET) from $BRANCH @ $SHA ==="

cd "$REPO"

# Sanity: warn if there are uncommitted changes — the image won't reflect
# them. (Use --no-push if you're intentionally bench-iterating.)
if ! git diff-index --quiet HEAD --; then
    echo "WARN: working tree has uncommitted changes; image will be built from HEAD"
fi

# Surface context size so accidental bloat (heavy gitignored dirs that
# .dockerignore doesn't cover) is obvious before we spend bandwidth on
# the build. Approximate via `git ls-files` — our .dockerignore mirrors
# .gitignore for the heavy paths, so this is a close estimate without
# the cost of building the actual tar.
CTX_BYTES=$(git ls-files --cached --others --exclude-standard -z 2>/dev/null \
    | xargs -0 -r du -bc 2>/dev/null \
    | tail -1 | awk '{print $1}')
CTX_BYTES=${CTX_BYTES:-0}
CTX_HR=$(numfmt --to=iec "$CTX_BYTES" 2>/dev/null || echo "${CTX_BYTES}B")
echo "Build context estimate (git-tracked + untracked-non-ignored): ${CTX_HR}"
if [ "$CTX_BYTES" -gt $((4 * 1024 * 1024 * 1024)) ]; then
    echo "WARN: build context > 4 GiB — check .dockerignore for missing exclusions"
fi
# BuildKit will also print 'transferring context: ...' below; that's the
# ground truth.

docker build \
    --target "$TARGET" \
    --tag "$IMAGE" \
    --tag "$SHA_TAG" \
    --label "org.opencontainers.image.source=https://github.com/thomas-schweich/pawn" \
    --label "org.opencontainers.image.revision=$SHA" \
    --label "org.opencontainers.image.branch=$BRANCH" \
    .

# Report resulting image size (helps catch base-image or COPY bloat early).
IMG_SIZE=$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null || echo 0)
IMG_HR=$(numfmt --to=iec "$IMG_SIZE" 2>/dev/null || echo "${IMG_SIZE}B")
echo "Image size: ${IMG_HR}"

if [ "$PUSH" -eq 1 ]; then
    echo "=== Pushing $IMAGE ==="
    docker push "$IMAGE"
    echo "=== Pushing $SHA_TAG ==="
    docker push "$SHA_TAG"

    # Reclaim local disk now that the registry has both tags. Keep the
    # floating :jax tag locally for fast re-use; drop the SHA tag so
    # we don't accumulate one multi-GB layer set per build.
    echo "=== Reclaiming local disk: dropping local SHA tag ==="
    docker image rm "$SHA_TAG" >/dev/null 2>&1 || true
    # Trim BuildKit cache too — it grows by gigabytes per iteration and
    # is recoverable from the registry on the next build.
    docker builder prune -f --filter "until=72h" >/dev/null 2>&1 || true
fi

echo "Done. Use with: bash deploy/vast.sh create <name> --image $IMAGE"
