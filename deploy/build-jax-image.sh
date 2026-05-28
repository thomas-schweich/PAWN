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

echo "=== Building $IMAGE (target=$TARGET) from $BRANCH @ $SHA ==="

cd "$REPO"

# Sanity: warn if there are uncommitted changes — the image won't reflect
# them. (Use --no-push if you're intentionally bench-iterating.)
if ! git diff-index --quiet HEAD --; then
    echo "WARN: working tree has uncommitted changes; image will be built from HEAD"
fi

docker build \
    --target "$TARGET" \
    --tag "$IMAGE" \
    --tag "$IMAGE_REPO:$TAG-$SHA" \
    --label "org.opencontainers.image.source=https://github.com/thomas-schweich/pawn" \
    --label "org.opencontainers.image.revision=$SHA" \
    --label "org.opencontainers.image.branch=$BRANCH" \
    .

if [ "$PUSH" -eq 1 ]; then
    echo "=== Pushing $IMAGE ==="
    docker push "$IMAGE"
    echo "=== Pushing $IMAGE_REPO:$TAG-$SHA ==="
    docker push "$IMAGE_REPO:$TAG-$SHA"
fi

echo "Done. Use with: bash deploy/vast.sh create <name> --image $IMAGE"
