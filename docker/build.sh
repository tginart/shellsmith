#!/usr/bin/env bash
set -euo pipefail

# Build the Shellsmith sandbox image from this checkout.
#
# The build context is a private temp dir holding only the committed docker/ recipe and src/.
# World files never enter the image (the host uploads one per episode). Agent
# CLIs are installed as pinned Linux npm packages by the Dockerfile.
#
# The image is tagged and labelled with a hash of the src/ and docker/ git trees, so a launcher
# can tell a stale image from a fresh one instead of trusting `:latest`. An
# overridden build (CLI versions or platform) gets a label that can never equal
# the plain build identity, so it is never mistaken for the default image.

here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
image_identity="$(git -C "$repo_root" rev-parse HEAD:src HEAD:docker | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
if [[ -n "$(git -C "$repo_root" status --porcelain -- src docker)" ]]; then
  echo "src/ or docker/ has uncommitted changes; commit them so the image label matches its contents." >&2
  exit 1
fi
# Read the committed source of truth without importing the whole package.
default_versions="$(python3 -c 'import runpy, sys; c = runpy.run_path(sys.argv[1]); print(c["PI_CODING_AGENT_VERSION"], c["CODEX_CLI_VERSION"])' "$repo_root/src/shellsmith/config.py")"
read -r default_pi_version default_codex_version <<< "$default_versions"
tree_label="$image_identity"
if [[ -n "${PI_CODING_AGENT_VERSION:-}${CODEX_VERSION:-}${DOCKER_PLATFORM:-}" ]]; then
  tree_label="${image_identity}+pi=${PI_CODING_AGENT_VERSION:-default}+codex=${CODEX_VERSION:-default}+platform=${DOCKER_PLATFORM:-default}"
fi
image_tag="${IMAGE_TAG:-shellsmith:${image_identity:0:12}}"

context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT
git -C "$repo_root" archive HEAD src docker | tar -x -C "$context"

# Native platform by default (local Docker is the only sandbox backend);
# set DOCKER_PLATFORM to cross-build.
platform_args=()
if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
  platform_args=(--platform "$DOCKER_PLATFORM")
fi
docker build \
  ${platform_args[@]+"${platform_args[@]}"} \
  --file "$context/docker/Dockerfile" \
  --build-arg "PI_CODING_AGENT_VERSION=${PI_CODING_AGENT_VERSION:-$default_pi_version}" \
  --build-arg "CODEX_VERSION=${CODEX_VERSION:-$default_codex_version}" \
  --label "shellsmith.image_identity=$tree_label" \
  --tag "$image_tag" \
  "$context"
echo "Built $image_tag (shellsmith.image_identity=$tree_label)"
