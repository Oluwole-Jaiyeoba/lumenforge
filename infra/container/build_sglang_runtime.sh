#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROFILE_PATH="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE:-nvidia_gh200}.json}"
SGLANG_RUNTIME_TAG="${SGLANG_RUNTIME_TAG:-agentic-sglang:0.5.10.post1}"
SGLANG_BASE_IMAGE="${SGLANG_BASE_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.10.post1}"

if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "Backend runtime profile not found: ${PROFILE_PATH}" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build a backend runtime image." >&2
  exit 2
fi

profile_value() {
  python3 - "${PROFILE_PATH}" "$1" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], ""))
PY
}

PROFILE_ID="$(profile_value profile_id)"
HOST_ARCH="$(profile_value host_architecture)"
EXPECTED_VERSION="$(profile_value expected_backend_version)"
if [[ "${SGLANG_VERSION}" != "${EXPECTED_VERSION}" ]]; then
  echo "Profile ${PROFILE_ID} expects SGLang ${EXPECTED_VERSION}, got ${SGLANG_VERSION}." >&2
  exit 2
fi

case "${HOST_ARCH}" in
  aarch64|arm64) TARGET_PLATFORM="linux/arm64" ;;
  x86_64|amd64) TARGET_PLATFORM="linux/amd64" ;;
  *)
    echo "Unsupported runtime-profile host architecture: ${HOST_ARCH}" >&2
    exit 2
    ;;
esac

echo "Building ${PROFILE_ID} backend image"
echo "  platform: ${TARGET_PLATFORM}"
echo "  base image: ${SGLANG_BASE_IMAGE}"
echo "  SGLang: ${SGLANG_VERSION}"
echo "  tag: ${SGLANG_RUNTIME_TAG}"

docker build \
  --platform "${TARGET_PLATFORM}" \
  --label "org.agentic-hardware.runtime-profile=${PROFILE_ID}" \
  --label "org.agentic-hardware.sglang-version=${SGLANG_VERSION}" \
  --build-arg "BASE_IMAGE=${SGLANG_BASE_IMAGE}" \
  --build-arg "SGLANG_VERSION=${SGLANG_VERSION}" \
  --file "${SCRIPT_DIR}/Dockerfile.sglang" \
  --tag "${SGLANG_RUNTIME_TAG}" \
  "${REPO_ROOT}"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${SGLANG_RUNTIME_TAG}")"
echo "Built ${SGLANG_RUNTIME_TAG}"
echo "Local image identity: local-image-id:${IMAGE_ID}"
echo "Next: SGLANG_DOCKER_IMAGE=${SGLANG_RUNTIME_TAG} BACKEND_RUNTIME_PROFILE=${PROFILE_ID} bash infra/container/probe_sglang_runtime.sh"
