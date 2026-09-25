#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROFILE_PATH="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE:-nvidia_gh200}.json}"
OUT_JSON="${BACKEND_RUNTIME_CONTRACT_OUT:-${REPO_ROOT}/sglang_direct_kv/artifacts/backend_runtime.json}"
DOCKER_PULL="${SGLANG_DOCKER_PULL:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "Backend runtime profile not found: ${PROFILE_PATH}" >&2
  exit 2
fi

profile_value() {
  python3 - "${PROFILE_PATH}" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], "")
if isinstance(value, list):
    print(" ".join(value))
elif isinstance(value, bool):
    print("1" if value else "0")
else:
    print(value)
PY
}

PROFILE_ID="$(profile_value profile_id)"
IMAGE_ENV="$(profile_value container_image_env)"
DEFAULT_IMAGE="$(profile_value default_container_image)"
GPU_VENDOR="$(profile_value gpu_vendor)"
GPU_ARCHITECTURE="$(profile_value gpu_architecture)"
HOST_ARCHITECTURE="$(profile_value host_architecture)"
GPU_ARGS="$(profile_value gpu_runtime_args)"
EXPECTED_VERSION="$(profile_value expected_backend_version)"
IMAGE="${SGLANG_DOCKER_IMAGE:-${DEFAULT_IMAGE}}"
if [[ -n "${IMAGE_ENV}" && -n "${!IMAGE_ENV:-}" ]]; then
  IMAGE="${!IMAGE_ENV}"
fi
if [[ -z "${IMAGE}" ]]; then
  echo "No container image configured. Set ${IMAGE_ENV:-SGLANG_DOCKER_IMAGE}." >&2
  exit 2
fi

echo "Backend runtime profile: ${PROFILE_ID}"
echo "Container image: ${IMAGE}"
echo "Expected SGLang version: ${EXPECTED_VERSION}"
echo "Runtime contract: ${OUT_JSON}"
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
mkdir -p "$(dirname "${OUT_JSON}")"
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the backend runtime probe." >&2
  exit 2
fi
if [[ "${DOCKER_PULL}" == "1" ]]; then
  docker pull "${IMAGE}"
fi

IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE}" 2>/dev/null || true)"
REQUIRE_DIGEST="$(profile_value reference_run_requires_image_digest)"
if [[ "${REQUIRE_DIGEST}" == "1" && -z "${IMAGE_DIGEST}" ]]; then
  echo "Reference runs require an immutable image digest; none was found for ${IMAGE}." >&2
  echo "Pull a published image or use a digest-qualified SGLANG_DOCKER_IMAGE." >&2
  exit 2
fi
PYTHONPATH_VALUE="/workspace/packages/agentic-core/src:/workspace/packages/agentic-backend-api/src:/workspace/packages/agentic-backend-sglang/src"
required_args=()
while IFS= read -r capability; do
  [[ -n "${capability}" ]] && required_args+=(--require-capability "${capability}")
done < <(python3 - "${PROFILE_PATH}" <<'PY'
import json
import sys
for value in json.load(open(sys.argv[1], encoding="utf-8")).get("required_capabilities", []):
    print(value)
PY
)

# shellcheck disable=SC2086
docker run --rm ${GPU_ARGS} \
  -e "PYTHONPATH=${PYTHONPATH_VALUE}" \
  -v "${REPO_ROOT}:/workspace:ro" \
  -v "$(dirname "${OUT_JSON}"):$(dirname "${OUT_JSON}")" \
  -w /workspace \
  "${IMAGE}" \
  python -m agentic_backends.sglang.runtime_contract \
    --out "${OUT_JSON}" \
    --runtime-profile "${PROFILE_ID}" \
    --container-image "${IMAGE}" \
    --container-image-digest "${IMAGE_DIGEST}" \
    --gpu-vendor "${GPU_VENDOR}" \
    --gpu-architecture "${GPU_ARCHITECTURE}" \
    --host-architecture "${HOST_ARCHITECTURE}" \
    "${required_args[@]}"

python3 - "${OUT_JSON}" "${EXPECTED_VERSION}" <<'PY'
import json
import sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
expected = sys.argv[2]
actual = row.get("backend_version", "")
if expected and actual != expected:
    raise SystemExit(f"SGLang version mismatch: expected {expected}, observed {actual}")
if not row.get("probe_ok"):
    raise SystemExit("Backend runtime capability probe failed")
print(f"Runtime handshake passed: SGLang {actual}, adapter {row.get('adapter')}")
PY
