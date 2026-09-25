#!/usr/bin/env bash
set -euo pipefail

# Host-side hybrid experiment entry point. The experiment driver, gateway,
# harnesses, controller, and report builder remain on the host. Only the
# SGLang server started by the existing testbed launcher runs in Docker.

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE}.json}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
REPORT_LABEL="${REPORT_LABEL:-hybrid_reference_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "Backend runtime profile not found: ${PROFILE_PATH}" >&2
  exit 2
fi
if [[ -z "${SGLANG_DOCKER_IMAGE}" ]]; then
  echo "Set SGLANG_DOCKER_IMAGE to an explicit backend image." >&2
  exit 2
fi
if [[ -z "${AGENTIC_MODEL_CACHE}" || ! -d "${AGENTIC_MODEL_CACHE}" ]]; then
  echo "Set AGENTIC_MODEL_CACHE to an existing host model-cache directory." >&2
  exit 2
fi

profile_value() {
  python3 - "${PROFILE_PATH}" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], "")
if isinstance(value, list):
    print(" ".join(value))
else:
    print(value)
PY
}

PROFILE_ID="$(profile_value profile_id)"
PROFILE_HARDWARE="$(profile_value hardware_profile)"
MODEL_CACHE_MOUNT="$(profile_value model_cache_mount)"
GPU_ARGS="$(profile_value gpu_runtime_args)"
if [[ -z "${HARDWARE_PROFILE:-}" && -n "${PROFILE_HARDWARE}" && "${PROFILE_HARDWARE}" != "none" ]]; then
  HARDWARE_PROFILE="${PROFILE_HARDWARE}"
fi

RUNTIME_DIR="${DIRECT_ROOT}/${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}/runtime"
BACKEND_RUNTIME_CONTRACT_OUT="${BACKEND_RUNTIME_CONTRACT_OUT:-${RUNTIME_DIR}/backend_runtime.json}"
RUN_MANIFEST_PATH="${RUN_MANIFEST_PATH:-${RUNTIME_DIR}/run_manifest.json}"
INSTRUMENTATION_CONTRACT="${CONTROLLER_INSTRUMENTATION_CONTRACT:-${REPO_ROOT}/configs/controller_instrumentation_contracts/controller_replay_v1.json}"
INSTRUMENTATION_STATIC_PREFLIGHT="${RUNTIME_DIR}/instrumentation_static_preflight.json"

mount_args=(
  -v "${AGENTIC_MODEL_CACHE}:${MODEL_CACHE_MOUNT}"
  -e "HF_HOME=${MODEL_CACHE_MOUNT}"
)
export SGLANG_DOCKER_EXTRA_ARGS="$(printf '%q ' "${mount_args[@]}") ${SGLANG_DOCKER_EXTRA_ARGS:-}"
export SGLANG_DOCKER_GPU_ARGS="${SGLANG_DOCKER_GPU_ARGS:-${GPU_ARGS}}"
export BACKEND_RUNTIME_PROFILE
export BACKEND_RUNTIME_CONTRACT_OUT
export SGLANG_DOCKER_IMAGE
export HARDWARE_PROFILE
export REPORT_LABEL
export RESULTS_ROOT
export DRY_RUN

if [[ -f "${DIRECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${DIRECT_ROOT}/.venv/bin/activate"
fi

echo "Hybrid reference run"
echo "  runtime profile: ${PROFILE_ID}"
echo "  hardware profile: ${HARDWARE_PROFILE:-none}"
echo "  backend image: ${SGLANG_DOCKER_IMAGE}"
echo "  host model cache: ${AGENTIC_MODEL_CACHE}"
echo "  report label: ${REPORT_LABEL}"

"${SCRIPT_DIR}/probe_sglang_runtime.sh"
if [[ "${INSTRUMENTATION_CONTRACT}" != /* ]]; then
  INSTRUMENTATION_CONTRACT="${REPO_ROOT}/${INSTRUMENTATION_CONTRACT}"
fi
if [[ ! -f "${INSTRUMENTATION_CONTRACT}" ]]; then
  echo "Controller instrumentation contract not found: ${INSTRUMENTATION_CONTRACT}" >&2
  exit 2
fi
PYTHONPATH="${REPO_ROOT}/packages/agentic-backend-sglang/src:${PYTHONPATH:-}" python3 \
  -m agentic_backends.sglang.instrumentation_preflight \
  --stage static \
  --contract "${INSTRUMENTATION_CONTRACT}" \
  --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --policy "${CONTROLLER_INSTRUMENTATION_POLICY:-strict}" \
  --out "${INSTRUMENTATION_STATIC_PREFLIGHT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. Runtime profile and container mount configuration validated."
  exit 0
fi

workload_json="$(python3 - <<'PY'
import json
import os
print(json.dumps({
    "harnesses": os.environ.get("HARNESSES", "").split(),
    "modes": os.environ.get("MODES", "").split(),
    "pressure_levels": os.environ.get("PRESSURE_LEVELS", "").split(),
    "tool_wait_profile": os.environ.get("TOOL_WAIT_PROFILE", ""),
    "task_replay_steps": os.environ.get("TASK_REPLAY_STEPS", ""),
}))
PY
)"
python3 "${REPO_ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_MANIFEST_PATH}" \
  --run-id "${REPORT_LABEL}" \
  --experiment "hybrid_reference" \
  --model "${MODEL}" \
  --hardware-profile "${HARDWARE_PROFILE:-}" \
  --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${workload_json}" \
  --instrumentation "${TRACE_PROFILE:-full_debug}" \
  --artifact "runtime_contract=${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --artifact "instrumentation_contract=${INSTRUMENTATION_CONTRACT}" \
  --artifact "instrumentation_static_preflight=${INSTRUMENTATION_STATIC_PREFLIGHT}" \
  --artifact "run_root=${DIRECT_ROOT}/${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}"

export AGENTIC_BACKEND_RUNTIME_CONTRACT="${BACKEND_RUNTIME_CONTRACT_OUT}"
export AGENTIC_RUN_MANIFEST="${RUN_MANIFEST_PATH}"
export CONTROLLER_INSTRUMENTATION_CONTRACT="${INSTRUMENTATION_CONTRACT}"

cd "${DIRECT_ROOT}"
exec bash scripts/run_harness_deadline_pressure.sh "${MODEL}"
