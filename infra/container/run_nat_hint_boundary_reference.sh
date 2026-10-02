#!/usr/bin/env bash
set -euo pipefail

# Host-native NAT capture and forwarding into a pinned containerized backend.
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:?Set SGLANG_DOCKER_IMAGE to the pinned backend image}"
AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:?Set AGENTIC_MODEL_CACHE to the host model cache}"
NAT_PYTHON="${NAT_PYTHON:-${REPO_ROOT}/.venvs/nat_py311/bin/python}"
REPORT_LABEL="${REPORT_LABEL:-nat_hint_boundary_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/hint_benchmark/${REPORT_LABEL}"
BACKEND_RUNTIME_CONTRACT_OUT="${RUN_ROOT}/backend_runtime.json"
SERVER_PID=""

[[ -d "${AGENTIC_MODEL_CACHE}" ]] || { echo "Model cache does not exist" >&2; exit 2; }
[[ -x "${NAT_PYTHON}" ]] || { echo "NAT Python not found: ${NAT_PYTHON}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run directory already exists: ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "${RUN_ROOT}"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export BACKEND_RUNTIME_PROFILE SGLANG_DOCKER_IMAGE BACKEND_RUNTIME_CONTRACT_OUT
export SGLANG_DOCKER_EXTRA_ARGS="-v ${AGENTIC_MODEL_CACHE}:/tmp/hfcache -e HF_HOME=/tmp/hfcache ${SGLANG_DOCKER_EXTRA_ARGS:-}"
export PYTHONPATH="$(printf '%s:' "${REPO_ROOT}"/packages/*/src)${PYTHONPATH:-}"
PYTHON="${DIRECT_ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="${NAT_PYTHON}"

"${SCRIPT_DIR}/probe_sglang_runtime.sh"
ADAPTER="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"])' "${BACKEND_RUNTIME_CONTRACT_OUT}")"
TRACE="${RUN_ROOT}/backend_trace.jsonl"
(
  cd "${DIRECT_ROOT}"
  export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${TRACE}"
  profile_values="$("${PYTHON}" -m agentic_backends.sglang.instrumentation_profiles request_boundary --shell)"
  while IFS='=' read -r name value; do
    [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
    printf -v "${name}" '%s' "${value}"
    export "${name}"
  done <<< "${profile_values}"
  export HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.70
  bash scripts/run_sglang_hicache_server.sh "${MODEL}"
) >"${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"

deadline=$((SECONDS + 240))
until curl --silent --fail http://127.0.0.1:30000/v1/models >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null || (( SECONDS >= deadline )); then
    echo "Backend did not become ready; inspect ${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  sleep 2
done

"${PYTHON}" -m agentic_backends.sglang.trace_contract \
  --adapter "${ADAPTER}" --profile request_boundary --installation-only \
  --trace "${TRACE}" --out "${RUN_ROOT}/hook_installation_audit.json"

"${NAT_PYTHON}" "${DIRECT_ROOT}/scripts/run_hint_benchmark.py" \
  --harness nemo_agent_toolkit --scenarios nat_priority_high \
  --nat-dynamo-transport-capture \
  --nat-backend-url http://127.0.0.1:30000/v1/chat/completions \
  --nat-backend-model "${MODEL}" \
  --run-id "${REPORT_LABEL}" --out-dir "${RUN_ROOT}/hint_run"

"${PYTHON}" -m agentic_backends.sglang.trace_contract \
  --adapter "${ADAPTER}" --profile request_boundary \
  --trace "${TRACE}" --out "${RUN_ROOT}/backend_audit.json" \
  --events-out "${RUN_ROOT}/backend_events.jsonl" --export-signal request.accepted
"${PYTHON}" -m agentic_experiments.runners.audit_hint_backend_evidence \
  --hint-observations "${RUN_ROOT}/hint_run/observed_hint_evidence.jsonl" \
  --backend-audit "${RUN_ROOT}/backend_audit.json" \
  --backend-events "${RUN_ROOT}/backend_events.jsonl" \
  --request-map "${RUN_ROOT}/hint_run/backend_request_map.json" \
  --require-same-request \
  --out "${RUN_ROOT}/hint_backend_audit.json"
echo "Complete: ${RUN_ROOT}/hint_backend_audit.json"
