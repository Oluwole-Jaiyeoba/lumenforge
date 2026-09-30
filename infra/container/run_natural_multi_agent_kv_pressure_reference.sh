#!/usr/bin/env bash
set -euo pipefail

# Production-shaped, observation-only workload. Ordinary session replays after
# tool waits create all cache activity; it never calls prepared-prefix control.

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE}.json}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
REPORT_LABEL="${REPORT_LABEL:-natural_multi_agent_kv_pressure_$(date +%Y%m%d_%H%M%S)}"
SESSION_COUNT="${SESSION_COUNT:-8}"
TOOL_WAITS="${TOOL_WAITS:-3}"
SESSION_PREFIX_TOKENS="${SESSION_PREFIX_TOKENS:-8192}"
REPLAY_TOKENS="${REPLAY_TOKENS:-384}"
INITIAL_STAGGER_MS="${INITIAL_STAGGER_MS:-75}"
TOOL_WAIT_MIN_MS="${TOOL_WAIT_MIN_MS:-400}"
TOOL_WAIT_MAX_MS="${TOOL_WAIT_MAX_MS:-2000}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-60}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
SERVER_PID=""

[[ -f "${PROFILE_PATH}" ]] || { echo "Backend runtime profile not found: ${PROFILE_PATH}" >&2; exit 2; }
[[ -n "${SGLANG_DOCKER_IMAGE}" ]] || { echo "Set SGLANG_DOCKER_IMAGE to the pinned backend image." >&2; exit 2; }
[[ -n "${AGENTIC_MODEL_CACHE}" && -d "${AGENTIC_MODEL_CACHE}" ]] || { echo "Set AGENTIC_MODEL_CACHE to an existing model-cache directory." >&2; exit 2; }

profile_value() { python3 - "${PROFILE_PATH}" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], "")
print(" ".join(value) if isinstance(value, list) else value)
PY
}

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -f "${DIRECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${DIRECT_ROOT}/.venv/bin/activate"
fi
PACKAGE_PYTHONPATH=""
for package_src in "${REPO_ROOT}"/packages/*/src; do
  [[ -d "${package_src}" ]] || continue
  PACKAGE_PYTHONPATH="${PACKAGE_PYTHONPATH:+${PACKAGE_PYTHONPATH}:}${package_src}"
done
export PYTHONPATH="${PACKAGE_PYTHONPATH}:${PYTHONPATH:-}"

HARDWARE_PROFILE="${HARDWARE_PROFILE:-$(profile_value hardware_profile)}"
MODEL_CACHE_MOUNT="$(profile_value model_cache_mount)"
GPU_ARGS="$(profile_value gpu_runtime_args)"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/hardware/${REPORT_LABEL}"
RUNTIME_DIR="${RUN_ROOT}/runtime"
TRACE_PATH="${RUN_ROOT}/backend_trace.jsonl"
CONTRACT_PATH="${RUNTIME_DIR}/backend_runtime.json"
mkdir -p "${RUNTIME_DIR}"
: >"${TRACE_PATH}"
export SGLANG_DOCKER_EXTRA_ARGS="-v ${AGENTIC_MODEL_CACHE}:${MODEL_CACHE_MOUNT} -e HF_HOME=${MODEL_CACHE_MOUNT} ${SGLANG_DOCKER_EXTRA_ARGS:-}"
export SGLANG_DOCKER_GPU_ARGS="${SGLANG_DOCKER_GPU_ARGS:-${GPU_ARGS}}"
export SGLANG_DOCKER_IMAGE BACKEND_RUNTIME_PROFILE HARDWARE_PROFILE BACKEND_RUNTIME_CONTRACT_OUT="${CONTRACT_PATH}"

"${SCRIPT_DIR}/probe_sglang_runtime.sh"
BACKEND_VERSION="$(python3 - "${CONTRACT_PATH}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["backend_version"])
PY
)"

cat >"${RUN_ROOT}/command.sh" <<EOF
SGLANG_DOCKER_IMAGE=${SGLANG_DOCKER_IMAGE@Q}
AGENTIC_MODEL_CACHE=${AGENTIC_MODEL_CACHE@Q}
BACKEND_RUNTIME_PROFILE=${BACKEND_RUNTIME_PROFILE@Q}
REPORT_LABEL=${REPORT_LABEL@Q}
SESSION_COUNT=${SESSION_COUNT@Q}
TOOL_WAITS=${TOOL_WAITS@Q}
SESSION_PREFIX_TOKENS=${SESSION_PREFIX_TOKENS@Q}
REPLAY_TOKENS=${REPLAY_TOKENS@Q}
INITIAL_STAGGER_MS=${INITIAL_STAGGER_MS@Q}
TOOL_WAIT_MIN_MS=${TOOL_WAIT_MIN_MS@Q}
TOOL_WAIT_MAX_MS=${TOOL_WAIT_MAX_MS@Q}
CONNECT_TIMEOUT_S=${CONNECT_TIMEOUT_S@Q}
HICACHE_SIZE_GB=${HICACHE_SIZE_GB@Q}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC@Q}
export SGLANG_DOCKER_IMAGE AGENTIC_MODEL_CACHE BACKEND_RUNTIME_PROFILE REPORT_LABEL SESSION_COUNT TOOL_WAITS SESSION_PREFIX_TOKENS REPLAY_TOKENS INITIAL_STAGGER_MS TOOL_WAIT_MIN_MS TOOL_WAIT_MAX_MS CONNECT_TIMEOUT_S HICACHE_SIZE_GB MEM_FRACTION_STATIC
bash infra/container/run_natural_multi_agent_kv_pressure_reference.sh ${MODEL@Q}
EOF
chmod +x "${RUN_ROOT}/command.sh"

(
  cd "${DIRECT_ROOT}"
  export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${TRACE_PATH}"
  export AGENTIC_KV_TRACE_SCHEDULER=1 AGENTIC_KV_TRACE_KV_POOL=1
  export HICACHE_SIZE_GB MEM_FRACTION_STATIC
  bash scripts/run_sglang_hicache_server.sh "${MODEL}"
) >"${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"
deadline=$((SECONDS + 240))
until curl --silent --show-error --fail http://127.0.0.1:30000/v1/models >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Backend exited; inspect ${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  (( SECONDS < deadline )) || { echo "Backend did not become ready" >&2; exit 1; }
  sleep 2
done

python3 -m agentic_experiments.runners.run_natural_multi_agent_kv_pressure \
  --run-id "${REPORT_LABEL}" --out-dir "${RUN_ROOT}" --backend-trace "${TRACE_PATH}" \
  --model "${MODEL}" --session-count "${SESSION_COUNT}" --tool-waits "${TOOL_WAITS}" \
  --session-prefix-tokens "${SESSION_PREFIX_TOKENS}" --replay-tokens "${REPLAY_TOKENS}" \
  --initial-stagger-ms "${INITIAL_STAGGER_MS}" --tool-wait-min-ms "${TOOL_WAIT_MIN_MS}" \
  --tool-wait-max-ms "${TOOL_WAIT_MAX_MS}" --connect-timeout-s "${CONNECT_TIMEOUT_S}"

python3 -m agentic_reports.builders.build_natural_multi_agent_kv_pressure_report \
  --summary "${RUN_ROOT}/natural_multi_agent_kv_pressure_summary.json" \
  --out "${RUN_ROOT}/natural_multi_agent_kv_pressure_report.html"
python3 "${REPO_ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${REPORT_LABEL}" \
  --experiment "natural_multi_agent_kv_pressure" --model "${MODEL}" --hardware-profile "${HARDWARE_PROFILE}" \
  --runtime-contract "${CONTRACT_PATH}" \
  --workload-json "{\"frontend_priority\":\"none\",\"session_count\":${SESSION_COUNT},\"tool_waits\":${TOOL_WAITS},\"session_prefix_tokens\":${SESSION_PREFIX_TOKENS},\"replay_tokens\":${REPLAY_TOKENS}}" \
  --instrumentation "lightweight_backend_trace" --instrumentation "natural_session_timeline" \
  --artifact "command=${RUN_ROOT}/command.sh" --artifact "summary=${RUN_ROOT}/natural_multi_agent_kv_pressure_summary.json" \
  --artifact "report=${RUN_ROOT}/natural_multi_agent_kv_pressure_report.html" --completion-status complete
echo "Complete: ${RUN_ROOT}/natural_multi_agent_kv_pressure_report.html"
