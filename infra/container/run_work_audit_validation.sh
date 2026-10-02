#!/usr/bin/env bash
set -euo pipefail

# One backend, two sequential cases. This validates evidence collection only.
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${ROOT}/configs/backend_runtimes/${PROFILE}.json"
IMAGE="${SGLANG_DOCKER_IMAGE:-}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
RUN_ID="${WORK_AUDIT_RUN_ID:-work_audit_$(date +%Y%m%d_%H%M%S)}"
SECOND_REPLAY="${WORK_AUDIT_SECOND_REPLAY:-0}"
CASE_ORDER="${WORK_AUDIT_CASE_ORDER:-warm-host}"
TRACE_PROFILE="${WORK_AUDIT_TRACE_PROFILE:-kv_lifecycle}"
RESULTS_BASE="${DIRECT_ROOT}/artifacts/results/work_audit"
RUN_ROOT="${RESULTS_BASE}/${RUN_ID}"
SERVER_PID=""

[[ -f "${PROFILE_PATH}" ]] || { echo "Missing runtime profile: ${PROFILE_PATH}" >&2; exit 2; }
[[ -n "${IMAGE}" ]] || { echo "Set SGLANG_DOCKER_IMAGE" >&2; exit 2; }
[[ -d "${MODEL_CACHE}" ]] || { echo "Set AGENTIC_MODEL_CACHE to a directory" >&2; exit 2; }
[[ "${SECOND_REPLAY}" == "0" || "${SECOND_REPLAY}" == "1" ]] || {
  echo "WORK_AUDIT_SECOND_REPLAY must be 0 or 1" >&2; exit 2;
}
[[ "${CASE_ORDER}" == "warm-host" || "${CASE_ORDER}" == "host-warm" ]] || {
  echo "WORK_AUDIT_CASE_ORDER must be warm-host or host-warm" >&2; exit 2;
}
[[ "${TRACE_PROFILE}" == "kv_lifecycle" || "${TRACE_PROFILE}" == "kv_lifecycle_lean" ]] || {
  echo "WORK_AUDIT_TRACE_PROFILE must be kv_lifecycle or kv_lifecycle_lean" >&2; exit 2;
}
if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
  echo "Port 30000 is already serving a model; refusing to disturb it." >&2
  exit 2
fi
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
mkdir -p "${RUN_ROOT}/runtime"

if [[ -f "${DIRECT_ROOT}/.venv/bin/activate" ]]; then
  source "${DIRECT_ROOT}/.venv/bin/activate"
fi
PACKAGE_PYTHONPATH=""
for source_dir in "${ROOT}"/packages/*/src; do
  [[ -d "${source_dir}" ]] || continue
  PACKAGE_PYTHONPATH="${PACKAGE_PYTHONPATH:+${PACKAGE_PYTHONPATH}:}${source_dir}"
done
export PYTHONPATH="${PACKAGE_PYTHONPATH}:${PYTHONPATH:-}"
export SGLANG_DOCKER_IMAGE="${IMAGE}"
export BACKEND_RUNTIME_PROFILE="${PROFILE}"
export BACKEND_RUNTIME_CONTRACT_OUT="${RUN_ROOT}/runtime/backend_runtime.json"
MODEL_CACHE_MOUNT="$(python3 - "${PROFILE_PATH}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model_cache_mount"])
PY
)"
export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:${MODEL_CACHE_MOUNT} -e HF_HOME=${MODEL_CACHE_MOUNT} ${SGLANG_DOCKER_EXTRA_ARGS:-}"

"${SCRIPT_DIR}/probe_sglang_runtime.sh"
python3 - "${BACKEND_RUNTIME_CONTRACT_OUT}" <<'PY'
import json
import sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))
if runtime.get("backend_version") != "0.5.10.post1" or runtime.get("adapter") != "v0510":
    raise SystemExit(f"Refusing unpinned audit runtime: {runtime.get('backend_version')} / {runtime.get('adapter')}")
PY

echo "Starting pinned backend for ${RUN_ID}"
(
  cd "${DIRECT_ROOT}"
  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${RUN_ROOT}/backend_trace.jsonl"
  profile_values="$(python3 -m agentic_backends.sglang.instrumentation_profiles "${TRACE_PROFILE}" --shell)"
  while IFS='=' read -r name value; do
    [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
    printf -v "${name}" '%s' "${!name:-${value}}"
    export "${name}"
  done <<< "${profile_values}"
  if [[ "${TRACE_PROFILE}" == "kv_lifecycle_lean" ]]; then
    export AGENTIC_KV_TRACE_CONTROL_ONLY=1
  fi
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=0
  export AGENTIC_KV_PREPARE_CONTROL_ENABLE=1
  export AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1
  export AGENTIC_KV_PREPARE_CONTROL_PORT=31991
  export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
  export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
  bash scripts/run_sglang_hicache_server.sh "${MODEL}"
) >"${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"

deadline=$((SECONDS + 240))
until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Backend exited; inspect ${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Backend startup timed out; inspect ${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  sleep 2
done

python3 - "${RUN_ROOT}/backend_trace.jsonl" "${TRACE_PROFILE}" <<'PY'
import json
import sys
from pathlib import Path
from agentic_backends.sglang.instrumentation_profiles import validate_installation

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    summaries = [row for line in handle if (row := json.loads(line)).get("event") == "trace.install.summary"]
if not summaries:
    raise SystemExit("Work-audit gate failed: missing backend hook installation summary")
result = validate_installation(sys.argv[2], "v0510", summaries[-1])
if not result["valid"]:
    raise SystemExit(f"Work-audit gate failed: {json.dumps(result['missing'])}")
if sys.argv[2] == "kv_lifecycle_lean":
    pump = "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run"
    if pump not in summaries[-1].get("installed_hooks", []):
        raise SystemExit("Work-audit gate failed: control-only scheduler pump is missing")
print("Work-audit lifecycle hooks installed")
PY

SECOND_REPLAY_RUN_ARGS=()
SECOND_REPLAY_ANALYSIS_ARGS=()
if [[ "${SECOND_REPLAY}" == "1" ]]; then
  SECOND_REPLAY_RUN_ARGS+=(--second-replay)
  SECOND_REPLAY_ANALYSIS_ARGS+=(--require-second-replay)
fi
python3 -m agentic_experiments.runners.run_work_audit_validation \
  --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" --case-order "${CASE_ORDER}" \
  "${SECOND_REPLAY_RUN_ARGS[@]}"
python3 -m agentic_backends.sglang.trace_contract \
  --adapter v0510 --profile "${TRACE_PROFILE}" \
  --trace "${RUN_ROOT}/backend_trace.jsonl" \
  --out "${RUN_ROOT}/instrumentation_audit.json"
python3 -m agentic_experiments.runners.analyze_work_audit_validation \
  --run-id "${RUN_ID}" --trace "${RUN_ROOT}/backend_trace.jsonl" \
  --harness "${RUN_ROOT}/harness_events.jsonl" --out-dir "${RUN_ROOT}" \
  "${SECOND_REPLAY_ANALYSIS_ARGS[@]}"
python3 -m agentic_reports.audits.build_work_audit_block_audit \
  --trace "${RUN_ROOT}/backend_trace.jsonl" \
  --harness "${RUN_ROOT}/harness_events.jsonl" \
  --summary "${RUN_ROOT}/summary.json" --out "${RUN_ROOT}/block_audit.json"
python3 -m agentic_reports.builders.build_work_audit_report \
  --results-dir "${RESULTS_BASE}" --out "${RUN_ROOT}/report.html"

HARDWARE_PROFILE="$(python3 - "${PROFILE_PATH}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["hardware_profile"])
PY
)"
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment "agentic_work_audit_validation" --model "${MODEL}" \
  --hardware-profile "${HARDWARE_PROFILE}" \
  --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "{\"cases\":[\"${CASE_ORDER%-*}\",\"${CASE_ORDER#*-}\"],\"frontend_priority\":\"none\",\"purpose\":\"evidence_validation\",\"replays_per_case\":$((SECOND_REPLAY + 1))}" \
  --instrumentation "v0510_backend_trace" --instrumentation "work_audit_event_join" \
  --instrumentation "logical_block_lifecycle_reuse" --instrumentation "${TRACE_PROFILE}" \
  --artifact "instrumentation_audit=${RUN_ROOT}/instrumentation_audit.json" \
  --artifact "summary=${RUN_ROOT}/summary.json" \
  --artifact "block_audit=${RUN_ROOT}/block_audit.json" \
  --artifact "report=${RUN_ROOT}/report.html" \
  --completion-status complete
echo "Validated: ${RUN_ROOT}/summary.json"
