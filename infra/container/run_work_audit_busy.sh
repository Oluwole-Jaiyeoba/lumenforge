#!/usr/bin/env bash
set -euo pipefail

# Paired busy-workload KV audit. Fresh pinned backend for each arm.
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${ROOT}/configs/backend_runtimes/${PROFILE}.json"
IMAGE="${SGLANG_DOCKER_IMAGE:-}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
RUN_ID="${WORK_AUDIT_RUN_ID:-work_audit_busy_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${WORK_AUDIT_SEEDS:-1 2}"
SESSION_COUNT="${WORK_AUDIT_SESSION_COUNT:-12}"
TOOL_WAITS="${WORK_AUDIT_TOOL_WAITS:-3}"
PREFIX_TOKENS="${WORK_AUDIT_PREFIX_TOKENS:-8192}"
REPLAY_TOKENS="${WORK_AUDIT_REPLAY_TOKENS:-64}"
WAIT_MIN_MS="${WORK_AUDIT_WAIT_MIN_MS:-800}"
WAIT_MAX_MS="${WORK_AUDIT_WAIT_MAX_MS:-3500}"
ESTIMATED_LOAD_MS="${WORK_AUDIT_ESTIMATED_LOAD_MS:-250}"
MARGIN_MS="${WORK_AUDIT_MARGIN_MS:-150}"
MINIMUM_HOST_TOKENS="${WORK_AUDIT_MINIMUM_HOST_TOKENS:-512}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
TRACE_PROFILE=kv_lifecycle_lean
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/work_audit/${RUN_ID}"
SERVER_PID=""
CONTAINER_CID=""

[[ -f "${PROFILE_PATH}" && -n "${IMAGE}" && -d "${MODEL_CACHE}" ]] || {
  echo "Set a valid runtime profile, SGLANG_DOCKER_IMAGE and AGENTIC_MODEL_CACHE." >&2
  exit 2
}
[[ "${SEEDS}" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "WORK_AUDIT_SEEDS must be space-separated integers" >&2; exit 2; }
for value in "${SESSION_COUNT}" "${TOOL_WAITS}" "${PREFIX_TOKENS}" "${REPLAY_TOKENS}" "${WAIT_MIN_MS}" "${WAIT_MAX_MS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Workload settings must be positive integers" >&2; exit 2; }
done
[[ "${WAIT_MAX_MS}" -ge "${WAIT_MIN_MS}" ]] || { echo "Wait bounds are reversed" >&2; exit 2; }
if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
  echo "Port 30000 already serves a model; refusing to disturb it." >&2
  exit 2
fi

stop_backend() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  if [[ -n "${CONTAINER_CID}" && -f "${CONTAINER_CID}" ]]; then
    docker rm -f "$(<"${CONTAINER_CID}")" >/dev/null 2>&1 || true
    rm -f "${CONTAINER_CID}"
  fi
  CONTAINER_CID=""
}
trap stop_backend EXIT
mkdir -p "${RUN_ROOT}/runtime" "${RUN_ROOT}/arms"
if [[ -f "${DIRECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${DIRECT_ROOT}/.venv/bin/activate"
fi
PACKAGE_PYTHONPATH=""
for source_dir in "${ROOT}"/packages/*/src; do
  [[ -d "${source_dir}" ]] || continue
  PACKAGE_PYTHONPATH="${PACKAGE_PYTHONPATH:+${PACKAGE_PYTHONPATH}:}${source_dir}"
done
export PYTHONPATH="${PACKAGE_PYTHONPATH}:${PYTHONPATH:-}"
export BACKEND_RUNTIME_PROFILE="${PROFILE}"
export BACKEND_RUNTIME_CONTRACT_OUT="${RUN_ROOT}/runtime/backend_runtime.json"
export SGLANG_DOCKER_IMAGE="${IMAGE}"
MODEL_CACHE_MOUNT="$(python3 - "${PROFILE_PATH}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model_cache_mount"])
PY
)"
MODEL_CACHE_ARGS="-v ${MODEL_CACHE}:${MODEL_CACHE_MOUNT} -e HF_HOME=${MODEL_CACHE_MOUNT}"
"${SCRIPT_DIR}/probe_sglang_runtime.sh"
NATIVE_LOAD_EVENT="$(python3 - <<'PY'
from agentic_backends.sglang.audit_v0510 import LIFECYCLE_ROLES
print(next(name for name, role in LIFECYCLE_ROLES.items() if role == "nested_load"))
PY
)"
python3 - "${BACKEND_RUNTIME_CONTRACT_OUT}" <<'PY'
import json, sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))
if runtime.get("backend_version") != "0.5.10.post1" or runtime.get("adapter") != "v0510":
    raise SystemExit("Busy audit requires the pinned v0510 SGLang backend")
PY

declare -a seed_values
read -r -a seed_values <<<"${SEEDS}"
for seed in "${seed_values[@]}"; do
  if (( seed % 2 )); then
    modes=(baseline controller)
  else
    modes=(controller baseline)
  fi
  for mode in "${modes[@]}"; do
    ARM_ID="seed${seed}_${mode}"
    ARM_ROOT="${RUN_ROOT}/arms/${ARM_ID}"
    mkdir -p "${ARM_ROOT}"
    CONTAINER_CID="${ARM_ROOT}/container.cid"
    export SGLANG_DOCKER_EXTRA_ARGS="${MODEL_CACHE_ARGS} --cidfile ${CONTAINER_CID}"
    echo "Starting ${ARM_ID} on a fresh backend"
    (
      cd "${DIRECT_ROOT}"
      export AGENTIC_KV_TRACE_ENABLE=1
      export AGENTIC_KV_TRACE_PATH="${ARM_ROOT}/backend_trace.jsonl"
      profile_values="$(python3 -m agentic_backends.sglang.instrumentation_profiles "${TRACE_PROFILE}" --shell)"
      while IFS='=' read -r name value; do
        [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
        printf -v "${name}" '%s' "${value}"
        export "${name}"
      done <<<"${profile_values}"
      export AGENTIC_KV_TRACE_CONTROL_ONLY=1
      export AGENTIC_KV_COPY_TELEMETRY_ENABLE=0
      export AGENTIC_KV_PREPARE_CONTROL_ENABLE=1
      export AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1
      export AGENTIC_KV_PREPARE_CONTROL_PORT=31991
      export HICACHE_SIZE_GB MEM_FRACTION_STATIC
      bash scripts/run_sglang_hicache_server.sh "${MODEL}"
    ) >"${ARM_ROOT}/server.log" 2>&1 &
    SERVER_PID="$!"
    deadline=$((SECONDS + 240))
    until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Backend exited; inspect ${ARM_ROOT}/server.log" >&2
        exit 1
      fi
      (( SECONDS < deadline )) || { echo "Backend startup timed out: ${ARM_ROOT}" >&2; exit 1; }
      sleep 2
    done
    python3 - "${ARM_ROOT}/backend_trace.jsonl" <<'PY'
import json, sys
from agentic_backends.sglang.instrumentation_profiles import validate_installation
with open(sys.argv[1], encoding="utf-8") as handle:
    summaries = [row for line in handle if (row := json.loads(line)).get("event") == "trace.install.summary"]
if not summaries or not validate_installation("kv_lifecycle_lean", "v0510", summaries[-1])["valid"]:
    raise SystemExit("Busy audit stopped: required KV lifecycle hooks were not installed")
if "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run" not in summaries[-1].get("installed_hooks", []):
    raise SystemExit("Busy audit stopped: control-only scheduler pump is missing")
PY
    python3 -m agentic_experiments.runners.run_busy_kv_audit \
      --run-id "${RUN_ID}_${ARM_ID}" --mode "${mode}" --seed "${seed}" \
      --out-dir "${ARM_ROOT}" --backend-trace "${ARM_ROOT}/backend_trace.jsonl" \
      --native-load-event "${NATIVE_LOAD_EVENT}" \
      --model "${MODEL}" --session-count "${SESSION_COUNT}" --tool-waits "${TOOL_WAITS}" \
      --prefix-tokens "${PREFIX_TOKENS}" --replay-tokens "${REPLAY_TOKENS}" \
      --wait-min-ms "${WAIT_MIN_MS}" --wait-max-ms "${WAIT_MAX_MS}" \
      --estimated-load-ms "${ESTIMATED_LOAD_MS}" --margin-ms "${MARGIN_MS}" \
      --minimum-host-tokens "${MINIMUM_HOST_TOKENS}"
    python3 -m agentic_backends.sglang.trace_contract \
      --adapter v0510 --profile "${TRACE_PROFILE}" \
      --trace "${ARM_ROOT}/backend_trace.jsonl" \
      --out "${ARM_ROOT}/instrumentation_audit.json"
    stop_backend
    sleep 3
  done
done

python3 -m agentic_experiments.runners.analyze_busy_kv_audit \
  --run-id "${RUN_ID}" --arms-dir "${RUN_ROOT}/arms" --out "${RUN_ROOT}/summary.json"
HARDWARE_PROFILE="$(python3 - "${PROFILE_PATH}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["hardware_profile"])
PY
)"
WORKLOAD_JSON="$(python3 - "${SEEDS}" "${SESSION_COUNT}" "${TOOL_WAITS}" "${PREFIX_TOKENS}" "${REPLAY_TOKENS}" "${WAIT_MIN_MS}" "${WAIT_MAX_MS}" "${HICACHE_SIZE_GB}" "${MEM_FRACTION_STATIC}" "${ESTIMATED_LOAD_MS}" "${MARGIN_MS}" "${MINIMUM_HOST_TOKENS}" <<'PY'
import json, sys
seeds, sessions, waits, prefix, replay, lo, hi, cache, mem, load, margin, host = sys.argv[1:]
print(json.dumps({"research_question_id":"RQ8", "frontend_priority":"none", "forced_eviction":False,
    "capacity_policy":"native_sglang", "modes":["baseline","controller"], "seeds":list(map(int,seeds.split())),
    "session_count":int(sessions), "tool_waits_per_session":int(waits), "prefix_tokens":int(prefix),
    "replay_tokens":int(replay), "wait_range_ms":[int(lo),int(hi)],
    "initial_stagger_ms":75, "estimated_load_ms":float(load), "load_margin_ms":float(margin),
    "minimum_host_tokens":int(host),
    "hicache_size_gb":float(cache), "mem_fraction_static":float(mem)}))
PY
)"
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment agentic_work_audit_busy --model "${MODEL}" \
  --hardware-profile "${HARDWARE_PROFILE}" \
  --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${WORKLOAD_JSON}" \
  --instrumentation v0510_backend_trace --instrumentation kv_lifecycle_lean \
  --artifact "summary=${RUN_ROOT}/summary.json" --completion-status complete
python3 -m agentic_reports.builders.build_work_audit_report \
  --results-dir "${DIRECT_ROOT}/artifacts/results/work_audit" \
  --progress-file "${ROOT}/docs/work_audit/research_progress.json" \
  --out "${RUN_ROOT}/report.html"
echo "Complete: ${RUN_ROOT}/summary.json"
