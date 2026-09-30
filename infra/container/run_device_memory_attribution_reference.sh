#!/usr/bin/env bash
set -euo pipefail

# Clean three-condition timing lane. The backend is restarted for every sample
# so each condition gets a fresh SGLang cache and scheduler state.

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE}.json}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
REPORT_LABEL="${REPORT_LABEL:-device_memory_attribution_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_SET_ID="${SAMPLE_SET_ID:-${REPORT_LABEL}_samples}"
TRIALS="${TRIALS:-1}"
SEED="${SEED:-7}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
WARMUP_CHUNKS="${WARMUP_CHUNKS:-24}"
DONOR_PROMPT_TOKENS="${DONOR_PROMPT_TOKENS:-4090}"
EVICTION_PROMPT_TOKENS="${EVICTION_PROMPT_TOKENS:-8192}"
EVICTION_ROUNDS="${EVICTION_ROUNDS:-8}"
MINIMUM_HOST_TOKENS="${MINIMUM_HOST_TOKENS:-512}"
MIN_LOAD_TOKENS="${MIN_LOAD_TOKENS:-}"
# This is deliberately opt-in. Profiler traces are diagnostic evidence and
# must never be pooled with normal timing results.
PROFILE_TORCH="${PROFILE_TORCH:-0}"
SERVER_PID=""

[[ -f "${PROFILE_PATH}" ]] || { echo "Backend runtime profile not found: ${PROFILE_PATH}" >&2; exit 2; }
[[ -n "${SGLANG_DOCKER_IMAGE}" ]] || { echo "Set SGLANG_DOCKER_IMAGE to the pinned backend image." >&2; exit 2; }
[[ -n "${AGENTIC_MODEL_CACHE}" && -d "${AGENTIC_MODEL_CACHE}" ]] || { echo "Set AGENTIC_MODEL_CACHE to a host model-cache directory." >&2; exit 2; }

profile_value() { python3 - "${PROFILE_PATH}" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], "")
print(" ".join(value) if isinstance(value, list) else value)
PY
}
cleanup_backend() { if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; fi; SERVER_PID=""; }
trap cleanup_backend EXIT

HARDWARE_PROFILE="${HARDWARE_PROFILE:-$(profile_value hardware_profile)}"
MODEL_CACHE_MOUNT="$(profile_value model_cache_mount)"
GPU_ARGS="$(profile_value gpu_runtime_args)"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/hardware/${REPORT_LABEL}"
RUNTIME_DIR="${RUN_ROOT}/runtime"
BACKEND_RUNTIME_CONTRACT_OUT="${RUNTIME_DIR}/backend_runtime.json"
mkdir -p "${RUNTIME_DIR}"
export SGLANG_DOCKER_EXTRA_ARGS="-v ${AGENTIC_MODEL_CACHE}:${MODEL_CACHE_MOUNT} -e HF_HOME=${MODEL_CACHE_MOUNT} ${SGLANG_DOCKER_EXTRA_ARGS:-}"
export SGLANG_DOCKER_GPU_ARGS="${SGLANG_DOCKER_GPU_ARGS:-${GPU_ARGS}}"
export SGLANG_DOCKER_IMAGE BACKEND_RUNTIME_PROFILE HARDWARE_PROFILE BACKEND_RUNTIME_CONTRACT_OUT

[[ ! -f "${DIRECT_ROOT}/.venv/bin/activate" ]] || source "${DIRECT_ROOT}/.venv/bin/activate"
PACKAGE_PYTHONPATH=""
for package_src in "${REPO_ROOT}"/packages/*/src; do [[ -d "${package_src}" ]] || continue; PACKAGE_PYTHONPATH="${PACKAGE_PYTHONPATH:+${PACKAGE_PYTHONPATH}:}${package_src}"; done
export PYTHONPATH="${PACKAGE_PYTHONPATH}:${PYTHONPATH:-}"

echo "Native KV reload attribution"
echo "  profile: ${BACKEND_RUNTIME_PROFILE}; image: ${SGLANG_DOCKER_IMAGE}; trials: ${TRIALS}"
"${SCRIPT_DIR}/probe_sglang_runtime.sh"
BACKEND_VERSION="$(python3 - "${BACKEND_RUNTIME_CONTRACT_OUT}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["backend_version"])
PY
)"

wait_for_server() {
  local trial_dir="$1" deadline=$((SECONDS + 240))
  until curl --silent --show-error --fail http://127.0.0.1:30000/v1/models >/dev/null; do
    [[ -z "${SERVER_PID}" || $(kill -0 "${SERVER_PID}" 2>/dev/null; echo $?) -eq 0 ]] || { echo "Backend exited; inspect ${trial_dir}/server.log" >&2; return 1; }
    (( SECONDS < deadline )) || { echo "Backend did not become ready; inspect ${trial_dir}/server.log" >&2; return 1; }
    sleep 2
  done
}

verify_trace() {
  local condition="$1" trace_path="$2"
  python3 - "${condition}" "${trace_path}" <<'PY'
import json, sys
condition, path = sys.argv[1:]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
summary = [row for row in rows if row.get("event") == "trace.install.summary"]
if not summary or summary[-1].get("missing_required_hooks"):
    raise SystemExit("Required trace hooks are not installed; refusing to trust this result.")
if condition == "host_reload_collision":
    if not any(str(row.get("event") or "").endswith("hostpool.load_to_device_per_layer.end") for row in rows):
        raise SystemExit("Host-reload condition captured no native host-to-device layer copy.")
PY
}

verify_profiler_output() {
  local trial_dir="$1"
  [[ "${PROFILE_TORCH}" == "1" ]] || return 0
  compgen -G "${trial_dir}/torch_cuda_profiles/torch_cuda_profile_*.json" >/dev/null || {
    echo "Torch CUDA profiler was enabled but exported no timeline: ${trial_dir}" >&2
    return 1
  }
}

run_condition() {
  local condition="$1"
  local condition_dir="${RUN_ROOT}/${condition}"
  rm -rf "${condition_dir}"; mkdir -p "${condition_dir}"
  local profiler_start_events="${AGENTIC_KV_TORCH_PROFILER_START_EVENTS:-scheduler.run_batch}"
  if [[ "${PROFILE_TORCH}" == "1" && "${condition}" == "host_reload_collision" && -z "${AGENTIC_KV_TORCH_PROFILER_START_EVENTS:-}" ]]; then
    profiler_start_events="hostpool.load_to_device_per_layer"
  fi
  for ((index=0; index<TRIALS; index++)); do
    local trial_dir="${condition_dir}/trial_$(printf '%03d' "${index}")"
    local trace_path="${trial_dir}/backend_trace.jsonl"
    mkdir -p "${trial_dir}"
    echo "Starting ${condition}, sample ${index}/${TRIALS}..."
    ( cd "${DIRECT_ROOT}"; export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${trace_path}" AGENTIC_KV_TRACE_SCHEDULER=1 AGENTIC_KV_TRACE_KV_POOL=1 AGENTIC_KV_COPY_TELEMETRY_ENABLE=1 AGENTIC_KV_COPY_TELEMETRY_PATH="${trial_dir}/kv_copy_telemetry.jsonl" AGENTIC_KV_PREPARE_CONTROL_ENABLE=1 AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1 AGENTIC_KV_PREPARE_CONTROL_PORT=31991 AGENTIC_KV_PREPARE_WAIT_TIMEOUT_MS=30000 HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}" AGENTIC_KV_TORCH_PROFILER_ENABLE="${PROFILE_TORCH}" AGENTIC_KV_TORCH_PROFILER_DIR="${trial_dir}/torch_cuda_profiles" AGENTIC_KV_TORCH_PROFILER_START_EVENTS="${profiler_start_events}" AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS="${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:-64}" AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY="${AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY:-0}"; bash scripts/run_sglang_hicache_server.sh "${MODEL}" ) >"${trial_dir}/server.log" 2>&1 &
    SERVER_PID="$!"; wait_for_server "${trial_dir}"
    append=(); (( index == 0 )) || append=(--append)
    min_load=(); [[ -z "${MIN_LOAD_TOKENS}" ]] || min_load=(--min-load-tokens "${MIN_LOAD_TOKENS}")
    python3 -m agentic_experiments.runners.run_device_memory_attribution --condition "${condition}" --run-id "${REPORT_LABEL}_${condition}" --sample-set-id "${SAMPLE_SET_ID}" --out-dir "${condition_dir}" --model "${MODEL}" --hardware-profile "${HARDWARE_PROFILE}" --backend-version "${BACKEND_VERSION}" --seed "${SEED}" --decode-tokens "${DECODE_TOKENS}" --warmup-chunks "${WARMUP_CHUNKS}" --donor-prompt-tokens "${DONOR_PROMPT_TOKENS}" --eviction-prompt-tokens "${EVICTION_PROMPT_TOKENS}" --eviction-rounds "${EVICTION_ROUNDS}" --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" --trials 1 --sample-offset "${index}" "${min_load[@]}" "${append[@]}"
    verify_trace "${condition}" "${trace_path}"; cleanup_backend; verify_profiler_output "${trial_dir}"
  done
}

run_condition target_only
run_condition device_resident_control
run_condition host_reload_collision
python3 -m agentic_reports.builders.build_device_memory_attribution_report --target-only "${RUN_ROOT}/target_only/probe_run.json" --device-resident "${RUN_ROOT}/device_resident_control/probe_run.json" --host-reload "${RUN_ROOT}/host_reload_collision/probe_run.json" --out "${RUN_ROOT}/device_memory_attribution_report.html" --summary-out "${RUN_ROOT}/device_memory_attribution_summary.json"
instrumentation=(--instrumentation "lightweight_backend_trace" --instrumentation "cuda_event_load_status" --instrumentation "streamed_decode_timing")
if [[ "${PROFILE_TORCH}" == "1" ]]; then
  instrumentation+=(--instrumentation "torch_cuda_profiler")
fi
python3 "${REPO_ROOT}/scripts/create_run_manifest.py" --out "${RUN_ROOT}/run_manifest.json" --run-id "${REPORT_LABEL}" --experiment "device_memory_attribution" --model "${MODEL}" --hardware-profile "${HARDWARE_PROFILE}" --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" --workload-json "{\"sample_set_id\":\"${SAMPLE_SET_ID}\",\"trials_per_condition\":${TRIALS},\"seed\":${SEED},\"frontend_priority\":\"none\",\"conditions\":[\"target_only\",\"device_resident_control\",\"host_reload_collision\"],\"decode_tokens\":${DECODE_TOKENS},\"warmup_chunks\":${WARMUP_CHUNKS},\"profiler\":${PROFILE_TORCH}}" "${instrumentation[@]}" --artifact "report=${RUN_ROOT}/device_memory_attribution_report.html" --artifact "summary=${RUN_ROOT}/device_memory_attribution_summary.json" --completion-status complete
echo "Complete: ${RUN_ROOT}/device_memory_attribution_report.html"
