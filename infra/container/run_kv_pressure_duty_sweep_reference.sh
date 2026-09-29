#!/usr/bin/env bash
set -euo pipefail

# Equal-frontend-semantics KV-pressure sweep.  The backend is restarted for
# every sample/condition; only the verified native reload budget changes.

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE}.json}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
REPORT_LABEL="${REPORT_LABEL:-kv_pressure_duty_sweep_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_SET_ID="${SAMPLE_SET_ID:-${REPORT_LABEL}_samples}"
TRIALS="${TRIALS:-3}"
SEED="${SEED:-7}"
DECODE_TOKENS="${DECODE_TOKENS:-384}"
WARMUP_CHUNKS="${WARMUP_CHUNKS:-24}"
DONOR_COUNT="${DONOR_COUNT:-8}"
DONOR_PROMPT_TOKENS="${DONOR_PROMPT_TOKENS:-8192}"
LOW_LOAD_COUNT="${LOW_LOAD_COUNT:-20}"
MEDIUM_LOAD_COUNT="${MEDIUM_LOAD_COUNT:-60}"
HIGH_LOAD_COUNT="${HIGH_LOAD_COUNT:-100}"
LOW_MIN_CUDA_SHARE_PCT="${LOW_MIN_CUDA_SHARE_PCT:-1}"
MEDIUM_MIN_CUDA_SHARE_PCT="${MEDIUM_MIN_CUDA_SHARE_PCT:-4}"
HIGH_MIN_CUDA_SHARE_PCT="${HIGH_MIN_CUDA_SHARE_PCT:-8}"
EVICTION_PROMPT_TOKENS="${EVICTION_PROMPT_TOKENS:-8192}"
EVICTION_ROUNDS="${EVICTION_ROUNDS:-8}"
MINIMUM_HOST_TOKENS="${MINIMUM_HOST_TOKENS:-512}"
DEVICE_FREE_TOKENS="${DEVICE_FREE_TOKENS:-40000}"
RECYCLE_EVICT_TOKENS="${RECYCLE_EVICT_TOKENS:-8192}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
WORKLOAD_ID="${WORKLOAD_ID:-kv_pressure_duty_sweep_v1}"
SERVER_PID=""

if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "Backend runtime profile not found: ${PROFILE_PATH}" >&2
  exit 2
fi
if [[ -z "${SGLANG_DOCKER_IMAGE}" ]]; then
  echo "Set SGLANG_DOCKER_IMAGE to the pinned backend image." >&2
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
print(" ".join(value) if isinstance(value, list) else value)
PY
}

cleanup_backend() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_backend EXIT

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

PROFILE_ID="$(profile_value profile_id)"
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

echo "KV pressure duty sweep"
echo "  profile: ${PROFILE_ID}; loads: low=${LOW_LOAD_COUNT}, medium=${MEDIUM_LOAD_COUNT}, high=${HIGH_LOAD_COUNT}"
"${SCRIPT_DIR}/probe_sglang_runtime.sh"
BACKEND_VERSION="$(python3 - "${BACKEND_RUNTIME_CONTRACT_OUT}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["backend_version"])
PY
)"

cat >"${RUN_ROOT}/command.sh" <<EOF
MODEL=${MODEL@Q}
BACKEND_RUNTIME_PROFILE=${BACKEND_RUNTIME_PROFILE@Q}
SGLANG_DOCKER_IMAGE=${SGLANG_DOCKER_IMAGE@Q}
AGENTIC_MODEL_CACHE=${AGENTIC_MODEL_CACHE@Q}
REPORT_LABEL=${REPORT_LABEL@Q}
TRIALS=${TRIALS@Q}
DECODE_TOKENS=${DECODE_TOKENS@Q}
WARMUP_CHUNKS=${WARMUP_CHUNKS@Q}
DONOR_COUNT=${DONOR_COUNT@Q}
DONOR_PROMPT_TOKENS=${DONOR_PROMPT_TOKENS@Q}
LOW_LOAD_COUNT=${LOW_LOAD_COUNT@Q}
MEDIUM_LOAD_COUNT=${MEDIUM_LOAD_COUNT@Q}
HIGH_LOAD_COUNT=${HIGH_LOAD_COUNT@Q}
LOW_MIN_CUDA_SHARE_PCT=${LOW_MIN_CUDA_SHARE_PCT@Q}
MEDIUM_MIN_CUDA_SHARE_PCT=${MEDIUM_MIN_CUDA_SHARE_PCT@Q}
HIGH_MIN_CUDA_SHARE_PCT=${HIGH_MIN_CUDA_SHARE_PCT@Q}
DEVICE_FREE_TOKENS=${DEVICE_FREE_TOKENS@Q}
HICACHE_SIZE_GB=${HICACHE_SIZE_GB@Q}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC@Q}
export BACKEND_RUNTIME_PROFILE SGLANG_DOCKER_IMAGE AGENTIC_MODEL_CACHE REPORT_LABEL TRIALS DECODE_TOKENS WARMUP_CHUNKS DONOR_COUNT DONOR_PROMPT_TOKENS LOW_LOAD_COUNT MEDIUM_LOAD_COUNT HIGH_LOAD_COUNT LOW_MIN_CUDA_SHARE_PCT MEDIUM_MIN_CUDA_SHARE_PCT HIGH_MIN_CUDA_SHARE_PCT DEVICE_FREE_TOKENS HICACHE_SIZE_GB MEM_FRACTION_STATIC
bash infra/container/run_kv_pressure_duty_sweep_reference.sh "\$MODEL"
EOF
chmod +x "${RUN_ROOT}/command.sh"

wait_for_server() {
  local trial_dir="$1"
  local deadline=$((SECONDS + 240))
  until curl --silent --show-error --fail http://127.0.0.1:30000/v1/models >/dev/null; do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "SGLang exited during startup; inspect ${trial_dir}/server.log" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "SGLang server did not become ready; inspect ${trial_dir}/server.log" >&2
      return 1
    fi
    sleep 2
  done
}

verify_trace() {
  local trace_path="$1"
  local expected_loads="$2"
  python3 - "${trace_path}" "${expected_loads}" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
summaries = [row for row in rows if row.get("event") == "trace.install.summary"]
if not summaries:
    raise SystemExit("Trace hook installation summary is missing.")
missing = summaries[-1].get("missing_required_hooks") or []
for hook in ("HiCacheController.start_loading", "HiRadixCache.load_back", "MHATokenToKVPoolHost.load_to_device_per_layer"):
    if any(hook in item for item in missing):
        raise SystemExit(f"Required pressure hook is missing: {hook}")
finished = [row for row in rows if row.get("event") == "agentic_kv.prepare_prefix.load_status" and row.get("status") == "finished"]
if len(finished) != int(sys.argv[2]):
    raise SystemExit(f"Expected {sys.argv[2]} finished native loads, observed {len(finished)}")
PY
}

load_count_for() {
  case "$1" in
    decode_control) printf '0' ;;
    pressure_low) printf '%s' "${LOW_LOAD_COUNT}" ;;
    pressure_medium) printf '%s' "${MEDIUM_LOAD_COUNT}" ;;
    pressure_high) printf '%s' "${HIGH_LOAD_COUNT}" ;;
  esac
}

minimum_share_for() {
  case "$1" in
    decode_control) printf '0' ;;
    pressure_low) printf '%s' "${LOW_MIN_CUDA_SHARE_PCT}" ;;
    pressure_medium) printf '%s' "${MEDIUM_MIN_CUDA_SHARE_PCT}" ;;
    pressure_high) printf '%s' "${HIGH_MIN_CUDA_SHARE_PCT}" ;;
  esac
}

run_condition() {
  local condition="$1"
  local sample_index="$2"
  local count share condition_dir trial_dir trace_path append_arg=()
  count="$(load_count_for "${condition}")"
  share="$(minimum_share_for "${condition}")"
  condition_dir="${RUN_ROOT}/${condition}"
  trial_dir="${condition_dir}/trial_$(printf '%03d' "${sample_index}")"
  trace_path="${trial_dir}/backend_trace.jsonl"
  mkdir -p "${trial_dir}"
  : >"${trace_path}"
  (( sample_index > 0 )) && append_arg=(--append)
  echo "Starting ${condition}, sample $((sample_index + 1))/${TRIALS}, native reloads=${count}..."
  (
    cd "${DIRECT_ROOT}"
    export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${trace_path}"
    export AGENTIC_KV_TRACE_SCHEDULER=1 AGENTIC_KV_TRACE_KV_POOL=1
    export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1 AGENTIC_KV_COPY_TELEMETRY_PATH="${trial_dir}/kv_copy_telemetry.jsonl"
    export AGENTIC_KV_PREPARE_CONTROL_ENABLE=1 AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1 AGENTIC_KV_PREPARE_CONTROL_PORT=31991
    export AGENTIC_KV_PREPARE_WAIT_TIMEOUT_MS=30000 HICACHE_SIZE_GB MEM_FRACTION_STATIC
    bash scripts/run_sglang_hicache_server.sh "${MODEL}"
  ) >"${trial_dir}/server.log" 2>&1 &
  SERVER_PID="$!"
  wait_for_server "${trial_dir}"
  python3 -m agentic_experiments.runners.run_kv_pressure_duty_sweep \
    --condition "${condition}" --load-count "${count}" --minimum-cuda-load-share-pct "${share}" \
    --run-id "${REPORT_LABEL}_${condition}" --sample-set-id "${SAMPLE_SET_ID}" --out-dir "${condition_dir}" \
    --model "${MODEL}" --hardware-profile "${HARDWARE_PROFILE}" --backend-version "${BACKEND_VERSION}" \
    --workload-id "${WORKLOAD_ID}" --seed "${SEED}" --decode-tokens "${DECODE_TOKENS}" --warmup-chunks "${WARMUP_CHUNKS}" \
    --donor-count "${DONOR_COUNT}" --donor-prompt-tokens "${DONOR_PROMPT_TOKENS}" \
    --eviction-prompt-tokens "${EVICTION_PROMPT_TOKENS}" --eviction-rounds "${EVICTION_ROUNDS}" \
    --direct-device-evict-for-stage \
    --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" --device-free-tokens "${DEVICE_FREE_TOKENS}" \
    --recycle-evict-tokens "${RECYCLE_EVICT_TOKENS}" --trials 1 --sample-offset "${sample_index}" "${append_arg[@]}"
  verify_trace "${trace_path}" "${count}"
  cleanup_backend
}

for condition in decode_control pressure_low pressure_medium pressure_high; do
  mkdir -p "${RUN_ROOT}/${condition}"
done
orders=(
  "decode_control pressure_low pressure_medium pressure_high"
  "pressure_low pressure_medium pressure_high decode_control"
  "pressure_medium pressure_high decode_control pressure_low"
  "pressure_high decode_control pressure_low pressure_medium"
)
for ((sample = 0; sample < TRIALS; sample++)); do
  read -r -a order <<<"${orders[$((sample % ${#orders[@]}))]}"
  for condition in "${order[@]}"; do
    run_condition "${condition}" "${sample}"
  done
done

python3 -m agentic_reports.builders.build_kv_pressure_duty_sweep_report \
  --control "${RUN_ROOT}/decode_control/probe_run.json" --low "${RUN_ROOT}/pressure_low/probe_run.json" \
  --medium "${RUN_ROOT}/pressure_medium/probe_run.json" --high "${RUN_ROOT}/pressure_high/probe_run.json" \
  --out "${RUN_ROOT}/kv_pressure_duty_sweep_report.html" --summary-out "${RUN_ROOT}/kv_pressure_duty_sweep_summary.json"
python3 "${REPO_ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${REPORT_LABEL}" --experiment "kv_pressure_duty_sweep" \
  --model "${MODEL}" --hardware-profile "${HARDWARE_PROFILE}" --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "{\"workload_id\":\"${WORKLOAD_ID}\",\"sample_set_id\":\"${SAMPLE_SET_ID}\",\"trials_per_condition\":${TRIALS},\"frontend_priority\":\"none\",\"conditions\":[\"decode_control\",\"pressure_low\",\"pressure_medium\",\"pressure_high\"],\"load_counts\":{\"low\":${LOW_LOAD_COUNT},\"medium\":${MEDIUM_LOAD_COUNT},\"high\":${HIGH_LOAD_COUNT}}}" \
  --instrumentation "lightweight_backend_trace" --instrumentation "cuda_event_load_status" --instrumentation "streamed_decode_timing" \
  --artifact "command=${RUN_ROOT}/command.sh" --artifact "summary=${RUN_ROOT}/kv_pressure_duty_sweep_summary.json" \
  --artifact "report=${RUN_ROOT}/kv_pressure_duty_sweep_report.html" --completion-status complete
echo "Complete: ${RUN_ROOT}/kv_pressure_duty_sweep_report.html"
