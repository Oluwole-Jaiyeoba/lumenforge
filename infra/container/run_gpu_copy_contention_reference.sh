#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
REPORT_LABEL="${REPORT_LABEL:-gpu_copy_contention_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/hardware/${REPORT_LABEL}"
PROFILE_TORCH="${PROFILE_TORCH:-0}"
MODES="${MODES:-idle,copy,copy,idle}"
BACKEND_NAME="agentic-copy-backend-${REPORT_LABEL}"
WORKER_NAME="agentic-copy-worker-${REPORT_LABEL}"

[[ -n "${SGLANG_DOCKER_IMAGE}" ]] || { echo "Set SGLANG_DOCKER_IMAGE." >&2; exit 2; }
[[ -n "${AGENTIC_MODEL_CACHE}" && -d "${AGENTIC_MODEL_CACHE}" ]] || { echo "Set AGENTIC_MODEL_CACHE to the model cache directory." >&2; exit 2; }
[[ "${PROFILE_TORCH}" == "0" || "${PROFILE_TORCH}" == "1" ]] || { echo "PROFILE_TORCH must be 0 or 1." >&2; exit 2; }
if [[ "${PROFILE_TORCH}" == "1" && "${MODES}" == *,* ]]; then
  echo "Profile one mode per run so the worker trace is not overwritten." >&2
  exit 2
fi
if [[ "${PROFILE_TORCH}" == "1" ]]; then
  DECODE_TIMEOUT_S="${DECODE_TIMEOUT_S:-600}"
else
  DECODE_TIMEOUT_S="${DECODE_TIMEOUT_S:-180}"
fi
if curl --silent --fail http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
  echo "Port 30000 is already serving a model; refusing to disturb it." >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

cleanup() {
  docker stop "${WORKER_NAME}" "${BACKEND_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

export SGLANG_DOCKER_EXTRA_ARGS="-v ${AGENTIC_MODEL_CACHE}:/tmp/hfcache -e HF_HOME=/tmp/hfcache --name ${BACKEND_NAME}"
export SGLANG_DOCKER_IMAGE AGENTIC_KV_TRACE_ENABLE="${PROFILE_TORCH}"
if [[ "${PROFILE_TORCH}" == "1" ]]; then
  export AGENTIC_KV_TRACE_PATH="${RUN_ROOT}/backend_trace.jsonl"
  export AGENTIC_KV_TRACE_SCHEDULER=1
  export AGENTIC_KV_TORCH_PROFILER_ENABLE=1
  export AGENTIC_KV_TORCH_PROFILER_DIR="${RUN_ROOT}/backend_profiles"
  export AGENTIC_KV_TORCH_PROFILER_START_EVENTS=scheduler.run_batch
  export AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=1024
  export AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY=0
fi
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
(
  cd "${DIRECT_ROOT}"
  bash scripts/run_sglang_hicache_server.sh "${MODEL}"
) >"${RUN_ROOT}/backend.log" 2>&1 &
BACKEND_PID="$!"

deadline=$((SECONDS + 240))
until curl --silent --fail http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "Backend exited; see ${RUN_ROOT}/backend.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Backend readiness timed out; see ${RUN_ROOT}/backend.log" >&2
    exit 1
  fi
  sleep 2
done

worker_args=(--buffer-mib "${COPY_BUFFER_MIB:-128}")
if [[ "${PROFILE_TORCH}" == "1" ]]; then
  worker_args+=(--profile-path "${RUN_ROOT}/copy_worker_profile.json")
fi
docker run --rm --gpus all --network host --ipc host \
  --name "${WORKER_NAME}" \
  -v "${REPO_ROOT}/packages:${REPO_ROOT}/packages:ro" \
  -v "${RUN_ROOT}:${RUN_ROOT}" \
  -e PYTHONPATH="${REPO_ROOT}/packages/agentic-hardware-probes/src" \
  "${SGLANG_DOCKER_IMAGE}" \
  python3 -m agentic_hardware_probes.copy_pressure_worker \
  "${worker_args[@]}" >"${RUN_ROOT}/copy_worker.log" 2>&1 &
WORKER_PID="$!"

deadline=$((SECONDS + 90))
until curl --silent --fail http://127.0.0.1:31992/status >/dev/null 2>&1; do
  if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    echo "Copy worker exited; see ${RUN_ROOT}/copy_worker.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Copy worker readiness timed out; see ${RUN_ROOT}/copy_worker.log" >&2
    exit 1
  fi
  sleep 1
done

if [[ -f "${DIRECT_ROOT}/.venv/bin/activate" ]]; then
  source "${DIRECT_ROOT}/.venv/bin/activate"
fi
export PYTHONPATH="${REPO_ROOT}/packages/agentic-experiments/src:${REPO_ROOT}/packages/agentic-hardware-probes/src:${PYTHONPATH:-}"
python3 -m agentic_experiments.runners.run_gpu_copy_contention \
  --model "${MODEL}" --modes "${MODES}" \
  --decode-tokens "${DECODE_TOKENS:-320}" \
  --warmup-chunks "${WARMUP_CHUNKS:-16}" \
  --window-s "${COPY_WINDOW_S:-5}" \
  --timeout-s "${DECODE_TIMEOUT_S}" \
  --out "${RUN_ROOT}/results.json"

echo "Complete: ${RUN_ROOT}/results.json"
