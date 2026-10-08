#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
RUN_ID="${SWAP_RUN_ID:-coordinated_swap_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/work_audit/${RUN_ID}"
MODEL="${SWAP_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PROFILE="${SWAP_TRACE_PROFILE:-kv_lifecycle_counts}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-${HOME}/.cache/huggingface/hub}"
SERVER_PID=""
CONTAINER_CID=""
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -d "${MODEL_CACHE}" ]] || { echo "Missing model cache" >&2; exit 2; }
if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
  echo "Backend already active; refusing to disturb it" >&2; exit 2
fi
stop_backend() {
  if [[ -n "${CONTAINER_CID}" && -f "${CONTAINER_CID}" ]]; then
    docker stop -t 15 "$(<"${CONTAINER_CID}")" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SERVER_PID}" ]]; then
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_backend EXIT
mkdir -p "${RUN_ROOT}/arms" "${RUN_ROOT}/runtime"
source "${DIRECT_ROOT}/.venv/bin/activate"
for source_dir in "${ROOT}"/packages/*/src; do
  export PYTHONPATH="${source_dir}:${PYTHONPATH:-}"
done
export BACKEND_RUNTIME_PROFILE=nvidia_standard
export BACKEND_RUNTIME_CONTRACT_OUT="${RUN_ROOT}/runtime/backend_runtime.json"
export SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-agentic-sglang-standard:0.5.10.post1}"
"${ROOT}/infra/container/probe_sglang_runtime.sh"
python3 "${ROOT}/scripts/create_run_manifest.py" --out "${RUN_ROOT}/run_manifest.json" \
  --run-id "${RUN_ID}" --experiment coordinated_cpu_gpu_swap --model "${MODEL}" \
  --hardware-profile nvidia_a10g_24gb --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json '{"research_question_id":"RQ21","frontend_priority":"equal","storage":false,"cuda_graph":true,"overlap_schedule":true}' \
  --instrumentation "${PROFILE}" --completion-status created
sha256sum "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/"*coordinated_swap.py \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/coordinated_audit.py" \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/sglang/versions/v0510_coordinated_kv.py" \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/sglang/trace/patch.py" \
  "${ROOT}/packages/agentic-controller/src/agentic_controller/coordinated_swap.py" \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/sglang/instrumentation_profiles.py" \
  "${BASH_SOURCE[0]}" >"${RUN_ROOT}/source_sha256.txt"
tar -czf "${RUN_ROOT}/source_bundle.tar.gz" -C "${ROOT}" \
  packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_coordinated_swap.py \
  packages/agentic-experiments/src/agentic_experiments/runners/analyze_work_audit_coordinated_swap.py \
  packages/agentic-backend-sglang/src/agentic_backends/coordinated_audit.py \
  packages/agentic-backend-sglang/src/agentic_backends/sglang/versions/v0510_coordinated_kv.py \
  packages/agentic-backend-sglang/src/agentic_backends/sglang/trace/patch.py \
  packages/agentic-controller/src/agentic_controller/coordinated_swap.py \
  packages/agentic-backend-sglang/src/agentic_backends/sglang/instrumentation_profiles.py \
  packages/agentic-instrumentation/src/agentic_instrumentation/catalog.py \
  infra/container/run_work_audit_coordinated_swap.sh \
  sglang_direct_kv/scripts/run_sglang_hicache_server.sh
read -r -a trials <<< "${SWAP_TRIALS:-1 2 3}"
read -r -a modes <<< "${SWAP_MODES:-independent coordinated resident}"
for trial in "${trials[@]}"; do
  for (( offset=0; offset<${#modes[@]}; offset++ )); do
    mode="${modes[$(( (offset + trial - 1) % ${#modes[@]} ))]}"
    available_kb="$(df -Pk "${RUN_ROOT}" | awk 'NR==2 {print $4}')"
    (( available_kb >= 2097152 )) || { echo "Less than 2 GiB free; preserve evidence and stop"; exit 2; }
    [[ "${mode}" =~ ^(independent|coordinated|resident)$ ]] || exit 2
    ARM_ROOT="${RUN_ROOT}/arms/trial${trial}_${mode}"
    mkdir -p "${ARM_ROOT}"
    CONTAINER_CID="${ARM_ROOT}/container.cid"
    export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:/model-cache -e HF_HOME=/model-cache --cidfile ${CONTAINER_CID}"
    export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${ARM_ROOT}/backend_trace.jsonl"
    export AGENTIC_KV_TRACE_INDEX_DETAIL=full
    profile_values="$(python3 -m agentic_backends.sglang.instrumentation_profiles "${PROFILE}" --shell)"
    while IFS='=' read -r name value; do
      [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
      printf -v "${name}" '%s' "${value}"
      export "${name}"
    done <<< "${profile_values}"
    export AGENTIC_KV_TRACE_CONTROL_ONLY=1 AGENTIC_KV_PREPARE_CONTROL_ENABLE=1
    export AGENTIC_KV_COORDINATED_AUDIT_ENABLE=1 AGENTIC_KV_STORAGE_AUDIT_ENABLE=0
    export AGENTIC_KV_PREPARE_LOAD_WORKER=0 AGENTIC_KV_ASYNC_VERIFY_FULL_COPY=0
    export AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1 AGENTIC_KV_PREPARE_CONTROL_PORT=31991
    export HICACHE_SIZE_GB=8 HICACHE_IO_BACKEND="${SWAP_IO_BACKEND:-direct}" HICACHE_MEM_LAYOUT=layer_first
    unset HICACHE_STORAGE_BACKEND HICACHE_STORAGE_PATH
    export MEM_FRACTION_STATIC=0.75 CUDA_GRAPH_FLAG="" OVERLAP_FLAG=""
    cap="${SWAP_GPU_TOKENS:-110592}"
    [[ "${mode}" == resident ]] && cap="${SWAP_RESIDENT_TOKENS:-262144}"
    export EXTRA_SERVER_ARGS="--hicache-write-policy write_through --page-size 64 --max-total-tokens ${cap}"
    echo "Starting ${mode}, trial ${trial}, cap ${cap}, turns ${SWAP_TURNS:-40}"
    (cd "${DIRECT_ROOT}"; bash scripts/run_sglang_hicache_server.sh "${MODEL}") >"${ARM_ROOT}/server.log" 2>&1 &
    SERVER_PID="$!"
    deadline=$((SECONDS + 300))
    until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
      kill -0 "${SERVER_PID}" 2>/dev/null || { echo "Backend exited: ${ARM_ROOT}/server.log"; exit 1; }
      (( SECONDS < deadline )) || { echo "Startup timed out"; exit 1; }
      sleep 2
    done
    python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
      --profile "${PROFILE}" --installation-only --trace "${ARM_ROOT}/backend_trace.jsonl" \
      --out "${ARM_ROOT}/installation_gate.json"
    python3 -m agentic_experiments.runners.run_work_audit_coordinated_swap \
      --out-dir "${ARM_ROOT}" --mode "${mode}" --trial "${trial}" --model "${MODEL}" \
      --turns "${SWAP_TURNS:-40}" --decode-tokens "${SWAP_DECODE_TOKENS:-32}" \
      --initial-tokens "${SWAP_INITIAL_TOKENS:-8192}" --tool-words "${SWAP_TOOL_WORDS:-16}" \
      --io-backend "${SWAP_IO_BACKEND:-direct}" \
      --restore-style "${SWAP_RESTORE_STYLE:-group}" \
      --control-style "${SWAP_CONTROL_STYLE:-group}" \
      --trace-profile "${PROFILE}" \
      --resident-gpu-tokens "${SWAP_RESIDENT_TOKENS:-262144}" \
      --restricted-gpu-tokens "${SWAP_GPU_TOKENS:-110592}"
    stop_backend
    live_flags=()
    [[ "${mode}" == resident ]] && live_flags=(--installation-only)
    python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
      --profile "${PROFILE}" --trace "${ARM_ROOT}/backend_trace.jsonl" \
      --out "${ARM_ROOT}/instrumentation_audit.json" "${live_flags[@]}"
    gzip "${ARM_ROOT}/backend_trace.jsonl"
  done
done
python3 -m agentic_experiments.runners.analyze_work_audit_coordinated_swap \
  --run-root "${RUN_ROOT}" --expected-trials "${trials[@]}" --expected-modes "${modes[@]}"
echo "Saved ${RUN_ROOT}/summary.json"
