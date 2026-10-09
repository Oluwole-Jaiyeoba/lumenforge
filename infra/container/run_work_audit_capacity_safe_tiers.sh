#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
MODEL="${CAPACITY_TIER_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
IMAGE="${SGLANG_DOCKER_IMAGE:-agentic-sglang-standard:0.5.10.post1}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-${HOME}/.cache/huggingface/hub}"
RUN_ID="${CAPACITY_TIER_RUN_ID:-capacity_safe_tiers_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/work_audit/${RUN_ID}"
SEEDS="${CAPACITY_TIER_SEEDS:-1 2 3}"
PATTERNS="${CAPACITY_TIER_PATTERNS:-spread burst}"
MODES="${CAPACITY_TIER_MODES:-resident host storage}"
PROFILE="${CAPACITY_TIER_TRACE_PROFILE:-kv_lifecycle_lean}"
SERVER_PID=""
CONTAINER_CID=""

[[ -d "${MODEL_CACHE}" ]] || { echo "Model cache missing: ${MODEL_CACHE}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run already exists: ${RUN_ROOT}" >&2; exit 2; }
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
export SGLANG_DOCKER_IMAGE="${IMAGE}"
"${ROOT}/infra/container/probe_sglang_runtime.sh"

read -r -a seed_values <<< "${SEEDS}"
read -r -a pattern_values <<< "${PATTERNS}"
read -r -a mode_values <<< "${MODES}"
for seed in "${seed_values[@]}"; do
  for pattern in "${pattern_values[@]}"; do
    for (( offset=0; offset<${#mode_values[@]}; offset++ )); do
      mode="${mode_values[$(( (offset + seed - 1) % ${#mode_values[@]} ))]}"
      [[ "${mode}" =~ ^(resident|host|storage)$ ]] || { echo "Invalid tier mode: ${mode}" >&2; exit 2; }
      [[ "${pattern}" =~ ^(spread|burst)$ ]] || { echo "Invalid return pattern: ${pattern}" >&2; exit 2; }
      available_kb="$(df -Pk "${RUN_ROOT}" | awk 'NR==2 {print $4}')"
      (( available_kb >= 4194304 )) || { echo "Less than 4 GiB free; preserve evidence and stop" >&2; exit 2; }

      ARM_ROOT="${RUN_ROOT}/arms/seed${seed}/${pattern}_${mode}"
      mkdir -p "${ARM_ROOT}/storage"
      CONTAINER_CID="${ARM_ROOT}/container.cid"
      export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:/model-cache -e HF_HOME=/model-cache --cidfile ${CONTAINER_CID}"
      export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${ARM_ROOT}/backend_trace.jsonl"
      export AGENTIC_KV_TRACE_INDEX_DETAIL="${CAPACITY_TIER_TRACE_INDEX_DETAIL:-count}"
      profile_values="$(python3 -m agentic_backends.sglang.instrumentation_profiles "${PROFILE}" --shell)"
      while IFS='=' read -r name value; do
        [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
        printf -v "${name}" '%s' "${value}"
        export "${name}"
      done <<< "${profile_values}"
      export AGENTIC_KV_TRACE_CONTROL_ONLY=1 AGENTIC_KV_PREPARE_CONTROL_ENABLE=1
      export AGENTIC_KV_COORDINATED_AUDIT_ENABLE=1
      export AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1 AGENTIC_KV_PREPARE_CONTROL_PORT=31991
      export AGENTIC_KV_PREPARE_LOAD_WORKER=0 AGENTIC_KV_ASYNC_VERIFY_FULL_COPY=0
      export HICACHE_IO_BACKEND=direct HICACHE_MEM_LAYOUT=layer_first
      export MEM_FRACTION_STATIC="${CAPACITY_TIER_MEM_FRACTION:-0.80}" CUDA_GRAPH_FLAG="" OVERLAP_FLAG=""

      if [[ "${mode}" == "resident" ]]; then
        cap="${CAPACITY_TIER_RESIDENT_GPU_TOKENS:-40960}"
        export HICACHE_SIZE_GB="${CAPACITY_TIER_RESIDENT_HOST_GB:-3}"
        export AGENTIC_KV_STORAGE_AUDIT_ENABLE=0
        unset HICACHE_STORAGE_BACKEND HICACHE_STORAGE_PATH HICACHE_STORAGE_PREFETCH_POLICY
        unset HICACHE_STORAGE_BACKEND_EXTRA_CONFIG
      elif [[ "${mode}" == "host" ]]; then
        cap="${CAPACITY_TIER_RESTRICTED_GPU_TOKENS:-12288}"
        export HICACHE_SIZE_GB="${CAPACITY_TIER_HOST_GB:-2}"
        export AGENTIC_KV_STORAGE_AUDIT_ENABLE=0
        unset HICACHE_STORAGE_BACKEND HICACHE_STORAGE_PATH HICACHE_STORAGE_PREFETCH_POLICY
        unset HICACHE_STORAGE_BACKEND_EXTRA_CONFIG
      else
        cap="${CAPACITY_TIER_RESTRICTED_GPU_TOKENS:-12288}"
        # This deliberately tight cache forces storage. Before staging, the
        # runner evicts an inactive, already-storage-backed host copy if needed.
        export HICACHE_SIZE_GB="${CAPACITY_TIER_STORAGE_HOST_GB:-1}"
        export AGENTIC_KV_STORAGE_AUDIT_ENABLE=1
        export HICACHE_STORAGE_BACKEND=file HICACHE_STORAGE_PATH="${ARM_ROOT}/storage"
        export HICACHE_STORAGE_PREFETCH_POLICY=wait_complete
        export HICACHE_STORAGE_BACKEND_EXTRA_CONFIG='{"prefetch_threshold":64}'
      fi
      export EXTRA_SERVER_ARGS="--hicache-write-policy write_through --page-size 64 --max-total-tokens ${cap}"

      echo "Starting seed=${seed} pattern=${pattern} mode=${mode} GPU_tokens=${cap} host_GiB=${HICACHE_SIZE_GB}"
      (cd "${DIRECT_ROOT}"; bash scripts/run_sglang_hicache_server.sh "${MODEL}") >"${ARM_ROOT}/server.log" 2>&1 &
      SERVER_PID="$!"
      deadline=$((SECONDS + 360))
      until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "Backend exited: ${ARM_ROOT}/server.log" >&2; exit 1; }
        (( SECONDS < deadline )) || { echo "Backend startup timed out" >&2; exit 1; }
        sleep 2
      done
      python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
        --profile "${PROFILE}" --installation-only --trace "${ARM_ROOT}/backend_trace.jsonl" \
        --out "${ARM_ROOT}/installation_gate.json"

      python3 -m agentic_experiments.runners.run_work_audit_capacity_safe_tiers \
        --run-id "${RUN_ID}" --out "${ARM_ROOT}/case_results.json" \
        --mode "${mode}" --pattern "${pattern}" --seed "${seed}" --model "${MODEL}" \
        --sessions "${CAPACITY_TIER_SESSIONS:-6}" --turns "${CAPACITY_TIER_TURNS:-10}" \
        --initial-tokens "${CAPACITY_TIER_INITIAL_TOKENS:-4096}" \
        --tool-words "${CAPACITY_TIER_TOOL_WORDS:-16}" \
        --decode-tokens "${CAPACITY_TIER_DECODE_TOKENS:-16}" \
        --wait-ms "${CAPACITY_TIER_WAIT_MS:-1000}" \
        --burst-window-ms "${CAPACITY_TIER_BURST_WINDOW_MS:-75}" \
        --spread-window-ms "${CAPACITY_TIER_SPREAD_WINDOW_MS:-1000}" \
        --max-active "${CAPACITY_TIER_MAX_ACTIVE:-2}" \
        --active-token-limit "${CAPACITY_TIER_ACTIVE_TOKEN_LIMIT:-12288}"
      stop_backend
      live_flags=()
      [[ "${mode}" == "resident" ]] && live_flags=(--installation-only)
      python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
        --profile "${PROFILE}" --trace "${ARM_ROOT}/backend_trace.jsonl" \
        --out "${ARM_ROOT}/instrumentation_audit.json" "${live_flags[@]}"
      gzip -f "${ARM_ROOT}/backend_trace.jsonl"
      find "${ARM_ROOT}/storage" -type f -delete
      sleep 2
    done
  done
done

python3 -m agentic_experiments.runners.analyze_work_audit_capacity_safe_tiers \
  --run-root "${RUN_ROOT}" --expected-seeds "${seed_values[@]}" \
  --expected-patterns "${pattern_values[@]}" --expected-modes "${mode_values[@]}"

WORKLOAD_JSON="$(python3 - "${SEEDS}" "${PATTERNS}" "${MODES}" "${MODEL}" <<'PY'
import json, os, sys
seeds, patterns, modes, model = sys.argv[1:]
value = {
    "research_question_id": "RQ24", "model": model,
    "seeds": list(map(int, seeds.split())), "return_patterns": patterns.split(), "modes": modes.split(),
    "session_count": int(os.getenv("CAPACITY_TIER_SESSIONS", "6")),
    "turns_per_session": int(os.getenv("CAPACITY_TIER_TURNS", "10")),
    "initial_tokens": int(os.getenv("CAPACITY_TIER_INITIAL_TOKENS", "4096")),
    "tool_result_words": int(os.getenv("CAPACITY_TIER_TOOL_WORDS", "16")),
    "decode_tokens": int(os.getenv("CAPACITY_TIER_DECODE_TOKENS", "16")),
    "tool_wait_ms": int(os.getenv("CAPACITY_TIER_WAIT_MS", "1000")),
    "burst_window_ms": int(os.getenv("CAPACITY_TIER_BURST_WINDOW_MS", "75")),
    "spread_window_ms": int(os.getenv("CAPACITY_TIER_SPREAD_WINDOW_MS", "1000")),
    "max_active": int(os.getenv("CAPACITY_TIER_MAX_ACTIVE", "2")),
    "active_token_limit": int(os.getenv("CAPACITY_TIER_ACTIVE_TOKEN_LIMIT", "12288")),
    "resident_gpu_tokens": int(os.getenv("CAPACITY_TIER_RESIDENT_GPU_TOKENS", "40960")),
    "resident_host_cache_gb": float(os.getenv("CAPACITY_TIER_RESIDENT_HOST_GB", "3")),
    "restricted_gpu_tokens": int(os.getenv("CAPACITY_TIER_RESTRICTED_GPU_TOKENS", "12288")),
    "host_cache_gb": float(os.getenv("CAPACITY_TIER_HOST_GB", "2")),
    "storage_host_cache_gb": float(os.getenv("CAPACITY_TIER_STORAGE_HOST_GB", "1")),
    "cuda_graph": True, "overlap_schedule": True, "frontend_priority": "equal",
    "prefetch": False, "admission_after_complete_restore": True,
    "storage_page_cache_flushed": False, "fresh_backend_per_arm": True,
    "trace_profile": os.getenv("CAPACITY_TIER_TRACE_PROFILE", "kv_lifecycle_lean"),
}
print(json.dumps(value))
PY
)"
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment agentic_work_audit_capacity_safe_tiers --model "${MODEL}" \
  --hardware-profile nvidia_a10g_24gb --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${WORKLOAD_JSON}" --instrumentation "${PROFILE}" \
  --artifact "summary=${RUN_ROOT}/summary.json" --completion-status complete

sha256sum \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_capacity_safe_tiers.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/analyze_work_audit_capacity_safe_tiers.py" \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/sglang/trace/patch.py" \
  "${BASH_SOURCE[0]}" >"${RUN_ROOT}/source_sha256.txt"
echo "complete: ${RUN_ROOT}/summary.json"
