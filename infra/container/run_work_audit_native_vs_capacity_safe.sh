#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
MODEL="${POLICY_COMPARE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
IMAGE="${SGLANG_DOCKER_IMAGE:-agentic-sglang-standard:0.5.10.post1}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-${HOME}/.cache/huggingface/hub}"
RUN_ID="${POLICY_COMPARE_RUN_ID:-native_vs_capacity_safe_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/work_audit/${RUN_ID}"
SEEDS="${POLICY_COMPARE_SEEDS:-1 2}"
PATTERNS="${POLICY_COMPARE_PATTERNS:-spread burst}"
PROFILE="${POLICY_COMPARE_TRACE_PROFILE:-kv_lifecycle_lean}"
GPU_TOKENS="${POLICY_COMPARE_GPU_TOKENS:-12288}"
HOST_GB="${POLICY_COMPARE_HOST_GB:-2}"
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
  if [[ -n "${SERVER_PID}" ]]; then wait "${SERVER_PID}" 2>/dev/null || true; fi
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
policy_values=(native_sglang capacity_safe)
pattern_index=0
for pattern in "${pattern_values[@]}"; do
  [[ "${pattern}" =~ ^(spread|burst)$ ]] || { echo "Invalid return pattern: ${pattern}" >&2; exit 2; }
  for seed in "${seed_values[@]}"; do
    start_offset=$(( (seed + pattern_index - 1) % 2 ))
    for policy_offset in 0 1; do
      policy="${policy_values[$(( (start_offset + policy_offset) % 2 ))]}"
      available_kb="$(df -Pk "${RUN_ROOT}" | awk 'NR==2 {print $4}')"
      (( available_kb >= 4194304 )) || { echo "Less than 4 GiB free; preserve evidence and stop" >&2; exit 2; }

      ARM_ROOT="${RUN_ROOT}/arms/seed${seed}/${pattern}_${policy}"
      mkdir -p "${ARM_ROOT}"
      CONTAINER_CID="${ARM_ROOT}/container.cid"
      export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:/model-cache -e HF_HOME=/model-cache --cidfile ${CONTAINER_CID}"
      export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${ARM_ROOT}/backend_trace.jsonl"
      export AGENTIC_KV_TRACE_INDEX_DETAIL=count
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
      export AGENTIC_KV_STORAGE_AUDIT_ENABLE=0
      export HICACHE_IO_BACKEND=direct HICACHE_MEM_LAYOUT=layer_first
      export HICACHE_SIZE_GB="${HOST_GB}"
      unset HICACHE_STORAGE_BACKEND HICACHE_STORAGE_PATH HICACHE_STORAGE_PREFETCH_POLICY
      unset HICACHE_STORAGE_BACKEND_EXTRA_CONFIG
      export MEM_FRACTION_STATIC="${POLICY_COMPARE_MEM_FRACTION:-0.80}" CUDA_GRAPH_FLAG="" OVERLAP_FLAG=""
      export EXTRA_SERVER_ARGS="--hicache-write-policy write_through --page-size 64 --max-total-tokens ${GPU_TOKENS}"

      echo "Starting seed=${seed} pattern=${pattern} policy=${policy} GPU_tokens=${GPU_TOKENS} host_GiB=${HOST_GB}"
      (cd "${DIRECT_ROOT}"; bash scripts/run_sglang_hicache_server.sh "${MODEL}") >"${ARM_ROOT}/server.log" 2>&1 &
      SERVER_PID="$!"
      deadline=$((SECONDS + 360))
      until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "Backend exited: ${ARM_ROOT}/server.log" >&2; exit 1; }
        (( SECONDS < deadline )) || { echo "Backend startup timed out" >&2; exit 1; }
        sleep 2
      done
      AGENTIC_KV_TRACE_ENABLE=0 python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
        --profile "${PROFILE}" --installation-only --trace "${ARM_ROOT}/backend_trace.jsonl" \
        --out "${ARM_ROOT}/installation_gate.json"

      python3 -m agentic_experiments.runners.run_work_audit_native_vs_capacity_safe \
        --run-id "${RUN_ID}" --out "${ARM_ROOT}/case_results.json" \
        --policy "${policy}" --pattern "${pattern}" --seed "${seed}" --model "${MODEL}" \
        --sessions "${POLICY_COMPARE_SESSIONS:-6}" --turns "${POLICY_COMPARE_TURNS:-10}" \
        --initial-tokens "${POLICY_COMPARE_INITIAL_TOKENS:-4096}" \
        --prime-tokens "${POLICY_COMPARE_PRIME_TOKENS:-2}" \
        --tool-words "${POLICY_COMPARE_TOOL_WORDS:-16}" \
        --decode-tokens "${POLICY_COMPARE_DECODE_TOKENS:-16}" \
        --wait-ms "${POLICY_COMPARE_WAIT_MS:-1000}" \
        --burst-window-ms "${POLICY_COMPARE_BURST_WINDOW_MS:-75}" \
        --spread-window-ms "${POLICY_COMPARE_SPREAD_WINDOW_MS:-1000}" \
        --max-active "${POLICY_COMPARE_MAX_ACTIVE:-2}" \
        --active-token-limit "${GPU_TOKENS}" --host-cache-gb "${HOST_GB}" \
        --trace-profile "${PROFILE}" --workload-namespace "paired-seed${seed}-${pattern}"
      stop_backend
      AGENTIC_KV_TRACE_ENABLE=0 python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
        --profile "${PROFILE}" --trace "${ARM_ROOT}/backend_trace.jsonl" \
        --out "${ARM_ROOT}/instrumentation_audit.json"
      gzip -f "${ARM_ROOT}/backend_trace.jsonl"
      sleep 2
    done
  done
  pattern_index=$((pattern_index + 1))
done

python3 -m agentic_experiments.runners.analyze_work_audit_native_vs_capacity_safe \
  --run-root "${RUN_ROOT}" --expected-seeds "${seed_values[@]}" \
  --expected-patterns "${pattern_values[@]}"

WORKLOAD_JSON="$(python3 - "${SEEDS}" "${PATTERNS}" "${MODEL}" <<'PY'
import json, os, sys
seeds, patterns, model = sys.argv[1:]
print(json.dumps({
    "research_question_id": "RQ25", "model": model,
    "seeds": list(map(int, seeds.split())), "return_patterns": patterns.split(),
    "policies": ["native_sglang", "capacity_safe"],
    "session_count": int(os.getenv("POLICY_COMPARE_SESSIONS", "6")),
    "turns_per_session": int(os.getenv("POLICY_COMPARE_TURNS", "10")),
    "initial_tokens": int(os.getenv("POLICY_COMPARE_INITIAL_TOKENS", "4096")),
    "prime_tokens": int(os.getenv("POLICY_COMPARE_PRIME_TOKENS", "2")),
    "tool_result_words": int(os.getenv("POLICY_COMPARE_TOOL_WORDS", "16")),
    "decode_tokens": int(os.getenv("POLICY_COMPARE_DECODE_TOKENS", "16")),
    "tool_wait_ms": int(os.getenv("POLICY_COMPARE_WAIT_MS", "1000")),
    "burst_window_ms": int(os.getenv("POLICY_COMPARE_BURST_WINDOW_MS", "75")),
    "spread_window_ms": int(os.getenv("POLICY_COMPARE_SPREAD_WINDOW_MS", "1000")),
    "gpu_token_limit": int(os.getenv("POLICY_COMPARE_GPU_TOKENS", "12288")),
    "host_cache_gb": float(os.getenv("POLICY_COMPARE_HOST_GB", "2")),
    "max_active_capacity_safe": int(os.getenv("POLICY_COMPARE_MAX_ACTIVE", "2")),
    "cuda_graph": True, "overlap_schedule": True, "frontend_priority": "equal",
    "prefetch": False, "fresh_backend_per_arm": True,
    "trace_profile": os.getenv("POLICY_COMPARE_TRACE_PROFILE", "kv_lifecycle_lean"),
}))
PY
)"
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment native_sglang_vs_capacity_safe --model "${MODEL}" \
  --hardware-profile nvidia_a10g_24gb --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${WORKLOAD_JSON}" --instrumentation "${PROFILE}" \
  --artifact "summary=${RUN_ROOT}/summary.json" --completion-status complete

sha256sum \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_native_vs_capacity_safe.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/analyze_work_audit_native_vs_capacity_safe.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_memory_tiers.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_capacity_safe_tiers.py" \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/sglang/trace/patch.py" \
  "${BASH_SOURCE[0]}" >"${RUN_ROOT}/source_sha256.txt"
echo "complete: ${RUN_ROOT}/summary.json"
