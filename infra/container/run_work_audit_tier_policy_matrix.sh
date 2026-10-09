#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
MODEL="${TIER_POLICY_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
IMAGE="${SGLANG_DOCKER_IMAGE:-agentic-sglang-standard:0.5.10.post1}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-${HOME}/.cache/huggingface/hub}"
RUN_ID="${TIER_POLICY_RUN_ID:-tier_policy_matrix_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/work_audit/${RUN_ID}"
SEEDS="${TIER_POLICY_SEEDS:-1 2}"
PATTERNS="${TIER_POLICY_PATTERNS:-spread burst}"
MODES="resident host storage"
POLICIES="native_sglang capacity_safe"
PROFILE="${TIER_POLICY_TRACE_PROFILE:-kv_lifecycle_lean}"
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
read -r -a default_policy_values <<< "${POLICIES}"

for seed in "${seed_values[@]}"; do
  if (( seed % 2 == 0 )); then
    policy_values=(capacity_safe native_sglang)
  else
    policy_values=("${default_policy_values[@]}")
  fi
  for pattern in "${pattern_values[@]}"; do
    for (( mode_offset=0; mode_offset<${#mode_values[@]}; mode_offset++ )); do
      mode="${mode_values[$(( (mode_offset + seed - 1) % ${#mode_values[@]} ))]}"
      for policy in "${policy_values[@]}"; do
        available_kb="$(df -Pk "${RUN_ROOT}" | awk 'NR==2 {print $4}')"
        (( available_kb >= 4194304 )) || { echo "Less than 4 GiB free; preserve evidence and stop" >&2; exit 2; }

        ARM_ROOT="${RUN_ROOT}/arms/seed${seed}/${pattern}_${mode}_${policy}"
        mkdir -p "${ARM_ROOT}/storage"
        CONTAINER_CID="${ARM_ROOT}/container.cid"
        export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:/model-cache -e HF_HOME=/model-cache --cidfile ${CONTAINER_CID}"
        export AGENTIC_KV_TRACE_ENABLE=1 AGENTIC_KV_TRACE_PATH="${ARM_ROOT}/backend_trace.jsonl"
        export AGENTIC_KV_TRACE_INDEX_DETAIL="${TIER_POLICY_TRACE_INDEX_DETAIL:-count}"
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
        export MEM_FRACTION_STATIC="${TIER_POLICY_MEM_FRACTION:-0.80}" CUDA_GRAPH_FLAG="" OVERLAP_FLAG=""

        storage_flag=()
        if [[ "${mode}" == "resident" ]]; then
          gpu_tokens="${TIER_POLICY_RESIDENT_GPU_TOKENS:-40960}"
          host_gb="${TIER_POLICY_RESIDENT_HOST_GB:-3}"
          max_active="${TIER_POLICY_RESIDENT_MAX_ACTIVE:-6}"
          export AGENTIC_KV_STORAGE_AUDIT_ENABLE=0
          unset HICACHE_STORAGE_BACKEND HICACHE_STORAGE_PATH HICACHE_STORAGE_PREFETCH_POLICY
          unset HICACHE_STORAGE_BACKEND_EXTRA_CONFIG
        elif [[ "${mode}" == "host" ]]; then
          gpu_tokens="${TIER_POLICY_RESTRICTED_GPU_TOKENS:-12288}"
          host_gb="${TIER_POLICY_HOST_GB:-2}"
          max_active="${TIER_POLICY_RESTRICTED_MAX_ACTIVE:-2}"
          export AGENTIC_KV_STORAGE_AUDIT_ENABLE=0
          unset HICACHE_STORAGE_BACKEND HICACHE_STORAGE_PATH HICACHE_STORAGE_PREFETCH_POLICY
          unset HICACHE_STORAGE_BACKEND_EXTRA_CONFIG
        else
          gpu_tokens="${TIER_POLICY_RESTRICTED_GPU_TOKENS:-12288}"
          host_gb="${TIER_POLICY_STORAGE_HOST_GB:-1}"
          max_active="${TIER_POLICY_RESTRICTED_MAX_ACTIVE:-2}"
          export AGENTIC_KV_STORAGE_AUDIT_ENABLE=1
          export HICACHE_STORAGE_BACKEND=file HICACHE_STORAGE_PATH="${ARM_ROOT}/storage"
          export HICACHE_STORAGE_PREFETCH_POLICY=wait_complete
          export HICACHE_STORAGE_BACKEND_EXTRA_CONFIG='{"prefetch_threshold":64}'
          storage_flag=(--storage-enabled)
        fi
        export HICACHE_SIZE_GB="${host_gb}"
        export EXTRA_SERVER_ARGS="--hicache-write-policy write_through --page-size 64 --max-total-tokens ${gpu_tokens}"

        echo "Starting seed=${seed} pattern=${pattern} mode=${mode} policy=${policy} GPU_tokens=${gpu_tokens} host_GiB=${host_gb}"
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
          --policy "${policy}" --mode "${mode}" --pattern "${pattern}" --seed "${seed}" \
          --model "${MODEL}" --sessions "${TIER_POLICY_SESSIONS:-6}" \
          --turns "${TIER_POLICY_TURNS:-10}" \
          --initial-tokens "${TIER_POLICY_INITIAL_TOKENS:-4096}" \
          --prime-tokens "${TIER_POLICY_PRIME_TOKENS:-2}" \
          --tool-words "${TIER_POLICY_TOOL_WORDS:-16}" \
          --decode-tokens "${TIER_POLICY_DECODE_TOKENS:-16}" \
          --wait-ms "${TIER_POLICY_WAIT_MS:-1000}" \
          --burst-window-ms "${TIER_POLICY_BURST_WINDOW_MS:-75}" \
          --spread-window-ms "${TIER_POLICY_SPREAD_WINDOW_MS:-1000}" \
          --max-active "${max_active}" --active-token-limit "${gpu_tokens}" \
          --host-cache-gb "${host_gb}" --trace-profile "${PROFILE}" \
          --workload-namespace "tier-policy-seed${seed}-${pattern}" "${storage_flag[@]}"
        stop_backend
        live_flags=()
        [[ "${mode}" == "resident" ]] && live_flags=(--installation-only)
        AGENTIC_KV_TRACE_ENABLE=0 python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
          --profile "${PROFILE}" --trace "${ARM_ROOT}/backend_trace.jsonl" \
          --out "${ARM_ROOT}/instrumentation_audit.json" "${live_flags[@]}"
        gzip -f "${ARM_ROOT}/backend_trace.jsonl"
        find "${ARM_ROOT}/storage" -type f -delete
        sleep 2
      done
    done
  done
done

python3 -m agentic_experiments.runners.analyze_work_audit_tier_policy_matrix \
  --run-root "${RUN_ROOT}" --expected-seeds "${seed_values[@]}" \
  --expected-patterns "${pattern_values[@]}"

WORKLOAD_JSON="$(python3 - "${SEEDS}" "${PATTERNS}" "${MODEL}" <<'PY'
import json, os, sys
seeds, patterns, model = sys.argv[1:]
value = {
    "research_question_id": "RQ26", "model": model,
    "seeds": list(map(int, seeds.split())), "return_patterns": patterns.split(),
    "modes": ["resident", "host", "storage"],
    "policies": ["native_sglang", "capacity_safe"],
    "session_count": int(os.getenv("TIER_POLICY_SESSIONS", "6")),
    "turns_per_session": int(os.getenv("TIER_POLICY_TURNS", "10")),
    "initial_tokens": int(os.getenv("TIER_POLICY_INITIAL_TOKENS", "4096")),
    "prime_tokens": int(os.getenv("TIER_POLICY_PRIME_TOKENS", "2")),
    "tool_result_words": int(os.getenv("TIER_POLICY_TOOL_WORDS", "16")),
    "decode_tokens": int(os.getenv("TIER_POLICY_DECODE_TOKENS", "16")),
    "tool_wait_ms": int(os.getenv("TIER_POLICY_WAIT_MS", "1000")),
    "burst_window_ms": int(os.getenv("TIER_POLICY_BURST_WINDOW_MS", "75")),
    "spread_window_ms": int(os.getenv("TIER_POLICY_SPREAD_WINDOW_MS", "1000")),
    "resident_gpu_tokens": int(os.getenv("TIER_POLICY_RESIDENT_GPU_TOKENS", "40960")),
    "resident_host_cache_gb": float(os.getenv("TIER_POLICY_RESIDENT_HOST_GB", "3")),
    "resident_max_active": int(os.getenv("TIER_POLICY_RESIDENT_MAX_ACTIVE", "6")),
    "restricted_gpu_tokens": int(os.getenv("TIER_POLICY_RESTRICTED_GPU_TOKENS", "12288")),
    "host_cache_gb": float(os.getenv("TIER_POLICY_HOST_GB", "2")),
    "storage_host_cache_gb": float(os.getenv("TIER_POLICY_STORAGE_HOST_GB", "1")),
    "restricted_max_active": int(os.getenv("TIER_POLICY_RESTRICTED_MAX_ACTIVE", "2")),
    "pairwise_capacity_contract": (
        "Within each tier, native SGLang and capacity-safe use identical GPU, host, and storage capacities."
    ),
    "same_workload_contract": True,
    "measurement_boundary": "after_initial_prefix_population",
    "cuda_graph": True, "overlap_schedule": True, "frontend_priority": "equal",
    "prefetch_before_tool_return": False, "storage_page_cache_flushed": False,
    "fresh_backend_per_arm": True,
    "trace_profile": os.getenv("TIER_POLICY_TRACE_PROFILE", "kv_lifecycle_lean"),
}
print(json.dumps(value))
PY
)"
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment agentic_work_audit_tier_policy_matrix --model "${MODEL}" \
  --hardware-profile nvidia_a10g_24gb --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${WORKLOAD_JSON}" --instrumentation "${PROFILE}" \
  --artifact "summary=${RUN_ROOT}/summary.json" --completion-status complete

sha256sum \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_native_vs_capacity_safe.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/analyze_work_audit_tier_policy_matrix.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_memory_tiers.py" \
  "${ROOT}/packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_capacity_safe_tiers.py" \
  "${ROOT}/packages/agentic-backend-sglang/src/agentic_backends/sglang/trace/patch.py" \
  "${BASH_SOURCE[0]}" >"${RUN_ROOT}/source_sha256.txt"
echo "complete: ${RUN_ROOT}/summary.json"
