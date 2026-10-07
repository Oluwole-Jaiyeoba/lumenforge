#!/usr/bin/env bash
set -euo pipefail

# Equal-priority, repeated tool-return audit. Capacity pressure is natural;
# this launcher never invokes the cache eviction control endpoints.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
MODEL="${WORK_AUDIT_CYCLES_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
IMAGE="${SGLANG_DOCKER_IMAGE:-agentic-sglang-standard:0.5.10.post1}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-${HOME}/.cache/huggingface/hub}"
RUN_ID="${WORK_AUDIT_CYCLES_RUN_ID:-storage_cycles_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${WORK_AUDIT_CYCLES_SEEDS:-1 2 3}"
ARMS="${WORK_AUDIT_CYCLES_ARMS:-on_demand}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/work_audit/${RUN_ID}"
SERVER_PID=""
CONTAINER_CID=""

[[ -d "${MODEL_CACHE}" ]] || { echo "Model cache missing: ${MODEL_CACHE}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run ID already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ "${SEEDS}" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "Invalid seed list" >&2; exit 2; }
if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
  echo "Port 30000 is already in use; refusing to disturb that server" >&2; exit 2
fi
stop_backend() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  if [[ -n "${CONTAINER_CID}" && -f "${CONTAINER_CID}" ]]; then
    docker rm -f "$(<"${CONTAINER_CID}")" >/dev/null 2>&1 || true
  fi
}
trap stop_backend EXIT
mkdir -p "${RUN_ROOT}/arms" "${RUN_ROOT}/runtime"
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
export BACKEND_RUNTIME_PROFILE=nvidia_standard
export BACKEND_RUNTIME_CONTRACT_OUT="${RUN_ROOT}/runtime/backend_runtime.json"
export SGLANG_DOCKER_IMAGE="${IMAGE}"
"${ROOT}/infra/container/probe_sglang_runtime.sh"

read -r -a seed_values <<< "${SEEDS}"
for seed in "${seed_values[@]}"; do
  read -r -a arm_values <<< "${ARMS}"
  if (( seed % 2 == 0 )) && [[ "${ARMS}" == "on_demand host_stage" ]]; then
    arm_values=(host_stage on_demand)
  fi
  for arm in "${arm_values[@]}"; do
    [[ "${arm}" == "on_demand" || "${arm}" == "host_stage" ]] || {
      echo "Unsupported arm: ${arm}; full_prepare is blocked after a live scheduler assertion" >&2; exit 2;
    }
    ARM_ROOT="${RUN_ROOT}/arms/seed${seed}_${arm}"
    mkdir -p "${ARM_ROOT}/storage"
    CONTAINER_CID="${ARM_ROOT}/container.cid"
    export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:/model-cache -e HF_HOME=/model-cache --cidfile ${CONTAINER_CID}"
    export AGENTIC_KV_TRACE_ENABLE=1
    export AGENTIC_KV_TRACE_PATH="${ARM_ROOT}/backend_trace.jsonl"
    profile_values="$(python3 -m agentic_backends.sglang.instrumentation_profiles kv_lifecycle_lean --shell)"
    while IFS='=' read -r name value; do
      [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
      printf -v "${name}" '%s' "${value}"
      export "${name}"
    done <<< "${profile_values}"
    export AGENTIC_KV_TRACE_CONTROL_ONLY=1
    export AGENTIC_KV_PREPARE_CONTROL_ENABLE=1
    export AGENTIC_KV_STORAGE_AUDIT_ENABLE=1
    export AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1
    export AGENTIC_KV_PREPARE_CONTROL_PORT=31991
    export HICACHE_SIZE_GB="${WORK_AUDIT_CYCLES_HOST_GB:-1}"
    export HICACHE_IO_BACKEND=direct
    export HICACHE_STORAGE_BACKEND=file
    export HICACHE_STORAGE_PATH="${ARM_ROOT}/storage"
    export HICACHE_STORAGE_PREFETCH_POLICY=wait_complete
    export HICACHE_STORAGE_BACKEND_EXTRA_CONFIG='{"prefetch_threshold":64}'
    export MEM_FRACTION_STATIC="${WORK_AUDIT_CYCLES_MEM_FRACTION:-0.7}"
    export EXTRA_SERVER_ARGS="--hicache-write-policy write_through --page-size 64 --max-total-tokens ${WORK_AUDIT_CYCLES_GPU_TOKENS:-16384}"
    echo "Starting seed${seed}_${arm} (GPU tokens ${WORK_AUDIT_CYCLES_GPU_TOKENS:-16384}, host GiB ${HICACHE_SIZE_GB})"
    (
      cd "${DIRECT_ROOT}"
      bash scripts/run_sglang_hicache_server.sh "${MODEL}"
    ) > "${ARM_ROOT}/server.log" 2>&1 &
    SERVER_PID="$!"
    deadline=$((SECONDS + 240))
    until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Backend exited; inspect ${ARM_ROOT}/server.log" >&2; exit 1
      fi
      (( SECONDS < deadline )) || { echo "Backend startup timed out" >&2; exit 1; }
      sleep 2
    done
    python3 - "${ARM_ROOT}/backend_trace.jsonl" <<'PY'
import json, sys
from agentic_backends.sglang.instrumentation_profiles import validate_installation
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
summaries = [row for row in rows if row.get("event") == "trace.install.summary"]
if not summaries or not validate_installation("kv_lifecycle_lean", "v0510", summaries[-1])["valid"]:
    raise SystemExit("Required pinned SGLang KV hooks are missing")
installed = summaries[-1].get("installed_hooks") or []
for hook in ("HiCacheController.prefetch", "HiRadixCache.evict_host", "PrefetchOperation.increment"):
    if not any(hook in value for value in installed):
        raise SystemExit(f"Required native storage hook missing: {hook}")
PY
    python3 -m agentic_experiments.runners.run_work_audit_storage_cycles \
      --run-id "${RUN_ID}_seed${seed}_${arm}" --arm "${arm}" --seed "${seed}" \
      --model "${MODEL}" --sessions "${WORK_AUDIT_CYCLES_SESSIONS:-4}" \
      --turns "${WORK_AUDIT_CYCLES_TURNS:-6}" \
      --initial-tokens "${WORK_AUDIT_CYCLES_INITIAL_TOKENS:-2048}" \
      --tool-result-words "${WORK_AUDIT_CYCLES_TOOL_WORDS:-900}" \
      --decode-tokens "${WORK_AUDIT_CYCLES_DECODE_TOKENS:-16}" \
      --wait-ms "${WORK_AUDIT_CYCLES_WAIT_MS:-1000}" \
      --wait-spread-ms "${WORK_AUDIT_CYCLES_WAIT_SPREAD_MS:-3000}" \
      --stagger-ms "${WORK_AUDIT_CYCLES_STAGGER_MS:-250}" \
      --out "${ARM_ROOT}/case_results.json"
    python3 -m agentic_backends.sglang.trace_contract --adapter v0510 \
      --profile kv_lifecycle_lean --trace "${ARM_ROOT}/backend_trace.jsonl" \
      --out "${ARM_ROOT}/instrumentation_audit.json"
    gzip -f "${ARM_ROOT}/backend_trace.jsonl"
    stop_backend
    # These are this arm's temporary file-cache payloads, not report evidence.
    find "${ARM_ROOT}/storage" -type f -delete
    sleep 3
  done
done
python3 -m agentic_experiments.runners.analyze_work_audit_storage_cycles \
  --arms-dir "${RUN_ROOT}/arms" --out "${RUN_ROOT}/summary.json" \
  --expected-seeds "${seed_values[@]}" --expected-arms "${arm_values[@]}"
RUN_STATUS="$(python3 - "${RUN_ROOT}/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)"
WORKLOAD_JSON="$(python3 - "${SEEDS}" "${ARMS}" "${MODEL}" \
  "${WORK_AUDIT_CYCLES_SESSIONS:-4}" "${WORK_AUDIT_CYCLES_TURNS:-6}" \
  "${WORK_AUDIT_CYCLES_INITIAL_TOKENS:-2048}" "${WORK_AUDIT_CYCLES_TOOL_WORDS:-900}" \
  "${WORK_AUDIT_CYCLES_DECODE_TOKENS:-16}" "${WORK_AUDIT_CYCLES_WAIT_MS:-1000}" \
  "${WORK_AUDIT_CYCLES_WAIT_SPREAD_MS:-3000}" "${WORK_AUDIT_CYCLES_STAGGER_MS:-250}" \
  "${WORK_AUDIT_CYCLES_GPU_TOKENS:-16384}" "${WORK_AUDIT_CYCLES_HOST_GB:-1}" <<'PY'
import json, sys
(seeds, arms, model, sessions, turns, initial, words, decode, wait, spread,
 stagger, gpu, host) = sys.argv[1:]
print(json.dumps({"research_question_id": "RQ17", "seeds": list(map(int, seeds.split())),
                  "arms": arms.split(), "model": model, "session_count": int(sessions),
                  "turns_per_session": int(turns), "initial_tokens": int(initial),
                  "tool_result_words": int(words), "decode_tokens": int(decode),
                  "tool_wait_base_ms": int(wait), "tool_wait_spread_ms": int(spread),
                  "session_stagger_ms": int(stagger), "gpu_kv_token_cap": int(gpu),
                  "host_cache_gb": int(host), "page_size_tokens": 64,
                  "storage_backend": "file", "write_policy": "write_through",
                  "frontend_priority": "equal", "fresh_backend_per_arm": True,
                  "trace_profile": "kv_lifecycle_lean", "manual_eviction": False}))
PY
)"
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment agentic_work_audit_storage_cycles --model "${MODEL}" \
  --hardware-profile nvidia_a10g_24gb --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${WORKLOAD_JSON}" --instrumentation kv_lifecycle_lean \
  --artifact "summary=${RUN_ROOT}/summary.json" --completion-status "${RUN_STATUS}"
echo "${RUN_STATUS}: ${RUN_ROOT}/summary.json"
if [[ "${RUN_STATUS}" == "blocked" || "${RUN_STATUS}" == "insufficient_exposure" ]]; then
  exit 2
fi
