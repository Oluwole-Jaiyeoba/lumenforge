#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${WORK_AUDIT_LADDER_RUN_ID:-storage_ladder_$(date +%Y%m%d_%H%M%S)}"
COUNTS="${WORK_AUDIT_LADDER_SESSION_COUNTS:-1 2 4 8}"
SEEDS="${WORK_AUDIT_LADDER_SEEDS:-1 2}"
[[ "${COUNTS}" =~ ^[1-9][0-9]*(\ [1-9][0-9]*)*$ ]] || {
  echo "Session counts must be positive integers" >&2; exit 2;
}
[[ "${SEEDS}" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "Invalid seed list" >&2; exit 2; }

RUN_ROOT="${ROOT}/sglang_direct_kv/artifacts/results/work_audit"
read -r -a session_counts <<< "${COUNTS}"
for count in "${session_counts[@]}"; do
  export WORK_AUDIT_STORAGE_RUN_ID="${RUN_ID}_n${count}"
  export WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ18
  export WORK_AUDIT_STORAGE_PROMPT_ID="${RUN_ID}_shared"
  export WORK_AUDIT_STORAGE_ARMS="on_demand host_stage"
  export WORK_AUDIT_STORAGE_SEEDS="${SEEDS}"
  export WORK_AUDIT_STORAGE_PEERS="$((count - 1))"
  export WORK_AUDIT_STORAGE_HOST_GB="${WORK_AUDIT_LADDER_HOST_GB:-14}"
  export WORK_AUDIT_STORAGE_CUDA_GRAPH=1
  export WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1
  export WORK_AUDIT_STORAGE_VERIFY_OUTPUT=1
  bash "${ROOT}/infra/container/run_work_audit_storage.sh"
done

PYTHONPATH="$(printf '%s:' "${ROOT}"/packages/*/src)${PYTHONPATH:-}" \
  python3 -m agentic_experiments.runners.analyze_work_audit_storage_ladder \
    --runs-root "${RUN_ROOT}" --run-id "${RUN_ID}" \
    --session-counts "${COUNTS}" --seeds "${SEEDS}" \
    --out "${RUN_ROOT}/${RUN_ID}/ladder_summary.json"
