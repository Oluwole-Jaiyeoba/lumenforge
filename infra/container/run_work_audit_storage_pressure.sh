#!/usr/bin/env bash
set -euo pipefail

# Paired storage-replay sweep with only peer pressure changing between runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP_ID="${WORK_AUDIT_STORAGE_SWEEP_ID:-storage_pressure_$(date +%Y%m%d_%H%M%S)}"
PEER_LEVELS="${WORK_AUDIT_STORAGE_PRESSURE_LEVELS:-0 2 6}"
[[ "${PEER_LEVELS}" =~ ^[0-9]+(\ [0-9]+)*$ ]] || {
  echo "Invalid pressure levels: ${PEER_LEVELS}" >&2
  exit 2
}

for peers in ${PEER_LEVELS}; do
  export WORK_AUDIT_STORAGE_PEERS="${peers}"
  export WORK_AUDIT_STORAGE_RUN_ID="${SWEEP_ID}_peers${peers}"
  export WORK_AUDIT_STORAGE_PROMPT_ID="${SWEEP_ID}_shared"
  export WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ16
  export WORK_AUDIT_STORAGE_PEER_START_MS=0
  echo "Storage pressure sweep: ${peers} peers (${WORK_AUDIT_STORAGE_RUN_ID})"
  bash "${ROOT}/infra/container/run_work_audit_storage.sh"
done
