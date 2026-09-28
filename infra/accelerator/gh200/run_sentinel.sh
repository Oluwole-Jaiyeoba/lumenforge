#!/usr/bin/env bash
set -euo pipefail

# Locked GH200 bring-up: equal-importance DeepAgents, P3, baseline vs RTG.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_gh200}"
if [[ "${BACKEND_RUNTIME_PROFILE}" != "nvidia_gh200" ]]; then
  echo "GH200 sentinel requires BACKEND_RUNTIME_PROFILE=nvidia_gh200." >&2
  exit 2
fi
if [[ -n "${BACKEND_RUNTIME_PROFILE_PATH:-}" || -n "${HARDWARE_PROFILE_PATH:-}" ]]; then
  echo "GH200 sentinel does not accept backend or hardware profile path overrides." >&2
  exit 2
fi
export AGENTIC_MODEL_CACHE="${AGENTIC_MODEL_CACHE:-${AGENTIC_GH200_MODEL_CACHE:-${HOME}/dynamo_model_cache}}"
export REPORT_LABEL="${REPORT_LABEL:-gh200_scenario1_equal_$(date +%Y%m%d_%H%M%S)}"
export UPDATE_LATEST=0

exec "${REPO_ROOT}/infra/container/run_scenario1_hybrid_reference.sh" "$@"
