#!/usr/bin/env bash
set -euo pipefail

# Locked reproduction of Scenario 1 through the host/controller + Docker-SGLang
# path. It intentionally has no application-level priority classes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
requested_hardware_profile="${HARDWARE_PROFILE:-}"

set -a
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/experiment_specs/scenario1_ready_time_reference.env"
set +a

export BACKEND_RUNTIME_PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
profile_path="${BACKEND_RUNTIME_PROFILE_PATH:-${REPO_ROOT}/configs/backend_runtimes/${BACKEND_RUNTIME_PROFILE}.json}"
if [[ ! -f "${profile_path}" ]]; then
  echo "Backend runtime profile not found: ${profile_path}" >&2
  exit 2
fi
profile_hardware="$(python3 - "${profile_path}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8")).get("hardware_profile", "")
if not isinstance(value, str) or not value:
    raise SystemExit("Backend runtime profile has no hardware_profile")
print(value)
PY
)"
if [[ -n "${requested_hardware_profile}" && "${requested_hardware_profile}" != "${profile_hardware}" ]]; then
  echo "HARDWARE_PROFILE=${requested_hardware_profile} disagrees with ${BACKEND_RUNTIME_PROFILE} (${profile_hardware})." >&2
  exit 2
fi
export HARDWARE_PROFILE="${profile_hardware}"
export REPORT_LABEL="${REPORT_LABEL:-scenario1_hybrid_reference_$(date +%Y%m%d_%H%M%S)}"
export UPDATE_LATEST="${UPDATE_LATEST:-0}"

expected_modes="no_prefetch controller_ready_time_gpu_backfill"
if [[ "${MODES}" != "${expected_modes}" ]]; then
  echo "Scenario 1 requires MODES='${expected_modes}', got '${MODES}'." >&2
  exit 2
fi
if [[ "${HARNESSES}" != "hatcher" || "${PRESSURE_LEVELS}" != "p3_high" ]]; then
  echo "Scenario 1 requires HARNESSES=hatcher and PRESSURE_LEVELS=p3_high." >&2
  exit 2
fi
if [[ "${WORKLOAD_SHAPE_MODE_INDEPENDENT}" != "1" ]]; then
  echo "Scenario 1 requires a mode-independent workload timeline." >&2
  exit 2
fi
if [[ "${AGENTIC_EQUAL_IMPORTANCE_WORKLOAD:-0}" != "1" || "${FILLER_REPLAY_DEADLINES}" != "1" ]]; then
  echo "Scenario 1 requires equal importance and replay deadlines for every session." >&2
  exit 2
fi

"${SCRIPT_DIR}/run_hybrid_reference.sh" "${MODEL}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 "${REPO_ROOT}/scripts/validate_scenario1_equal_importance.py" \
    --run-root "${REPO_ROOT}/sglang_direct_kv/artifacts/results/runs/controlled/${REPORT_LABEL}" \
    --report-dir "${REPO_ROOT}/sglang_direct_kv/artifacts/results/reports/${REPORT_LABEL}" \
    --expected-replays 32
fi
