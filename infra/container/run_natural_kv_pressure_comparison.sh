#!/usr/bin/env bash
set -euo pipefail

# Repeat isolated, equal-priority natural-pressure cases. The child wrapper
# starts a clean backend for every case so cache state cannot cross conditions.

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
REPORT_LABEL="${REPORT_LABEL:-natural_kv_pressure_comparison_$(date +%Y%m%d_%H%M%S)}"
SESSION_LEVELS="${SESSION_LEVELS:-4 8 12}"
TRIALS="${TRIALS:-3}"
TOOL_WAITS="${TOOL_WAITS:-3}"
SESSION_PREFIX_TOKENS="${SESSION_PREFIX_TOKENS:-8192}"
REPLAY_TOKENS="${REPLAY_TOKENS:-64}"
INITIAL_STAGGER_MS="${INITIAL_STAGGER_MS:-75}"
TOOL_WAIT_MIN_MS="${TOOL_WAIT_MIN_MS:-400}"
TOOL_WAIT_MAX_MS="${TOOL_WAIT_MAX_MS:-2000}"
RUN_ROOT="${DIRECT_ROOT}/artifacts/results/hardware/${REPORT_LABEL}"
CASES_ROOT="${RUN_ROOT}/cases"

[[ -n "${SGLANG_DOCKER_IMAGE:-}" ]] || { echo "Set SGLANG_DOCKER_IMAGE." >&2; exit 2; }
[[ -n "${AGENTIC_MODEL_CACHE:-}" && -d "${AGENTIC_MODEL_CACHE}" ]] || { echo "Set AGENTIC_MODEL_CACHE to an existing model-cache directory." >&2; exit 2; }
mkdir -p "${CASES_ROOT}"

for sessions in ${SESSION_LEVELS}; do
  for ((trial = 1; trial <= TRIALS; trial++)); do
    case_id="s${sessions}_t${trial}"
    case_label="${REPORT_LABEL}_${case_id}"
    echo "Starting ${case_id}: ${sessions} equal-priority sessions, trial ${trial}/${TRIALS}"
    REPORT_LABEL="${case_label}" \
    SESSION_COUNT="${sessions}" TOOL_WAITS="${TOOL_WAITS}" \
    SESSION_PREFIX_TOKENS="${SESSION_PREFIX_TOKENS}" REPLAY_TOKENS="${REPLAY_TOKENS}" \
    INITIAL_STAGGER_MS="${INITIAL_STAGGER_MS}" \
    TOOL_WAIT_MIN_MS="${TOOL_WAIT_MIN_MS}" TOOL_WAIT_MAX_MS="${TOOL_WAIT_MAX_MS}" \
    bash "${SCRIPT_DIR}/run_natural_multi_agent_kv_pressure_reference.sh" "${MODEL}"
    case_dir="${DIRECT_ROOT}/artifacts/results/hardware/${case_label}"
    mkdir -p "${CASES_ROOT}/${case_id}/runtime"
    cp "${case_dir}/command.sh" "${case_dir}/natural_multi_agent_kv_pressure_summary.json" \
      "${case_dir}/natural_multi_agent_kv_pressure_report.html" "${case_dir}/run_manifest.json" \
      "${CASES_ROOT}/${case_id}/"
    cp "${case_dir}/runtime/backend_runtime.json" "${CASES_ROOT}/${case_id}/runtime/"
  done
done

PACKAGE_PYTHONPATH=""
for package_src in "${REPO_ROOT}"/packages/*/src; do
  [[ -d "${package_src}" ]] || continue
  PACKAGE_PYTHONPATH="${PACKAGE_PYTHONPATH:+${PACKAGE_PYTHONPATH}:}${package_src}"
done
export PYTHONPATH="${PACKAGE_PYTHONPATH}:${PYTHONPATH:-}"
python3 -m agentic_reports.builders.build_natural_kv_pressure_comparison_report \
  --cases-root "${CASES_ROOT}" \
  --out "${RUN_ROOT}/natural_kv_pressure_comparison_report.html" \
  --summary-out "${RUN_ROOT}/natural_kv_pressure_comparison_summary.json"
python3 "${REPO_ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${REPORT_LABEL}" \
  --experiment "natural_kv_pressure_comparison" --model "${MODEL}" \
  --hardware-profile "${HARDWARE_PROFILE:-nvidia_standard}" \
  --workload-json "{\"frontend_priority\":\"none\",\"session_levels\":\"${SESSION_LEVELS}\",\"trials\":${TRIALS},\"tool_waits\":${TOOL_WAITS},\"session_prefix_tokens\":${SESSION_PREFIX_TOKENS},\"replay_tokens\":${REPLAY_TOKENS}}" \
  --instrumentation "lightweight_backend_trace" --instrumentation "natural_session_timeline" \
  --artifact "summary=${RUN_ROOT}/natural_kv_pressure_comparison_summary.json" \
  --artifact "report=${RUN_ROOT}/natural_kv_pressure_comparison_report.html" --completion-status complete
echo "Complete: ${RUN_ROOT}/natural_kv_pressure_comparison_report.html"
