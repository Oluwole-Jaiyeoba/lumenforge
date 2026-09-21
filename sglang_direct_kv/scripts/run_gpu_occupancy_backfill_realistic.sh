#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ -x ".venv/bin/python" ]]; then
  export PATH="${PWD}/.venv/bin:${PATH}"
  export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
fi

export REPORT_LABEL="${REPORT_LABEL:-gpu_occupancy_backfill_h1_$(date +%Y%m%d_%H%M%S)}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p3_high}"

# Hardware-telemetry follow-up:
# - no_prefetch shows ordinary behavior
# - controller_predictive_deadline_queue_admission_guard isolates ready-time ordering
# - controller_ready_time_gpu_backfill adds cached GPU-idle telemetry to admission
export MODES="${MODES:-no_prefetch controller_predictive_deadline_queue_admission_guard controller_ready_time_gpu_backfill}"

# Create staggered ready times plus a constant stream of candidate work.
# The backfill stream is what gives GPU telemetry a chance to matter: the
# controller can admit small/later work only when the next replay is protected
# and the GPU has physical idle headroom.
export P3_HIGH_KNOBS="${P3_HIGH_KNOBS:-tool_wait_ms=1000 target_prompt_tokens=2048 filler_sessions=8 filler_prompt_tokens=1024 session_count=1 concurrency=4}"
export TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS:-2}"
export FILLER_REPLAY_DEADLINES="${FILLER_REPLAY_DEADLINES:-1}"
export FILLER_BACKLOG_MODE="${FILLER_BACKLOG_MODE:-constant}"
export FILLER_BACKLOG_TARGET="${FILLER_BACKLOG_TARGET:-2}"
export FILLER_BACKLOG_TOTAL="${FILLER_BACKLOG_TOTAL:-24}"

export TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE:-hardware_backfill_staggered}"
export TOOL_WAIT_PROFILE_SPEC="${TOOL_WAIT_PROFILE_SPEC:-short:45:2000-6000,medium:40:10000-25000,long:15:45000-90000}"
export TOOL_WAIT_SEED="${TOOL_WAIT_SEED:-20260921}"
export WORKLOAD_SHAPE_MODE_INDEPENDENT="${WORKLOAD_SHAPE_MODE_INDEPENDENT:-1}"
export AGENTIC_WORKLOAD_PROFILE="${AGENTIC_WORKLOAD_PROFILE:-synthetic_pressure}"

export TRACE_PROFILE="${TRACE_PROFILE:-full_debug}"
export TRACE_CONTROLLER_DECISIONS="${TRACE_CONTROLLER_DECISIONS:-1}"
export TRACE_CONTROLLER_COMPLETION_LINKAGE="${TRACE_CONTROLLER_COMPLETION_LINKAGE:-1}"
export TRACE_REPLAY_FRICTION_DEEP_DIVE="${TRACE_REPLAY_FRICTION_DEEP_DIVE:-1}"
export TRACE_REPLAY_BLOCKERS="${TRACE_REPLAY_BLOCKERS:-1}"
export TRACE_REPLAY_BLOCKERS_MAX_IDS="${TRACE_REPLAY_BLOCKERS_MAX_IDS:-16}"
export TRACE_REPLAY_BLOCKERS_EVENTS="${TRACE_REPLAY_BLOCKERS_EVENTS:-before_acquire,submit}"
export AGENTIC_KV_GPU_UTIL_SAMPLER="${AGENTIC_KV_GPU_UTIL_SAMPLER:-1}"

export CONTROLLER_REPLAY_ADMISSION_GUARD="${CONTROLLER_REPLAY_ADMISSION_GUARD:-1}"
export CONTROLLER_REPLAY_LOOKAHEAD_MS="${CONTROLLER_REPLAY_LOOKAHEAD_MS:-6000}"
export CONTROLLER_REPLAY_SAFETY_MARGIN_MS="${CONTROLLER_REPLAY_SAFETY_MARGIN_MS:-500}"
export CONTROLLER_MAX_HOLD_MS="${CONTROLLER_MAX_HOLD_MS:-10000}"

# Force the RTG mode to justify risky candidate admission through GPU-idle
# backfill rather than through the generic "small work" shortcut.
export CONTROLLER_ALLOW_SMALL_WORK_TOKENS="${CONTROLLER_ALLOW_SMALL_WORK_TOKENS:-0}"
export CONTROLLER_GPU_TELEMETRY_POLL_MS="${CONTROLLER_GPU_TELEMETRY_POLL_MS:-250}"
export CONTROLLER_GPU_IDLE_UTIL_THRESHOLD_PCT="${CONTROLLER_GPU_IDLE_UTIL_THRESHOLD_PCT:-5}"
export CONTROLLER_GPU_IDLE_RECENT_WINDOW_MS="${CONTROLLER_GPU_IDLE_RECENT_WINDOW_MS:-500}"
export CONTROLLER_GPU_BACKFILL_MAX_RUNTIME_MS="${CONTROLLER_GPU_BACKFILL_MAX_RUNTIME_MS:-2500}"

export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-rich}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"

echo "GPU occupancy-aware backfill run"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "MODES=${MODES}"
echo "P3_HIGH_KNOBS=${P3_HIGH_KNOBS}"
echo "FILLER_BACKLOG_MODE=${FILLER_BACKLOG_MODE}"
echo "FILLER_BACKLOG_TARGET=${FILLER_BACKLOG_TARGET}"
echo "FILLER_BACKLOG_TOTAL=${FILLER_BACKLOG_TOTAL}"
echo "TOOL_WAIT_PROFILE_SPEC=${TOOL_WAIT_PROFILE_SPEC}"
echo "TOOL_WAIT_SEED=${TOOL_WAIT_SEED}"
echo "CONTROLLER_ALLOW_SMALL_WORK_TOKENS=${CONTROLLER_ALLOW_SMALL_WORK_TOKENS}"
echo "CONTROLLER_GPU_IDLE_UTIL_THRESHOLD_PCT=${CONTROLLER_GPU_IDLE_UTIL_THRESHOLD_PCT}"
echo "CONTROLLER_GPU_BACKFILL_MAX_RUNTIME_MS=${CONTROLLER_GPU_BACKFILL_MAX_RUNTIME_MS}"

bash scripts/run_harness_deadline_pressure.sh "${MODEL}"
