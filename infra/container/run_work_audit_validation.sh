#!/usr/bin/env bash
set -euo pipefail

# One pinned backend for controlled lifecycle, timing, or concurrent timeline probes.
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIRECT_ROOT="${ROOT}/sglang_direct_kv"
PROFILE="${BACKEND_RUNTIME_PROFILE:-nvidia_standard}"
PROFILE_PATH="${ROOT}/configs/backend_runtimes/${PROFILE}.json"
IMAGE="${SGLANG_DOCKER_IMAGE:-}"
MODEL_CACHE="${AGENTIC_MODEL_CACHE:-}"
RUN_ID="${WORK_AUDIT_RUN_ID:-work_audit_$(date +%Y%m%d_%H%M%S)}"
STUDY="${WORK_AUDIT_STUDY:-validation}"
SECOND_REPLAY="${WORK_AUDIT_SECOND_REPLAY:-0}"
CASE_ORDER="${WORK_AUDIT_CASE_ORDER:-}"
if [[ -z "${CASE_ORDER}" ]]; then
  case "${STUDY}" in
    timing) CASE_ORDER="early-late" ;;
    multisession_compare) CASE_ORDER="late_nonblocking-early" ;;
    multisession_window) CASE_ORDER="late_nonblocking-early-post_short" ;;
    multisession_controller) CASE_ORDER="late_nonblocking-early-post_short-controller_window" ;;
    multisession_overlap) CASE_ORDER="early-post_short" ;;
    overlap_dose) CASE_ORDER="dose" ;;
    tool_cycles) CASE_ORDER="cycles" ;;
    multisession) CASE_ORDER="long-short-ends" ;;
    *) CASE_ORDER="warm-host" ;;
  esac
fi
TRACE_PROFILE="${WORK_AUDIT_TRACE_PROFILE:-$( if [[ "${STUDY}" == "tool_cycles" ]]; then echo tool_cycle_timing; elif [[ "${STUDY}" == "multisession_overlap" || "${STUDY}" == "overlap_dose" ]]; then echo kv_decode_overlap; elif [[ "${STUDY}" == "timing" || "${STUDY}" == "multisession" || "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" ]]; then echo kv_lifecycle_lean; else echo kv_lifecycle; fi )}"
TRACE_ENABLE="${WORK_AUDIT_TRACE_ENABLE:-1}"
FORWARD_TRACE="${WORK_AUDIT_FORWARD_TRACE:-0}"
NSYS_ENABLE="${WORK_AUDIT_NSYS_ENABLE:-0}"
CUDA_GRAPH="${WORK_AUDIT_CUDA_GRAPH:-0}"
OVERLAP_SCHEDULE="${WORK_AUDIT_OVERLAP_SCHEDULE:-0}"
PAIRS="${WORK_AUDIT_PAIRS:-1}"
WARMUP_PAIRS="${WORK_AUDIT_WARMUP_PAIRS:-1}"
WAIT_MS="${WORK_AUDIT_WAIT_MS:-$( [[ "${STUDY}" == "timing" ]] && echo 2000 || echo 500 )}"
SHORT_WAIT_MS="${WORK_AUDIT_SHORT_WAIT_MS:-900}"
LONG_WAIT_MS="${WORK_AUDIT_LONG_WAIT_MS:-$( [[ "${STUDY}" == "multisession_overlap" ]] && echo 10000 || echo 2500 )}"
EARLY_AT_MS="${WORK_AUDIT_EARLY_AT_MS:-1200}"
ESTIMATED_LOAD_MS="${WORK_AUDIT_ESTIMATED_LOAD_MS:-250}"
LOAD_MARGIN_MS="${WORK_AUDIT_LOAD_MARGIN_MS:-150}"
EXACT_INDICES="${WORK_AUDIT_EXACT_INDICES:-256}"
REQUIRE_SLOT_PROOF="${WORK_AUDIT_REQUIRE_SLOT_PROOF:-0}"
PROMPT_WORDS="${WORK_AUDIT_PROMPT_WORDS:-4090}"
ACTIVE_PROMPT_WORDS="${WORK_AUDIT_ACTIVE_PROMPT_WORDS:-${PROMPT_WORDS}}"
DONOR_PROMPT_WORDS="${WORK_AUDIT_DONOR_PROMPT_WORDS:-${PROMPT_WORDS}}"
MAX_OUTPUT_TOKENS="${WORK_AUDIT_MAX_OUTPUT_TOKENS:-16}"
MINIMUM_HOST_TOKENS="${WORK_AUDIT_MINIMUM_HOST_TOKENS:-512}"
EVICTION_ROUNDS="${WORK_AUDIT_EVICTION_ROUNDS:-4}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
SESSION_COUNT="${WORK_AUDIT_SESSION_COUNT:-6}"
DONOR_COUNT="${WORK_AUDIT_DONOR_COUNT:-4}"
PLANNED_OVERLAP="${WORK_AUDIT_PLANNED_OVERLAP:-1}"
SEED="${WORK_AUDIT_SEED:-1}"
PAIR_ID="${WORK_AUDIT_PAIR_ID:-}"
DECODE_TOKENS="${WORK_AUDIT_DECODE_TOKENS:-96}"
DONOR_WAIT_MS="${WORK_AUDIT_DONOR_WAIT_MS:-10000}"
TOOL_CYCLE_TURNS="${WORK_AUDIT_TOOL_CYCLE_TURNS:-12}"
TOOL_CYCLE_ACTIVE_COUNT="${WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT:-2}"
TOOL_CYCLE_INITIAL_TOKENS="${WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS:-768}"
TOOL_CYCLE_DONOR_INITIAL_TOKENS="${WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS:-512}"
TOOL_CYCLE_RESULT_WORDS="${WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS:-96}"
TOOL_CYCLE_WAIT_MS="${WORK_AUDIT_TOOL_CYCLE_WAIT_MS:-800}"
RESULTS_BASE="${WORK_AUDIT_RESULTS_BASE:-${DIRECT_ROOT}/artifacts/results/work_audit}"
RUN_ROOT="${RESULTS_BASE}/${RUN_ID}"
SERVER_PID=""
NSYS_CONTAINER_NAME=""

[[ -f "${PROFILE_PATH}" ]] || { echo "Missing runtime profile: ${PROFILE_PATH}" >&2; exit 2; }
[[ -n "${IMAGE}" ]] || { echo "Set SGLANG_DOCKER_IMAGE" >&2; exit 2; }
[[ -d "${MODEL_CACHE}" ]] || { echo "Set AGENTIC_MODEL_CACHE to a directory" >&2; exit 2; }
[[ "${STUDY}" == "validation" || "${STUDY}" == "timing" || "${STUDY}" == "multisession" || "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" || "${STUDY}" == "multisession_overlap" || "${STUDY}" == "overlap_dose" || "${STUDY}" == "tool_cycles" ]] || {
  echo "Unsupported WORK_AUDIT_STUDY: ${STUDY}" >&2; exit 2;
}
[[ "${TRACE_ENABLE}" == "1" || ( "${TRACE_ENABLE}" == "0" && "${STUDY}" == "tool_cycles" ) ]] || {
  echo "Trace-off control is supported only for tool_cycles" >&2; exit 2;
}
[[ "${SECOND_REPLAY}" == "0" || "${SECOND_REPLAY}" == "1" ]] || {
  echo "WORK_AUDIT_SECOND_REPLAY must be 0 or 1" >&2; exit 2;
}
[[ "${WAIT_MS}" =~ ^[1-9][0-9]*$ ]] || { echo "WORK_AUDIT_WAIT_MS must be positive" >&2; exit 2; }
for value in "${PROMPT_WORDS}" "${MAX_OUTPUT_TOKENS}" "${MINIMUM_HOST_TOKENS}" "${EVICTION_ROUNDS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Work-audit request settings must be positive integers" >&2; exit 2; }
done
if [[ "${STUDY}" == "tool_cycles" ]]; then
  [[ "${CASE_ORDER}" == "cycles" && "${TRACE_PROFILE}" == "tool_cycle_timing" &&
     "${TOOL_CYCLE_TURNS}" =~ ^[1-9][0-9]*$ &&
     "${TOOL_CYCLE_ACTIVE_COUNT}" =~ ^[1-9][0-9]*$ &&
     "${DONOR_COUNT}" =~ ^[0-9]+$ &&
     "${TOOL_CYCLE_INITIAL_TOKENS}" =~ ^[1-9][0-9]*$ &&
     "${TOOL_CYCLE_DONOR_INITIAL_TOKENS}" =~ ^[1-9][0-9]*$ &&
     "${TOOL_CYCLE_RESULT_WORDS}" =~ ^[1-9][0-9]*$ &&
     "${TOOL_CYCLE_WAIT_MS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Tool cycles need valid counts and tool_cycle_timing tracing" >&2; exit 2;
  }
elif [[ "${STUDY}" == "overlap_dose" ]]; then
  [[ "${CASE_ORDER}" == "dose" && "${TRACE_PROFILE}" == "kv_decode_overlap" &&
     "${AGENTIC_KV_PREPARE_LOAD_WORKER:-0}" == "1" &&
     "${ACTIVE_PROMPT_WORDS}" =~ ^[1-9][0-9]*$ &&
     "${DONOR_PROMPT_WORDS}" =~ ^[1-9][0-9]*$ &&
     "${SESSION_COUNT}" =~ ^[1-9][0-9]*$ && "${DONOR_COUNT}" =~ ^[1-9][0-9]*$ &&
     "${PLANNED_OVERLAP}" =~ ^[0-9]+$ && "${SEED}" =~ ^[1-9][0-9]*$ &&
     "${DECODE_TOKENS}" =~ ^[1-9][0-9]*$ && "${DONOR_WAIT_MS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Overlap dose requires worker KV loads, kv_decode_overlap, and valid workload counts" >&2; exit 2;
  }
  (( SESSION_COUNT > DONOR_COUNT && PLANNED_OVERLAP <= DONOR_COUNT )) || {
    echo "Overlap dose needs active sessions and no more overlapping loads than donors" >&2; exit 2;
  }
elif [[ "${STUDY}" == "multisession" || "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" || "${STUDY}" == "multisession_overlap" ]]; then
  [[ "${SHORT_WAIT_MS}" =~ ^[1-9][0-9]*$ &&
     "${LONG_WAIT_MS}" =~ ^[1-9][0-9]*$ && "${SHORT_WAIT_MS}" -lt "${LONG_WAIT_MS}" ]] || {
    echo "Multisession requires positive, ordered tool waits" >&2; exit 2;
  }
  if [[ "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" || "${STUDY}" == "multisession_overlap" ]]; then
    if [[ "${STUDY}" == "multisession_controller" ]]; then
      [[ "${CASE_ORDER}" == "late_nonblocking-early-post_short-controller_window" ||
         "${CASE_ORDER}" == "controller_window-post_short-early-late_nonblocking" ]] || {
        echo "Controller study requires all four timing modes" >&2; exit 2;
      }
      [[ "${ESTIMATED_LOAD_MS}" =~ ^[1-9][0-9]*$ && "${LOAD_MARGIN_MS}" =~ ^[0-9]+$ ]] || {
        echo "Controller load estimate must be positive and margin nonnegative" >&2; exit 2;
      }
    elif [[ "${STUDY}" == "multisession_window" ]]; then
      [[ "${CASE_ORDER}" == "late_nonblocking-early-post_short" ||
         "${CASE_ORDER}" == "post_short-early-late_nonblocking" ]] || {
        echo "Window study requires all three timing modes" >&2; exit 2;
      }
    elif [[ "${STUDY}" == "multisession_overlap" ]]; then
      [[ "${CASE_ORDER}" == "early-post_short" || "${CASE_ORDER}" == "post_short-early" ]] || {
        echo "Decode overlap study requires early and post_short only" >&2; exit 2;
      }
      [[ "${AGENTIC_KV_PREPARE_LOAD_WORKER:-0}" == "1" ]] || {
        echo "Decode overlap study requires worker KV loading" >&2; exit 2;
      }
    else
      [[ "${CASE_ORDER}" == "late_nonblocking-early" || "${CASE_ORDER}" == "early-late_nonblocking" ]] || {
        echo "Comparison requires late_nonblocking-early or early-late_nonblocking" >&2; exit 2;
      }
    fi
    [[ "${EARLY_AT_MS}" =~ ^[1-9][0-9]*$ && "${SHORT_WAIT_MS}" -lt "${EARLY_AT_MS}" &&
       "${EARLY_AT_MS}" -lt "${LONG_WAIT_MS}" && "${PAIRS}" =~ ^[1-9][0-9]*$ &&
       "${WARMUP_PAIRS}" =~ ^[0-9]+$ ]] || {
      echo "Comparison requires short wait < early load < long wait, positive pairs, and nonnegative warmups" >&2; exit 2;
    }
  else
    [[ "${CASE_ORDER}" == "long-short-ends" ]] || { echo "Multisession requires long-short-ends" >&2; exit 2; }
  fi
  if [[ "${STUDY}" == "multisession_overlap" ]]; then
    [[ "${TRACE_PROFILE}" == "kv_decode_overlap" ]] || {
      echo "Decode overlap study requires kv_decode_overlap" >&2; exit 2;
    }
  else
    [[ "${TRACE_PROFILE}" == "kv_lifecycle_lean" ]] || {
      echo "Multisession requires kv_lifecycle_lean" >&2; exit 2;
    }
  fi
elif [[ "${STUDY}" == "validation" ]]; then
  [[ "${CASE_ORDER}" == "warm-host" || "${CASE_ORDER}" == "host-warm" ]] || {
    echo "WORK_AUDIT_CASE_ORDER must be warm-host or host-warm for validation" >&2; exit 2;
  }
else
  [[ "${CASE_ORDER}" == "early-late" || "${CASE_ORDER}" == "late-early" ||
     "${CASE_ORDER}" == "early-late-late_nonblocking" ||
     "${CASE_ORDER}" == "late_nonblocking-late-early" ]] || {
    echo "Unsupported WORK_AUDIT_CASE_ORDER for timing" >&2; exit 2;
  }
  [[ "${TRACE_PROFILE}" == "kv_lifecycle_lean" ]] || {
    echo "Timing study requires kv_lifecycle_lean" >&2; exit 2;
  }
  [[ "${PAIRS}" =~ ^[1-9][0-9]*$ && "${WAIT_MS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "WORK_AUDIT_PAIRS and WORK_AUDIT_WAIT_MS must be positive integers" >&2; exit 2;
  }
  [[ "${WARMUP_PAIRS}" =~ ^[0-9]+$ ]] || {
    echo "WORK_AUDIT_WARMUP_PAIRS must be a nonnegative integer" >&2; exit 2;
  }
  [[ "${EXACT_INDICES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "WORK_AUDIT_EXACT_INDICES must be positive" >&2; exit 2;
  }
  [[ "${REQUIRE_SLOT_PROOF}" == "0" || "${REQUIRE_SLOT_PROOF}" == "1" ]] || {
    echo "WORK_AUDIT_REQUIRE_SLOT_PROOF must be 0 or 1" >&2; exit 2;
  }
fi
[[ "${TRACE_PROFILE}" == "kv_lifecycle" || "${TRACE_PROFILE}" == "kv_lifecycle_lean" || "${TRACE_PROFILE}" == "kv_decode_overlap" || "${TRACE_PROFILE}" == "tool_cycle_timing" ]] || {
  echo "Unsupported WORK_AUDIT_TRACE_PROFILE" >&2; exit 2;
}
[[ "${FORWARD_TRACE}" == "0" || "${FORWARD_TRACE}" == "1" ]] || {
  echo "WORK_AUDIT_FORWARD_TRACE must be 0 or 1" >&2; exit 2;
}
[[ "${NSYS_ENABLE}" == "0" || "${NSYS_ENABLE}" == "1" ]] || {
  echo "WORK_AUDIT_NSYS_ENABLE must be 0 or 1" >&2; exit 2;
}
[[ "${CUDA_GRAPH}" == "0" || "${CUDA_GRAPH}" == "1" ]] || {
  echo "WORK_AUDIT_CUDA_GRAPH must be 0 or 1" >&2; exit 2;
}
[[ "${OVERLAP_SCHEDULE}" == "0" || "${OVERLAP_SCHEDULE}" == "1" ]] || {
  echo "WORK_AUDIT_OVERLAP_SCHEDULE must be 0 or 1" >&2; exit 2;
}
[[ "${NSYS_ENABLE}" == "0" || ( ( "${STUDY}" == "multisession_overlap" || "${STUDY}" == "overlap_dose" ) && "${FORWARD_TRACE}" == "1" ) ]] || {
  echo "Nsight capture requires an overlap study with forward tracing" >&2; exit 2;
}
[[ "${FORWARD_TRACE}" == "0" || "${STUDY}" == "multisession_overlap" || "${STUDY}" == "overlap_dose" ]] || {
  echo "Forward-only tracing is supported only for overlap studies" >&2; exit 2;
}
if [[ "${STUDY}" == "validation" ]]; then
  DEFAULT_QUESTION_ID="RQ1"
elif [[ "${STUDY}" == "multisession_controller" ]]; then
  DEFAULT_QUESTION_ID="RQ7"
elif [[ "${STUDY}" == "multisession_window" ]]; then
  DEFAULT_QUESTION_ID="RQ6"
elif [[ "${STUDY}" == "multisession_compare" ]]; then
  DEFAULT_QUESTION_ID="RQ5"
elif [[ "${STUDY}" == "multisession_overlap" ]]; then
  DEFAULT_QUESTION_ID="RQ10"
elif [[ "${STUDY}" == "tool_cycles" ]]; then
  DEFAULT_QUESTION_ID="RQ14"
elif [[ "${STUDY}" == "overlap_dose" ]]; then
  DEFAULT_QUESTION_ID="RQ11"
elif [[ "${STUDY}" == "multisession" ]]; then
  DEFAULT_QUESTION_ID="RQ4"
elif [[ "${CASE_ORDER}" == *late_nonblocking* ]]; then
  DEFAULT_QUESTION_ID="RQ3"
else
  DEFAULT_QUESTION_ID="RQ2"
fi
RESEARCH_QUESTION_ID="${WORK_AUDIT_RESEARCH_QUESTION_ID:-${DEFAULT_QUESTION_ID}}"
[[ "${RESEARCH_QUESTION_ID}" =~ ^RQ[1-9][0-9]*$ ]] || {
  echo "WORK_AUDIT_RESEARCH_QUESTION_ID must look like RQ1" >&2; exit 2;
}
[[ "${PAIR_ID}" =~ ^[A-Za-z0-9_-]*$ ]] || {
  echo "WORK_AUDIT_PAIR_ID must contain only letters, numbers, underscores, or hyphens" >&2; exit 2;
}
[[ "${AGENTIC_KV_PREPARE_LOAD_WORKER:-0}" == "0" || "${AGENTIC_KV_PREPARE_LOAD_WORKER:-0}" == "1" ]] || {
  echo "AGENTIC_KV_PREPARE_LOAD_WORKER must be 0 or 1" >&2; exit 2;
}
LOAD_EXECUTION="scheduler"
[[ "${AGENTIC_KV_PREPARE_LOAD_WORKER:-0}" == "1" ]] && LOAD_EXECUTION="worker"
if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
  echo "Port 30000 is already serving a model; refusing to disturb it." >&2
  exit 2
fi
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
mkdir -p "${RUN_ROOT}/runtime"

if [[ -f "${DIRECT_ROOT}/.venv/bin/activate" ]]; then
  source "${DIRECT_ROOT}/.venv/bin/activate"
fi
PACKAGE_PYTHONPATH=""
for source_dir in "${ROOT}"/packages/*/src; do
  [[ -d "${source_dir}" ]] || continue
  PACKAGE_PYTHONPATH="${PACKAGE_PYTHONPATH:+${PACKAGE_PYTHONPATH}:}${source_dir}"
done
export PYTHONPATH="${PACKAGE_PYTHONPATH}:${PYTHONPATH:-}"
export SGLANG_DOCKER_IMAGE="${IMAGE}"
export BACKEND_RUNTIME_PROFILE="${PROFILE}"
export BACKEND_RUNTIME_CONTRACT_OUT="${RUN_ROOT}/runtime/backend_runtime.json"
export CUDA_GRAPH_FLAG="--disable-cuda-graph"
export OVERLAP_FLAG="--disable-overlap-schedule"
[[ "${CUDA_GRAPH}" == "1" ]] && CUDA_GRAPH_FLAG=""
[[ "${OVERLAP_SCHEDULE}" == "1" ]] && OVERLAP_FLAG=""
MODEL_CACHE_MOUNT="$(python3 - "${PROFILE_PATH}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model_cache_mount"])
PY
)"
export SGLANG_DOCKER_EXTRA_ARGS="-v ${MODEL_CACHE}:${MODEL_CACHE_MOUNT} -e HF_HOME=${MODEL_CACHE_MOUNT} ${SGLANG_DOCKER_EXTRA_ARGS:-}"
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  command -v nsys >/dev/null || { echo "Nsight Systems is required on the host" >&2; exit 2; }
  NSYS_HOST_DIR="$(dirname "$(dirname "$(readlink -f "$(command -v nsys)")")")"
  NSYS_CONTAINER_NAME="agentic-work-audit-${RUN_ID//[^a-zA-Z0-9_.-]/-}"
  mkdir -p "${RUN_ROOT}/nsys"
  export AGENTIC_NSYS_BIN="/opt/agentic_nsight/target-linux-x64/nsys"
  export AGENTIC_NSYS_OUTPUT="${RUN_ROOT}/nsys/backend"
  export AGENTIC_NSYS_INTERACTIVE=1
  export AGENTIC_KV_NVTX_ENABLE=1
  export SGLANG_DOCKER_EXTRA_ARGS="${SGLANG_DOCKER_EXTRA_ARGS} --cap-add SYS_ADMIN -v ${NSYS_HOST_DIR}:/opt/agentic_nsight:ro --name ${NSYS_CONTAINER_NAME}"
fi

"${SCRIPT_DIR}/probe_sglang_runtime.sh"
python3 - "${BACKEND_RUNTIME_CONTRACT_OUT}" <<'PY'
import json
import sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))
if runtime.get("backend_version") != "0.5.10.post1" or runtime.get("adapter") != "v0510":
    raise SystemExit(f"Refusing unpinned audit runtime: {runtime.get('backend_version')} / {runtime.get('adapter')}")
PY

echo "Starting pinned backend for ${RUN_ID}"
(
  cd "${DIRECT_ROOT}"
  export AGENTIC_KV_TRACE_ENABLE="${TRACE_ENABLE}"
  export AGENTIC_KV_TRACE_PATH="${RUN_ROOT}/backend_trace.jsonl"
  profile_values="$(python3 -m agentic_backends.sglang.instrumentation_profiles "${TRACE_PROFILE}" --shell)"
  while IFS='=' read -r name value; do
    [[ -z "${name}" || "${name}" == *_DEFAULT ]] && continue
    printf -v "${name}" '%s' "${!name:-${value}}"
    export "${name}"
  done <<< "${profile_values}"
  export AGENTIC_KV_TRACE_MODEL_FORWARD_ONLY="${FORWARD_TRACE}"
  if [[ "${TRACE_PROFILE}" == "kv_lifecycle_lean" || "${TRACE_PROFILE}" == "kv_decode_overlap" || "${TRACE_PROFILE}" == "tool_cycle_timing" ]]; then
    export AGENTIC_KV_TRACE_CONTROL_ONLY=1
  fi
  if [[ "${STUDY}" == "timing" || "${STUDY}" == "multisession" || "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" || "${STUDY}" == "multisession_overlap" || "${STUDY}" == "overlap_dose" || "${STUDY}" == "tool_cycles" ]]; then
    export AGENTIC_KV_TRACE_MAX_EXACT_INDICES="${EXACT_INDICES}"
  fi
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=0
  export AGENTIC_KV_PREPARE_CONTROL_ENABLE=1
  export AGENTIC_KV_PREPARE_CONTROL_HOST=127.0.0.1
  export AGENTIC_KV_PREPARE_CONTROL_PORT=31991
  export HICACHE_SIZE_GB MEM_FRACTION_STATIC
  bash scripts/run_sglang_hicache_server.sh "${MODEL}"
) >"${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"

deadline=$((SECONDS + 240))
until curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Backend exited; inspect ${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Backend startup timed out; inspect ${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  sleep 2
done

python3 - "${RUN_ROOT}/server.log" "${CUDA_GRAPH}" "${OVERLAP_SCHEDULE}" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
args = next((line for line in log.splitlines() if "server_args=ServerArgs(" in line), "")
for field, enabled in (("disable_cuda_graph", sys.argv[2]),
                       ("disable_overlap_schedule", sys.argv[3])):
    match = re.search(rf"\b{field}=(True|False)\b", args)
    if match is None or (match.group(1) == "False") != (enabled == "1"):
        raise SystemExit(f"Backend flag gate failed: expected {field} disabled={enabled == '0'}")
print("Work-audit backend scheduling flags verified")
PY

if [[ "${TRACE_ENABLE}" == "1" ]]; then
python3 - "${RUN_ROOT}/backend_trace.jsonl" "${TRACE_PROFILE}" "${FORWARD_TRACE}" <<'PY'
import json
import sys
from pathlib import Path
from agentic_backends.sglang.instrumentation_profiles import validate_installation

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    summaries = [row for line in handle if (row := json.loads(line)).get("event") == "trace.install.summary"]
if not summaries:
    raise SystemExit("Work-audit gate failed: missing backend hook installation summary")
result = validate_installation(sys.argv[2], "v0510", summaries[-1])
if not result["valid"]:
    raise SystemExit(f"Work-audit gate failed: {json.dumps(result['missing'])}")
if sys.argv[2] in ("kv_lifecycle_lean", "kv_decode_overlap", "tool_cycle_timing"):
    pump = "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run"
    if pump not in summaries[-1].get("installed_hooks", []):
        raise SystemExit("Work-audit gate failed: control-only scheduler pump is missing")
if sys.argv[2] in ("kv_decode_overlap", "tool_cycle_timing"):
    installed = set(summaries[-1].get("installed_hooks", []))
    required = {f"sglang.srt.managers.scheduler.Scheduler.{name}" for name in
                ("run_batch", "process_batch_result_decode")}
    if sys.argv[3] == "1":
        required.add("sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation")
    if required - installed:
        raise SystemExit(f"Work-audit gate failed: missing decode hooks {sorted(required - installed)}")
print("Work-audit lifecycle hooks installed")
PY
fi

SECOND_REPLAY_RUN_ARGS=()
SECOND_REPLAY_ANALYSIS_ARGS=()
STREAM_ARGS=()
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  docker exec "${NSYS_CONTAINER_NAME}" "${AGENTIC_NSYS_BIN}" start \
    --sample=none --cpuctxsw=none --output="${RUN_ROOT}/nsys/backend"
fi
[[ "${STUDY}" == "multisession_overlap" ]] && STREAM_ARGS+=(--capture-stream-chunks)
if [[ "${SECOND_REPLAY}" == "1" ]]; then
  SECOND_REPLAY_RUN_ARGS+=(--second-replay)
  SECOND_REPLAY_ANALYSIS_ARGS+=(--require-second-replay)
fi
if [[ "${STUDY}" == "timing" ]]; then
  python3 -m agentic_experiments.runners.run_work_audit_timing \
    --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" \
    --case-order "${CASE_ORDER}" --pairs "${PAIRS}" --warmup-pairs "${WARMUP_PAIRS}" \
    --wait-ms "${WAIT_MS}" --prompt-tokens "${PROMPT_WORDS}" \
    --max-tokens "${MAX_OUTPUT_TOKENS}" --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" \
    --eviction-rounds "${EVICTION_ROUNDS}"
elif [[ "${STUDY}" == "tool_cycles" ]]; then
  python3 -m agentic_experiments.runners.run_work_audit_tool_cycles \
    --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" \
    --research-question-id "${RESEARCH_QUESTION_ID}" --seed "${SEED}" \
    --active-count "${TOOL_CYCLE_ACTIVE_COUNT}" --donor-count "${DONOR_COUNT}" \
    --turns "${TOOL_CYCLE_TURNS}" --initial-tokens "${TOOL_CYCLE_INITIAL_TOKENS}" \
    --donor-initial-tokens "${TOOL_CYCLE_DONOR_INITIAL_TOKENS}" \
    --tool-result-words "${TOOL_CYCLE_RESULT_WORDS}" --decode-tokens "${DECODE_TOKENS}" \
    --wait-ms "${TOOL_CYCLE_WAIT_MS}"
elif [[ "${STUDY}" == "overlap_dose" ]]; then
  python3 -m agentic_experiments.runners.run_work_audit_overlap_sweep \
    --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" \
    --research-question-id "${RESEARCH_QUESTION_ID}" \
    --seed "${SEED}" --session-count "${SESSION_COUNT}" --donor-count "${DONOR_COUNT}" \
    --planned-overlap "${PLANNED_OVERLAP}" --decode-tokens "${DECODE_TOKENS}" \
    --donor-wait-ms "${DONOR_WAIT_MS}" --target-wait-ms "${SHORT_WAIT_MS}" \
    --active-prompt-tokens "${ACTIVE_PROMPT_WORDS}" --donor-prompt-tokens "${DONOR_PROMPT_WORDS}" \
    --eviction-prompt-tokens "${DONOR_PROMPT_WORDS}" \
    --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" --eviction-rounds "${EVICTION_ROUNDS}"
elif [[ "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" || "${STUDY}" == "multisession_overlap" ]]; then
  python3 -m agentic_experiments.runners.run_work_audit_multisession \
    --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" \
    --case-order "${CASE_ORDER}" --pairs "${PAIRS}" --warmup-pairs "${WARMUP_PAIRS}" \
    --early-at-ms "${EARLY_AT_MS}" --estimated-load-ms "${ESTIMATED_LOAD_MS}" \
    --load-margin-ms "${LOAD_MARGIN_MS}" \
    --short-wait-ms "${SHORT_WAIT_MS}" --long-wait-ms "${LONG_WAIT_MS}" \
    --prompt-tokens "${PROMPT_WORDS}" --max-tokens "${MAX_OUTPUT_TOKENS}" \
    --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" --eviction-rounds "${EVICTION_ROUNDS}" \
    "${STREAM_ARGS[@]}"
elif [[ "${STUDY}" == "multisession" ]]; then
  python3 -m agentic_experiments.runners.run_work_audit_multisession \
    --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" \
    --case-order "${CASE_ORDER}" \
    --short-wait-ms "${SHORT_WAIT_MS}" --long-wait-ms "${LONG_WAIT_MS}" \
    --prompt-tokens "${PROMPT_WORDS}" --max-tokens "${MAX_OUTPUT_TOKENS}" \
    --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" --eviction-rounds "${EVICTION_ROUNDS}"
else
  python3 -m agentic_experiments.runners.run_work_audit_validation \
    --run-id "${RUN_ID}" --out-dir "${RUN_ROOT}" --model "${MODEL}" --case-order "${CASE_ORDER}" \
    --wait-ms "${WAIT_MS}" --prompt-tokens "${PROMPT_WORDS}" \
    --max-tokens "${MAX_OUTPUT_TOKENS}" --minimum-host-tokens "${MINIMUM_HOST_TOKENS}" \
    --eviction-rounds "${EVICTION_ROUNDS}" \
    "${SECOND_REPLAY_RUN_ARGS[@]}"
fi
python3 - "${RUN_ROOT}/server.log" "${RUN_ROOT}/runtime/backend_features.json" "${CUDA_GRAPH}" "${OVERLAP_SCHEDULE}" <<'PY'
import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
args = next((line for line in log.splitlines() if "server_args=ServerArgs(" in line), "")
decode_lines = [line for line in log.splitlines() if "Decode batch," in line]
graph_true = sum("cuda graph: True" in line for line in decode_lines)
graph_false = sum("cuda graph: False" in line for line in decode_lines)
features = {
    "schema": "agentic_work_audit.backend_features.v1",
    "cuda_graph_requested": sys.argv[3] == "1",
    "overlap_schedule_requested": sys.argv[4] == "1",
    "cuda_graph_disabled": re.search(r"\bdisable_cuda_graph=(True|False)\b", args).group(1) == "True",
    "overlap_schedule_disabled": re.search(r"\bdisable_overlap_schedule=(True|False)\b", args).group(1) == "True",
    "decode_graph_true_log_count": graph_true,
    "decode_graph_false_log_count": graph_false,
}
Path(sys.argv[2]).write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
if features["cuda_graph_requested"] and not graph_true:
    raise SystemExit("CUDA graphs were requested but no logged decode batch used one")
print(json.dumps(features, sort_keys=True))
PY
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  docker exec "${NSYS_CONTAINER_NAME}" "${AGENTIC_NSYS_BIN}" stop
fi
if [[ "${TRACE_ENABLE}" == "1" ]]; then
  python3 -m agentic_backends.sglang.trace_contract \
    --adapter v0510 --profile "${TRACE_PROFILE}" \
    --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --out "${RUN_ROOT}/instrumentation_audit.json"
fi
if [[ "${STUDY}" == "tool_cycles" && "${TRACE_ENABLE}" == "1" ]]; then
  python3 -m agentic_experiments.runners.analyze_work_audit_tool_cycles \
    --summary "${RUN_ROOT}/summary.json" --trace "${RUN_ROOT}/backend_trace.jsonl"
elif [[ "${STUDY}" == "tool_cycles" ]]; then
  python3 - "${RUN_ROOT}/summary.json" "${RUN_ROOT}/instrumentation_audit.json" <<'PY'
import json
import sys
from pathlib import Path
summary_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["status"] = "trace_off_control"
summary["evidence_status"] = "trace_disabled"
summary["limitations"] = ["Trace disabled: no backend stage or KV-load attribution is available."]
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
audit_path.write_text(json.dumps({"profile": "disabled", "valid": True,
                                  "validation_level": "trace_off_control"}, indent=2) + "\n", encoding="utf-8")
PY
elif [[ "${STUDY}" == "overlap_dose" ]]; then
  : # The dose runner writes its own summary after the shared trace gate.
elif [[ "${STUDY}" == "multisession_overlap" ]]; then
  FORWARD_ARGS=()
  [[ "${FORWARD_TRACE}" == "1" ]] && FORWARD_ARGS+=(--require-model-forward)
  python3 -m agentic_experiments.runners.analyze_work_audit_decode_overlap \
    --run-id "${RUN_ID}" --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --harness "${RUN_ROOT}/harness_events.jsonl" \
    --case-results "${RUN_ROOT}/case_results.json" --out-dir "${RUN_ROOT}" \
    "${FORWARD_ARGS[@]}"
elif [[ "${STUDY}" == "timing" ]]; then
  SLOT_PROOF_ARGS=()
  if [[ "${REQUIRE_SLOT_PROOF}" == "1" ]]; then
    SLOT_PROOF_ARGS+=(--require-slot-proof)
  fi
  python3 -m agentic_experiments.runners.analyze_work_audit_timing \
    --run-id "${RUN_ID}" --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --harness "${RUN_ROOT}/harness_events.jsonl" \
    --case-results "${RUN_ROOT}/case_results.json" --out-dir "${RUN_ROOT}" \
    "${SLOT_PROOF_ARGS[@]}"
elif [[ "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" ]]; then
  python3 -m agentic_experiments.runners.analyze_work_audit_multisession \
    --run-id "${RUN_ID}" --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --harness "${RUN_ROOT}/harness_events.jsonl" \
    --case-results "${RUN_ROOT}/case_results.json" --out-dir "${RUN_ROOT}"
elif [[ "${STUDY}" == "multisession" ]]; then
  python3 -m agentic_experiments.runners.analyze_work_audit_multisession \
    --run-id "${RUN_ID}" --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --harness "${RUN_ROOT}/harness_events.jsonl" --out-dir "${RUN_ROOT}"
else
  python3 -m agentic_experiments.runners.analyze_work_audit_validation \
    --run-id "${RUN_ID}" --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --harness "${RUN_ROOT}/harness_events.jsonl" --out-dir "${RUN_ROOT}" \
    "${SECOND_REPLAY_ANALYSIS_ARGS[@]}"
  python3 -m agentic_reports.audits.build_work_audit_block_audit \
    --trace "${RUN_ROOT}/backend_trace.jsonl" \
    --harness "${RUN_ROOT}/harness_events.jsonl" \
    --summary "${RUN_ROOT}/summary.json" --out "${RUN_ROOT}/block_audit.json"
fi
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  docker exec "${NSYS_CONTAINER_NAME}" python3 -c '
import os
import signal
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    try:
        args = open(f"{entry.path}/cmdline", "rb").read().split(b"\0")
    except (OSError, PermissionError):
        continue
    if len(args) >= 3 and args[0].endswith(b"python3") and args[1:3] == [b"-m", b"sglang.launch_server"]:
        os.kill(int(entry.name), signal.SIGTERM)
        break
else:
    raise SystemExit("Nsight shutdown failed: SGLang launcher not found")
'
  for _ in {1..45}; do
    [[ "$(docker inspect -f '{{.State.Running}}' "${NSYS_CONTAINER_NAME}" 2>/dev/null)" == "true" ]] || break
    sleep 1
  done
  if [[ "$(docker inspect -f '{{.State.Running}}' "${NSYS_CONTAINER_NAME}" 2>/dev/null)" == "true" ]]; then
    docker stop --time 20 "${NSYS_CONTAINER_NAME}" >/dev/null
  fi
  wait "${SERVER_PID}" || true
  SERVER_PID=""
  [[ -s "${RUN_ROOT}/nsys/backend.nsys-rep" ]] || {
    echo "Nsight capture failed: no backend.nsys-rep" >&2; exit 1;
  }
  nsys export --type sqlite --force-overwrite=true \
    -o "${RUN_ROOT}/nsys/backend.sqlite" "${RUN_ROOT}/nsys/backend.nsys-rep" >/dev/null
  if [[ "${STUDY}" == "overlap_dose" ]]; then
    python3 -m agentic_experiments.runners.analyze_work_audit_physical_overlap \
      --sqlite "${RUN_ROOT}/nsys/backend.sqlite" \
      --summary "${RUN_ROOT}/summary.json" --trace "${RUN_ROOT}/backend_trace.jsonl" \
      --out "${RUN_ROOT}/nsys/physical_overlap.json"
    python3 -m agentic_experiments.runners.analyze_work_audit_decode_submission \
      --sqlite "${RUN_ROOT}/nsys/backend.sqlite" \
      --summary "${RUN_ROOT}/summary.json" --trace "${RUN_ROOT}/backend_trace.jsonl" \
      --out "${RUN_ROOT}/nsys/decode_submission.json"
  else
    python3 -m agentic_experiments.runners.analyze_work_audit_cuda_kernels \
      --sqlite "${RUN_ROOT}/nsys/backend.sqlite" \
      --summary "${RUN_ROOT}/summary.json" \
      --out "${RUN_ROOT}/nsys/kernel_attribution.json"
  fi
fi
HARDWARE_PROFILE="$(python3 - "${PROFILE_PATH}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["hardware_profile"])
PY
)"
CASE_ORDER_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1].split("-")))' "${CASE_ORDER}")"
if [[ "${STUDY}" == "tool_cycles" ]]; then
  WORKLOAD_JSON="{\"research_question_id\":\"${RESEARCH_QUESTION_ID}\",\"frontend_priority\":\"none\",\"purpose\":\"tool_cycles\",\"seed\":${SEED},\"active_count\":${TOOL_CYCLE_ACTIVE_COUNT},\"donor_count\":${DONOR_COUNT},\"turn_count\":${TOOL_CYCLE_TURNS},\"initial_tokens\":${TOOL_CYCLE_INITIAL_TOKENS},\"donor_initial_tokens\":${TOOL_CYCLE_DONOR_INITIAL_TOKENS},\"tool_result_words\":${TOOL_CYCLE_RESULT_WORDS},\"wait_ms\":${TOOL_CYCLE_WAIT_MS},\"decode_tokens\":${DECODE_TOKENS},\"cuda_graph_requested\":${CUDA_GRAPH},\"overlap_schedule_requested\":${OVERLAP_SCHEDULE},\"trace_enabled\":${TRACE_ENABLE},\"hicache_size_gb\":${HICACHE_SIZE_GB},\"mem_fraction_static\":${MEM_FRACTION_STATIC}}"
elif [[ "${STUDY}" == "overlap_dose" ]]; then
  WORKLOAD_JSON="{\"research_question_id\":\"${RESEARCH_QUESTION_ID}\",\"pair_id\":\"${PAIR_ID}\",\"frontend_priority\":\"none\",\"purpose\":\"overlap_dose\",\"load_execution\":\"worker\",\"session_count\":${SESSION_COUNT},\"donor_count\":${DONOR_COUNT},\"planned_overlap\":${PLANNED_OVERLAP},\"seed\":${SEED},\"decode_tokens\":${DECODE_TOKENS},\"active_prompt_words\":${ACTIVE_PROMPT_WORDS},\"donor_prompt_words\":${DONOR_PROMPT_WORDS},\"target_wait_ms\":${SHORT_WAIT_MS},\"donor_wait_ms\":${DONOR_WAIT_MS},\"forward_trace_enabled\":${FORWARD_TRACE},\"nsys_enabled\":${NSYS_ENABLE},\"cuda_graph_requested\":${CUDA_GRAPH},\"overlap_schedule_requested\":${OVERLAP_SCHEDULE},\"hicache_size_gb\":${HICACHE_SIZE_GB},\"mem_fraction_static\":${MEM_FRACTION_STATIC}}"
elif [[ "${STUDY}" == "multisession_compare" || "${STUDY}" == "multisession_window" || "${STUDY}" == "multisession_controller" || "${STUDY}" == "multisession_overlap" ]]; then
  WORKLOAD_JSON="{\"cases\":${CASE_ORDER_JSON},\"research_question_id\":\"${RESEARCH_QUESTION_ID}\",\"frontend_priority\":\"none\",\"purpose\":\"${STUDY}\",\"load_execution\":\"${LOAD_EXECUTION}\",\"forward_trace_enabled\":${FORWARD_TRACE},\"nsys_enabled\":${NSYS_ENABLE},\"session_count\":3,\"capacity_policy\":\"explicit_two_prefix_budget\",\"short_wait_ms\":${SHORT_WAIT_MS},\"long_wait_ms\":${LONG_WAIT_MS},\"early_at_ms\":${EARLY_AT_MS},\"estimated_load_ms\":${ESTIMATED_LOAD_MS},\"load_margin_ms\":${LOAD_MARGIN_MS},\"pairs\":${PAIRS},\"warmup_pairs\":${WARMUP_PAIRS},\"prompt_words_target\":${PROMPT_WORDS},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"minimum_host_tokens\":${MINIMUM_HOST_TOKENS},\"eviction_rounds\":${EVICTION_ROUNDS},\"hicache_size_gb\":${HICACHE_SIZE_GB},\"mem_fraction_static\":${MEM_FRACTION_STATIC}}"
elif [[ "${STUDY}" == "multisession" ]]; then
  WORKLOAD_JSON="{\"cases\":${CASE_ORDER_JSON},\"research_question_id\":\"${RESEARCH_QUESTION_ID}\",\"frontend_priority\":\"none\",\"purpose\":\"multisession\",\"load_execution\":\"${LOAD_EXECUTION}\",\"session_count\":3,\"capacity_policy\":\"explicit_two_prefix_budget\",\"short_wait_ms\":${SHORT_WAIT_MS},\"long_wait_ms\":${LONG_WAIT_MS},\"prompt_words_target\":${PROMPT_WORDS},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"minimum_host_tokens\":${MINIMUM_HOST_TOKENS},\"eviction_rounds\":${EVICTION_ROUNDS},\"hicache_size_gb\":${HICACHE_SIZE_GB},\"mem_fraction_static\":${MEM_FRACTION_STATIC}}"
else
  WORKLOAD_JSON="{\"cases\":${CASE_ORDER_JSON},\"research_question_id\":\"${RESEARCH_QUESTION_ID}\",\"frontend_priority\":\"none\",\"purpose\":\"${STUDY}\",\"load_execution\":\"${LOAD_EXECUTION}\",\"replays_per_case\":$( [[ "${STUDY}" == "timing" ]] && echo 2 || echo $((SECOND_REPLAY + 1)) ),\"pairs\":$( [[ "${STUDY}" == "timing" ]] && echo "${PAIRS}" || echo 1 ),\"warmup_pairs\":$( [[ "${STUDY}" == "timing" ]] && echo "${WARMUP_PAIRS}" || echo 0 ),\"tool_wait_ms\":${WAIT_MS},\"prompt_words_target\":${PROMPT_WORDS},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"minimum_host_tokens\":${MINIMUM_HOST_TOKENS},\"eviction_rounds\":${EVICTION_ROUNDS},\"hicache_size_gb\":${HICACHE_SIZE_GB},\"mem_fraction_static\":${MEM_FRACTION_STATIC},\"exact_trace_indices\":$( [[ "${STUDY}" == "timing" ]] && echo "${EXACT_INDICES}" || echo 256 ),\"slot_proof_required\":$( [[ "${STUDY}" == "timing" ]] && [[ "${REQUIRE_SLOT_PROOF}" == "1" ]] && echo true || echo false )}"
fi
MANIFEST_ARTIFACTS=(--artifact "instrumentation_audit=${RUN_ROOT}/instrumentation_audit.json"
  --artifact "summary=${RUN_ROOT}/summary.json"
  --artifact "backend_features=${RUN_ROOT}/runtime/backend_features.json")
[[ "${TRACE_ENABLE}" == "1" ]] && MANIFEST_ARTIFACTS+=(--artifact "report=${RUN_ROOT}/report.html")
MANIFEST_INSTRUMENTATION=(--instrumentation "v0510_backend_trace"
  --instrumentation "work_audit_event_join"
  --instrumentation "logical_block_lifecycle_reuse"
  --instrumentation "${TRACE_PROFILE}")
if [[ "${TRACE_ENABLE}" == "0" ]]; then
  MANIFEST_INSTRUMENTATION=(--instrumentation "trace_disabled_control")
fi
if [[ "${NSYS_ENABLE}" == "1" ]]; then
  if [[ "${STUDY}" == "overlap_dose" ]]; then
    MANIFEST_ARTIFACTS+=(--artifact "physical_overlap=${RUN_ROOT}/nsys/physical_overlap.json")
    MANIFEST_ARTIFACTS+=(--artifact "decode_submission=${RUN_ROOT}/nsys/decode_submission.json")
  else
    MANIFEST_ARTIFACTS+=(--artifact "kernel_attribution=${RUN_ROOT}/nsys/kernel_attribution.json")
  fi
fi
if [[ "${STUDY}" == "validation" ]]; then
  MANIFEST_ARTIFACTS+=(--artifact "block_audit=${RUN_ROOT}/block_audit.json")
fi
python3 "${ROOT}/scripts/create_run_manifest.py" \
  --out "${RUN_ROOT}/run_manifest.json" --run-id "${RUN_ID}" \
  --experiment "agentic_work_audit_${STUDY}" --model "${MODEL}" \
  --hardware-profile "${HARDWARE_PROFILE}" \
  --runtime-contract "${BACKEND_RUNTIME_CONTRACT_OUT}" \
  --workload-json "${WORKLOAD_JSON}" \
  "${MANIFEST_INSTRUMENTATION[@]}" \
  "${MANIFEST_ARTIFACTS[@]}" \
  --completion-status complete
if [[ "${TRACE_ENABLE}" == "1" ]]; then
  python3 -m agentic_reports.builders.build_work_audit_report \
    --results-dir "${RESULTS_BASE}" --out "${RUN_ROOT}/report.html"
fi
echo "Validated: ${RUN_ROOT}/summary.json"
