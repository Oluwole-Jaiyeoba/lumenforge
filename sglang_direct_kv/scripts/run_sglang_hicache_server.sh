#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-14}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"
HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"
HICACHE_STORAGE_PREFETCH_POLICY="${HICACHE_STORAGE_PREFETCH_POLICY:-}"
HICACHE_STORAGE_BACKEND_EXTRA_CONFIG="${HICACHE_STORAGE_BACKEND_EXTRA_CONFIG:-}"
HICACHE_STORAGE_PATH="${HICACHE_STORAGE_PATH:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
CUDA_GRAPH_FLAG="${CUDA_GRAPH_FLAG:---disable-cuda-graph}"
OVERLAP_FLAG="${OVERLAP_FLAG:---disable-overlap-schedule}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-triton}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"
AGENTIC_KV_TRACE_ENABLE="${AGENTIC_KV_TRACE_ENABLE:-1}"
AGENTIC_KV_TRACE_PATH="${AGENTIC_KV_TRACE_PATH:-artifacts/kv_movement_trace.jsonl}"
AGENTIC_KV_TRACE_INSTALL_REPORT_PATH="${AGENTIC_KV_TRACE_INSTALL_REPORT_PATH:-}"
AGENTIC_KV_COPY_TELEMETRY_ENABLE="${AGENTIC_KV_COPY_TELEMETRY_ENABLE:-0}"
AGENTIC_KV_COPY_TELEMETRY_PATH="${AGENTIC_KV_COPY_TELEMETRY_PATH:-}"
AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE="${AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE:-0}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
SGLANG_DOCKER_PULL="${SGLANG_DOCKER_PULL:-0}"
SGLANG_DOCKER_GPU_ARGS="${SGLANG_DOCKER_GPU_ARGS:---gpus all}"
SGLANG_DOCKER_EXTRA_ARGS="${SGLANG_DOCKER_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AGENTIC_RUNTIME_TELEMETRY="${AGENTIC_RUNTIME_TELEMETRY:-0}"
AGENTIC_RUNTIME_TELEMETRY_PATH="${AGENTIC_RUNTIME_TELEMETRY_PATH:-}"
AGENTIC_RUNTIME_TELEMETRY_BACKEND="${AGENTIC_RUNTIME_TELEMETRY_BACKEND:-sglang}"
AGENTIC_KV_TORCH_PROFILER_ENABLE="${AGENTIC_KV_TORCH_PROFILER_ENABLE:-0}"
AGENTIC_KV_TORCH_PROFILER_DIR="${AGENTIC_KV_TORCH_PROFILER_DIR:-}"
AGENTIC_KV_TORCH_PROFILER_START_EVENTS="${AGENTIC_KV_TORCH_PROFILER_START_EVENTS:-}"
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS="${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:-0}"
AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY="${AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY:-1}"

mkdir -p "$(dirname "${AGENTIC_KV_TRACE_PATH}")"
if [[ -n "${AGENTIC_KV_TRACE_INSTALL_REPORT_PATH}" ]]; then
  mkdir -p "$(dirname "${AGENTIC_KV_TRACE_INSTALL_REPORT_PATH}")"
fi
if [[ -n "${AGENTIC_KV_COPY_TELEMETRY_PATH}" ]]; then
  mkdir -p "$(dirname "${AGENTIC_KV_COPY_TELEMETRY_PATH}")"
fi
if [[ -n "${AGENTIC_RUNTIME_TELEMETRY_PATH}" ]]; then
  mkdir -p "$(dirname "${AGENTIC_RUNTIME_TELEMETRY_PATH}")"
fi
if [[ -n "${AGENTIC_KV_TORCH_PROFILER_DIR}" ]]; then
  mkdir -p "${AGENTIC_KV_TORCH_PROFILER_DIR}"
fi
if [[ -n "${HICACHE_STORAGE_PATH}" ]]; then
  mkdir -p "${HICACHE_STORAGE_PATH}"
fi
export AGENTIC_KV_TRACE_ENABLE
export AGENTIC_KV_TRACE_PATH
export AGENTIC_KV_TRACE_INSTALL_REPORT_PATH
export AGENTIC_KV_COPY_TELEMETRY_ENABLE
export AGENTIC_KV_COPY_TELEMETRY_PATH
export AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE
export AGENTIC_RUNTIME_TELEMETRY
export AGENTIC_RUNTIME_TELEMETRY_PATH
export AGENTIC_RUNTIME_TELEMETRY_BACKEND
export AGENTIC_KV_TORCH_PROFILER_ENABLE
export AGENTIC_KV_TORCH_PROFILER_DIR
export AGENTIC_KV_TORCH_PROFILER_START_EVENTS
export AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS
export AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY
if [[ "${HICACHE_STORAGE_BACKEND}" == "file" && -n "${HICACHE_STORAGE_PATH}" ]]; then
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${HICACHE_STORAGE_PATH}"
fi
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN_DIR="$(dirname "$(command -v nvcc)")"
  export CUDA_HOME="${CUDA_HOME:-$(cd "${CUDA_BIN_DIR}/.." && pwd)}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

launch_args=(
  "${PYTHON_BIN}" -m sglang.launch_server
  --model-path "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --enable-hierarchical-cache
  --hicache-size "${HICACHE_SIZE_GB}"
  --hicache-io-backend "${HICACHE_IO_BACKEND}"
  --hicache-mem-layout "${HICACHE_MEM_LAYOUT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --attention-backend "${ATTENTION_BACKEND}"
  --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}"
  --decode-attention-backend "${DECODE_ATTENTION_BACKEND}"
)

if [[ -n "${HICACHE_STORAGE_BACKEND}" ]]; then
  launch_args+=(--hicache-storage-backend "${HICACHE_STORAGE_BACKEND}")
fi
if [[ "${HICACHE_STORAGE_BACKEND}" == "file" && -n "${HICACHE_STORAGE_PATH}" ]]; then
  launch_args+=(--file-storage-path "${HICACHE_STORAGE_PATH}")
fi
if [[ -n "${HICACHE_STORAGE_BACKEND}" && -n "${HICACHE_STORAGE_PREFETCH_POLICY}" ]]; then
  launch_args+=(--hicache-storage-prefetch-policy "${HICACHE_STORAGE_PREFETCH_POLICY}")
fi
if [[ -n "${HICACHE_STORAGE_BACKEND}" && -n "${HICACHE_STORAGE_BACKEND_EXTRA_CONFIG}" ]]; then
  launch_args+=(--hicache-storage-backend-extra-config "${HICACHE_STORAGE_BACKEND_EXTRA_CONFIG}")
fi

if [[ -n "${CUDA_GRAPH_FLAG}" ]]; then
  # shellcheck disable=SC2206
  launch_args+=( ${CUDA_GRAPH_FLAG} )
fi
if [[ -n "${OVERLAP_FLAG}" ]]; then
  # shellcheck disable=SC2206
  launch_args+=( ${OVERLAP_FLAG} )
fi
if [[ -n "${EXTRA_SERVER_ARGS}" ]]; then
  # shellcheck disable=SC2206
  launch_args+=( ${EXTRA_SERVER_ARGS} )
fi

# Static check that the installed SGLang still accepts every flag above
# (warning only unless AGENTIC_SGLANG_STRICT=1). Skipped for Docker launches,
# where the image's SGLang is not the host's.
if [[ -z "${SGLANG_DOCKER_IMAGE}" && "${AGENTIC_SGLANG_PREFLIGHT:-1}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/sglang_preflight.py -- "${launch_args[@]:3}" || [[ "${AGENTIC_SGLANG_STRICT:-0}" != "1" ]]
fi

if [[ -n "${SGLANG_DOCKER_IMAGE}" ]]; then
  if [[ "${SGLANG_DOCKER_PULL}" == "1" ]]; then
    docker pull "${SGLANG_DOCKER_IMAGE}"
  fi
  docker_mount_args=(-v "$(pwd):$(pwd)")
  # SGLang adapter/trace code lives in <repo>/packages (agentic-backend-sglang);
  # mount it at the same absolute path so agentic_kv's sys.path fallback finds it.
  if [[ -d "$(pwd)/../packages" ]]; then
    packages_dir="$(cd "$(pwd)/../packages" && pwd)"
    docker_mount_args+=(-v "${packages_dir}:${packages_dir}:ro")
  fi
  if [[ -n "${HICACHE_STORAGE_PATH}" ]]; then
    docker_mount_args+=(-v "${HICACHE_STORAGE_PATH}:${HICACHE_STORAGE_PATH}")
  fi
  # The host runner may select a virtual-environment interpreter via PYTHON_BIN.
  # That path/name is not meaningful inside the backend image.
  docker_launch_args=("${launch_args[@]}")
  docker_launch_args[0]="python3"
  echo "Launching SGLang in Docker image: ${SGLANG_DOCKER_IMAGE}"
  exec docker run --rm \
    ${SGLANG_DOCKER_GPU_ARGS} \
    --network host \
    --ipc host \
    --shm-size 16g \
    "${docker_mount_args[@]}" \
    -w "$(pwd)" \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e AGENTIC_KV_TRACE_ENABLE \
    -e AGENTIC_KV_TRACE_PATH \
    -e AGENTIC_KV_TRACE_INSTALL_REPORT_PATH \
    -e AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-0}" \
    -e AGENTIC_KV_TRACE_SCHEDULER_INGRESS_ONLY="${AGENTIC_KV_TRACE_SCHEDULER_INGRESS_ONLY:-0}" \
    -e AGENTIC_KV_TRACE_CONTROL_ONLY="${AGENTIC_KV_TRACE_CONTROL_ONLY:-0}" \
    -e AGENTIC_KV_TRACE_MAX_EXACT_INDICES="${AGENTIC_KV_TRACE_MAX_EXACT_INDICES:-256}" \
    -e AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-0}" \
    -e AGENTIC_KV_PREPARE_CONTROL_ENABLE="${AGENTIC_KV_PREPARE_CONTROL_ENABLE:-0}" \
    -e AGENTIC_KV_PREPARE_CONTROL_HOST="${AGENTIC_KV_PREPARE_CONTROL_HOST:-127.0.0.1}" \
    -e AGENTIC_KV_PREPARE_CONTROL_PORT="${AGENTIC_KV_PREPARE_CONTROL_PORT:-31991}" \
    -e AGENTIC_KV_PREPARE_WAIT_TIMEOUT_MS="${AGENTIC_KV_PREPARE_WAIT_TIMEOUT_MS:-5000}" \
    -e AGENTIC_KV_COPY_TELEMETRY_ENABLE \
    -e AGENTIC_KV_COPY_TELEMETRY_PATH \
    -e AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE \
    -e AGENTIC_RUNTIME_TELEMETRY \
    -e AGENTIC_RUNTIME_TELEMETRY_PATH \
    -e AGENTIC_RUNTIME_TELEMETRY_BACKEND \
    -e AGENTIC_KV_TORCH_PROFILER_ENABLE \
    -e AGENTIC_KV_TORCH_PROFILER_DIR \
    -e AGENTIC_KV_TORCH_PROFILER_START_EVENTS \
    -e AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS \
    -e AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY \
    -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-}" \
    -e PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    ${SGLANG_DOCKER_EXTRA_ARGS} \
    "${SGLANG_DOCKER_IMAGE}" \
    sh -c 'umask 000; exec "$@"' -- \
    "${docker_launch_args[@]}"
fi

exec "${launch_args[@]}"
