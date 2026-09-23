#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

if [[ "${AGENTIC_SGLANG_PREFLIGHT:-1}" == "1" ]]; then
  # shellcheck disable=SC2086
  python "$(dirname "${BASH_SOURCE[0]}")/sglang_preflight.py" -- --trust-remote-code ${EXTRA_SERVER_ARGS} \
    || [[ "${AGENTIC_SGLANG_STRICT:-0}" != "1" ]]
fi

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  ${EXTRA_SERVER_ARGS}
