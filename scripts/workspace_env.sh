#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

workspace_src_paths=(
  "${REPO_ROOT}/packages/agentic-core/src"
  "${REPO_ROOT}/packages/agentic-backend-api/src"
  "${REPO_ROOT}/packages/agentic-controller/src"
  "${REPO_ROOT}/packages/agentic-harnesses/src"
  "${REPO_ROOT}/packages/agentic-prompt-codec/src"
  "${REPO_ROOT}/sglang_direct_kv/src"
)

joined_paths="$(IFS=:; echo "${workspace_src_paths[*]}")"
export PYTHONPATH="${joined_paths}${PYTHONPATH:+:${PYTHONPATH}}"
