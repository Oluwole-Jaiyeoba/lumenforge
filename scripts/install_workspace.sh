#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SGLANG_TESTBED="${INSTALL_SGLANG_TESTBED:-1}"

package_paths=(
  "packages/agentic-core"
  "packages/agentic-backend-api"
  "packages/agentic-controller"
  "packages/agentic-harnesses"
  "packages/agentic-prompt-codec"
  "packages/agentic-gateway"
  "packages/agentic-backend-sglang"
  "packages/agentic-harness-scenarios"
  "packages/agentic-hardware-probes"
  "packages/agentic-experiments"
  "packages/agentic-reports"
)

for package_path in "${package_paths[@]}"; do
  "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}/${package_path}"
done

if [[ "${INSTALL_SGLANG_TESTBED}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}/sglang_direct_kv"
fi

echo "Installed agentic workspace packages from ${REPO_ROOT}."
