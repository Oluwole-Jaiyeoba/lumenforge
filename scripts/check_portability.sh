#!/usr/bin/env bash
# Run every check that does not need SGLang or a GPU.
# Usage: bash scripts/check_portability.sh        (after scripts/install_workspace.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi
cd "${REPO_ROOT}"

run() {
  echo "==> $*"
  "$@"
}

run "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests/architecture
run "${PYTHON_BIN}" scripts/build_hardware_experiments.py --check
for package in agentic-core agentic-backend-api agentic-controller agentic-harnesses agentic-gateway agentic-backend-sglang agentic-experiments agentic-reports agentic-hardware-probes; do
  if [[ -d "packages/${package}/tests" ]]; then
    (cd "packages/${package}" && run "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests)
  fi
done
(cd packages/agentic-harness-scenarios && run "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests)
(cd sglang_direct_kv && run "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests)
echo "All portability checks passed."
