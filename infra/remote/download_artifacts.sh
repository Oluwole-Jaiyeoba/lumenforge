#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./infra/remote/download_artifacts.sh        Download artifacts from host index 0
  ./infra/remote/download_artifacts.sh <idx>  Download artifacts from host index <idx>

Downloads:
  remote sglang_direct_kv/artifacts/ -> local sglang_direct_kv/artifacts/
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 1
fi

INDEX="${1:-0}"
validate_host_index "${INDEX}"

if [[ -n "${SSH_KEY}" ]]; then
  chmod 400 "${SSH_KEY}"
fi

ip="${REMOTE_HOSTS[$INDEX]}"
label="${REMOTE_LABELS[$INDEX]}"
remote_host="${REMOTE_USER}@${ip}"

REMOTE_ARTIFACTS_DIR="${REMOTE_PROJECT_DIR}/sglang_direct_kv/artifacts"
LOCAL_ARTIFACTS_DIR="${REPO_ROOT}/sglang_direct_kv/artifacts"
mkdir -p "${LOCAL_ARTIFACTS_DIR}"

SSH_CMD="ssh $(ssh_opts download)"

RSYNC_COMMON_OPTS=(
  -az
  --human-readable
  --itemize-changes
  --stats
  --omit-dir-times
  --no-perms
  --no-owner
  --no-group
)

echo "==== Downloading artifacts from ${label} (${ip}) ===="
echo "Remote source: ${REMOTE_ARTIFACTS_DIR}/"
echo "Local dest:    ${LOCAL_ARTIFACTS_DIR}/"

if ssh $(ssh_opts download) "${remote_host}" "test -d '${REMOTE_ARTIFACTS_DIR}'"; then
  rsync \
    "${RSYNC_COMMON_OPTS[@]}" \
    -e "${SSH_CMD}" \
    "${remote_host}:${REMOTE_ARTIFACTS_DIR}/" \
    "${LOCAL_ARTIFACTS_DIR}/"
  echo "Download complete."
else
  echo "Remote artifacts directory not found; nothing to download:" >&2
  echo "  ${REMOTE_ARTIFACTS_DIR}" >&2
fi
