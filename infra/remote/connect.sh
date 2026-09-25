#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./infra/remote/connect.sh <idx>
  ./infra/remote/connect.sh <idx> '<command>'

Configure servers by editing infra/remote/config.sh or exporting:
  AGENTIC_HW_REMOTE_HOSTS="1.2.3.4"
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

INDEX="$1"
validate_host_index "${INDEX}"
if [[ -n "${SSH_KEY}" ]]; then
  chmod 400 "${SSH_KEY}"
fi

ip="${REMOTE_HOSTS[$INDEX]}"
remote_host="${REMOTE_USER}@${ip}"

echo "Connecting to ${remote_host} ..."
if [[ $# -gt 1 ]]; then
  shift
  ssh $(ssh_opts shell) "${remote_host}" "$@"
else
  ssh $(ssh_opts shell) "${remote_host}"
fi
