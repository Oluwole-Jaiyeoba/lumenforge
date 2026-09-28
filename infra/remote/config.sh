#!/usr/bin/env bash

# Shared, provider-neutral remote-host configuration.
#
# Keep machine-specific credentials and addresses outside this repository.
# The default local location is ~/.config/agentic_hardware/remote.env. Its
# values may also be supplied directly as environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_NAME="$(basename "${REPO_ROOT}")"

LOCAL_CONFIG="${AGENTIC_HW_REMOTE_CONFIG:-${HOME}/.config/agentic_hardware/remote.env}"
# Capture explicit process-level overrides before sourcing the optional local
# file. The local file is a default, while a caller-supplied value must win.
EXPLICIT_SSH_KEY="${AGENTIC_HW_SSH_KEY:-}"
EXPLICIT_REMOTE_USER="${AGENTIC_HW_REMOTE_USER:-}"
EXPLICIT_REMOTE_DIR="${AGENTIC_HW_REMOTE_DIR:-}"
EXPLICIT_REMOTE_HOSTS="${AGENTIC_HW_REMOTE_HOSTS:-}"
if [[ -f "${LOCAL_CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_CONFIG}"
fi

[[ -n "${EXPLICIT_SSH_KEY}" ]] && AGENTIC_HW_SSH_KEY="${EXPLICIT_SSH_KEY}"
[[ -n "${EXPLICIT_REMOTE_USER}" ]] && AGENTIC_HW_REMOTE_USER="${EXPLICIT_REMOTE_USER}"
[[ -n "${EXPLICIT_REMOTE_DIR}" ]] && AGENTIC_HW_REMOTE_DIR="${EXPLICIT_REMOTE_DIR}"
[[ -n "${EXPLICIT_REMOTE_HOSTS}" ]] && AGENTIC_HW_REMOTE_HOSTS="${EXPLICIT_REMOTE_HOSTS}"

SSH_KEY="${AGENTIC_HW_SSH_KEY:-}"
REMOTE_USER="${AGENTIC_HW_REMOTE_USER:-}"
REMOTE_PROJECT_DIR="${AGENTIC_HW_REMOTE_DIR:-}"

if [[ -n "${AGENTIC_HW_REMOTE_HOSTS:-}" ]]; then
  read -r -a REMOTE_HOSTS <<< "${AGENTIC_HW_REMOTE_HOSTS}"
else
  REMOTE_HOSTS=()
fi

if [[ -z "${REMOTE_PROJECT_DIR}" && -n "${REMOTE_USER}" ]]; then
  REMOTE_PROJECT_DIR="/home/${REMOTE_USER}/${REPO_NAME}"
fi

REMOTE_LABELS=()
for i in "${!REMOTE_HOSTS[@]}"; do
  REMOTE_LABELS+=("host-${i}")
done

ssh_opts() {
  local control_name="$1"
  local key_args=()
  if [[ -n "${SSH_KEY}" ]]; then
    key_args=(-i "${SSH_KEY}")
  fi
  printf '%q ' \
    "${key_args[@]}" \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o "ControlPath=/tmp/agentic-hardware-${control_name}-%r@%h:%p" \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3
}

validate_remote_config() {
  if [[ -z "${REMOTE_USER}" ]]; then
    echo "Set AGENTIC_HW_REMOTE_USER in ${LOCAL_CONFIG} or the environment." >&2
    return 1
  fi
  if [[ -z "${REMOTE_PROJECT_DIR}" ]]; then
    echo "Set AGENTIC_HW_REMOTE_DIR in ${LOCAL_CONFIG} or the environment." >&2
    return 1
  fi
}

validate_host_index() {
  local index="$1"
  validate_remote_config || return 1
  if ! [[ "${index}" =~ ^[0-9]+$ ]]; then
    echo "Host index must be numeric: ${index}" >&2
    return 1
  fi
  if (( index < 0 || index >= ${#REMOTE_HOSTS[@]} )); then
    echo "Host index out of range: ${index}" >&2
    echo "Valid indices: 0..$(( ${#REMOTE_HOSTS[@]} - 1 ))" >&2
    return 1
  fi
  if [[ -z "${REMOTE_HOSTS[$index]}" ]]; then
    echo "Host index ${index} has no address configured." >&2
    return 1
  fi
}
