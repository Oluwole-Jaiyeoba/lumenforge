#!/usr/bin/env bash
set -euo pipefail

# remote NVIDIA A10G host-scale pressure on NVIDIA GH200. Use this for apples-to-apples comparison with
# the remote NVIDIA A10G host/A10G-class run before scaling pressure upward.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-nvidia_a10g_24gb}"
export SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline harness_emitted frontend_supplied gateway_injected}"
export HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
export REPORT_LABEL="${REPORT_LABEL:-gh200_apples_to_apples_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_host_signal_design_space.sh"
