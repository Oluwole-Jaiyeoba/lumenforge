"""Exercise the portable controller decision and boundary-action path.

This is deliberately a host-side probe: controller actions in these portable
experiments are acknowledged at the gateway/request boundary, not by mutating
private SGLang scheduler state in place.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentic_backend_api import BackendCapabilities
from agentic_backends.sglang.adapters import GatewayPriorityBackendAdapter
from agentic_controller import ControllerPolicy, ControllerStateStore, PolicyConfig
from agentic_core import ControllerEvent, EventType


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    event_id = f"instrumentation-controller-probe-{time.time_ns()}"
    store = ControllerStateStore()
    backend = GatewayPriorityBackendAdapter(
        BackendCapabilities(priority_queue=True, live_metrics=True, observe_only=False, backend_name=args.mode)
    )
    policy = ControllerPolicy(PolicyConfig(observe_only=False))
    started = ControllerEvent(
        event_id=f"{event_id}-started",
        event=EventType.TOOL_STARTED,
        session_id="instrumentation-sentinel",
        prefix_id="instrumentation-sentinel-prefix",
        session_generation=1,
        monotonic_ms=0,
        expected_completion_ms=500,
        deadline_after_completion_ms=25,
        execution_priority=100,
    )
    store.apply_event(started)
    event = ControllerEvent(
        event_id=f"{event_id}-ready",
        event=EventType.TOOL_COMPLETED,
        session_id="instrumentation-sentinel",
        prefix_id="instrumentation-sentinel-prefix",
        session_generation=1,
        monotonic_ms=500,
        execution_priority=100,
    )
    state, accepted = store.apply_event(event)
    decision = policy.plan(state, backend.capabilities(), now_ms=0)
    results = [backend.apply(command).to_dict() for command in decision.commands]
    rows = [
        {
            "event": "m27.controller.preflight.decision",
            "ts_ns": time.time_ns(),
            "mode": args.mode,
            "controller_event_accepted": accepted,
            "controller_decision": decision.to_dict(),
        },
        {
            "event": "m27.controller.preflight.action_ack",
            "ts_ns": time.time_ns(),
            "mode": args.mode,
            "controller_backend_results": results,
            "acted_count": sum(bool(result.get("acted")) for result in results),
        },
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    if not accepted or not results or not any(result.get("accepted") for result in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
