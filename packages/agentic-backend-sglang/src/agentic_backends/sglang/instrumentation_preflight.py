"""Contract-driven, fail-loud validation for controller instrumentation."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _matches_any(values: list[str], patterns: list[str]) -> bool:
    return not patterns or any(value.startswith(pattern) for value in values for pattern in patterns)


def _check_environment(entry: dict[str, Any]) -> dict[str, Any]:
    required = entry.get("environment", {})
    observed = {name: os.environ.get(name) for name in required}
    passed = all(str(observed[name]) == str(expected) for name, expected in required.items())
    return {"id": entry["id"], "kind": "environment", "passed": passed, "expected": required, "observed": observed}


def _check_hooks(entry: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    installed = report.get("installed_hooks", [])
    alternatives = entry.get("installed_hooks_any", [])
    found = [hook for hook in installed if hook in alternatives]
    return {"id": entry["id"], "kind": "installation", "passed": bool(found), "expected_any": alternatives, "observed": found}


def _check_events(entry: dict[str, Any], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefixes = entry.get("source_event_prefixes_any", [])
    observed = [str(row.get("source_event") or row.get("event") or "") for row in trace_rows]
    found = [event for event in observed if any(event.startswith(prefix) for prefix in prefixes)]
    return {"id": entry["id"], "kind": "live_trace", "passed": bool(found), "expected_any": prefixes, "observed": sorted(set(found))}


def _static(contract: dict[str, Any], runtime: dict[str, Any] | None) -> list[dict[str, Any]]:
    if runtime is None:
        return [{"id": "runtime_capability_handshake", "kind": "runtime", "passed": False, "reason": "runtime contract is missing"}]
    adapter = runtime.get("adapter") or runtime.get("selected_adapter") or runtime.get("capabilities", {}).get("selected_adapter")
    supported = contract.get("supported_adapters", [])
    passed = bool(adapter) and (not supported or adapter in supported)
    return [{"id": "runtime_capability_handshake", "kind": "runtime", "passed": passed, "adapter": adapter, "supported_adapters": supported}]


def _send_sentinel(gateway_base: str, model: str, timeout: float) -> dict[str, Any]:
    request_id = f"instrumentation-sentinel-{uuid.uuid4().hex}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply only: ready."}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "cache_salt": request_id,
        "custom_params": {"agentic_kv": {"agent_label": "instrumentation_sentinel", "agent_phase": "instrumentation_sentinel", "agent_request_id": request_id}},
    }
    started = time.time_ns()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{gateway_base.rstrip('/')}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()
        return {"sent": True, "request_id": request_id, "started_ns": started}
    except Exception as exc:  # pragma: no cover - exercised against a live gateway
        return {"sent": False, "request_id": request_id, "started_ns": started, "error": f"{type(exc).__name__}: {exc}"}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = Path(args.contract).resolve()
    contract = _load(contract_path)
    runtime = _load(Path(args.runtime_contract)) if args.runtime_contract and Path(args.runtime_contract).exists() else None
    checks = _static(contract, runtime) if runtime is not None or args.stage == "static" else []
    if args.stage == "static":
        valid = all(item["passed"] for item in checks)
        return {"schema_version": "agentic_controller_instrumentation_preflight.v1", "stage": "static", "policy": args.policy, "valid": valid, "experiment_allowed": valid or args.policy == "observe_only", "contract": str(contract_path), "checks": checks}

    installation = _load(Path(args.installation_report)) if Path(args.installation_report).exists() else {}
    for entry in contract.get("required", []):
        if entry.get("environment"):
            checks.append(_check_environment(entry))
        if entry.get("installed_hooks_any"):
            checks.append(_check_hooks(entry, installation))

    sentinel = _send_sentinel(args.gateway_base, args.model, args.timeout)
    deadline = time.monotonic() + args.trace_wait
    trace_rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        trace_rows = _read_jsonl(Path(args.trace))
        if sentinel.get("sent") and any(int(row.get("ts_ns", 0) or 0) >= sentinel["started_ns"] for row in trace_rows):
            break
        time.sleep(0.2)
    # A zero wait is useful in unit tests and when the request completed before
    # the first poll. Always take one final snapshot before evaluating events.
    trace_rows = _read_jsonl(Path(args.trace))
    trace_rows = [row for row in trace_rows if int(row.get("ts_ns", 0) or 0) >= int(sentinel.get("started_ns", 0))]
    for entry in contract.get("required", []):
        if entry.get("source_event_prefixes_any"):
            checks.append(_check_events(entry, trace_rows))

    valid = bool(sentinel.get("sent")) and all(item["passed"] for item in checks)
    return {
        "schema_version": "agentic_controller_instrumentation_preflight.v1",
        "stage": "live",
        "policy": args.policy,
        "valid": valid,
        "experiment_allowed": valid or args.policy == "observe_only",
        "contract": str(contract_path),
        "installation_report": str(Path(args.installation_report).resolve()),
        "trace": str(Path(args.trace).resolve()),
        "checks": checks,
        "sentinel": {**sentinel, "excluded_from_measurement": True, "trace_rows_considered": len(trace_rows)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("static", "live"), required=True)
    parser.add_argument("--runtime-contract")
    parser.add_argument("--installation-report")
    parser.add_argument("--gateway-base")
    parser.add_argument("--model")
    parser.add_argument("--trace")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--trace-wait", type=float, default=15)
    parser.add_argument("--policy", choices=("strict", "observe_only"), default="strict")
    args = parser.parse_args()
    if args.stage == "live":
        for name in ("installation_report", "gateway_base", "model", "trace"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for live preflight")
    result = run_preflight(args)
    _write(Path(args.out), result)
    if not result["valid"]:
        print(f"Instrumentation preflight INVALID; details: {args.out}")
        return 0 if args.policy == "observe_only" else 2
    print(f"Instrumentation preflight valid: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
