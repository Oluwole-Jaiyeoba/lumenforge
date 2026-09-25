"""Contract-driven, fail-loud validation for controller instrumentation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .hook_registry import resolve_hook


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


def _check_hook(entry: dict[str, Any], report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    adapter = str(report.get("adapter") or "")
    resolved = resolve_hook(str(entry["hook_id"]), adapter) if adapter else {"targets": [], "hook_id": entry["hook_id"]}
    installed = set(report.get("installed_hooks", []))
    statuses = dict(report.get("hook_statuses", {}))
    targets = [item["target"] for item in resolved["targets"]]
    observed = [{"target": target, "status": "installed" if target in installed else statuses.get(target, "missing_target")} for target in targets]
    return (
        {
            "id": entry["id"],
            "hook_id": entry["hook_id"],
            "kind": "installation",
            "passed": bool(targets) and any(item["status"] == "installed" for item in observed),
            "adapter": adapter,
            "targets": observed,
        },
        [item["event_prefix"] for item in resolved["targets"]],
    )


def _check_trace(entry: dict[str, Any], rows: list[dict[str, Any]], prefixes: list[str], *, source: bool = False, require_all: bool = False) -> dict[str, Any]:
    key = "source_event" if source else "event"
    observed = [str(row.get(key) or "") for row in rows]
    found = {prefix: sum(value.startswith(prefix) for value in observed) for prefix in prefixes}
    fields = entry.get("required_fields", [])
    matching = [row for row in rows if any(str(row.get(key) or "").startswith(prefix) for prefix in prefixes)]
    fields_ok = all(any(row.get(field) not in (None, "") for row in matching) for field in fields)
    passed = (all(found.values()) if require_all else any(found.values())) and fields_ok
    return {"id": entry["id"], "kind": "live_trace", "passed": passed, "expected": prefixes, "event_counts": found, "required_fields": fields, "fields_observed": fields_ok}


def _controller_probe(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.controller_probe_report)
    command = [sys.executable, "-m", "agentic_experiments.runners.controller_preflight_probe", "--out", str(out), "--mode", args.controller_mode]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
    except Exception as exc:  # pragma: no cover - runtime integration
        return {"ran": False, "error": f"{type(exc).__name__}: {exc}", "rows": []}
    rows = _read_jsonl(out)
    return {"ran": completed.returncode == 0, "returncode": completed.returncode, "stderr": completed.stderr[-2000:], "rows": rows}


def _check_controller_probe(entry: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    action = "action_acknowledgement" in entry["id"]
    expected = "m27.controller.preflight.action_ack" if action else "m27.controller.preflight.decision"
    matching = [row for row in probe["rows"] if row.get("event") == expected]
    acted = any(int(row.get("acted_count", 0) or 0) > 0 for row in matching) if action else True
    return {"id": entry["id"], "kind": "controller_probe", "passed": bool(probe.get("ran")) and bool(matching) and acted, "expected": expected, "event_count": len(matching), "acted": acted}


def _static(contract: dict[str, Any], runtime: dict[str, Any] | None) -> list[dict[str, Any]]:
    if runtime is None:
        return [{"id": "runtime_capability_handshake", "kind": "runtime", "passed": False, "reason": "runtime contract is missing"}]
    adapter = runtime.get("adapter") or runtime.get("selected_adapter") or runtime.get("capabilities", {}).get("selected_adapter")
    supported = contract.get("supported_adapters", [])
    passed = bool(adapter) and (not supported or adapter in supported)
    return [{"id": "runtime_capability_handshake", "kind": "runtime", "passed": passed, "adapter": adapter, "supported_adapters": supported}]


def _send_sentinel(gateway_base: str, model: str, timeout: float) -> dict[str, Any]:
    request_id = f"instrumentation-sentinel-{uuid.uuid4().hex}"
    marker = {
        "session_id": "instrumentation-sentinel",
        "phase": "replay",
        "mode": "instrumentation_sentinel",
        "harness": "instrumentation",
        "label": request_id,
        "deadline_offset_ms": 1000,
        "expected_replay_request_id": request_id,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(marker, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Reply only: ready.\nHARNESS_REPLAY_EXPERIMENT_JSON:{encoded}"}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "cache_salt": request_id,
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
        return {"schema_version": "agentic_controller_instrumentation_preflight.v1", "stage": "static", "policy": args.policy, "valid": valid, "experiment_allowed": valid or args.policy == "observe_only", "invalid_for_controller_claims": not valid, "contract": str(contract_path), "checks": checks}

    installation = _load(Path(args.installation_report)) if Path(args.installation_report).exists() else {}
    source_prefixes: dict[str, list[str]] = {}
    for entry in contract.get("required", []):
        if entry.get("hook_id"):
            check, source_prefixes[entry["id"]] = _check_hook(entry, installation)
            checks.append(check)
    for entry in contract.get("optional", []):
        if entry.get("hook_id"):
            check, _ = _check_hook(entry, installation)
            check["required"] = False
            check["status"] = "installed" if check["passed"] else "optional_unavailable"
            checks.append(check)

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
    sentinel_trace = Path(args.sentinel_trace)
    sentinel_trace.parent.mkdir(parents=True, exist_ok=True)
    sentinel_trace.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows), encoding="utf-8")
    controller_probe = _controller_probe(args)
    for entry in contract.get("required", []):
        if entry.get("hook_id"):
            checks.append(_check_trace(entry, trace_rows, source_prefixes[entry["id"]], source=True))
        if entry.get("trace_event_prefixes_all"):
            checks.append(_check_trace(entry, trace_rows, entry["trace_event_prefixes_all"], require_all=True))
        if entry.get("controller_probe"):
            checks.append(_check_controller_probe(entry, controller_probe))

    valid = bool(sentinel.get("sent")) and all(item["passed"] for item in checks if item.get("required", True))
    return {
        "schema_version": "agentic_controller_instrumentation_preflight.v1",
        "stage": "live",
        "policy": args.policy,
        "valid": valid,
        "experiment_allowed": valid or args.policy == "observe_only",
        "invalid_for_controller_claims": not valid,
        "contract": str(contract_path),
        "installation_report": str(Path(args.installation_report).resolve()),
        "installation": {
            "adapter": installation.get("adapter", ""),
            "sglang_version": installation.get("sglang_version", ""),
            "missing_required_hooks": installation.get("missing_required_hooks", []),
            "missing_optional_hooks": installation.get("missing_optional_hooks", []),
        },
        "trace": str(Path(args.trace).resolve()),
        "live_sentinel_trace": str(sentinel_trace.resolve()),
        "controller_probe": {key: value for key, value in controller_probe.items() if key != "rows"},
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
    parser.add_argument("--sentinel-trace")
    parser.add_argument("--controller-probe-report")
    parser.add_argument("--controller-mode", default="controller_scheduler_priority")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--trace-wait", type=float, default=15)
    parser.add_argument("--policy", choices=("strict", "observe_only"), default="strict")
    args = parser.parse_args()
    if args.stage == "live":
        for name in ("installation_report", "gateway_base", "model", "trace", "sentinel_trace", "controller_probe_report"):
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
