import json
from pathlib import Path

from agentic_backends.sglang.hook_registry import resolve_hook
from agentic_backends.sglang.trace_contract import hook_inventory, inspect_trace, validate_bundle


def _installation():
    installed = []
    for hook_id in ("request_lifecycle", "scheduler_batch_observation", "request_completion_timing"):
        installed.extend(item["target"] for item in resolve_hook(hook_id, "v0510")["targets"])
    return {"adapter": "v0510", "installed_hooks": installed}


def test_controller_queue_evidence_gate_and_inventory():
    rows = [
        {"event": "trace.install.summary", **_installation()},
        {"event": "scheduler.handle_generate_request.end", "ts_ns": 1, "agent_request_id": "r"},
        {"event": "scheduler.get_next_batch_to_run.end", "ts_ns": 2},
        {"event": "scheduler.process_batch_result.end", "ts_ns": 3},
    ]
    inspection = inspect_trace(rows, "v0510")
    assert inspection["embedded_installation"]["adapter"] == "v0510"
    assert validate_bundle("controller_queue", "v0510", _installation(), inspection)["valid"]
    assert any(item["source_event_prefix"] == "scheduler.handle_generate_request" and
               item["observed_count"] == 1 for item in inspection["hooks"])
    assert len(hook_inventory("v0510")) >= 30


def test_gate_rejects_missing_field_or_wrong_adapter():
    rows = [
        {"event": "scheduler.handle_generate_request.end", "ts_ns": 1},
        {"event": "scheduler.get_next_batch_to_run.end", "ts_ns": 2},
        {"event": "scheduler.process_batch_result.end", "ts_ns": 3},
    ]
    inspection = inspect_trace(rows, "v0510")
    assert validate_bundle("controller_queue", "v0510", _installation(), inspection)["invalid_fields"] == ["request.accepted"]
    assert not validate_bundle("controller_queue", "v0511", _installation(), inspection)["valid"]


def test_sanitized_reference_trace_preserves_load_semantics():
    fixture = Path(__file__).with_name("fixtures") / "v0510_trace.jsonl"
    inspection = inspect_trace((json.loads(line) for line in fixture.read_text().splitlines()), "v0510")
    assert inspection["signal_counts"] == {
        "request.accepted": 1, "batch.scheduled": 1, "batch.completed": 1,
        "kv.load_gpu": 1, "kv.layer_copy": 1, "kv.prefix_match": 1,
    }


def test_copy_gate_requires_live_copy_but_control_can_check_install_only():
    installed = []
    for hook_id in ("kv_gpu_load", "kv_layer_copy"):
        installed.extend(item["target"] for item in resolve_hook(hook_id, "v0510")["targets"])
    installation = {"adapter": "v0510", "installed_hooks": installed}
    empty = inspect_trace([], "v0510")
    assert not validate_bundle("copy_timing", "v0510", installation, empty)["valid"]
    gate = validate_bundle("copy_timing", "v0510", installation, empty, installation_only=True)
    assert gate["valid"] and gate["validation_level"] == "installation_only"
    installation["installed_hooks"] = []
    assert not validate_bundle("copy_timing", "v0510", installation, empty, installation_only=True)["valid"]


def test_request_boundary_needs_acceptance_with_request_identity():
    installed = [item["target"] for item in resolve_hook("request_lifecycle", "v0510")["targets"]]
    installation = {"adapter": "v0510", "installed_hooks": installed}
    missing_identity = inspect_trace([{"event": "scheduler.handle_generate_request.end", "ts_ns": 1}], "v0510")
    assert not validate_bundle("request_boundary", "v0510", installation, missing_identity)["valid"]
    emitted = []
    valid = inspect_trace([{"event": "scheduler.handle_generate_request.end", "ts_ns": 1,
                            "kv_context": {"agent_request_id": "request-1", "agent_correlation_id": "corr-1"}}],
                          "v0510", emitted.append)
    assert validate_bundle("request_boundary", "v0510", installation, valid)["valid"]
    assert emitted[0].request_id == "request-1"
    assert emitted[0].correlation_id == "corr-1"
