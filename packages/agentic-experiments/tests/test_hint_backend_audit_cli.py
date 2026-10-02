import json
import sys

from agentic_experiments.runners.audit_hint_backend_evidence import main


def test_cli_requires_live_gate_and_preserves_separate_proof_claims(tmp_path, monkeypatch):
    observations = tmp_path / "native.jsonl"
    observations.write_text(json.dumps({"scenario_id": "s", "payload_index": 1,
                                        "hint_id": "priority", "evidence_tier": "native_client_or_transport_capture",
                                        "evidence_source": "nat_dynamo_transport_capture",
                                        "capture_correlation_id": "c"}) + "\n")
    events = tmp_path / "backend.jsonl"
    events.write_text(json.dumps({"signal_id": "request.accepted", "request_id": "r",
                                  "correlation_id": "c"}) + "\n")
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps([{"scenario_id": "s", "payload_index": 1,
                                    "backend_request_id": "r", "correlation_id": "c"}]))
    audit = tmp_path / "gate.json"
    audit.write_text(json.dumps({"gate": {"valid": True, "profile": "request_boundary",
                                          "validation_level": "live_evidence", "adapter": "v0510"}}))
    out = tmp_path / "out.json"
    args = ["audit", "--hint-observations", str(observations), "--backend-events", str(events),
            "--backend-audit", str(audit), "--request-map", str(mapping), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    result = json.loads(out.read_text())
    assert result["mappings"][0]["same_request_proven"]
    assert result["mappings"][0]["backend_hint_effect"] == "not_proven"

    observations.write_text(json.dumps({"scenario_id": "s", "payload_index": 1,
                                        "hint_id": "priority", "evidence_tier": "native_client_or_transport_capture",
                                        "evidence_source": "nat_dynamo_transport_capture",
                                        "capture_correlation_id": ""}) + "\n")
    monkeypatch.setattr(sys, "argv", args + ["--require-same-request"])
    assert main() == 2
    strict_result = json.loads(out.read_text())
    assert not strict_result["valid"]
    assert strict_result["errors"][0]["reason"] == "same_request_link_not_proven"

    audit.write_text(json.dumps({"gate": {"valid": True, "profile": "request_boundary",
                                          "validation_level": "installation_only"}}))
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("installation-only audit was accepted as live evidence")
