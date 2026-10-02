from agentic_harnesses.hint_benchmark.backend_evidence import audit_backend_mappings
from agentic_harnesses.hint_benchmark.runner import capture_correlation_id


def _native():
    return [{"scenario_id": "cache", "payload_index": 1, "hint_id": "cache_control",
             "evidence_tier": "native_client_or_transport_capture", "evidence_source": "claude_native_capture",
             "capture_correlation_id": "c1"}]


def _backend():
    return [{"signal_id": "request.accepted", "request_id": "r1", "correlation_id": "c1"}]


def test_matching_correlation_proves_same_request_but_not_hint_effect():
    result = audit_backend_mappings(_native(), [{"scenario_id": "cache", "payload_index": 1,
                                                "backend_request_id": "r1", "correlation_id": "c1"}], _backend())
    assert result["valid"]
    assert result["mappings"][0]["same_request_proven"]
    assert result["mappings"][0]["backend_hint_effect"] == "not_proven"


def test_external_mapping_alone_does_not_prove_same_request():
    native = [dict(_native()[0], capture_correlation_id="")]
    result = audit_backend_mappings(native, [{"scenario_id": "cache", "payload_index": 1,
                                              "backend_request_id": "r1", "correlation_id": "c1"}], _backend())
    assert result["valid"]
    assert not result["mappings"][0]["same_request_proven"]


def test_fixture_and_wrong_correlation_are_rejected():
    fixture = [dict(_native()[0], evidence_tier="fixture_plumbing_only")]
    mapping = [{"scenario_id": "cache", "payload_index": 1,
                "backend_request_id": "r1", "correlation_id": "wrong"}]
    result = audit_backend_mappings(fixture, mapping, _backend())
    assert not result["valid"]
    assert len(result["errors"]) == 2


def test_native_capture_correlation_comes_from_captured_request():
    assert capture_correlation_id({"_capture": {"headers": {"x-agentic-correlation-id": "c1"}}}) == "c1"
    assert capture_correlation_id({"metadata": {"agentic_correlation_id": "c2"}}) == "c2"
    assert capture_correlation_id({}) == ""
