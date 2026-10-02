from agentic_instrumentation import EvidenceEvent, assess_loaded_match, validate_events, validate_profile


def _event(kind, time_ns, indices, *, request=""):
    key = "matched_indices" if kind == "kv.prefix_match" else "device_indices"
    return EvidenceEvent(kind, time_ns, "fixture", session_id="session", request_id=request,
                         payload={key: {"values": indices, "index_count": len(indices)}})


def test_split_node_replay_match_is_supported_by_gpu_slots():
    result = assess_loaded_match([_event("kv.load_gpu", 10, [1, 2, 3, 4])],
                                 [_event("kv.prefix_match", 20, [3, 4], request="replay")])
    assert result["loaded_slots_matched_by_replay"] == 2
    assert result["exact_model_consumption"] == "not_proven"


def test_eviction_or_missing_identity_cannot_claim_reuse():
    load = _event("kv.load_gpu", 10, [1, 2])
    match = _event("kv.prefix_match", 20, [1], request="replay")
    assert assess_loaded_match([load], [match], [_event("kv.evict_gpu", 15, [])])["loaded_slots_match_status"] == "unknown"
    no_request = EvidenceEvent("kv.prefix_match", 20, "fixture", session_id="session", payload=match.payload)
    assert assess_loaded_match([load], [no_request])["loaded_slots_match_status"] == "unknown"


def test_range_requires_digest_before_claiming_overlap():
    load = EvidenceEvent("kv.load_gpu", 10, "fixture", session_id="session",
                         payload={"device_indices": {"min": 1, "max": 4, "index_count": 4}})
    match = _event("kv.prefix_match", 20, [3, 4], request="replay")
    assert assess_loaded_match([load], [match])["loaded_slots_match_status"] == "unknown"


def test_profile_validation_fails_loud_on_missing_evidence():
    result = validate_profile("kv_lifecycle", {"kv.load_gpu"})
    assert result["valid"] is False
    assert result["missing"] == ["kv.prefix_match"]


def test_event_validation_rejects_match_without_index_evidence():
    load = _event("kv.load_gpu", 10, [1, 2])
    match = EvidenceEvent("kv.prefix_match", 20, "fixture", session_id="session", request_id="replay")
    result = validate_events("kv_lifecycle", [load, match])
    assert result["valid"] is False
    assert result["invalid_fields"] == ["kv.prefix_match"]
