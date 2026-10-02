from agentic_reports.audits.build_work_audit_block_audit import analyze_block_audit
from agentic_work_audit import AuditEvent


def _event(kind, ts, *, request="", node=None):
    return AuditEvent(kind=kind, ts_ns=ts, source="test", session_id="r-host",
                      request_id=request, evidence={"node_id": node} if node else {})


def _raw(name, ts, *, request="initial", host="host", device="device", result=None):
    context = {
        "agent_session_id": "r-host", "agent_request_id": request,
        "request_id": request, "node_id": 13,
        "host_indices": {"index_count": 64, "sha1_16": host},
        "device_indices": {"index_count": 64, "sha1_16": device},
    }
    row = {"event": name, "ts_ns": ts, "kv_context": context}
    if result is not None:
        row["result"] = [{"index_count": result, "sha1_16": "prefix"},
                         {"id": 16, "parent_id": 13, "value": {"index_count": result,
                          "values": list(range(result))}}]
    return row


def _fixture():
    validation = {
        "status": "validated", "run_id": "r",
        "runtime": {"backend_version": "0.5.10.post1", "adapter": "v0510"},
        "cases": [{"session_id": "r-host", "case_type": "host_backed",
                   "native_loaded_tokens": 64}],
    }
    harness = [
        _event("tool_start", 200), _event("tool_end", 300),
        _event("host_resident_proof", 310, node=13),
        _event("replay_sent", 400, request="replay"),
        _event("replay_first_token", 450, request="replay"),
        _event("replay_finished", 600, request="replay"),
    ]
    trace = [
        _raw("hicache.write.end", 150),
        _raw("hicache.evict_device.end", 250),
        _raw("hiradix.load_back.end", 349),
        _raw("hicache.load.end", 350),
        _raw("hostpool.load_to_device_per_layer.end", 351),
        _raw("hostpool.load_to_device_per_layer.end", 352),
        _raw("hiradix.match_prefix.end", 420, request="replay", result=62),
        _raw("hiradix.match_prefix.end", 550, request="replay", result=63),
    ]
    return trace, harness, validation


def test_existing_ledger_counts_one_load_and_links_per_layer_copies():
    trace, harness, validation = _fixture()
    audit = analyze_block_audit(trace, harness, validation)
    case = audit["cases"][0]
    assert audit["status"] == "validated"
    assert case["semantic_load_transitions"] == 1
    assert case["nested_load_observations"] == 1
    assert case["layer_copy_observations"] == 2
    assert case["exactly_linked_layer_copies"] == 2
    assert case["selected_host_node_id"] == "13"
    assert case["semantic_loaded_tokens"] == 64
    assert case["load_periods"] == ["ready_before_replay"]
    assert case["replay_prefix_match_observations"] == 1
    assert case["max_replay_matched_tokens"] == 62
    assert case["same_loaded_block_used_by_replay"] == "not_proven"
    assert audit["logical_loaded_records"] == 1


def test_unlinked_copy_fails_the_evidence_gate():
    trace, harness, validation = _fixture()
    trace[5]["kv_context"]["device_indices"]["sha1_16"] = "different"
    audit = analyze_block_audit(trace, harness, validation)
    assert audit["status"] == "failed"
    assert "some per-layer copies could not be linked" in audit["failures"][0]


def test_backend_version_mismatch_fails_closed():
    trace, harness, validation = _fixture()
    validation["runtime"]["backend_version"] = "other"
    audit = analyze_block_audit(trace, harness, validation)
    assert audit["status"] == "failed"
    assert "pinned v0510" in audit["failures"][0]


def test_replay_match_of_loaded_slots_does_not_claim_model_consumption():
    trace, harness, validation = _fixture()
    trace[3]["kv_context"]["device_indices"] = {"index_count": 4, "values": [10, 11, 12, 13]}
    trace[4]["kv_context"]["device_indices"] = trace[3]["kv_context"]["device_indices"]
    trace[5]["kv_context"]["device_indices"] = trace[3]["kv_context"]["device_indices"]
    trace[6]["result"] = [{"index_count": 2}, {"id": 16, "parent_id": 13,
                           "value": {"index_count": 2, "values": [12, 13]}}]
    result = analyze_block_audit(trace, harness, validation)
    case = result["cases"][0]
    assert case["loaded_slots_matched_by_replay"] == 2
    assert case["loaded_slots_match_status"] == "supported"
    assert case["exact_model_consumption"] == "not_proven"
