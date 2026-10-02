import json

from agentic_backends.sglang.audit_v0510 import lifecycle_role, normalize_lifecycle_evidence, translate_trace


def test_lifecycle_roles_distinguish_transition_from_copy():
    assert lifecycle_role("hicache.load.end") == "semantic_load"
    assert lifecycle_role("hiradix.load_back.end") == "nested_load"
    assert lifecycle_role("hostpool.load_to_device_per_layer.end") == "layer_copy"
    assert lifecycle_role("other.event") == "unknown"


def test_normalize_match_preserves_index_lineage_and_request_identity():
    row = {"event": "hiradix.match_prefix.end", "ts_ns": 123,
           "kv_context": {"agent_session_id": "s", "agent_request_id": "replay"},
           "result": [{"index_count": 4}, {"id": 16, "parent_id": 12,
                       "value": {"index_count": 2, "values": [7, 8]}}]}
    event = normalize_lifecycle_evidence(row)
    assert event.signal_id == "kv.prefix_match"
    assert event.session_id == "s" and event.request_id == "replay"
    assert event.payload["matched_indices"]["values"] == [7, 8]


def test_translate_keeps_only_identity_linked_evidence(tmp_path):
    trace = tmp_path / "backend.jsonl"
    rows = [
        {"event": "trace.install.summary", "ts_ns": 1, "sglang_version": "0.5.10.post1",
         "adapter": "v0510", "installed_hooks": ["HiRadixCache.load_back"], "missing_required_hooks": []},
        {"event": "kv_telemetry.prefill.start", "ts_ns": 2, "agent_session_id": "s",
         "agent_request_id": "r", "batch_cached_prefix_token_sum": 64},
        {"event": "kv_telemetry.cache.end", "ts_ns": 3, "request_id": "unjoined"},
        {"event": "agentic_kv.prepare_prefix.result", "ts_ns": 4, "load_id": "l",
         "command": {"session_id": "s", "request_id": "control"}},
        {"event": "agentic_kv.prepare_prefix.load_status", "ts_ns": 5,
         "load_id": "l", "status": "finished", "loaded_tokens": 512},
        {"event": "kv_telemetry.request_stage", "ts_ns": 6, "agent_session_id": "s",
         "agent_request_id": "control", "category": "host_to_device_copy", "phase": "end"},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    normalized = translate_trace(trace)
    assert [row.kind for row in normalized] == ["runtime_hooks", "cache_match", "load_complete", "layer_copy"]
    assert normalized[2].session_id == "s"
    assert normalized[2].evidence["join_method"] == "load_id"
    assert normalized[0].evidence["audit_capabilities"] == ["native_load"]


def test_translate_uses_raw_prefix_match_without_scheduler_telemetry(tmp_path):
    trace = tmp_path / "lean.jsonl"
    rows = [
        {"event": "hiradix.match_prefix.end", "ts_ns": 10,
         "kv_context": {"agent_session_id": "s", "agent_request_id": "replay"},
         "result": [{"index_count": 64}, {"value": {"index_count": 2, "values": [62, 63]}}]},
        {"event": "hiradix.match_prefix.end", "ts_ns": 11,
         "result": [{"index_count": 32}, {"value": {"index_count": 32, "values": list(range(32))}}]},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    translated = translate_trace(trace)
    assert len(translated) == 1
    assert translated[0].kind == "cache_match"
    assert translated[0].request_id == "replay"
    assert translated[0].evidence == {"cached_prefix_tokens": 64, "join_method": "raw_prefix_match"}
