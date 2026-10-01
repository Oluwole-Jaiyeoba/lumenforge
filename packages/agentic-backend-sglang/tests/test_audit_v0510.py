import json

from agentic_backends.sglang.audit_v0510 import translate_trace


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
