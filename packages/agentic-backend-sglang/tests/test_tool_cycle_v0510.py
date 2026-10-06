from agentic_backends.sglang.tool_cycle_v0510 import normalize_tool_cycle_stage


def test_batch_request_ids_are_preserved():
    row = {"event": "scheduler.run_batch.start", "ts_ns": 10,
           "kv_context": {"batch": {"requests": [
               {"agent_request_id": "r1"}, {"agent_request_id": "r2"}]}}}
    assert normalize_tool_cycle_stage(row) == {
        "kind": "batch_start", "ts_ns": 10, "request_ids": ["r1", "r2"]}


def test_native_load_maps_to_neutral_duration():
    row = {"event": "hiradix.load_back.end", "ts_ns": 20,
           "duration_ms": 4.5, "kv_context": {"agent_request_id": "r1"}}
    assert normalize_tool_cycle_stage(row) == {
        "kind": "load_end", "ts_ns": 20, "request_ids": ["r1"], "duration_ms": 4.5}


def test_unrelated_event_is_ignored():
    assert normalize_tool_cycle_stage({"event": "unrelated", "ts_ns": 30}) is None
