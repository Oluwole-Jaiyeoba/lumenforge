from agentic_experiments.runners.analyze_work_audit_decode_overlap import (
    _overlap_metrics,
    _short_decode,
)


def test_short_decode_links_only_its_batches_and_counts_output():
    request_id = "trial-short-replay"
    request = {"agent_request_id": request_id, "output_ids": {"count": 2}}
    other = {"agent_request_id": "other-short-replay", "output_ids": {"count": 2}}
    trace = [
        {"event": "scheduler.run_batch.start", "call_id": "a", "ts_ns": 1_100_000_000,
         "kv_context": {"batch": {"requests": [request]}}},
        {"event": "worker.forward_batch_generation.start", "call_id": "f", "ts_ns": 1_102_000_000},
        {"event": "worker.forward_batch_generation.end", "call_id": "f", "ts_ns": 1_117_000_000},
        {"event": "scheduler.run_batch.end", "call_id": "a", "ts_ns": 1_120_000_000},
        {"event": "scheduler.process_batch_result_decode.end", "ts_ns": 1_125_000_000,
         "kv_context": {"batch": {"requests": [request]}}},
        {"event": "scheduler.run_batch.start", "call_id": "b", "ts_ns": 1_130_000_000,
         "kv_context": {"batch": {"requests": [other]}}},
        {"event": "scheduler.run_batch.end", "call_id": "b", "ts_ns": 1_140_000_000},
    ]
    case = {"case_id": "trial", "short": {"replay": {
        "first_token_ns": 1_100_000_000, "request_end_ns": 1_200_000_000,
        "content_chunks": [{"ts_ns": 1_100_000_000, "characters": 3}],
    }}, "long": {"load_status": {"worker_started_ns": 1_110_000_000,
                                 "committed_ns": 1_150_000_000}}}

    short = _short_decode(case, trace)
    overlap = _overlap_metrics(case, short)

    assert short["final_output_ids_count"] == 2
    assert len(short["decode_batches"]) == 1
    assert short["decode_batches"][0]["model_forward_ms"] == 15
    assert short["decode_batches"][0]["model_forward_call_ids"] == ["f"]
    assert short["decode_batches"][0]["non_forward_ms"] == 5
    assert overlap["overlapping_batch_count"] == 1
    assert overlap["overlapping_forward_median_ms"] == 15
    assert overlap["short_decode_overlap_ms"] == 40
