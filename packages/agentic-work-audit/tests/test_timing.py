from agentic_work_audit import AuditEvent, analyze_timing


def event(kind, ms, session="", request="", **evidence):
    return AuditEvent(kind=kind, ts_ns=int(ms * 1_000_000), source="test",
                      session_id=session, request_id=request, evidence=evidence)


def pair_events(condition, base):
    session = f"pair01-{condition}"
    initial, replay, second = (f"{session}-{part}" for part in ("initial", "replay", "replay-2"))
    request_at = 145 if condition == "early" else 305
    complete_at = 240 if condition == "early" else 370
    token_at = 345 if condition == "early" else 425
    rows = [
        event("initial_sent", base + 10, session, initial),
        event("device_evict_proof", base + 80, session, initial),
        event("host_resident_proof", base + 90, session, initial, host_tokens=2048),
        event("tool_start", base + 100, session, initial),
        event("load_requested", base + request_at, session, initial, load_id=f"{session}-load"),
        event("load_accepted", base + request_at + 10, session, initial,
              load_id=f"{session}-load", loaded_tokens=2048),
        event("layer_copy", base + complete_at - 10, session, initial),
        event("load_complete", base + complete_at, session, initial,
              load_id=f"{session}-load", loaded_tokens=2048),
        event("tool_end", base + 300, session, initial),
        event("replay_sent", base + 320 if condition == "early" else base + 330, session, replay),
        event("cache_match", base + 330 if condition == "early" else base + 350,
              session, replay, cached_prefix_tokens=2048),
        event("replay_first_token", base + token_at, session, replay),
        event("replay_finished", base + token_at + 20, session, replay),
        event("tool_2_start", base + token_at + 25, session, replay),
        event("tool_2_end", base + token_at + 225, session, replay),
        event("replay_2_sent", base + token_at + 230, session, second),
        event("cache_match", base + token_at + 235, session, second, cached_prefix_tokens=2048),
        event("replay_2_first_token", base + token_at + 245, session, second),
        event("replay_2_finished", base + token_at + 260, session, second),
    ]
    return rows, {"session_id": session, "pair": 1, "condition": condition,
                  "prompt_words": 4090, "initial": {"total_latency_ms": 2000}}


def test_timing_pair_separates_due_delay_from_request_ttft():
    early, early_spec = pair_events("early", 1000)
    late, late_spec = pair_events("late", 2000)
    runtime = event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510",
                    missing_required_hooks=[])
    result = analyze_timing([runtime, *early, *late], [early_spec, late_spec])
    assert result["status"] == "validated"
    assert result["pairs"][0]["comparable"]
    assert result["pairs"][0]["late_minus_early_first_token_after_due_ms"] == 80
    assert result["pairs"][0]["late_minus_early_submission_after_due_ms"] == 10
    assert result["cases"][0]["completion_observed_before_due"] is True
    assert result["cases"][1]["completion_observed_before_due"] is False
    assert result["cases"][1]["submission_after_due_ms"] == 30
    assert result["cases"][0]["load_request_from_due_ms"] == -155
    assert result["cases"][1]["load_request_from_due_ms"] == 5
    assert result["cases"][1]["replay_ttft_ms"] == 95


def test_timing_fails_on_missing_completion_or_second_match():
    early, spec = pair_events("early", 1000)
    runtime = event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510",
                    missing_required_hooks=[])
    rows = [row for row in early if row.kind != "load_complete"
            and not (row.kind == "cache_match" and row.request_id.endswith("replay-2"))]
    result = analyze_timing([runtime, *rows], [spec])
    assert result["status"] == "failed"
    assert any("CUDA completion" in failure for failure in result["failures"])
    assert any("prefix match" in failure for failure in result["failures"])


def test_second_replay_outlier_does_not_invalidate_first_replay_pair():
    early, early_spec = pair_events("early", 1000)
    late, late_spec = pair_events("late", 2000)
    late = [event(row.kind, row.ts_ns / 1_000_000 + 700, row.session_id, row.request_id,
                  **row.evidence) if row.kind in {"replay_2_first_token", "replay_2_finished"} else row
            for row in late]
    runtime = event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510",
                    missing_required_hooks=[])
    result = analyze_timing([runtime, *early, *late], [early_spec, late_spec])
    assert result["pairs"][0]["comparable"]
    assert not result["pairs"][0]["task_comparable"]
    assert result["pairs"][0]["late_minus_early_task_latency_ms"] is None
