from agentic_work_audit import AuditEvent, analyze_validation


def event(kind, t, session="", request="", **evidence):
    return AuditEvent(kind=kind, ts_ns=t, source="test", session_id=session,
                      request_id=request, evidence=evidence)


def test_validation_requires_linked_native_copy_and_never_infers_stall():
    run = "tiny"
    warm = f"{run}-warm"
    host = f"{run}-host"
    hooks = ["HiRadixCache.match_prefix", "HiRadixCache.load_back", "HiCacheController.start_loading"]
    rows = [event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510",
                  installed_hooks=hooks, missing_required_hooks=[],
                  audit_capabilities=["cache_match", "native_load", "host_copy"])]
    for session, base in ((warm, 100), (host, 200)):
        initial, replay = f"{session}-initial", f"{session}-replay"
        rows.extend([
            event("tool_start", base, session, initial),
            event("tool_end", base + 20, session, initial),
            event("replay_sent", base + 60, session, replay),
            event("cache_match", base + 65, session, replay, cached_prefix_tokens=64),
            event("replay_first_token", base + 90, session, replay),
        ])
    rows.extend([
        event("device_evict_proof", 225, host, f"{host}-initial"),
        event("host_resident_proof", 226, host, f"{host}-initial", host_tokens=2048),
        event("layer_copy", 230, host, f"{host}-initial", duration_ms=1),
        event("load_complete", 240, host, f"{host}-initial", loaded_tokens=2048),
    ])
    summary = analyze_validation(rows, expected_sessions=(warm, host))
    assert summary["status"] == "validated"
    assert summary["cases"][1]["native_layer_copies"] == 1
    assert summary["cases"][1]["replay_stalled_on_load"] is None
    assert summary["cases"][0]["max_cached_prefix_tokens"] == 64
    assert summary["opportunity_ledgers"]["avoidable_eviction"].startswith("unknown")

    missing = [row for row in rows if row.kind != "layer_copy"]
    failed = analyze_validation(missing, expected_sessions=(warm, host))
    assert failed["status"] == "failed"
    assert any("native layer copy" in item for item in failed["failures"])


def test_missing_runtime_fails_closed():
    summary = analyze_validation([], expected_sessions=("tiny-warm", "tiny-host"))
    assert summary["status"] == "failed"
    assert "backend hook installation proof is missing" in summary["failures"]


def test_second_replay_requires_ordered_timeline_and_time_linked_match():
    warm, host = "two-warm", "two-host"
    rows = [event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510",
                  missing_required_hooks=[],
                  audit_capabilities=["cache_match", "native_load", "host_copy"])]
    for session, base in ((warm, 100), (host, 300)):
        initial, first, second = (f"{session}-{suffix}" for suffix in ("initial", "replay", "replay-2"))
        rows.extend([
            event("tool_start", base, session, initial),
            event("tool_end", base + 20, session, initial),
            event("replay_sent", base + 60, session, first),
            event("cache_match", base + 65, session, first, cached_prefix_tokens=64),
            event("replay_first_token", base + 90, session, first),
            event("replay_finished", base + 100, session, first),
            event("tool_2_start", base + 110, session, first),
            event("tool_2_end", base + 130, session, first),
            event("replay_2_sent", base + 140, session, second),
            event("cache_match", base + 145, session, second, cached_prefix_tokens=72),
            event("replay_2_first_token", base + 170, session, second),
            event("replay_2_finished", base + 180, session, second),
        ])
    rows.extend([
        event("device_evict_proof", 325, host, f"{host}-initial"),
        event("host_resident_proof", 326, host, f"{host}-initial", host_tokens=2048),
        event("layer_copy", 330, host, f"{host}-initial"),
        event("load_complete", 340, host, f"{host}-initial", loaded_tokens=2048),
    ])
    summary = analyze_validation(rows, expected_sessions=(warm, host), require_second_replay=True)
    assert summary["status"] == "validated"
    assert [case["second_replay_cache_matches"] for case in summary["cases"]] == [1, 1]
    assert [case["second_replay_ttft_ms"] for case in summary["cases"]] == [0.0, 0.0]

    missing = [row for row in rows if not (row.session_id == host and row.kind == "cache_match"
                                          and row.request_id.endswith("replay-2"))]
    failed = analyze_validation(missing, expected_sessions=(warm, host), require_second_replay=True)
    assert failed["status"] == "failed"
    assert any("second replay has no" in issue for issue in failed["failures"])
