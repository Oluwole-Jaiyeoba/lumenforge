from agentic_work_audit.events import AuditEvent
from agentic_work_audit.multisession import analyze_multisession


def event(kind, tick, label="", request="", **evidence):
    return AuditEvent(kind=kind, ts_ns=tick * 1_000_000, source="test",
                      session_id=f"run-{label}" if label else "",
                      request_id=f"run-{label}-{request}" if request else "", evidence=evidence)


def sample():
    return [
        event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510"),
        event("initial_sent", 1, "long", "initial"),
        event("initial_sent", 1, "short", "initial"),
        event("initial_sent", 1, "ends", "initial"),
        event("initial_finished", 2, "long", "initial"),
        event("initial_finished", 2, "short", "initial"),
        event("initial_finished", 2, "ends", "initial"),
        event("tool_start", 3, "long", "initial", expected_ms=10, importance="equal"),
        event("tool_start", 3, "short", "initial", expected_ms=5, importance="equal"),
        event("capacity_policy", 4, "long", "initial", active_prefix_budget=2,
              action="explicit_evict_long_prefix"),
        event("device_evict_proof", 5, "long", "initial"),
        event("host_resident_proof", 6, "long", "initial"),
        event("session_end", 7, "ends", "initial"),
        event("tool_end", 8, "short", "initial"),
        event("replay_sent", 9, "short", "replay"),
        event("cache_match", 10, "short", "replay", cached_prefix_tokens=512),
        event("replay_first_token", 11, "short", "replay"),
        event("tool_end", 13, "long", "initial"),
        event("load_requested", 14, "long", "initial"),
        event("replay_sent", 15, "long", "replay"),
        event("load_accepted", 16, "long", "initial"),
        event("layer_copy", 17, "long", "initial"),
        event("load_complete", 18, "long", "initial"),
        event("cache_match", 19, "long", "replay", cached_prefix_tokens=256),
        event("replay_first_token", 20, "long", "replay"),
        event("replay_2_sent", 22, "long", "replay-2"),
        event("cache_match", 23, "long", "replay-2", cached_prefix_tokens=512),
        event("replay_2_first_token", 24, "long", "replay-2"),
    ]


def test_multisession_marks_opportunity_without_claiming_avoidable_work():
    result = analyze_multisession(sample(), "run", expected_runtime=("0.5.10.post1", "v0510"))
    assert result["status"] == "validated"
    assert result["setup"]["tool_waits_overlap"]
    assert result["sessions"]["long"]["load_requested_after_tool_return"]
    assert result["sessions"]["long"]["native_load_accepted_before_replay"] is False
    assert result["sessions"]["long"]["load_acceptance_after_tool_ms"] == 3
    assert result["plausibly_mistimed"]
    assert result["avoidable_work"].startswith("unknown")


def test_multisession_requires_second_replay_linkage():
    result = analyze_multisession([row for row in sample() if row.kind != "cache_match" or
                                   not row.request_id.endswith("replay-2")], "run",
                                  expected_runtime=("0.5.10.post1", "v0510"))
    assert result["status"] == "failed"
    assert any("second-replay reuse" in failure for failure in result["failures"])


def test_multisession_rejects_frontend_importance():
    rows = [event(row.kind, row.ts_ns // 1_000_000,
                  row.session_id.removeprefix("run-") if row.session_id else "",
                  row.request_id.removeprefix(row.session_id + "-") if row.request_id else "",
                  **({**row.evidence, "importance": "high"} if row.kind == "tool_start" else row.evidence))
            for row in sample()]
    result = analyze_multisession(rows, "run", expected_runtime=("0.5.10.post1", "v0510"))
    assert result["status"] == "failed"
    assert any("importance" in failure for failure in result["failures"])
