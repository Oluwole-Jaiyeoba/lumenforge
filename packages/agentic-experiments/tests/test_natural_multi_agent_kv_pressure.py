from agentic_experiments.runners.run_natural_multi_agent_kv_pressure import (
    bucket_for_load_count,
    correlate,
    native_loads,
)


def test_natural_reload_correlation_uses_only_replay_loads() -> None:
    trace = [
        {
            "event": "hiradix.load_back.end",
            "ts_ns": 2_000_000_000,
            "duration_ms": 10.0,
            "kv_context": {
                "agent_mode": "natural_multi_agent_tool_workload",
                "agent_phase": "replay_1",
                "agent_session_id": "other-session",
                "agent_request_id": "other-replay",
            },
        },
        {
            "event": "hiradix.load_back.end",
            "ts_ns": 2_100_000_000,
            "duration_ms": 10.0,
            "kv_context": {
                "agent_mode": "natural_multi_agent_tool_workload",
                "agent_phase": "initial",
                "agent_session_id": "ignored-session",
            },
        },
    ]
    loads = native_loads(trace)
    rows = correlate(
        [
            {
                "session_id": "target-session",
                "request_id": "target-replay",
                "tool_wait_step": 1,
                "tool_wait_ms": 500,
                "ttft_ms": 20.0,
                "total_latency_ms": 100.0,
                "request_start_ns": 1_900_000_000,
                "request_end_ns": 2_050_000_000,
            }
        ],
        loads,
    )
    assert len(loads) == 1
    assert rows[0]["natural_reload_count"] == 1
    assert rows[0]["cross_session_reload_count"] == 1
    assert rows[0]["natural_reload_sessions"] == ["other-session"]
    assert rows[0]["pressure_bucket"] == "low_like"


def test_pressure_buckets_match_the_controlled_reference_ranges() -> None:
    assert bucket_for_load_count(0) == "control_like"
    assert bucket_for_load_count(20) == "low_like"
    assert bucket_for_load_count(21) == "medium_like"
    assert bucket_for_load_count(60) == "medium_like"
    assert bucket_for_load_count(61) == "high_like"
