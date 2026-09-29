from agentic_reports.builders.build_natural_multi_agent_kv_pressure_report import build


def test_report_labels_natural_reloads_without_claiming_injection() -> None:
    page = build(
        {
            "workload": {
                "session_count": 8,
                "tool_waits_per_session": 3,
                "session_prefix_tokens": 8192,
                "replay_tokens": 384,
                "tool_wait_range_ms": [400, 2000],
            },
            "native_reload_events": 1,
            "replay_count": 24,
            "pressure_buckets": {"control_like": 23, "low_like": 1, "medium_like": 0, "high_like": 0},
            "observations": [
                {
                    "session_id": "session-0",
                    "tool_wait_step": 1,
                    "tool_wait_ms": 500,
                    "ttft_ms": 30.0,
                    "total_decode_ms": 100.0,
                    "natural_reload_count": 1,
                    "pressure_bucket": "low_like",
                    "natural_reload_sessions": ["session-1"],
                }
            ],
            "interpretation": "Natural means ordinary replay requests caused the observed load-back event.",
        }
    )
    assert "No prepared-prefix control request" in page
    assert "low like" in page
    assert "session-1" in page
