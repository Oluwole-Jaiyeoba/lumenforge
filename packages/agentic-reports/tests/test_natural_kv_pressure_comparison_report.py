from agentic_reports.builders.build_natural_kv_pressure_comparison_report import aggregate, build


def test_aggregate_separates_cross_session_overlap() -> None:
    report = aggregate(
        [
            {
                "run_id": "case-1",
                "workload": {"session_count": 4},
                "native_reload_events": 2,
                "observations": [
                    {"ttft_ms": 10, "replay_duration_ms": 20, "cross_session_reload_count": 0},
                    {"ttft_ms": 30, "replay_duration_ms": 40, "cross_session_reload_count": 1},
                ],
            }
        ]
    )
    level = report["levels"][0]
    assert level["cross_session_overlap"]["replays"] == 1
    assert level["no_cross_session_overlap"]["replays"] == 1
    assert "Cross-session overlap" in build(report)
