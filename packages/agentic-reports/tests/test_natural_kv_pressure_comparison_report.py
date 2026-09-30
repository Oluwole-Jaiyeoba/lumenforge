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


def test_aggregate_allows_a_level_without_cross_session_overlap() -> None:
    report = aggregate(
        [
            {
                "run_id": "case-1",
                "workload": {"session_count": 4},
                "native_reload_events": 0,
                "observations": [
                    {"ttft_ms": 10, "replay_duration_ms": 20, "cross_session_reload_count": 0},
                ],
            }
        ]
    )
    group = report["levels"][0]["cross_session_overlap"]
    assert group["replays"] == 0
    assert group["p95_ttft_ms"] is None
