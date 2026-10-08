from pathlib import Path

from agentic_reports.builders.build_work_audit_report import render, render_markdown
from agentic_reports.builders.coordinated_swap_report import (
    SCHEMA, comparison_rows, reproduction, session_rows,
)


def test_blocked_swap_has_no_success_claim_and_includes_runner():
    summary = {"schema": SCHEMA, "run_id": "swap_test", "status": "blocked",
               "issues": ["missing arm"], "arms": [], "_started_ns": 1791485498893546751}
    for document in (render([(Path("summary.json"), summary)]),
                     render_markdown([(Path("summary.json"), summary)])):
        assert "no performance conclusion" in document
        assert "run_work_audit_coordinated_swap.sh" in document
    assert "SWAP_TURNS=40" in reproduction(summary)


def test_comparison_keeps_each_trial_and_session_visible():
    summary = {"comparisons": [{"trial": 2, "reference": "resident", "workload_change_pct": 25,
                "due_to_first_token_change_ms": 40, "sessions_finished_sooner": 0,
                "sessions_finished_later": 20}], "arms": [
        {"trial": 2, "mode": "coordinated", "per_session_completion_ms": {"s1": 1250}},
        {"trial": 2, "mode": "resident", "per_session_completion_ms": {"s1": 1000}},
    ]}
    assert comparison_rows(summary) == [("2 / resident", "+25.0%", "+40.0", "0 / 20")]
    assert session_rows(summary) == [("2 / s1", "unavailable", "1.250", "1.000")]
