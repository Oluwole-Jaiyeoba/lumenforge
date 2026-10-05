from __future__ import annotations

import json

from agentic_experiments.runners.compare_async_kv_loads import _arm


def test_comparison_counts_only_accepted_loads(tmp_path):
    root = tmp_path / "arms" / "seed1_controller"
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps({
        "replay_count": 1,
        "total_replay_ttft_ms": 12,
        "total_return_to_first_token_ms": 14,
        "workflow_makespan_ms": 100,
        "replays": [],
    }), encoding="utf-8")
    rows = [
        {"kind": "controller_load", "session_id": "one", "load_id": "load-1",
         "loaded_tokens": 2048, "load_request_ns": 1_000_000, "load_response_ns": 3_000_000},
        {"kind": "controller_load", "session_id": "two", "load_id": "",
         "loaded_tokens": 0, "load_request_ns": 4_000_000, "load_response_ns": 9_000_000},
        {"kind": "controller_load_outcome", "session_id": "one", "load_id": "load-1"},
    ]
    (root / "harness_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    result = _arm(tmp_path, "controller")
    assert result["load_attempts"] == 2
    assert result["accepted_loads"] == 1
    assert result["confirmed_loads"] == 1
    assert result["load_sessions"] == ["one"]
    assert result["load_submit_ms"] == [2.0]
