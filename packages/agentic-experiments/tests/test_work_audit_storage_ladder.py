import json

import pytest

from agentic_experiments.runners.analyze_work_audit_storage_ladder import analyze


def _rung(root, count, *, peer_delta):
    path = root / f"ladder_n{count}"
    path.mkdir()
    pair = {"seed": 1, "host_stage_delta_ms": -150,
            "host_stage_workflow_delta_ms": -100,
            "peer_ttft_deltas_ms": [peer_delta] if count > 1 else [],
            "peer_completion_deltas_ms": [peer_delta] if count > 1 else [],
            "observed_no_peer_harm": peer_delta <= 0,
            "observed_whole_workload_improvement": True}
    (path / "summary.json").write_text(json.dumps({
        "status": "complete", "paired": [pair],
        "rows": [{"peer_count": count - 1} for _ in range(2)],
    }), encoding="utf-8")


def test_ladder_keeps_target_and_peer_tradeoff_separate(tmp_path):
    _rung(tmp_path, 1, peer_delta=0)
    _rung(tmp_path, 2, peer_delta=40)
    result = analyze(tmp_path, "ladder", [1, 2], [1])
    assert result["first_observed_peer_tradeoff_sessions"] == 2
    assert result["rungs"][1]["median_replay_delay_delta_ms"] == -150
    assert result["rungs"][1]["peer_ttft_worsened_count"] == 1


def test_ladder_rejects_missing_seed(tmp_path):
    _rung(tmp_path, 1, peer_delta=0)
    with pytest.raises(ValueError, match="Incomplete paired rung"):
        analyze(tmp_path, "ladder", [1], [1, 2])
