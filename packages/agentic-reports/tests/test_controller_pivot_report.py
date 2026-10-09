from copy import deepcopy
from pathlib import Path

from agentic_reports.analysis.analyze_controller_audit_pivots import fingerprint, model_prompt_hash, pair_comparison
from agentic_reports.builders.build_work_audit_report import render, render_markdown
from agentic_backends.controller_audit import runtime_issues


def sample_summary():
    arms = []
    pairs = []
    for trial in (1, 2):
        for mode, scale in (("no_prefetch", 1), ("controller_ready_time_gpu_backfill", 0.8)):
            arms.append({
                "trial": trial, "mode": mode, "label": f"t{trial}_{mode}", "issues": [],
                "metrics": {"workload_ms": 10000 * scale, "total_all_ttft_ms": 20000 * scale,
                            "total_replay_ttft_ms": 15000 * scale, "total_replay_lateness_ms": 18000 * scale,
                            "median_replay_ttft_ms": 400 * scale,
                            "median_replay_delay_ms": 500 * scale, "p95_replay_delay_ms": 700 * scale,
                            "requests": 48, "replays": 32},
                "request_fingerprint": [["a", "replay", "1", "hash", "8", "1000"]],
                "backend_args": {"max_total_tokens": 24576},
                "case_path": f"raw/case{trial}_{mode}", "report_path": f"raw/report{trial}_{mode}",
            })
        pairs.append(pair_comparison(*arms[-2:]))
    return {
        "schema": "agentic_work_audit.controller_pivot.summary.v1", "run_id": "sample",
        "scenario": "1", "name": "Replay scheduling", "status": "complete", "arms": arms, "pairs": pairs,
        "source_revision": "a" * 40, "limits": "Synthetic test", "matching": "Same workload",
        "_manifest": {"model": "Qwen", "image": "pinned-image", "spec": {"seeds": [1, 2]}},
    }


def test_html_and_markdown_show_each_trial_and_exact_reproduction():
    summary = sample_summary()
    path = Path("docs/reports/work_audit/sample/summary.json")
    for document in (render([(path, summary)]), render_markdown([(path, summary)])):
        assert "All requests: total TTFT" in document
        assert "Replay TTFT median (ms)" in document
        assert "Total replay lateness" in document
        assert "10.000" in document and "8.000" in document
        assert "source.tar.gz" in document
        assert "experiment_spec.json" in document
        assert "run_controller_audit_pivots" in document
        assert "--scenarios 1" in document
        assert "a" * 40 in document
        assert "Trial 2" in document


def test_pair_gate_rejects_workload_and_capacity_mismatch():
    baseline, treatment = deepcopy(sample_summary()["arms"][:2])
    assert pair_comparison(baseline, treatment)["comparable"]
    treatment["request_fingerprint"][0][-1] = "2000"
    result = pair_comparison(baseline, treatment)
    assert not result["comparable"]
    assert result["metrics"]["workload_ms"]["change_pct"] is None
    treatment["request_fingerprint"] = baseline["request_fingerprint"]
    treatment["backend_args"]["max_total_tokens"] = 12288
    assert not pair_comparison(baseline, treatment)["comparable"]


def test_invalid_pair_is_not_reported_as_win():
    summary = sample_summary()
    summary["pairs"][0]["comparable"] = False
    summary["pairs"][0]["issues"] = ["Request/prompt/output/wait identity mismatch"]
    summary["status"] = "invalid_comparison"
    markdown = render_markdown([(Path("sample/summary.json"), summary)])
    assert "Comparison not established" in markdown
    assert "withheld" in markdown


def test_pairing_uses_model_prompt_not_transport_metadata_hash():
    baseline = {"session_id": "a", "phase": "replay", "tool_wait_step": 1,
                "max_tokens": 8, "tool_wait_ms": 1000, "prompt_hash": "baseline-body",
                "harness_controller_signal": {"cache": {"conversation_prefix_hash": "same-model-prompt"}}}
    treatment = deepcopy(baseline)
    treatment["prompt_hash"] = "controller-body"
    assert fingerprint([baseline]) == fingerprint([treatment])
    treatment["harness_controller_signal"]["cache"]["conversation_prefix_hash"] = "different-model-prompt"
    assert fingerprint([baseline]) != fingerprint([treatment])
    del treatment["harness_controller_signal"]
    assert model_prompt_hash(treatment) == ""


def test_runtime_gate_requires_graphs_overlap_and_isolated_queue_policy():
    args = {"disable_cuda_graph": False, "disable_overlap_schedule": False,
            "enable_priority_scheduling": False, "radix_eviction_policy": "priority"}
    assert runtime_issues(args, queue_ranking=False, retention_ranking=True) == []
    assert "Unexpected retention policy outside scenario 3" in runtime_issues(args, queue_ranking=False, retention_ranking=False)
    assert "Queue isolation mismatch" in runtime_issues(args, queue_ranking=True, retention_ranking=False)
    args["disable_overlap_schedule"] = True
    assert "CUDA graphs/overlap setting mismatch" in runtime_issues(args, queue_ranking=False, retention_ranking=True)
    del args["disable_cuda_graph"]
    args["radix_eviction_policy"] = "lru"
    assert "Retention policy missing" in runtime_issues(args, queue_ranking=False, retention_ranking=True)
