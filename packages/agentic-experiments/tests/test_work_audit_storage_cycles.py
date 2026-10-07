import asyncio
import gzip
import json
import time
from argparse import Namespace

from agentic_experiments.runners.analyze_work_audit_storage_cycles import analyze, native_hits
from agentic_experiments.runners import run_work_audit_storage_cycles as runner


def _case(root, seed, arm, *, due_ms, workflow_ms, hit_tokens):
    folder = root / f"seed{seed}_{arm}"
    folder.mkdir()
    request_id = f"s{seed}-turn01"
    case = {
        "arm": arm, "seed": seed, "started_ns": 100,
        "session_count": 1, "workflow_duration_ms": workflow_ms,
        "sessions": [{"completion_ms": workflow_ms}],
        "natural_storage_candidate_waits": 1,
        "inspections_before_due": 1,
        "stage_before_due_count": int(arm == "host_stage"),
        "turns": [{"turn": 1, "request_id": request_id,
                   "due_to_first_token_ms": due_ms, "ttft_ms": due_ms,
                   "prompt_tokens": 3000,
                   "preparation": {"stage": {"completed": {"storage_loaded_tokens": 512}}}
                   if arm == "host_stage" else None}],
    }
    (folder / "case_results.json").write_text(json.dumps(case), encoding="utf-8")
    with gzip.open(folder / "backend_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        for event in (
            {"event": "hiradix.storage_hit_tokens.end", "result": hit_tokens,
             "kv_context": {"request_id": "backend-rid"}},
            {"event": "hiradix.match_prefix.end",
             "kv_context": {"request": {"rid": "backend-rid"},
                            "agent_request_id": request_id}},
        ):
            handle.write(json.dumps(event) + "\n")
    return folder


def test_native_hit_joins_backend_rid_to_replay(tmp_path):
    folder = _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000,
                   hit_tokens=2048)
    assert native_hits(folder / "backend_trace.jsonl.gz") == {"s1-turn01": 2048}


def test_paired_storage_cycles_keep_workflow_and_stage_cost(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    _case(tmp_path, 1, "host_stage", due_ms=200, workflow_ms=9500, hit_tokens=0)
    summary = analyze(tmp_path)
    assert summary["paired_comparisons"] == [{
        "seed": 1, "mode": "host_stage", "workflow_delta_ms": 500,
        "median_due_to_first_token_delta_ms": -200,
        "median_replay_ttft_delta_ms": -200,
    }]
    staged = next(row for row in summary["arms"] if row["arm"] == "host_stage")
    assert staged["stage_loaded_tokens"] == 512
    assert staged["stage_before_due_count"] == 1


def test_failed_arm_blocks_three_seed_conclusion(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    _case(tmp_path, 1, "host_stage", due_ms=200, workflow_ms=9500, hit_tokens=0)
    failure = tmp_path / "seed2_host_stage"
    failure.mkdir()
    (failure / "case_failure.json").write_text(json.dumps({"error": "HTTP 500"}), encoding="utf-8")
    (failure / "server.log").write_text("AssertionError: cache tree\n", encoding="utf-8")
    summary = analyze(tmp_path, expected_seeds=[1, 2, 3],
                      expected_arms=["on_demand", "host_stage"])
    assert summary["status"] == "blocked"
    assert len(summary["paired_comparisons"]) == 1
    assert summary["failed_arms"][0]["backend_assertion"] == "cache tree"
    assert "seed3_host_stage" in summary["missing_arm_ids"]


def test_capacity_skip_is_visible_but_does_not_claim_failure(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    staged = _case(tmp_path, 1, "host_stage", due_ms=200, workflow_ms=9500, hit_tokens=0)
    path = staged / "case_results.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    case["turns"].append({"turn": 2, "request_id": "s1-turn02",
                          "due_to_first_token_ms": 300, "ttft_ms": 300,
                          "prompt_tokens": 3500,
                          "preparation": {"skip_reason": "host_capacity_insufficient"}})
    case["turns"].append({"turn": 3, "request_id": "s1-turn03",
                          "due_to_first_token_ms": 320, "ttft_ms": 320,
                          "prompt_tokens": 3600,
                          "preparation": {"skip_reason": "storage_prefetch_rate_limited"}})
    path.write_text(json.dumps(case), encoding="utf-8")
    result = analyze(tmp_path, expected_seeds=[1], expected_arms=["on_demand", "host_stage"])
    assert result["status"] == "complete"
    assert result["arms"][1]["preparation_skips"] == [
        "host_capacity_insufficient", "storage_prefetch_rate_limited",
    ]


def test_unexpected_preparation_error_blocks_claim(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    staged = _case(tmp_path, 1, "host_stage", due_ms=200, workflow_ms=9500, hit_tokens=0)
    path = staged / "case_results.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    case["turns"][0]["preparation"] = {"error": "storage_prefix_anchor_mismatch"}
    path.write_text(json.dumps(case), encoding="utf-8")
    result = analyze(tmp_path, expected_seeds=[1], expected_arms=["on_demand", "host_stage"])
    assert result["status"] == "blocked"


def test_selective_stage_keeps_session_and_stage_outcomes(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    staged = _case(tmp_path, 1, "selective_stage", due_ms=200, workflow_ms=8800,
                   hit_tokens=0)
    path = staged / "case_results.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    case["sessions"][0]["session_id"] = "session-1"
    case["stage_before_due_count"] = 1
    case["turns"][0]["preparation"] = {
        "stage": {"completed": {"storage_loaded_tokens": 512}},
        "stage_before_due": True,
    }
    case["turns"].append({"turn": 2, "request_id": "s1-turn02",
                          "due_to_first_token_ms": 300, "ttft_ms": 300,
                          "prompt_tokens": 3500,
                          "preparation": {"skip_reason": "stage_already_in_flight"}})
    path.write_text(json.dumps(case), encoding="utf-8")
    baseline_path = tmp_path / "seed1_on_demand" / "case_results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["sessions"][0]["session_id"] = "session-1"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    summary = analyze(tmp_path, expected_seeds=[1], expected_arms=["on_demand", "selective_stage"])
    assert summary["status"] == "complete"
    assert summary["paired_comparisons"][0]["session_completion_delta_by_id_ms"] == {"session-1": -200}
    assert summary["arms"][1]["stage_admit_count"] == 1
    assert summary["arms"][1]["preparation_skips"] == ["stage_already_in_flight"]
    assert summary["staged_replay_outcomes"][0]["delta_ms"] == -200
    assert summary["staged_replay_outcomes"][0]["staged_storage_tokens"] == 512


def test_selective_stage_skips_short_window_and_inflight_stage(monkeypatch):
    async def residency(*_args, **_kwargs):
        return {"storage_candidate_tokens": 512}

    async def unexpected_stage(*_args, **_kwargs):
        raise AssertionError("stage should have been skipped")

    monkeypatch.setattr(runner, "control", residency)
    monkeypatch.setattr(runner, "prepare_during_wait", unexpected_stage)
    args = Namespace(control_url="http://unused", page_size=64,
                     arm="selective_stage", min_stage_slack_ms=1200)

    async def check():
        semaphore = asyncio.Semaphore(1)
        short = await runner.inspect_during_wait(None, args, {}, "id",
                                                 time.time_ns() + 500_000_000, 0, semaphore)
        assert short["preparation"]["skip_reason"] == "insufficient_slack"
        async with semaphore:
            busy = await runner.inspect_during_wait(None, args, {}, "id",
                                                    time.time_ns() + 2_000_000_000, 0, semaphore)
        assert busy["preparation"]["skip_reason"] == "stage_already_in_flight"

    asyncio.run(check())


def test_selective_stage_requires_exposure_in_each_paired_seed(tmp_path):
    for seed in (1, 2):
        _case(tmp_path, seed, "on_demand", due_ms=400, workflow_ms=9000,
              hit_tokens=2048)
        staged = _case(tmp_path, seed, "selective_stage", due_ms=200,
                       workflow_ms=8800, hit_tokens=0)
        if seed == 1:
            path = staged / "case_results.json"
            case = json.loads(path.read_text(encoding="utf-8"))
            case["stage_before_due_count"] = 1
            path.write_text(json.dumps(case), encoding="utf-8")
    summary = analyze(tmp_path, expected_seeds=[1, 2],
                      expected_arms=["on_demand", "selective_stage"])
    assert summary["status"] == "insufficient_exposure"
    assert summary["selective_seed_exposure"][1]["valid"] is True
    assert summary["selective_seed_exposure"][2]["valid"] is False
