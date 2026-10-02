import asyncio
import time
from types import SimpleNamespace

import pytest

from agentic_work_audit import read_events
from agentic_experiments.runners import run_work_audit_multisession as study


@pytest.mark.parametrize("condition", ["early", "late_nonblocking", "post_short"])
def test_multisession_comparison_releases_slot_and_times_load(tmp_path, monkeypatch, condition):
    async def fake_completion(client, **kwargs):
        now = time.time_ns()
        return {"request_start_ns": now, "first_token_ns": now + 1,
                "request_end_ns": now + 2, "ttft_ms": 0.001}

    async def fake_prepare(client, **kwargs):
        now = time.time_ns()
        if kwargs["plan_only"]:
            return {"ok": True, "status": "would_load_back", "host_tokens": 1024,
                    "minimum_host_tokens": 512, "node_id": "host-1"}
        return {"load_id": "load-1", "loaded_tokens": 1024,
                "control_request_started_ns": now, "control_response_ns": now + 1}

    async def fake_evict(client, **kwargs):
        return {"ok": True, "evicted_tokens": 1024}

    async def fake_confirm(client, url, load_id):
        return {"loaded_tokens": 1024, "cuda_elapsed_ms": 1,
                "finished_observed_ns": time.time_ns()}

    monkeypatch.setattr(study, "completion", fake_completion)
    monkeypatch.setattr(study, "prepare_prefix", fake_prepare)
    monkeypatch.setattr(study, "evict_device_prefix", fake_evict)
    monkeypatch.setattr(study, "confirm_load", fake_confirm)
    args = SimpleNamespace(model="model", prompt_tokens=512, max_tokens=1,
                           eviction_rounds=1, short_wait_ms=5, long_wait_ms=30,
                           early_at_ms=10, minimum_host_tokens=512,
                           base_url="", prepare_control_url="")
    path = tmp_path / "events.jsonl"
    log = study.EventLog(path)
    try:
        result = asyncio.run(study.one_case(object(), args, log, "case", condition))
    finally:
        log.close()
    rows = read_events(path)
    def at(kind, label=None):
        return next(row.ts_ns for row in rows if row.kind == kind and
                    (label is None or row.session_id.endswith(f"-{label}")))
    assert result["condition"] == condition
    assert at("session_end") <= at("ended_prefix_evict_proof")
    assert at("ended_prefix_evict_proof") < at("load_requested")
    if condition == "early":
        assert at("load_requested") < at("tool_end", "long")
    elif condition == "post_short":
        assert at("replay_finished", "short") <= at("load_requested")
        assert at("load_requested") < at("tool_end", "long")
    else:
        assert at("tool_end", "long") <= at("load_requested")
