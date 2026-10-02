import asyncio
import time
from types import SimpleNamespace

from agentic_work_audit import read_events

from agentic_experiments.runners import run_work_audit_timing as timing


def test_nonblocking_replay_does_not_wait_for_control_response(tmp_path, monkeypatch):
    replay_started = asyncio.Event()

    async def fake_completion(client, **kwargs):
        started = time.time_ns()
        if kwargs["request_context"]["agentic_kv"]["phase"] == "audit_replay":
            replay_started.set()
        ended = time.time_ns()
        return {"request_start_ns": started, "first_token_ns": ended,
                "request_end_ns": ended, "total_latency_ms": 1}

    async def fake_prepare(client, **kwargs):
        if kwargs["plan_only"]:
            return {"host_tokens": 512, "node_id": "host-node"}
        started = time.time_ns()
        await asyncio.wait_for(replay_started.wait(), timeout=1)
        return {"load_id": "load-1", "loaded_tokens": 512,
                "control_request_started_ns": started,
                "control_response_ns": time.time_ns(), "control_duration_ms": 1}

    async def fake_evict(client, **kwargs):
        return {"ok": True, "evicted_tokens": 512}

    async def fake_confirm(client, url, load_id):
        return {"loaded_tokens": 512, "cuda_elapsed_ms": 1,
                "finished_observed_ns": time.time_ns()}

    monkeypatch.setattr(timing, "completion", fake_completion)
    monkeypatch.setattr(timing, "prepare_prefix", fake_prepare)
    monkeypatch.setattr(timing, "evict_device_prefix", fake_evict)
    monkeypatch.setattr(timing, "eligible_host_prefix", lambda plan: True)
    monkeypatch.setattr(timing, "confirm_load", fake_confirm)
    args = SimpleNamespace(run_id="test", model="test-model", prompt_tokens=512, max_tokens=1,
                           eviction_rounds=1, wait_ms=1, minimum_host_tokens=512,
                           base_url="", prepare_control_url="")
    path = tmp_path / "harness_events.jsonl"
    log = timing.EventLog(path)
    try:
        result = asyncio.run(timing.one_case(object(), args, log, 1, "late_nonblocking"))
    finally:
        log.close()
    events = read_events(path)
    requested = next(row.ts_ns for row in events if row.kind == "load_requested")
    replay = next(row.ts_ns for row in events if row.kind == "replay_sent")
    accepted = next(row.ts_ns for row in events if row.kind == "load_accepted")
    assert requested <= replay < accepted
    assert result["load_acceptance"]["load_id"] == "load-1"
