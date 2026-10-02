import pytest

from agentic_controller.kv_busy_prepare import BusyPreparePolicy


def test_busy_prepare_requires_host_proof_and_time() -> None:
    policy = BusyPreparePolicy(estimated_load_ms=250, margin_ms=150)
    now = 1_000_000_000
    assert policy.decide(now_ns=now, tool_due_ns=now + 1_000_000_000,
                         host_resident=False, active_replays=2).action == "hold"
    assert policy.decide(now_ns=now, tool_due_ns=now + 300_000_000,
                         host_resident=True, active_replays=2).action == "defer"
    decision = policy.decide(now_ns=now, tool_due_ns=now + 600_000_000,
                             host_resident=True, active_replays=2)
    assert decision.action == "load"
    assert decision.active_replays == 2


def test_busy_prepare_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        BusyPreparePolicy(estimated_load_ms=0, margin_ms=0)
    with pytest.raises(ValueError):
        BusyPreparePolicy(estimated_load_ms=1, margin_ms=-1)
    with pytest.raises(ValueError):
        BusyPreparePolicy(estimated_load_ms=1, margin_ms=0).decide(
            now_ns=0, tool_due_ns=1, host_resident=True, active_replays=-1)
