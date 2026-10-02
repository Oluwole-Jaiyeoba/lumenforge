from agentic_controller.kv_prepare_window import KVPrepareWindowPolicy


def test_policy_holds_until_observed_replay_completion():
    policy = KVPrepareWindowPolicy(estimated_load_ms=250, safety_margin_ms=150)
    inputs = dict(now_ns=1_000_000_000, tool_return_due_ns=1_700_000_000,
                  host_resident=True, slot_released=True, active_replays=0)
    before = policy.decide(**inputs, blocking_replay_finished=False)
    after = policy.decide(**inputs, blocking_replay_finished=True)
    assert (before.action, before.reason) == ("hold", "blocking_replay_not_finished")
    assert (after.action, after.reason, after.remaining_ms) == (
        "load", "quiet_window_before_tool_return", 700)


def test_policy_defers_if_unsafe_or_missing_evidence():
    policy = KVPrepareWindowPolicy(estimated_load_ms=250, safety_margin_ms=150)
    inputs = dict(now_ns=1_000_000_000, tool_return_due_ns=1_399_000_000,
                  host_resident=True, slot_released=True, active_replays=0,
                  blocking_replay_finished=True)
    assert policy.decide(**inputs).reason == "insufficient_window"
    assert policy.decide(**{**inputs, "host_resident": False}).reason == "host_residency_unproved"
    assert policy.decide(**{**inputs, "slot_released": False}).reason == "slot_not_released"
    assert policy.decide(**{**inputs, "active_replays": 1}).reason == "replay_active"
