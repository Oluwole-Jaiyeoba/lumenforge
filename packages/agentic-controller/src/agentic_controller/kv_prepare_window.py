"""Choose a cache load window from observable session events, without backend APIs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVPrepareDecision:
    action: str
    reason: str
    remaining_ms: float
    estimated_load_ms: float
    safety_margin_ms: float
    active_replays: int


class KVPrepareWindowPolicy:
    def __init__(self, *, estimated_load_ms: float, safety_margin_ms: float):
        if estimated_load_ms <= 0 or safety_margin_ms < 0:
            raise ValueError("load estimate must be positive and margin nonnegative")
        self.estimated_load_ms = estimated_load_ms
        self.safety_margin_ms = safety_margin_ms

    def decide(self, *, now_ns: int, tool_return_due_ns: int,
               host_resident: bool, slot_released: bool,
               active_replays: int, blocking_replay_finished: bool) -> KVPrepareDecision:
        if active_replays < 0:
            raise ValueError("active_replays cannot be negative")
        remaining_ms = (tool_return_due_ns - now_ns) / 1e6
        if not host_resident:
            action, reason = "hold", "host_residency_unproved"
        elif not slot_released:
            action, reason = "hold", "slot_not_released"
        elif not blocking_replay_finished:
            action, reason = "hold", "blocking_replay_not_finished"
        elif active_replays:
            action, reason = "hold", "replay_active"
        elif remaining_ms < self.estimated_load_ms + self.safety_margin_ms:
            action, reason = "defer", "insufficient_window"
        else:
            action, reason = "load", "quiet_window_before_tool_return"
        return KVPrepareDecision(
            action=action, reason=reason, remaining_ms=round(remaining_ms, 3),
            estimated_load_ms=self.estimated_load_ms,
            safety_margin_ms=self.safety_margin_ms, active_replays=active_replays,
        )
