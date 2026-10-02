"""Conservative, priority-free KV preparation during an observed tool wait."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BusyPrepareDecision:
    action: str
    reason: str
    remaining_ms: float
    active_replays: int


class BusyPreparePolicy:
    def __init__(self, *, estimated_load_ms: float, margin_ms: float):
        if estimated_load_ms <= 0 or margin_ms < 0:
            raise ValueError("estimated load must be positive and margin nonnegative")
        self.minimum_window_ms = estimated_load_ms + margin_ms

    def decide(self, *, now_ns: int, tool_due_ns: int, host_resident: bool,
               active_replays: int) -> BusyPrepareDecision:
        if active_replays < 0:
            raise ValueError("active_replays cannot be negative")
        remaining_ms = (tool_due_ns - now_ns) / 1_000_000
        if not host_resident:
            action, reason = "hold", "host_residency_not_proved"
        elif remaining_ms < self.minimum_window_ms:
            action, reason = "defer", "too_little_wait_remaining"
        else:
            action, reason = "load", "host_prefix_ready_before_tool_return"
        return BusyPrepareDecision(action, reason, round(remaining_ms, 3), active_replays)
