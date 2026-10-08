"""Backend-neutral schedules used by the coordinated KV lifecycle audit."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class PairRotation:
    sessions: int = 20
    wait_ms: int = 1000

    def __post_init__(self):
        if self.sessions < 4 or self.sessions % 4 or self.wait_ms <= 0:
            raise ValueError("Use four equally sized groups and a positive tool wait")

    def pair(self, session_index: int) -> int:
        if not 0 <= session_index < self.sessions:
            raise ValueError("Unknown session")
        return session_index // (self.sessions // 2)

    def boundary(self, previous_boundary: float, pair_finished: float) -> float:
        return max(previous_boundary + self.wait_ms / 1000, pair_finished)

    def tool_due(self, wait_started: float) -> float:
        return wait_started + self.wait_ms / 1000


@dataclass(frozen=True)
class SessionPipelinePolicy:
    """Deadline-ordered, session-level preparation without semantic priority."""

    sessions: int = 20
    wait_ms: int = 1000
    prefetch_lead_ms: int = 750
    headroom_sessions: int = 2
    max_inflight: int = 8

    def __post_init__(self) -> None:
        if self.sessions < 2 or self.wait_ms <= 0:
            raise ValueError("Use at least two sessions and a positive tool wait")
        if not 0 < self.prefetch_lead_ms < self.wait_ms:
            raise ValueError("Prefetch lead must fit inside the tool wait")
        if self.headroom_sessions < 1:
            raise ValueError("Reserve at least one session of preparation space")
        if not 0 < self.max_inflight < self.sessions:
            raise ValueError("Max inflight must leave waiting sessions to prepare")

    def initial_due_ns(self, started_ns: int, session_index: int) -> int:
        """Spread the first arrivals across one tool-wait interval."""
        if not 0 <= session_index < self.sessions:
            raise ValueError("Unknown session")
        return started_ns + session_index * self.wait_ms * 1_000_000 // self.sessions

    def prefetch_at_ns(self, tool_due_ns: int) -> int:
        return tool_due_ns - self.prefetch_lead_ms * 1_000_000

    def next_tool_due_ns(self, response_end_ns: int) -> int:
        return response_end_ns + self.wait_ms * 1_000_000

    @staticmethod
    def victim(states: Iterable[Mapping[str, Any]], incoming_due_ns: int) -> Mapping[str, Any] | None:
        """Choose the safely idle resident session needed furthest in the future."""
        candidates = [state for state in states
                      if state.get("resident") and not state.get("active")
                      and not state.get("loading") and not state.get("claimed")
                      and state.get("due_ns", 0) > incoming_due_ns]
        return max(candidates, key=lambda state: state["due_ns"], default=None)
