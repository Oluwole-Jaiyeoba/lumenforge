"""Backend-neutral, ideal paired schedule used by the KV lifecycle audit."""

from dataclasses import dataclass


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
