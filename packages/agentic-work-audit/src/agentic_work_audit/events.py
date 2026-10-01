"""Versioned, backend-neutral audit events."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "agentic_work_audit.event.v1"


@dataclass(frozen=True)
class AuditEvent:
    kind: str
    ts_ns: int
    source: str
    session_id: str = ""
    request_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or not self.kind or not self.source or self.ts_ns <= 0:
            raise ValueError("invalid audit event")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AuditEvent":
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_events(path: Path) -> list[AuditEvent]:
    events: list[AuditEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(AuditEvent.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return events


def write_events(path: Path, events: Iterable[AuditEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
