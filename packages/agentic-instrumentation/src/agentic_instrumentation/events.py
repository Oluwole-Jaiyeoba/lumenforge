"""Versioned, nanosecond-precision evidence passed between backend and lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceEvent:
    signal_id: str
    time_ns: int
    source: str
    session_id: str = ""
    request_id: str = ""
    correlation_id: str = ""
    phase: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
