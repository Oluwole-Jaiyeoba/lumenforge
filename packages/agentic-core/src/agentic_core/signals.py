from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .serialization import to_primitive


HARNESS_SIGNAL_SCHEMA_VERSION = "agentic_harness_signal.v1"


class AttachmentLevel(str, Enum):
    CONFIGURATION = "configuration"
    SESSION = "session"
    TASK = "task"
    REQUEST = "request"


class SignalProvenance(str, Enum):
    NATIVE_HARNESS = "native_harness"
    CONFIGURATION_DRIVEN = "configuration_driven"
    ADAPTER_DERIVED = "adapter_derived"
    INJECTED = "injected"
    REPLAYED = "replayed"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class HarnessSignal:
    signal_id: str
    name: str
    harness: str
    observed_at_ms: int
    value: Any = None
    attachment_level: AttachmentLevel = AttachmentLevel.REQUEST
    provenance: SignalProvenance = SignalProvenance.NATIVE_HARNESS
    source_lane: str = "native_harness"
    native_field: str = ""
    raw_location: str = ""
    client_id: str = ""
    session_id: str = ""
    task_id: str = ""
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = HARNESS_SIGNAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
