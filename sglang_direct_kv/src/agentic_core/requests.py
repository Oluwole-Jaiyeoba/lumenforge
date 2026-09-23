from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import to_primitive


REQUEST_ENVELOPE_SCHEMA_VERSION = "agentic_request_envelope.v1"


@dataclass(frozen=True)
class RequestEnvelope:
    request_id: str
    session_id: str
    created_at_ms: int
    client_id: str = ""
    task_id: str = ""
    parent_session_id: str = ""
    harness: str = ""
    model: str = ""
    phase: str = ""
    work_class: str = ""
    prefix_id: str = ""
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    expected_output_tokens: int | None = None
    expected_runtime_ms: int | None = None
    ready_at_ms: int | None = None
    deadline_at_ms: int | None = None
    tool_started_at_ms: int | None = None
    tool_expected_done_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = REQUEST_ENVELOPE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
