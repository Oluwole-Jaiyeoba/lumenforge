from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BACKEND_API_SCHEMA_VERSION = "agentic_backend_api.v1"


@dataclass(frozen=True)
class BackendCapabilities:
    priority_queue: bool = False
    background_prefill_budget: bool = False
    safe_preemption: bool = False
    kv_demote: bool = False
    kv_prefetch: bool = False
    kv_release: bool = False
    live_metrics: bool = False
    observe_only: bool = True
    backend_name: str = "unknown"
    backend_version: str = ""

    @property
    def schema_version(self) -> str:
        return BACKEND_API_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendActionResult:
    command_id: str
    accepted: bool
    acted: bool
    reason: str
    backend_name: str = "observe_only"

    @property
    def schema_version(self) -> str:
        return BACKEND_API_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "accepted": self.accepted,
            "acted": self.acted,
            "reason": self.reason,
            "backend_name": self.backend_name,
        }
