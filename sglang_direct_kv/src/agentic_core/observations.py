from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .serialization import to_primitive


BACKEND_OBSERVATION_SCHEMA_VERSION = "agentic_backend_observation.v1"
SCHEDULER_STATE_SCHEMA_VERSION = "agentic_scheduler_state.v1"
KV_RESIDENCY_SCHEMA_VERSION = "agentic_kv_residency.v1"
HARDWARE_TELEMETRY_SCHEMA_VERSION = "agentic_hardware_telemetry.v1"


class KVMemoryTier(str, Enum):
    GPU = "gpu"
    HOST = "host"
    STORAGE = "storage"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BackendObservation:
    observation_id: str
    observed_at_ms: int
    backend_name: str
    kind: str
    backend_version: str = ""
    request_id: str = ""
    session_id: str = ""
    correlation_id: str = ""
    source_event: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = BACKEND_OBSERVATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class SchedulerState:
    observed_at_ms: int
    backend_name: str
    waiting_request_ids: tuple[str, ...] = ()
    running_request_ids: tuple[str, ...] = ()
    held_request_ids: tuple[str, ...] = ()
    queue_depth: int = 0
    batch_size: int = 0
    max_concurrency: int | None = None
    available_slots: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEDULER_STATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class KVResidency:
    prefix_id: str
    observed_at_ms: int
    tier: KVMemoryTier = KVMemoryTier.UNKNOWN
    block_ids: tuple[str, ...] = ()
    bytes_resident: int | None = None
    reusable_tokens: int | None = None
    reuse_probability: float | None = None
    movement_state: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = KV_RESIDENCY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class HardwareTelemetry:
    observed_at_ms: int
    device_id: str
    gpu_utilization_pct: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    memory_bandwidth_utilization_pct: float | None = None
    power_watts: float | None = None
    temperature_c: float | None = None
    throttling_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = HARDWARE_TELEMETRY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
