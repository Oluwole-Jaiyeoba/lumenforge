from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


BACKEND_API_SCHEMA_VERSION = "agentic_backend_api.v1"
BACKEND_RUNTIME_SCHEMA_VERSION = "agentic_backend_runtime.v1"


class EffectLevel(str, Enum):
    """How far a controller command actually got.

    ``BackendActionResult.acted`` is kept for backward compatibility with
    existing reports, but it historically meant "accepted for lowering", not
    "the backend confirmed it".  ``effect_level`` makes the difference explicit:

    - ``recorded_only``: intent was recorded; nothing will change in the backend.
    - ``lowered_at_request_boundary``: the command will be encoded into outgoing
      request fields (e.g. a priority field).  The backend has not confirmed it.
    - ``dispatched_to_backend``: handed to a backend control hook (e.g. the
      SGLang prepare-prefix control server); completion is reported elsewhere.
    - ``backend_confirmed``: the backend reported that it executed the action.
    - ``unsupported``: the selected backend/version cannot perform the action.
    """

    RECORDED_ONLY = "recorded_only"
    LOWERED_AT_REQUEST_BOUNDARY = "lowered_at_request_boundary"
    DISPATCHED_TO_BACKEND = "dispatched_to_backend"
    BACKEND_CONFIRMED = "backend_confirmed"
    UNSUPPORTED = "unsupported"


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
class BackendRuntimeInfo:
    """Portable startup handshake returned by a backend runtime.

    The host consumes this record before an experiment starts.  Backend-specific
    probe details remain in ``raw_capabilities``; controller and experiment code
    should use the normalized ``capabilities`` field.
    """

    runtime_profile: str
    backend_name: str
    backend_version: str
    adapter: str
    endpoint: str = ""
    probe_ok: bool = False
    healthy: bool = False
    health_status: str = "not_checked"
    container_image: str = ""
    container_image_digest: str = ""
    gpu_vendor: str = ""
    gpu_architecture: str = ""
    host_architecture: str = ""
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    raw_capabilities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = BACKEND_RUNTIME_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendActionResult:
    command_id: str
    accepted: bool
    acted: bool
    reason: str
    backend_name: str = "observe_only"
    effect_level: str = ""

    @property
    def schema_version(self) -> str:
        return BACKEND_API_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "command_id": self.command_id,
            "accepted": self.accepted,
            "acted": self.acted,
            "reason": self.reason,
            "backend_name": self.backend_name,
        }
        # Only emitted when set, so records produced by adapters that predate
        # ``effect_level`` serialize exactly as before.
        if self.effect_level:
            row["effect_level"] = str(getattr(self.effect_level, "value", self.effect_level))
        return row


@dataclass(frozen=True)
class LaunchSpec:
    """A fully resolved command line for starting a backend server."""

    backend_name: str
    backend_version: str
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "argv": list(self.argv),
            "env": dict(self.env),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CompatibilityFinding:
    """One required or optional backend surface element and whether it exists."""

    requirement: str
    feature: str
    required: bool
    present: bool
    detail: str = ""


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of checking an installed/unpacked backend against an adapter."""

    backend_name: str
    backend_version: str
    adapter: str
    findings: tuple[CompatibilityFinding, ...] = ()

    @property
    def missing_required(self) -> tuple[CompatibilityFinding, ...]:
        return tuple(f for f in self.findings if f.required and not f.present)

    @property
    def missing_optional(self) -> tuple[CompatibilityFinding, ...]:
        return tuple(f for f in self.findings if not f.required and not f.present)

    @property
    def ok(self) -> bool:
        return not self.missing_required

    def broken_features(self) -> tuple[str, ...]:
        return tuple(sorted({f.feature for f in self.missing_required}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "adapter": self.adapter,
            "ok": self.ok,
            "broken_features": list(self.broken_features()),
            "missing_required": [asdict(f) for f in self.missing_required],
            "missing_optional": [asdict(f) for f in self.missing_optional],
            "checked": len(self.findings),
        }


class UnsupportedBackendVersion(RuntimeError):
    """Raised when no adapter can safely drive the installed backend version."""


class BackendCapabilityError(RuntimeError):
    """Raised when a command needs a capability the selected backend lacks."""
