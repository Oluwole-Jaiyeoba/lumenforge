from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import to_primitive


RUN_MANIFEST_SCHEMA_VERSION = "agentic_run_manifest.v2"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_ms: int
    experiment: str
    seed: int | None = None
    git_commit: str = ""
    model: str = ""
    hardware_profile: str = ""
    backend_name: str = ""
    backend_version: str = ""
    harness_versions: dict[str, str] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    enabled_instrumentation: tuple[str, ...] = ()
    artifact_locations: dict[str, str] = field(default_factory=dict)
    backend_runtime_contract: dict[str, Any] = field(default_factory=dict)
    completion_status: str = "created"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
