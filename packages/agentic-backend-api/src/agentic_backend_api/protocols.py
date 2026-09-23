"""Interfaces every inference-backend integration implements.

Controller, harness and gateway code depend only on these protocols and on
``agentic_core`` records.  A concrete backend (``agentic_backends.sglang``,
or a future ``agentic_backends.vllm``) implements them for specific versions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from agentic_core import BackendCommand, BackendObservation, BackendRequest

from .models import BackendActionResult, BackendCapabilities, CompatibilityReport, LaunchSpec


@runtime_checkable
class BackendAdapter(Protocol):
    """Receives controller commands and reports what happened to them."""

    def capabilities(self) -> BackendCapabilities:
        ...

    def apply(self, command: BackendCommand) -> BackendActionResult:
        ...


@runtime_checkable
class RequestLowering(Protocol):
    """Encodes a backend-neutral request into the backend's native JSON body."""

    def lower(self, request: BackendRequest, *, model: str) -> dict[str, Any]:
        ...


@runtime_checkable
class TelemetryNormalizer(Protocol):
    """Turns a raw backend trace/event row into a normalized observation."""

    def normalize(self, raw_event: Mapping[str, Any]) -> BackendObservation | None:
        ...


@runtime_checkable
class LaunchPlanner(Protocol):
    """Builds and validates the command line that starts the backend server."""

    def launch_spec(
        self,
        *,
        model: str,
        host: str = "0.0.0.0",
        port: int = 30000,
        options: Mapping[str, Any] | None = None,
    ) -> LaunchSpec:
        ...


@runtime_checkable
class CompatibilityProbe(Protocol):
    """Checks that an installed backend exposes every surface an adapter needs."""

    def check(self, source_root: str | None = None) -> CompatibilityReport:
        ...
