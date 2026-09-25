"""Portable capability and command interface for inference backends."""

from .models import (
    BACKEND_API_SCHEMA_VERSION,
    BACKEND_RUNTIME_SCHEMA_VERSION,
    BackendActionResult,
    BackendCapabilities,
    BackendCapabilityError,
    BackendRuntimeInfo,
    CompatibilityFinding,
    CompatibilityReport,
    EffectLevel,
    LaunchSpec,
    UnsupportedBackendVersion,
)
from .protocols import (
    BackendAdapter,
    CompatibilityProbe,
    LaunchPlanner,
    RequestLowering,
    TelemetryNormalizer,
)

__all__ = [
    "BACKEND_API_SCHEMA_VERSION",
    "BACKEND_RUNTIME_SCHEMA_VERSION",
    "BackendActionResult",
    "BackendAdapter",
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendRuntimeInfo",
    "CompatibilityFinding",
    "CompatibilityProbe",
    "CompatibilityReport",
    "EffectLevel",
    "LaunchPlanner",
    "LaunchSpec",
    "RequestLowering",
    "TelemetryNormalizer",
    "UnsupportedBackendVersion",
]
