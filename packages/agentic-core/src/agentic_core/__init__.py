"""Backend-neutral contracts shared by harnesses, controllers, and reports."""

from .backend_request import BACKEND_REQUEST_SCHEMA_VERSION, BackendRequest
from .controller import (
    CONTROLLER_SCHEMA_VERSION,
    BackendCommand,
    ControllerCommand,
    ControllerDecision,
    ControllerEvent,
    EventType,
    KVAction,
    SchedulerAction,
    SessionPhase,
)
from .observations import (
    BackendObservation,
    HardwareTelemetry,
    KVMemoryTier,
    KVResidency,
    SchedulerState,
)
from .requests import RequestEnvelope
from .run_manifest import RunManifest
from .signals import AttachmentLevel, HarnessSignal, SignalProvenance

__all__ = [
    "AttachmentLevel",
    "BACKEND_REQUEST_SCHEMA_VERSION",
    "BackendRequest",
    "BackendCommand",
    "BackendObservation",
    "CONTROLLER_SCHEMA_VERSION",
    "ControllerCommand",
    "ControllerDecision",
    "ControllerEvent",
    "EventType",
    "HardwareTelemetry",
    "HarnessSignal",
    "KVAction",
    "KVMemoryTier",
    "KVResidency",
    "RequestEnvelope",
    "RunManifest",
    "SchedulerAction",
    "SchedulerState",
    "SessionPhase",
    "SignalProvenance",
]
