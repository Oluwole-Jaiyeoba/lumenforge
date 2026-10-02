"""Stable signal names and minimum evidence requirements across research lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .events import EvidenceEvent


@dataclass(frozen=True)
class Signal:
    signal_id: str
    meaning: str
    scope: str
    cost: str
    required_fields: tuple[str, ...]
    producer: str = "backend hook"
    proof_limit: str = "The event alone does not prove task completion or performance benefit."


SIGNALS: dict[str, Signal] = {
    item.signal_id: item for item in (
        Signal("request.accepted", "Backend accepted a request", "request", "low", ("time_ns", "request_id")),
        Signal("batch.scheduled", "Scheduler selected backend work", "batch", "medium", ("time_ns",)),
        Signal("batch.completed", "Scheduler finished processing a batch result", "batch", "medium", ("time_ns",),
               proof_limit="Not proof that each request in the batch finished."),
        Signal("kv.write_host", "KV entered host cache", "cache-node", "medium", ("time_ns", "session_id")),
        Signal("kv.evict_gpu", "KV left device cache", "cache-node", "medium", ("time_ns", "session_id")),
        Signal("kv.evict_host", "KV left host cache", "cache-node", "medium", ("time_ns", "session_id")),
        Signal("kv.load_gpu", "A semantic host-to-device KV load call returned", "cache-node", "medium", ("time_ns", "session_id", "device_indices"),
               proof_limit="May enqueue asynchronous work; confirm completion and replay match separately."),
        Signal("kv.nested_load", "Nested cache load operation returned", "cache-node", "medium", ("time_ns",),
               proof_limit="A nested operation is not another semantic load."),
        Signal("kv.layer_copy", "One layer copy completed as part of a load", "layer", "high", ("time_ns", "session_id")),
        Signal("kv.layer_backup", "Per-layer device-to-host backup detail", "layer", "high", ("time_ns",),
               proof_limit="Not another semantic host-cache write."),
        Signal("kv.detail", "Version-specific cache detail", "backend", "high", ("time_ns",),
               proof_limit="Requires explicit version-adapter interpretation before making a lifecycle claim."),
        Signal("kv.prefix_match", "Replay matched cached KV slots", "request", "medium", ("time_ns", "session_id", "request_id", "matched_indices"),
               proof_limit="Prefix match is not proof the model consumed those slots."),
        Signal("model.forward", "A model forward pass occurred", "batch", "high", ("time_ns",)),
    )
}


@dataclass(frozen=True)
class Profile:
    name: str
    required_signals: tuple[str, ...]
    purpose: str
    optional_signals: tuple[str, ...] = ()


PROFILES: dict[str, Profile] = {
    item.name: item for item in (
        Profile("request_boundary", ("request.accepted",), "Backend request acceptance for hint-boundary audits"),
        Profile("controller_queue", ("request.accepted", "batch.scheduled", "batch.completed"), "Queue and replay timing"),
        Profile("kv_lifecycle", ("kv.load_gpu", "kv.prefix_match"), "Cache movement and replay match"),
        Profile("copy_timing", ("kv.load_gpu", "kv.layer_copy"), "Host-to-device copy detail"),
        Profile("full_debug", ("request.accepted", "batch.scheduled", "batch.completed"),
                "All available lifecycle and scheduler evidence",
                tuple(signal for signal in SIGNALS if signal not in ("request.accepted", "batch.scheduled", "batch.completed"))),
    )
}


def validate_profile(profile: str, observed: set[str]) -> dict[str, object]:
    """Fail closed on missing evidence; this checks events, not hook installation."""
    required = set(PROFILES[profile].required_signals)
    missing = sorted(required - observed)
    return {"profile": profile, "valid": not missing, "missing": missing, "observed": sorted(observed)}


def validate_events(profile: str, events: Iterable[EvidenceEvent]) -> dict[str, object]:
    """Check both signal presence and required evidence fields for a live case."""
    rows = list(events)
    result = validate_profile(profile, {row.signal_id for row in rows})
    invalid_fields = []
    for signal_id in PROFILES[profile].required_signals:
        signal = SIGNALS[signal_id]
        matching = [row for row in rows if row.signal_id == signal_id]
        if matching and not any(all(
            (getattr(row, field, None) if hasattr(row, field) else row.payload.get(field))
            not in (None, "", {}) for field in signal.required_fields
        ) for row in matching):
            invalid_fields.append(signal_id)
    result["invalid_fields"] = invalid_fields
    result["valid"] = result["valid"] and not invalid_fields
    return result
