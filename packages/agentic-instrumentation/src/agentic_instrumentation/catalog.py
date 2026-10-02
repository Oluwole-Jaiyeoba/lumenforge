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


SIGNALS: dict[str, Signal] = {
    item.signal_id: item for item in (
        Signal("request.accepted", "Backend accepted a request", "request", "low", ("time_ns", "request_id")),
        Signal("batch.scheduled", "Scheduler selected backend work", "batch", "medium", ("time_ns",)),
        Signal("request.completed", "Backend completed a request", "request", "low", ("time_ns", "request_id")),
        Signal("kv.write_host", "KV entered host cache", "cache-node", "medium", ("time_ns", "session_id")),
        Signal("kv.evict_gpu", "KV left device cache", "cache-node", "medium", ("time_ns", "session_id")),
        Signal("kv.evict_host", "KV left host cache", "cache-node", "medium", ("time_ns", "session_id")),
        Signal("kv.load_gpu", "A semantic host-to-device KV load completed", "cache-node", "medium", ("time_ns", "session_id", "device_indices")),
        Signal("kv.layer_copy", "One layer copy completed as part of a load", "layer", "high", ("time_ns", "session_id")),
        Signal("kv.prefix_match", "Replay matched cached KV slots", "request", "medium", ("time_ns", "session_id", "request_id", "matched_indices")),
        Signal("model.forward", "A model forward pass occurred", "batch", "high", ("time_ns",)),
    )
}


@dataclass(frozen=True)
class Profile:
    name: str
    required_signals: tuple[str, ...]
    purpose: str


PROFILES: dict[str, Profile] = {
    item.name: item for item in (
        Profile("controller_queue", ("request.accepted", "batch.scheduled", "request.completed"), "Queue and replay timing"),
        Profile("kv_lifecycle", ("kv.load_gpu", "kv.prefix_match"), "Cache movement and replay match"),
        Profile("copy_timing", ("kv.load_gpu", "kv.layer_copy"), "Host-to-device copy detail"),
        Profile("full_debug", tuple(SIGNALS), "All available lifecycle and scheduler evidence"),
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
