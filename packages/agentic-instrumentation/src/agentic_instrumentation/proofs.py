"""Conservative cache-slot lineage checks, independent of the backend version."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .events import EvidenceEvent


def _indices(summary: Any) -> set[int] | None:
    if not isinstance(summary, dict):
        return None
    values = summary.get("values")
    count = summary.get("index_count", summary.get("numel"))
    digest = summary.get("sha1_16")
    if isinstance(values, list) and values and all(type(value) is int for value in values):
        if count is not None and int(count) != len(values):
            return None
    else:
        low, high = summary.get("min"), summary.get("max")
        if not all(type(value) is int for value in (low, high, count)) or high - low + 1 != count:
            return None
        values = list(range(low, high + 1))
    if digest:
        actual = hashlib.sha1(",".join(map(str, values)).encode("utf-8")).hexdigest()[:16]
        if actual != digest:
            return None
    elif "values" not in summary:
        return None
    return set(values)


def assess_loaded_match(
    loads: Iterable[EvidenceEvent], matches: Iterable[EvidenceEvent],
    evictions: Iterable[EvidenceEvent] = (),
) -> dict[str, object]:
    """Find loaded device slots reused in a later replay cache match.

    A supported match is stronger than a prefix-length correlation, but is not
    proof that model kernels consumed these slots. An intervening eviction is
    deliberately treated as ambiguous because slot indices can be reused.
    """
    load_rows = list(loads)
    match_rows = list(matches)
    eviction_rows = list(evictions)
    best = 0
    for match in match_rows:
        matched = _indices(match.payload.get("matched_indices"))
        if not matched or not match.session_id or not match.request_id:
            continue
        for load in load_rows:
            if load.session_id != match.session_id or load.time_ns >= match.time_ns:
                continue
            loaded = _indices(load.payload.get("device_indices"))
            if not loaded:
                continue
            if any(load.time_ns < eviction.time_ns <= match.time_ns
                   for eviction in eviction_rows):
                continue
            best = max(best, len(loaded & matched))
    return {
        "loaded_slots_matched_by_replay": best,
        "loaded_slots_match_status": "supported" if best else "unknown",
        "exact_model_consumption": "not_proven" if load_rows else "not_applicable",
        "limitation": "A matched GPU slot is not proof that a model kernel read that exact slot; unobserved eviction or slot reuse remains possible.",
    }
