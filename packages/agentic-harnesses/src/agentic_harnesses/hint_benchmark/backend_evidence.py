"""Compare native hint capture with normalized backend request evidence.

An explicit mapping identifies the candidate backend request. A matching
correlation ID is needed to prove both records came from the same request.
Neither check alone proves that the backend used a scheduling/cache hint.
"""

from __future__ import annotations

from typing import Any


_NATIVE_TIERS = {"native_client_or_transport_capture", "native_client_real_provider_response"}


def audit_backend_mappings(
    observations: list[dict[str, Any]], mappings: list[dict[str, Any]],
    backend_events: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    errors = []
    for mapping in mappings:
        scenario = str(mapping.get("scenario_id") or "")
        payload_index = mapping.get("payload_index")
        request_id = str(mapping.get("backend_request_id") or "")
        correlation_id = str(mapping.get("correlation_id") or "")
        if not scenario or not isinstance(payload_index, int) or payload_index < 1 or not request_id:
            errors.append({"mapping": mapping, "reason": "scenario_id, positive payload_index, and backend_request_id required"})
            continue
        native = []
        for row in observations:
            if (row.get("scenario_id") != scenario or row.get("payload_index") != payload_index
                    or row.get("evidence_tier") not in _NATIVE_TIERS):
                continue
            raw = row.get("raw_emitted_value")
            capture = raw.get("_capture") if isinstance(raw, dict) else None
            if isinstance(capture, dict) and capture.get("kind") == "response":
                continue
            native.append(row)
        candidates = [row for row in backend_events if row.get("signal_id") == "request.accepted"
                      and row.get("request_id") == request_id]
        matched = [row for row in candidates if correlation_id and row.get("correlation_id") == correlation_id]
        native_matched = [row for row in native if correlation_id and row.get("capture_correlation_id") == correlation_id]
        if not native:
            errors.append({"scenario_id": scenario, "payload_index": payload_index,
                           "reason": "no native harness/provider hint observation for mapped payload"})
        if not candidates:
            errors.append({"backend_request_id": request_id, "reason": "backend acceptance not observed"})
        if correlation_id and not matched:
            errors.append({"backend_request_id": request_id, "reason": "correlation_id does not match backend evidence"})
        rows.append({
            "scenario_id": scenario,
            "payload_index": payload_index,
            "backend_request_id": request_id,
            "native_hint_ids": sorted({str(row["hint_id"]) for row in native}),
            "native_evidence_sources": sorted({str(row.get("evidence_source") or "") for row in native}),
            "backend_request_accepted": bool(candidates),
            "same_request_proven": bool(native_matched and matched),
            "join_basis": "matching_native_backend_correlation" if native_matched and matched else "external_mapping_only",
            "backend_hint_effect": "not_proven",
        })
    return {
        "schema_version": "hint.backend_boundary_audit.v1",
        "valid": bool(mappings) and not errors,
        "mappings": rows,
        "errors": errors,
        "proof_limit": "Backend acceptance does not prove translation, cache reuse, scheduling effect, or task success.",
    }
