"""Version-aware translation of SGLang trace rows to shared evidence.

Raw traces remain the source of truth. This module only extracts identities and
small proof fields; it does not copy arbitrary request bodies into evidence.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agentic_instrumentation import EvidenceEvent

from .versions import get_adapter


_KV_SIGNALS = {
    "hicache.write.end": "kv.write_host",
    "hicache.evict_device.end": "kv.evict_gpu",
    "hicache.evict_host.end": "kv.evict_host",
    "hicache.load.end": "kv.load_gpu",
    "hiradix.init_load_back.end": "kv.nested_load",
    "hiradix.load_back.end": "kv.nested_load",
    "hostpool.load_to_device_per_layer.end": "kv.layer_copy",
    "hostpool.backup_from_device_all_layer.end": "kv.layer_backup",
    "hiradix.match_prefix.end": "kv.prefix_match",
}

_SCHEDULER_SIGNALS = {
    "scheduler.handle_generate_request.end": "request.accepted",
    "scheduler.get_next_batch_to_run.end": "batch.scheduled",
    "scheduler.process_batch_result.end": "batch.completed",
    "scheduler.process_batch_result_decode.end": "batch.decode_step",
    "worker.forward_batch_generation.end": "model.forward",
}


@lru_cache(maxsize=None)
def _adapter_events(adapter_name: str) -> tuple[set[str], dict[str, str]]:
    adapter = get_adapter(adapter_name)
    prefixes = {prefix for target in adapter.hook_targets for prefix in target.methods.values()}
    return prefixes, adapter.raw_event_map


def _context(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("kv_context")
    return value if isinstance(value, dict) else row


def _identity(row: dict[str, Any], context: dict[str, Any]) -> tuple[str, str]:
    session = row.get("agent_session_id") or context.get("agent_session_id") or ""
    request = row.get("agent_request_id") or context.get("agent_request_id") or ""
    sessions = row.get("agent_sessions") or context.get("agent_sessions")
    if isinstance(sessions, list) and len(sessions) == 1 and isinstance(sessions[0], dict):
        session = session or sessions[0].get("agent_session_id") or ""
        request = request or sessions[0].get("agent_request_id") or ""
    nested = context.get("request")
    if isinstance(nested, dict):
        session = session or nested.get("agent_session_id") or ""
        request = request or nested.get("agent_request_id") or nested.get("rid") or ""
    return str(session), str(request)


def normalize_trace_event(row: dict[str, Any], adapter_name: str = "v0510") -> EvidenceEvent | None:
    """Map a recognized raw hook row; unrecognized/untimed rows stay raw-only."""
    source_event = str(row.get("source_event") or row.get("event") or "")
    timestamp = int(row.get("ts_ns") or 0)
    if timestamp <= 0:
        return None
    prefixes, raw_event_map = _adapter_events(adapter_name)
    if source_event.rsplit(".", 1)[0] not in prefixes:
        return None
    kv_type = raw_event_map.get(source_event)
    signal = _KV_SIGNALS.get(source_event) or _SCHEDULER_SIGNALS.get(source_event)
    if signal is None and kv_type is None:
        return None
    context = _context(row)
    session, request = _identity(row, context)
    matched = None
    result = row.get("result")
    if signal == "kv.prefix_match" and isinstance(result, list) and len(result) > 1:
        node = result[1]
        if isinstance(node, dict):
            matched = node.get("value")
    payload = {
        "source_event": source_event,
        "kv_event_type": kv_type,
        "node_id": context.get("node_id"),
        "host_indices": context.get("host_indices"),
        "device_indices": context.get("device_indices"),
        "matched_indices": matched,
    }
    return EvidenceEvent(
        signal_id=signal or "kv.detail",
        time_ns=timestamp,
        source=f"sglang_{adapter_name}",
        session_id=session,
        request_id=request,
        correlation_id=str(context.get("agent_correlation_id") or context.get("correlation_id") or ""),
        phase=str(context.get("agent_phase") or ""),
        payload=payload,
    )
