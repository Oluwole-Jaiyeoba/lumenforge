"""Normalize raw SGLang trace rows into ``agentic_core.BackendObservation``.

Raw rows are what ``trace/patch.py`` writes (``hiradix.load_back.end``,
``hicache.write.end`` ...).  Their names are SGLang-internal and may change
between releases; the adapter's ``raw_event_map`` translates them into the
stable KV event vocabulary (``KV_LOAD_GPU``, ``KV_WRITE_HOST`` ...).

Downstream code (reports, ledgers) should consume observations produced here
instead of matching raw SGLang event names.  NOTE: the richer block ledger in
``sglang_direct_kv/src/agentic_kv/block_ledger`` and many report scripts
still read raw rows directly; migrating them is listed as follow-up work in
README_RESTRUCTURING.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agentic_core import BackendObservation

from .versions import AdapterSpec, newest_adapter

# Trace bookkeeping events that are useful as observations too.
_META_EVENTS = {
    "trace.adapter.selected": "backend_adapter_selected",
    "trace.install.summary": "backend_hook_install",
}


def _context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    context = row.get("kv_context")
    return context if isinstance(context, Mapping) else row


def _first(context: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = context.get(key)
        if value not in (None, ""):
            return str(value)
    request = context.get("request")
    if isinstance(request, Mapping):
        for key in keys:
            value = request.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


class SGLangTelemetryNormalizer:
    """``agentic_backend_api.TelemetryNormalizer`` for SGLang trace rows."""

    def __init__(self, adapter: AdapterSpec | None = None, backend_version: str = "") -> None:
        self.adapter = adapter or newest_adapter()
        self.backend_version = backend_version

    def normalize(self, raw_event: Mapping[str, Any]) -> BackendObservation | None:
        source_event = str(raw_event.get("event") or "")
        kind = self.adapter.raw_event_map.get(source_event) or _META_EVENTS.get(source_event)
        if kind is None:
            return None
        context = _context(raw_event)
        ts_ns = raw_event.get("ts_ns")
        try:
            observed_at_ms = int(float(ts_ns) // 1_000_000) if ts_ns not in (None, "") else 0
        except (TypeError, ValueError):
            observed_at_ms = 0
        identity = json.dumps(
            [source_event, raw_event.get("call_id"), ts_ns], sort_keys=True, default=str
        ).encode("utf-8")
        payload = {
            key: context.get(key)
            for key in ("node_id", "token_count", "host_indices", "device_indices", "duration_ms", "direction")
            if context.get(key) not in (None, "")
        }
        if raw_event.get("duration_ms") not in (None, ""):
            payload["duration_ms"] = raw_event.get("duration_ms")
        if kind in _META_EVENTS.values():
            payload = {k: v for k, v in raw_event.items() if k not in ("event", "ts_ns", "pid")}
        return BackendObservation(
            observation_id=hashlib.sha256(identity).hexdigest()[:16],
            observed_at_ms=observed_at_ms,
            backend_name="sglang",
            backend_version=self.backend_version,
            kind=kind,
            request_id=_first(context, "agent_request_id", "request_id", "rid"),
            session_id=_first(context, "agent_session_id", "session_id"),
            correlation_id=_first(context, "agent_correlation_id", "correlation_id"),
            source_event=source_event,
            payload=payload,
        )
