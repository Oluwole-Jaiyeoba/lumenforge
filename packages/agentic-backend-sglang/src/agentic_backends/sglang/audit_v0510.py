"""Translate the pinned v0510 trace into backend-neutral work-audit evidence.

This module reads existing trace hooks; it does not install a second plugin or
alter the SGLang runtime. Raw events without a trustworthy session identity
remain unjoined and are not used to grade a session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_work_audit.events import AuditEvent


def _agent(row: dict[str, Any]) -> tuple[str, str]:
    session = row.get("agent_session_id") or ""
    request = row.get("agent_request_id") or ""
    sessions = row.get("agent_sessions")
    if not session and isinstance(sessions, list) and len(sessions) == 1:
        session = sessions[0].get("agent_session_id") or ""
        request = request or sessions[0].get("agent_request_id") or ""
    context = row.get("kv_context")
    if not session and isinstance(context, dict):
        session = context.get("agent_session_id") or ""
        request = request or context.get("agent_request_id") or ""
    return str(session), str(request)


def translate_trace(path: Path) -> list[AuditEvent]:
    raw: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid backend trace") from exc

    load_sessions: dict[str, tuple[str, str]] = {}
    for row in raw:
        if row.get("event") != "agentic_kv.prepare_prefix.result":
            continue
        load_id = row.get("load_id")
        command = row.get("command") or {}
        if load_id and command.get("session_id"):
            load_sessions[str(load_id)] = (str(command["session_id"]), str(command.get("request_id") or ""))

    out: list[AuditEvent] = []
    for row in raw:
        name = str(row.get("event") or "")
        ts = int(row.get("ts_ns") or 0)
        if ts <= 0:
            continue
        session, request = _agent(row)
        kind = ""
        evidence: dict[str, Any] = {}
        if name == "trace.install.summary":
            kind = "runtime_hooks"
            installed = row.get("installed_hooks") or []
            required = {
                "cache_match": "HiRadixCache.match_prefix",
                "native_load": "HiRadixCache.load_back",
                "host_copy": "HiCacheController.start_loading",
            }
            evidence = {
                "backend_version": row.get("sglang_version"),
                "adapter": row.get("adapter"),
                "missing_required_hooks": row.get("missing_required_hooks") or [],
                "installed_hooks": installed,
                "audit_capabilities": [key for key, hook in required.items() if any(hook in item for item in installed)],
            }
        elif name == "kv_telemetry.prefill.start" and session:
            kind = "cache_match"
            evidence = {
                "cached_prefix_tokens": row.get("batch_cached_prefix_token_sum"),
                "uncached_tokens": row.get("batch_uncached_token_sum"),
                "join_method": "agent_context",
            }
        elif name == "kv_telemetry.request_stage" and row.get("phase") == "end" and session:
            category = row.get("category")
            if category == "host_to_device_copy":
                kind = "layer_copy"
                evidence = {"duration_ms": row.get("duration_ms"), "join_method": "agent_context"}
            elif category == "hicache_evict_device":
                kind = "backend_device_evict"
                evidence = {"duration_ms": row.get("duration_ms"), "join_method": "agent_context"}
        elif name == "agentic_kv.prepare_prefix.load_status" and row.get("status") == "finished":
            load_id = str(row.get("load_id") or "")
            if load_id in load_sessions:
                session, request = load_sessions[load_id]
                kind = "load_complete"
                evidence = {
                    "loaded_tokens": row.get("loaded_tokens"),
                    "cuda_elapsed_ms": row.get("cuda_elapsed_ms"),
                    "load_id": load_id,
                    "join_method": "load_id",
                }
        elif name in ("agentic_work_audit.hook_error", "trace.hook_error"):
            kind = "hook_error"
            evidence = {"error": row.get("error")}
        if kind:
            out.append(AuditEvent(kind=kind, ts_ns=ts, source="backend_v0510", session_id=session,
                                  request_id=request, evidence=evidence))
    return out
