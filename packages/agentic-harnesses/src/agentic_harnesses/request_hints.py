"""Parse scheduling and cache hints that harness clients put on requests.

These helpers read OpenAI/Anthropic/NAT-style request bodies and headers and
report which native hint fields were present (``service_tier``,
``metadata.priority_class``, ``extra_body.agentic_hints``,
``nvext.agent_hints``, ``prompt_cache_key``, ``cache_control`` ...).

They are backend-neutral: nothing here knows how a hint will be lowered into a
serving engine.  Moved verbatim from ``sglang_direct_kv/scripts/
harness_sglang_gateway.py`` during the SGLang-portability refactor; output
strings (report column names) are unchanged on purpose.
"""

from __future__ import annotations

import json
from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def is_present(value: Any) -> bool:
    return value not in (None, "", [], {})


CACHE_SIGNAL_KEYS = {
    "cache_control",
    "cacheControl",
    "cache_control_format",
    "cacheControlFormat",
    "cache_retention",
    "cacheRetention",
    "prompt_cache_key",
    "promptCacheKey",
    "prompt_cache_retention",
    "promptCacheRetention",
    "cache_key",
    "cacheKey",
}


CACHE_IDENTITY_KEYS = {
    "session_id",
    "sessionId",
    "thread_id",
    "threadId",
    "conversation_id",
    "conversationId",
}


CACHE_SIGNAL_HEADERS = {
    "prompt-cache-key",
    "prompt_cache_key",
    "x-prompt-cache-key",
    "prompt-cache-retention",
    "prompt_cache_retention",
    "x-prompt-cache-retention",
}


CACHE_IDENTITY_HEADERS = {
    "session_id",
    "session-id",
    "x-session-id",
    "x-session-affinity",
    "x-client-request-id",
}


def payload_nvext_priority(payload: dict[str, Any]) -> int | None:
    nvext_priority = as_dict(as_dict(payload.get("nvext")).get("agent_hints")).get("priority")
    try:
        return int(float(nvext_priority))
    except (TypeError, ValueError):
        return None


def emitted_priority_signal(payload: dict[str, Any]) -> dict[str, str]:
    signals: list[str] = []
    sources: list[str] = []
    service_tier = payload.get("service_tier")
    if service_tier not in (None, "", [], {}):
        signals.append(f"service_tier={service_tier}")
        sources.append("service_tier")
    speed = payload.get("speed")
    if speed not in (None, "", [], {}):
        signals.append(f"speed={speed}")
        sources.append("speed")
    metadata = as_dict(payload.get("metadata"))
    metadata_priority = metadata.get("priority_class") or metadata.get("urgency")
    if metadata_priority not in (None, "", [], {}):
        signals.append(f"metadata.priority_class={metadata_priority}")
        sources.append("metadata")
    extra_body = as_dict(payload.get("extra_body"))
    agentic_hints = as_dict(extra_body.get("agentic_hints"))
    extra_priority = agentic_hints.get("priority_class") or agentic_hints.get("urgency")
    if extra_priority not in (None, "", [], {}):
        signals.append(f"extra_body.agentic_hints.priority_class={extra_priority}")
        sources.append("extra_body.agentic_hints")
    top_level_agentic_hints = as_dict(payload.get("agentic_hints"))
    top_level_priority = top_level_agentic_hints.get("priority_class") or top_level_agentic_hints.get("urgency")
    if top_level_priority not in (None, "", [], {}):
        signals.append(f"agentic_hints.priority_class={top_level_priority}")
        sources.append("agentic_hints")
    nvext = as_dict(payload.get("nvext"))
    nvext_agent_hints = as_dict(nvext.get("agent_hints"))
    nvext_priority = nvext_agent_hints.get("priority")
    if nvext_priority not in (None, "", [], {}):
        signals.append(f"nvext.agent_hints.priority={nvext_priority}")
        sources.append("nvext.agent_hints")
    nvext_latency = nvext_agent_hints.get("latency_sensitivity")
    if nvext_latency not in (None, "", [], {}):
        signals.append(f"nvext.agent_hints.latency_sensitivity={nvext_latency}")
        sources.append("nvext.agent_hints")
    nvext_prefix = nvext_agent_hints.get("prefix_id")
    if nvext_prefix not in (None, "", [], {}):
        signals.append(f"nvext.agent_hints.prefix_id={nvext_prefix}")
        sources.append("nvext.agent_hints")
    nvext_osl = nvext_agent_hints.get("osl")
    if nvext_osl not in (None, "", [], {}):
        signals.append(f"nvext.agent_hints.osl={nvext_osl}")
        sources.append("nvext.agent_hints")
    nvext_iat = nvext_agent_hints.get("iat")
    if nvext_iat not in (None, "", [], {}):
        signals.append(f"nvext.agent_hints.iat={nvext_iat}")
        sources.append("nvext.agent_hints")
    nvext_total = nvext_agent_hints.get("total_requests")
    if nvext_total not in (None, "", [], {}):
        signals.append(f"nvext.agent_hints.total_requests={nvext_total}")
        sources.append("nvext.agent_hints")
    nvext_cache_control = as_dict(nvext.get("cache_control"))
    if nvext_cache_control:
        signals.append(f"nvext.cache_control={compact_json(nvext_cache_control)}")
        sources.append("nvext.cache_control")
    return {
        "harness_emit_priority_signal": "; ".join(signals),
        "harness_emit_priority_signal_source": ", ".join(dict.fromkeys(sources)),
    }


def iter_structured_signals(value: Any, path: str = "$") -> list[tuple[str, str, Any]]:
    signals: list[tuple[str, str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in CACHE_SIGNAL_KEYS and is_present(child):
                signals.append(("cache", child_path, child))
            elif key in CACHE_IDENTITY_KEYS and is_present(child):
                signals.append(("identity", child_path, child))
            signals.extend(iter_structured_signals(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            signals.extend(iter_structured_signals(child, f"{path}[{index}]"))
    return signals


def header_signal_items(headers: dict[str, str]) -> list[tuple[str, str, Any]]:
    signals: list[tuple[str, str, Any]] = []
    for key, value in headers.items():
        normalized = key.lower()
        if normalized in CACHE_SIGNAL_HEADERS and is_present(value):
            signals.append(("cache", f"@headers.{normalized}", value))
        elif normalized in CACHE_IDENTITY_HEADERS and is_present(value):
            signals.append(("identity", f"@headers.{normalized}", value))
    return signals


def emitted_cache_signal(payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, str]:
    structured = iter_structured_signals(payload)
    structured.extend(header_signal_items(headers or {}))
    cache_items = [(path, value) for kind, path, value in structured if kind == "cache"]
    identity_items = [(path, value) for kind, path, value in structured if kind == "identity"]
    signals = [f"{path}={compact_json(value)}" for path, value in cache_items]
    identity_signals = [f"{path}={compact_json(value)}" for path, value in identity_items]
    return {
        "harness_native_cache_signal_seen": "yes" if cache_items else "no",
        "harness_native_cache_signal": "; ".join(signals),
        "harness_native_cache_signal_source": ", ".join(path for path, _ in cache_items),
        "harness_native_cache_identity_signal": "; ".join(identity_signals),
        "harness_native_cache_identity_source": ", ".join(path for path, _ in identity_items),
    }


def first_signal_value(payload: dict[str, Any], keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and is_present(value):
                return value
            found = first_signal_value(value, keys) if isinstance(value, (dict, list)) else None
            if is_present(found):
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = first_signal_value(item, keys) if isinstance(item, (dict, list)) else None
            if is_present(found):
                return found
    return None


def first_header_signal_value(headers: dict[str, str], keys: set[str]) -> Any:
    wanted = {key.lower() for key in keys}
    for key, value in headers.items():
        if key.lower() in wanted and is_present(value):
            return value
    return None


def emitted_signal_is_urgent(payload: dict[str, Any]) -> bool:
    service_tier = str(payload.get("service_tier") or "").lower()
    if service_tier in {"priority", "scale"}:
        return True
    metadata_priority = str(as_dict(payload.get("metadata")).get("priority_class") or "").lower()
    if metadata_priority in {"urgent", "high", "priority"}:
        return True
    extra_priority = str(as_dict(as_dict(payload.get("extra_body")).get("agentic_hints")).get("priority_class") or "").lower()
    if extra_priority in {"urgent", "high", "priority"}:
        return True
    top_priority = str(as_dict(payload.get("agentic_hints")).get("priority_class") or "").lower()
    if top_priority in {"urgent", "high", "priority"}:
        return True
    nvext_priority = as_dict(as_dict(payload.get("nvext")).get("agent_hints")).get("priority")
    try:
        return int(float(nvext_priority)) >= 100
    except (TypeError, ValueError):
        return False


__all__ = ["as_dict", "compact_json", "is_present", "CACHE_SIGNAL_KEYS", "CACHE_IDENTITY_KEYS", "CACHE_SIGNAL_HEADERS", "CACHE_IDENTITY_HEADERS", "payload_nvext_priority", "emitted_priority_signal", "iter_structured_signals", "header_signal_items", "emitted_cache_signal", "first_signal_value", "first_header_signal_value", "emitted_signal_is_urgent"]
