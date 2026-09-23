"""Client-API parsing for the harness gateway.

Understands the request shapes harness clients send (OpenAI chat,
Anthropic messages, OpenAI responses) and the experiment marker the driver
embeds in prompts (``HARNESS_REPLAY_EXPERIMENT_JSON:<base64>``).

Backend-neutral.  Moved verbatim from ``harness_sglang_gateway.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any

from agentic_harnesses.request_hints import as_dict


MARKER = "HARNESS_REPLAY_EXPERIMENT_JSON:"


def parse_b64_json(value: str) -> dict[str, Any]:
    padded = value + "=" * (-len(value) % 4)
    return as_dict(json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")))


def iter_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(iter_text(item))
        return out
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return [value["text"]]
        if isinstance(value.get("content"), str):
            return [value["content"]]
        out: list[str] = []
        for key in ("content", "input", "messages", "system"):
            if key in value:
                out.extend(iter_text(value[key]))
        return out
    return []


def strip_marker(text: str) -> str:
    return re.sub(rf"\n?{re.escape(MARKER)}[A-Za-z0-9_=-]+", "", text).strip()


def marker_from_payload(payload: Any) -> dict[str, Any]:
    joined = "\n".join(iter_text(payload))
    match = re.search(rf"{re.escape(MARKER)}([A-Za-z0-9_=-]+)", joined)
    if not match:
        return {}
    try:
        return parse_b64_json(match.group(1))
    except Exception as exc:  # noqa: BLE001
        return {"marker_parse_error": str(exc)}


def payload_text_chars(payload: Any) -> int:
    return sum(len(part) for part in iter_text(payload))


def payload_stream_requested(payload: Any) -> bool:
    return bool(as_dict(payload).get("stream"))


def payload_model(payload: Any) -> str:
    model = as_dict(payload).get("model")
    return str(model) if model is not None else ""


def request_shape(body: bytes, payload: Any, api_kind: str) -> dict[str, Any]:
    text_parts = iter_text(payload)
    joined = "\n".join(text_parts)
    return {
        "api_kind": api_kind,
        "body_size_bytes": len(body),
        "text_part_count": len(text_parts),
        "prompt_chars": payload_text_chars(payload),
        "prompt_hash": hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:32] if joined else "",
        "stream_requested": payload_stream_requested(payload),
        "model_requested": payload_model(payload),
    }


def is_bookkeeping_payload(payload: Any) -> bool:
    joined = "\n".join(iter_text(payload))
    bookkeeping_needles = (
        "Write the title in the predominant language of the session",
        "You are naming a coding session",
    )
    return any(needle in joined for needle in bookkeeping_needles)


def text_from_openai_chat(payload: dict[str, Any]) -> tuple[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return "", ""
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = "\n".join(iter_text(message.get("content")))
        if not content:
            continue
        if role == "system":
            system_parts.append(strip_marker(content))
        else:
            user_parts.append(strip_marker(content))
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def text_from_anthropic_messages(payload: dict[str, Any]) -> tuple[str, str]:
    system = "\n\n".join(strip_marker(part) for part in iter_text(payload.get("system")) if strip_marker(part))
    user_parts: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = strip_marker("\n".join(iter_text(message.get("content"))))
            if content:
                user_parts.append(f"{role}: {content}")
    return system, "\n\n".join(user_parts)


def text_from_responses(payload: dict[str, Any]) -> tuple[str, str]:
    system = strip_marker(str(payload.get("instructions") or ""))
    input_value = payload.get("input")
    if isinstance(input_value, str):
        user = strip_marker(input_value)
    else:
        user = "\n\n".join(strip_marker(part) for part in iter_text(input_value) if strip_marker(part))
    return system, user


def api_kind_from_path(path: str) -> str:
    if "/messages" in path:
        return "anthropic"
    if "/responses" in path:
        return "responses"
    return "chat"


__all__ = ["MARKER", "parse_b64_json", "iter_text", "strip_marker", "marker_from_payload", "payload_text_chars", "payload_stream_requested", "payload_model", "request_shape", "is_bookkeeping_payload", "text_from_openai_chat", "text_from_anthropic_messages", "text_from_responses", "api_kind_from_path"]
