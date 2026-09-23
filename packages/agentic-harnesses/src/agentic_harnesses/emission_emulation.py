"""Emulate harness-emitted priority hints for controlled experiments.

``pre_harness_priority_hints`` mode asks the experiment driver to shape the
outgoing client request the way a priority-aware harness *would* (OpenAI
``service_tier`` + ``metadata`` + ``extra_body.agentic_hints``, or Anthropic
``service_tier``).  Signals produced here have ``INJECTED`` provenance: they
prove the plumbing, not that a real harness emitted them.

Moved verbatim from ``run_multi_harness_replay_driver.py``.
"""

from __future__ import annotations

from typing import Any


def pre_harness_priority_enabled(mode: str) -> bool:
    return mode == "pre_harness_priority_hints"


def outbound_priority_fields(meta: dict[str, Any], api_kind: str) -> dict[str, Any]:
    if not pre_harness_priority_enabled(str(meta.get("mode") or "")):
        return {}
    intent = meta.get("priority_intent")
    if not isinstance(intent, dict):
        return {}
    priority_class = str(intent.get("class") or "")
    if priority_class != "urgent":
        return {
            "metadata": {
                "priority_class": priority_class,
                "priority_reason": str(intent.get("reason") or ""),
            }
        }
    metadata = {
        "priority_class": "urgent",
        "priority_reason": str(intent.get("reason") or "tool_replay_deadline"),
        "priority_deadline_ms": str(intent.get("deadline_ms") or ""),
    }
    if api_kind == "anthropic":
        return {"service_tier": "auto", "metadata": metadata}
    return {
        "service_tier": "priority",
        "metadata": metadata,
        "extra_body": {"agentic_hints": {"priority_class": "urgent", "reason": metadata["priority_reason"]}},
    }


__all__ = ["outbound_priority_fields", "pre_harness_priority_enabled"]
