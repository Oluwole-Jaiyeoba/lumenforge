"""Value-aware KV eviction metadata for the ``controller_value_aware_eviction`` mode.

Assigns each replay-capable request a value class (protected / normal /
evictable) from a stable hash bucket, reuse probability, recompute cost and
tool ETA, and derives the scheduler priority the backend should use for its
radix-cache priority eviction.

Moved verbatim from ``run_multi_harness_replay_driver.py``.  The output keys
``controller_sglang_priority`` / ``sglang_radix_priority_eviction`` are kept
because they are recorded artifact columns (see docs/architecture/README_RESTRUCTURING.md,
"Vocabulary allowlist").
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from .modes import controller_value_aware_eviction_mode


def _stable_percent(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _float_meta(meta: dict[str, Any], key: str, default: float) -> float:
    try:
        value = meta.get(key)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def value_aware_eviction_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    if not controller_value_aware_eviction_mode(str(meta.get("mode") or "")):
        return {}

    session_id = str(meta.get("session_id") or "")
    prefix_id = str(meta.get("prefix_id") or f"{session_id}:prefix")
    phase = str(meta.get("phase") or "")
    step = str(meta.get("tool_wait_step") or "0")
    bucket = _stable_percent(f"{session_id}|{prefix_id}|{step}")
    prompt_tokens = int(_float_meta(meta, "prompt_tokens", 0.0))

    if bucket < 40:
        value_class = "protected_high_value"
        reuse_probability = _float_meta(meta, "reuse_probability", 0.95)
        recompute_cost_tokens = int(_float_meta(meta, "recompute_cost_tokens", float(max(prompt_tokens, 2048))))
        priority = int(float(meta.get("high_priority") or 100))
        work_value = "critical_path"
        criticality = "high"
        cancelable = False
    elif bucket < 75:
        value_class = "normal_value"
        reuse_probability = _float_meta(meta, "reuse_probability", 0.65)
        recompute_cost_tokens = int(_float_meta(meta, "recompute_cost_tokens", float(max(prompt_tokens // 2, 1024))))
        priority = int(float(os.environ.get("CONTROLLER_EVICTION_NORMAL_PRIORITY", "0") or "0"))
        work_value = "normal"
        criticality = "normal"
        cancelable = False
    else:
        value_class = "evictable_low_value"
        reuse_probability = _float_meta(
            meta,
            "reuse_probability",
            float(os.environ.get("CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY", "0.15") or "0.15"),
        )
        recompute_cost_tokens = int(
            _float_meta(
                meta,
                "recompute_cost_tokens",
                float(os.environ.get("CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS", "256") or "256"),
            )
        )
        priority = int(float(meta.get("low_priority") or -100))
        work_value = "low_value_replay"
        criticality = "low"
        cancelable = True

    urgency_factor = 1.0
    try:
        eta_ms = float(meta.get("expected_tool_return_ms") or meta.get("next_ready_eta_ms") or meta.get("tool_wait_ms") or 0)
        if eta_ms > 0:
            urgency_factor = max(0.25, min(4.0, 60_000.0 / eta_ms))
    except (TypeError, ValueError):
        urgency_factor = 1.0
    value_score = int(round(reuse_probability * max(1, recompute_cost_tokens) * urgency_factor))
    return {
        "reuse_probability": reuse_probability,
        "recompute_cost_tokens": recompute_cost_tokens,
        "work_value": work_value,
        "criticality": criticality,
        "cancelable": cancelable,
        "priority_label": "low"
        if value_class == "evictable_low_value"
        else "normal"
        if value_class == "normal_value"
        else "high",
        "controller_sglang_priority": priority,
        "controller_eviction_policy": "sglang_radix_priority_eviction",
        "controller_eviction_value_class": value_class,
        "controller_eviction_value_score": value_score,
        "controller_eviction_bucket": bucket,
        "controller_eviction_signal_source": "session_id,prefix_id,phase,expected_tool_return_ms,reuse_probability,recompute_cost_tokens,deadline_after_ready_ms",
        "controller_eviction_translation": f"sglang.priority={priority};radix_eviction_policy=priority",
        "controller_eviction_scope": "all_replay_capable_requests",
        "controller_eviction_phase_seen": phase,
    }


__all__ = ["value_aware_eviction_metadata"]
