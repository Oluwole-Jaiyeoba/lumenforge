"""Deterministic input matrices for behavior-preservation (golden) tests.

These matrices were used to record ``*.golden.json`` from the code *before*
the SGLang-portability refactor (see ``docs/architecture/README_RESTRUCTURING.md``).  The tests
in ``test_golden_behavior.py`` replay the same inputs against the current
code and require identical outputs.

Do not edit the matrices casually: changing an input invalidates the recorded
golden output.  If behavior is changed on purpose, regenerate with
``python tests/golden/generate_golden.py`` and explain the change in the
restructuring README.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
import os
from typing import Any, Iterator

MODES = [
    "",
    "no_prefetch",
    "no_cache_signal",
    "e2e_priority_hints",
    "e2e_priority_hints_speculative_prefill",
    "pre_harness_priority_hints",
    "nat_inferred_priority_hints",
    "harness_native_cache_lowered",
    "harness_emitted_signals",
    "controller_observe_only",
    "controller_scheduler_priority",
    "controller_speculative_preload",
    "controller_targeted_kv_prefetch",
    "controller_demote_restore",
    "controller_priority_demote",
    "controller_priority_demotion_admission",
    "controller_priority_demotion_admission_soft",
    "controller_priority_demotion_admission_medium",
    "controller_priority_demotion_admission_hard",
    "controller_priority_demotion_admission_earlyprepare",
    "controller_priority_demotion_admission_shorthand",
    "controller_oracle_safe_sjf",
    "controller_oracle_safe_sjf_balanced",
    "controller_oracle_safe_sjf_aggressive",
    "controller_oracle_safe_sjf_maxfill",
    "controller_priority_demotion_calibrated_admission",
    "controller_oracle_exact_runtime_admission",
    "controller_deadline_fair",
    "controller_predictive_deadline_queue",
    "controller_predictive_deadline_queue_admission_guard",
    "controller_ready_time_gpu_backfill",
    "controller_value_aware_eviction",
    "controller_memory_admission",
    "controller_admission_control",
    "controller_full",
    "controller_full_chunked_prefill",
    "controller_harness_aware_full",
    "storage_hicache_baseline",
]

PHASES = ["replay", "pressure_filler", "pressure_filler_2", "initial_turn", "speculative_prefill", "request"]

META_VARIANTS: list[dict[str, Any]] = [
    {},
    {"controller_sglang_priority": 250},
    {"controller_sglang_priority": "not-a-number", "controller_eviction_value_class": "protected_high_value"},
    {"controller_eviction_value_class": "normal_value"},
    {"controller_eviction_value_class": "evictable_low_value", "low_priority": -7},
    {"controller_predictive_deadline_queue": True, "controller_sglang_priority": 99990},
    {"controller_deadline_fair": True, "controller_sglang_priority": 99991},
    {"priority_intent": {"class": "urgent", "reason": "tool_replay_deadline", "deadline_ms": 250}},
    {"priority_intent": {"class": "low"}},
    {"controller_priority_ladder": "rank=1/3", "controller_sglang_priority": 300, "high_priority": 120},
    {"controller_demote_priority": -55, "speculative_prefill_priority": 42, "speculative_prefill": True},
    {
        "harness_controller_signal": {"ignored": True},
        "deadline_offset_ms": 500,
        "controller_decision_id": "d-1",
        "controller_command_id": "c-1",
        "controller_replay_rank": 2,
        "kv_storage_enabled": "1",
        "max_tokens": 16,
        "prompt_hash": "abcdef0123456789abcdef0123456789ffff",
    },
]

PAYLOAD_VARIANTS: list[tuple[str, dict[str, Any], dict[str, str]]] = [
    ("openai_chat", {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]}, {}),
    ("openai_chat", {"messages": [{"role": "user", "content": "x"}], "nvext": {"agent_hints": {"priority": 100, "latency_sensitivity": 1.0, "osl": 64, "iat": 750, "total_requests": 4, "prefix_id": "p"}}}, {}),
    ("openai_chat", {"messages": [{"role": "user", "content": "x"}], "nvext": {"agent_hints": {"priority": 2}, "cache_control": {"type": "ephemeral", "ttl": "1s"}}}, {}),
    ("openai_chat", {"messages": [{"role": "user", "content": "x"}], "service_tier": "priority", "metadata": {"priority_class": "urgent"}}, {}),
    ("openai_chat", {"messages": [{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}]}], "extra_body": {"agentic_hints": {"priority_class": "urgent"}}, "prompt_cache_key": "k1", "prompt_cache_retention": "24h"}, {"x-session-affinity": "sa", "prompt-cache-key": "hk"}),
    ("openai_chat", {"messages": [{"role": "user", "content": "x"}], "agentic_hints": {"urgency": "high"}, "max_tokens": 32, "speed": "fast", "session_id": "sess-9"}, {}),
    ("anthropic", {"system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "service_tier": "auto"}, {}),
    ("responses", {"instructions": "be brief", "input": [{"role": "user", "content": [{"type": "input_text", "text": "q"}]}], "max_output_tokens": 12, "promptCacheKey": "rk"}, {"x-prompt-cache-retention": "long"}),
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:16]


def gateway_cases() -> Iterator[dict[str, Any]]:
    index = 0
    for mode in MODES:
        for phase in PHASES:
            for meta_index, meta_extra in enumerate(META_VARIANTS):
                api_kind, payload, headers = PAYLOAD_VARIANTS[index % len(PAYLOAD_VARIANTS)]
                meta = {
                    "mode": mode,
                    "phase": phase,
                    "session_id": f"s{index % 7}",
                    "label": f"s{index % 7}_{phase}",
                    "harness": ["hatcher", "claude_code", "qwen_code", "nat"][index % 4],
                    "task_index": index % 3,
                    **meta_extra,
                }
                yield {
                    "case": f"{mode or 'none'}|{phase}|m{meta_index}|p{index % len(PAYLOAD_VARIANTS)}",
                    "api_kind": api_kind,
                    "payload": payload,
                    "headers": headers,
                    "meta": meta,
                }
                index += 1


ENV_PROFILES: dict[str, dict[str, str]] = {
    "defaults": {},
    "overrides": {
        "CONTROLLER_TARGETED_PREFETCH_LEAD_MS": "333",
        "CONTROLLER_DIRECT_LOAD_FIXED_MS": "200",
        "CONTROLLER_DIRECT_LOAD_MS_PER_TOKEN": "0.5",
        "CONTROLLER_DIRECT_LOAD_ESTIMATE_MULTIPLIER": "1.5",
        "CONTROLLER_DIRECT_LOAD_ALLOWED_WAIT_CLASSES": "short, medium",
        "CONTROLLER_DIRECT_LOAD_REQUIRE_COMPLETION": "1",
        "CONTROLLER_DIRECT_LOAD_SAFETY_MARGIN_MS": "100",
        "CONTROLLER_DEADLINE_FAIR_BASE_PRIORITY": "5000",
        "CONTROLLER_DEADLINE_FAIR_BUCKET_MS": "25",
        "CONTROLLER_EVICTION_NORMAL_PRIORITY": "7",
        "CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY": "0.3",
        "CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS": "99",
    },
}

ENV_KEYS = sorted({key for profile in ENV_PROFILES.values() for key in profile})


@contextmanager
def env_profile(name: str) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(ENV_PROFILES[name])
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


WAITS_MS = [0, 100, 499, 500, 1_000, 3_000, 3_001, 10_000, 10_001, 60_000, 120_000]
DUES_MS = [0.0, 9.0, 10.0, 12_345.6, 1_000_000.0, -5.0]
WINDOW_CASES = [
    {"tool_start_ms": 0.0, "replay_due_ms": 10_000.0, "prompt_tokens": 1000, "wait_class": "short"},
    {"tool_start_ms": 0.0, "replay_due_ms": 10_000.0, "prompt_tokens": 1000, "wait_class": "long"},
    {"tool_start_ms": 5_000.0, "replay_due_ms": 5_500.0, "prompt_tokens": 10, "wait_class": None},
    {"tool_start_ms": 0.0, "replay_due_ms": 2_000.0, "prompt_tokens": 4000, "wait_class": "medium"},
    {"tool_start_ms": 100.0, "replay_due_ms": 60_000.0, "prompt_tokens": 0, "wait_class": "MEDIUM "},
]


def eviction_meta_cases() -> Iterator[dict[str, Any]]:
    for index in range(40):
        yield {
            "mode": "controller_value_aware_eviction" if index % 5 else "controller_full",
            "session_id": f"sess_{index}",
            "prefix_id": "" if index % 3 else f"pref_{index}",
            "phase": "replay",
            "tool_wait_step": index % 4,
            "prompt_tokens": [0, 512, 4096, "bad"][index % 4],
            "expected_tool_return_ms": [0, 1_000, 30_000, 500_000, "x"][index % 5],
            "high_priority": [None, 150][index % 2],
            "low_priority": [None, -3][index % 2],
            "reuse_probability": [None, 0.5, ""][index % 3],
        }


def outbound_cases() -> Iterator[tuple[dict[str, Any], str]]:
    intents: list[Any] = [
        None,
        "urgent",
        {"class": "urgent"},
        {"class": "urgent", "reason": "r", "deadline_ms": 99},
        {"class": "low", "reason": "bg"},
        {"class": ""},
    ]
    for mode in ["pre_harness_priority_hints", "e2e_priority_hints"]:
        for intent in intents:
            for api_kind in ["openai_chat", "anthropic", "responses"]:
                yield {"mode": mode, "priority_intent": intent}, api_kind
