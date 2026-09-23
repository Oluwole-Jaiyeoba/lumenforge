"""Gateway request translation: experiment mode + harness hints -> BackendRequest.

This module decides *what* a model request should carry:

- which scheduler priority to send (``resolve_backend_priority``),
- how harness-emitted priority/cache hints are interpreted
  (``priority_translation_context`` / ``cache_translation_context``),
- which correlation metadata travels with the request (``metadata_context``).

``translate_request`` packages all of that into a backend-neutral
``agentic_core.BackendRequest``.  Encoding it into a specific backend's wire
format is done by a backend integration, e.g.
``agentic_backends.sglang.lowering.lower_translation``.

Moved from ``sglang_direct_kv/scripts/harness_sglang_gateway.py`` during the
SGLang-portability refactor.  All functions except ``translate_request`` are
verbatim moves (``sglang_priority`` was renamed ``resolve_backend_priority``;
the old name is kept as an alias in the gateway script).  Mode sets are kept
exactly as the gateway had them -- they intentionally still differ from
``agentic_controller.modes`` (see docs/architecture/README_RESTRUCTURING.md, "Known divergences").

Some *data keys* still contain the word ``sglang`` (``controller_sglang_priority``,
``sglang_cache_salt``).  They are part of recorded artifacts and report
columns, so they were not renamed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentic_core import BackendRequest
from agentic_harnesses.request_hints import (
    as_dict,
    compact_json,
    emitted_cache_signal,
    emitted_priority_signal,
    emitted_signal_is_urgent,
    first_header_signal_value,
    first_signal_value,
    is_present,
    payload_nvext_priority,
)
from agentic_harnesses.signals import build_harness_controller_signal

from .client_api import text_from_anthropic_messages, text_from_openai_chat, text_from_responses


PRIORITY_ENABLED_MODES = {
    "e2e_priority_hints",
    "pre_harness_priority_hints",
    "nat_inferred_priority_hints",
    "e2e_priority_hints_speculative_prefill",
    "harness_emitted_signals",
    "controller_scheduler_priority",
    "controller_demote_restore",
    "controller_priority_demote",
    "controller_priority_demotion_admission",
    "controller_priority_demotion_admission_soft",
    "controller_priority_demotion_admission_medium",
    "controller_priority_demotion_admission_hard",
    "controller_priority_demotion_admission_earlyprepare",
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
}


PRE_HARNESS_PRIORITY_MODE = "pre_harness_priority_hints"


NAT_INFERRED_PRIORITY_MODE = "nat_inferred_priority_hints"


CACHE_LOWER_MODE = "harness_native_cache_lowered"


HARNESS_EMITTED_SIGNAL_MODE = "harness_emitted_signals"


CONTROLLER_SCHEDULER_PRIORITY_MODE = "controller_scheduler_priority"


CONTROLLER_DEMOTE_RESTORE_MODE = "controller_demote_restore"


CONTROLLER_PRIORITY_DEMOTE_MODE = "controller_priority_demote"


CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE = "controller_priority_demotion_admission"


CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE = "controller_priority_demotion_admission_soft"


CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE = "controller_priority_demotion_admission_medium"


CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE = "controller_priority_demotion_admission_hard"


CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE = "controller_priority_demotion_admission_earlyprepare"


CONTROLLER_ORACLE_SAFE_SJF_MODE = "controller_oracle_safe_sjf"


CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE = "controller_oracle_safe_sjf_balanced"


CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE = "controller_oracle_safe_sjf_aggressive"


CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE = "controller_oracle_safe_sjf_maxfill"


CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE = "controller_priority_demotion_calibrated_admission"


CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE = "controller_oracle_exact_runtime_admission"


CONTROLLER_DEADLINE_FAIR_MODE = "controller_deadline_fair"


CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_MODE = "controller_predictive_deadline_queue"


CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_ADMISSION_GUARD_MODE = "controller_predictive_deadline_queue_admission_guard"


CONTROLLER_READY_TIME_GPU_BACKFILL_MODE = "controller_ready_time_gpu_backfill"


CONTROLLER_VALUE_AWARE_EVICTION_MODE = "controller_value_aware_eviction"


CONTROLLER_MEMORY_ADMISSION_MODE = "controller_memory_admission"


CONTROLLER_ORACLE_SAFE_SJF_MODES = {
    CONTROLLER_ORACLE_SAFE_SJF_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE,
    CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE,
    CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE,
}


CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES = {
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE,
    *CONTROLLER_ORACLE_SAFE_SJF_MODES,
}


CONTROLLER_ADMISSION_CONTROL_MODE = "controller_admission_control"


CONTROLLER_FULL_MODE = "controller_full"


CONTROLLER_FULL_CHUNKED_PREFILL_MODE = "controller_full_chunked_prefill"


CONTROLLER_PRIORITY_MODES = {
    CONTROLLER_SCHEDULER_PRIORITY_MODE,
    CONTROLLER_DEMOTE_RESTORE_MODE,
    CONTROLLER_PRIORITY_DEMOTE_MODE,
    *CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
    CONTROLLER_DEADLINE_FAIR_MODE,
    CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_MODE,
    CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_ADMISSION_GUARD_MODE,
    CONTROLLER_READY_TIME_GPU_BACKFILL_MODE,
    CONTROLLER_VALUE_AWARE_EVICTION_MODE,
    CONTROLLER_MEMORY_ADMISSION_MODE,
    CONTROLLER_ADMISSION_CONTROL_MODE,
    CONTROLLER_FULL_MODE,
    CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
}


CACHE_SIGNAL_MODES = {
    "no_cache_signal",
    CACHE_LOWER_MODE,
    HARNESS_EMITTED_SIGNAL_MODE,
}


def metadata_context(meta: dict[str, Any], prompt_hash: str = "") -> dict[str, Any]:
    session_id = str(meta.get("session_id") or "harness_session")
    phase = str(meta.get("phase") or "request")
    label = str(meta.get("label") or f"{session_id}_{phase}")
    harness = str(meta.get("harness") or "unknown")
    priority_label = str(meta.get("priority_label") or ("low" if phase == "pressure_filler" else "high"))
    return {
        "experiment": "multi_harness_replay_deadline_pressure",
        "harness": harness,
        "request_role": priority_label,
        "request_id": label,
        "parent_run_id": session_id,
        "phase": phase,
        "case_id": session_id,
        "gap_id": str(meta.get("task_index") or "0"),
        "task_index": str(meta.get("task_index") or "0"),
        "correlation_id": f"{session_id}:{phase}:{label}",
        "prompt_hash": prompt_hash,
    }


def priority_enabled(mode: str) -> bool:
    return mode in PRIORITY_ENABLED_MODES


def resolve_backend_priority(meta: dict[str, Any], payload: dict[str, Any] | None = None) -> int | None:
    if not priority_enabled(str(meta.get("mode") or "")):
        return None
    phase = str(meta.get("phase") or "")
    mode = str(meta.get("mode") or "")
    if mode in CONTROLLER_PRIORITY_MODES:
        if mode == CONTROLLER_VALUE_AWARE_EVICTION_MODE:
            try:
                return int(float(meta.get("controller_sglang_priority")))
            except (TypeError, ValueError):
                value_class = str(meta.get("controller_eviction_value_class") or "")
                if value_class == "protected_high_value":
                    return int(meta.get("high_priority") or 100)
                if value_class == "normal_value":
                    return int(os.environ.get("CONTROLLER_EVICTION_NORMAL_PRIORITY", "0") or "0")
                if value_class == "evictable_low_value":
                    return int(meta.get("low_priority") or -100)
                return None
        if (
            mode
            in {
                CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_MODE,
                CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_ADMISSION_GUARD_MODE,
                CONTROLLER_READY_TIME_GPU_BACKFILL_MODE,
            }
            and meta.get("controller_predictive_deadline_queue")
        ):
            try:
                return int(float(meta.get("controller_sglang_priority")))
            except (TypeError, ValueError):
                return None
        if mode == CONTROLLER_DEADLINE_FAIR_MODE:
            if not meta.get("controller_deadline_fair"):
                return None
            try:
                return int(float(meta.get("controller_sglang_priority")))
            except (TypeError, ValueError):
                return None
        if mode in {
            CONTROLLER_DEMOTE_RESTORE_MODE,
            CONTROLLER_PRIORITY_DEMOTE_MODE,
            *CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
            CONTROLLER_FULL_MODE,
            CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
        } and phase == "pressure_filler":
            return int(meta.get("controller_demote_priority") or meta.get("low_priority") or -100)
        if phase != "replay":
            return None
        try:
            return int(float(meta.get("controller_sglang_priority")))
        except (TypeError, ValueError):
            return None
    if phase == "speculative_prefill":
        return int(meta.get("speculative_prefill_priority") or 50)
    if str(meta.get("mode") or "") == HARNESS_EMITTED_SIGNAL_MODE:
        nvext_priority = payload_nvext_priority(payload or {})
        if nvext_priority is not None:
            return nvext_priority
        if emitted_signal_is_urgent(payload or {}):
            return int(meta.get("high_priority") or 100)
        return None
    if str(meta.get("mode") or "") == NAT_INFERRED_PRIORITY_MODE:
        return payload_nvext_priority(payload or {})
    if phase.startswith("pressure_filler"):
        return int(meta.get("low_priority") or -100)
    if str(meta.get("mode") or "") == PRE_HARNESS_PRIORITY_MODE:
        intent = as_dict(meta.get("priority_intent"))
        priority_class = str(intent.get("class") or "")
        if priority_class not in {"urgent", "high"}:
            return None
    return int(meta.get("high_priority") or 100)


def cache_translation_context(meta: dict[str, Any], payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = headers or {}
    emitted = emitted_cache_signal(payload, request_headers)
    mode = str(meta.get("mode") or "")
    should_lower = mode in {CACHE_LOWER_MODE, HARNESS_EMITTED_SIGNAL_MODE} and emitted["harness_native_cache_signal_seen"] == "yes"
    lower_to_preload = mode == HARNESS_EMITTED_SIGNAL_MODE
    lowered: dict[str, Any] = {}
    if should_lower:
        cache_key = first_signal_value(payload, {"prompt_cache_key", "promptCacheKey", "cache_key", "cacheKey"})
        if not is_present(cache_key):
            cache_key = first_header_signal_value(
                request_headers,
                {"prompt-cache-key", "prompt_cache_key", "x-prompt-cache-key"},
            )
        retention = first_signal_value(
            payload,
            {"prompt_cache_retention", "promptCacheRetention", "cache_retention", "cacheRetention"},
        )
        if not is_present(retention):
            retention = first_header_signal_value(
                request_headers,
                {"prompt-cache-retention", "prompt_cache_retention", "x-prompt-cache-retention"},
            )
        control = first_signal_value(payload, {"cache_control", "cacheControl"})
        lowered = {
            "schema": "harness_native_cache_lowering.v1",
            "source": "harness_emitted_cache_signal",
            "mode": mode,
            "backend_action": "gateway_speculative_kv_preload" if lower_to_preload else "sglang_cache_salt",
            "cache_key": cache_key if is_present(cache_key) else "",
            "sglang_cache_salt": "" if lower_to_preload else str(cache_key) if is_present(cache_key) else "",
            "cache_retention": retention if is_present(retention) else "",
            "cache_control": control if is_present(control) else "",
        }
    return {
        **emitted,
        "gateway_cache_translation": compact_json(lowered),
        "gateway_cache_translation_source": "harness_emitted_cache_signal" if should_lower else "none",
        "gateway_cache_lowered": "yes" if should_lower else "no",
        "gateway_cache_salt": lowered.get("sglang_cache_salt", "") if should_lower else "",
        "gateway_cache_invented_signal": "false",
    }


def priority_translation_context(meta: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    emitted = emitted_priority_signal(payload)
    priority = resolve_backend_priority(meta, payload)
    mode = str(meta.get("mode") or "")
    if priority is None:
        source = "none"
    elif mode == NAT_INFERRED_PRIORITY_MODE:
        if emitted["harness_emit_priority_signal"]:
            source = "harness_emitted_nvext_priority"
        else:
            source = "none"
    elif mode == PRE_HARNESS_PRIORITY_MODE:
        if emitted["harness_emit_priority_signal"] and emitted_signal_is_urgent(payload):
            source = "harness_emitted_signal"
        elif meta.get("priority_intent"):
            source = "experiment_marker_priority_intent"
        else:
            source = "none"
    elif mode == CONTROLLER_VALUE_AWARE_EVICTION_MODE:
        source = "controller_value_aware_eviction_priority"
    elif mode in CONTROLLER_PRIORITY_MODES:
        if priority is not None:
            source = (
                "controller_demote_window"
                if mode in {
                    CONTROLLER_DEMOTE_RESTORE_MODE,
                    CONTROLLER_PRIORITY_DEMOTE_MODE,
                    *CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
                    CONTROLLER_FULL_MODE,
                    CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
                }
                and str(meta.get("phase") or "") == "pressure_filler"
                else "controller_full_ready_ladder"
                if mode in {CONTROLLER_FULL_MODE, CONTROLLER_FULL_CHUNKED_PREFILL_MODE}
                and is_present(meta.get("controller_priority_ladder"))
                else "controller_ready_decision"
            )
        else:
            source = "none"
    elif str(meta.get("phase") or "") == "speculative_prefill":
        source = "speculative_prefill_background_priority"
    elif mode == HARNESS_EMITTED_SIGNAL_MODE:
        if emitted["harness_emit_priority_signal"]:
            source = "harness_emitted_signal"
        else:
            source = "none"
    else:
        source = "gateway_boundary_priority_mode"
    return {
        "experiment_priority_intent": compact_json(meta.get("priority_intent")),
        "harness_input_priority_signal": compact_json(meta.get("harness_input_priority_signal")),
        "harness_input_priority_signal_source": compact_json(meta.get("harness_input_priority_signal_source")),
        **emitted,
        "gateway_priority_translation": priority if priority is not None else "",
        "gateway_priority_translation_source": source,
    }



def _controller_hint_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Controller decision fields copied verbatim from request metadata."""

    return {
        "controller_decision_id": meta.get("controller_decision_id", ""),
        "controller_command_id": meta.get("controller_command_id", ""),
        "controller_priority_translation": meta.get("controller_priority_translation", ""),
        "controller_demote_restore_active": meta.get("controller_demote_restore_active", ""),
        "controller_demote_decision_id": meta.get("controller_demote_decision_id", ""),
        "controller_demote_command_id": meta.get("controller_demote_command_id", ""),
        "controller_demote_translation": meta.get("controller_demote_translation", ""),
        "controller_admission_decision": meta.get("controller_admission_decision", ""),
        "controller_admission_reason": meta.get("controller_admission_reason", ""),
        "controller_memory_admission_decision": meta.get("controller_memory_admission_decision", ""),
        "controller_memory_admission_reason": meta.get("controller_memory_admission_reason", ""),
        "controller_memory_admission_predicted_replay_count": meta.get(
            "controller_memory_admission_predicted_replay_count", ""
        ),
        "controller_memory_admission_predicted_replay_tokens": meta.get(
            "controller_memory_admission_predicted_replay_tokens", ""
        ),
        "controller_memory_admission_reserved_headroom_tokens": meta.get(
            "controller_memory_admission_reserved_headroom_tokens", ""
        ),
        "controller_memory_admission_candidate_tokens": meta.get("controller_memory_admission_candidate_tokens", ""),
        "controller_memory_admission_delayed_for_ms": meta.get("controller_memory_admission_delayed_for_ms", ""),
        "controller_replay_rank": meta.get("controller_replay_rank", ""),
        "controller_urgent_replay_count": meta.get("controller_urgent_replay_count", ""),
        "controller_priority_ladder": meta.get("controller_priority_ladder", ""),
        "controller_eviction_policy": meta.get("controller_eviction_policy", ""),
        "controller_eviction_value_score": meta.get("controller_eviction_value_score", ""),
        "controller_eviction_value_class": meta.get("controller_eviction_value_class", ""),
        "controller_eviction_translation": meta.get("controller_eviction_translation", ""),
    }


def _kv_storage_fields(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "kv_storage_enabled": meta.get("kv_storage_enabled", ""),
        "kv_storage_backend": meta.get("kv_storage_backend", ""),
        "kv_storage_prefetch_policy": meta.get("kv_storage_prefetch_policy", ""),
        "kv_storage_path": meta.get("kv_storage_path", ""),
        "kv_storage_signal_source": meta.get("kv_storage_signal_source", ""),
    }


def translate_request(payload: dict[str, Any], meta: dict[str, Any], api_kind: str) -> BackendRequest:
    """Translate one incoming harness request into a backend-neutral request.

    This is the backend-neutral half of the former ``build_sglang_payload``.
    Key order inside the produced dictionaries is kept identical to the old
    implementation so that lowered JSON bodies are byte-for-byte unchanged.
    """

    if api_kind == "anthropic":
        system, user = text_from_anthropic_messages(payload)
    elif api_kind == "responses":
        system, user = text_from_responses(payload)
    else:
        system, user = text_from_openai_chat(payload)
    prompt_text = "\n\n".join(part for part in [system, user] if part).strip() or "Continue."
    context = metadata_context(meta, prompt_hash=str(meta.get("prompt_hash") or "")[:32])
    priority = resolve_backend_priority(meta, payload)
    priority_chain = priority_translation_context(meta, payload)
    cache_chain = cache_translation_context(meta, payload)
    observed_meta = {
        **{key: value for key, value in meta.items() if key != "harness_controller_signal"},
        **priority_chain,
        **cache_chain,
        "sglang_priority": priority if priority is not None else "",
    }
    harness_controller_signal = build_harness_controller_signal(observed_meta)
    controller_fields = _controller_hint_fields(meta)
    agent_hints = {
        "schema": "nvext.agent_hints",
        "session_id": context["parent_run_id"],
        "request_id": context["request_id"],
        "phase": context["phase"],
        "task_index": context["task_index"],
        "prompt_hash": context["prompt_hash"],
        "priority": priority if priority is not None else 0,
        "priority_label": context["request_role"],
        "expected_action": "consume_kv" if context["phase"] == "replay" else "mark_session_priority",
        "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
        "kv_cache_relevant": context["phase"] in {"initial_turn", "replay"},
        "speculative_prefill": bool(meta.get("speculative_prefill")),
        "speculative_prefill_role": meta.get("speculative_prefill_role", ""),
        "speculative_prefill_strategy": meta.get("speculative_prefill_strategy", ""),
        "parent_request_id": meta.get("parent_request_id", ""),
        "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
        "experiment_priority_intent": priority_chain["experiment_priority_intent"],
        "harness_input_priority_signal": priority_chain["harness_input_priority_signal"],
        "harness_emit_priority_signal": priority_chain["harness_emit_priority_signal"],
        "gateway_priority_translation": priority_chain["gateway_priority_translation"],
        "gateway_priority_translation_source": priority_chain["gateway_priority_translation_source"],
        **controller_fields,
        "harness_native_cache_signal_seen": cache_chain["harness_native_cache_signal_seen"],
        "harness_native_cache_signal": cache_chain["harness_native_cache_signal"],
        "gateway_cache_translation": cache_chain["gateway_cache_translation"],
        "gateway_cache_translation_source": cache_chain["gateway_cache_translation_source"],
        "gateway_cache_lowered": cache_chain["gateway_cache_lowered"],
        "gateway_cache_salt": cache_chain["gateway_cache_salt"],
        "gateway_cache_invented_signal": cache_chain["gateway_cache_invented_signal"],
        **_kv_storage_fields(meta),
        "harness_controller_signal": harness_controller_signal,
    }
    if context["phase"] == "speculative_prefill":
        agent_hints["expected_action"] = "warm_next_turn_prefix"
        agent_hints["kv_cache_relevant"] = True
    request_metadata = {
        "session_id": context["parent_run_id"],
        "phase": context["phase"],
        "label": context["request_id"],
        "mode": str(meta.get("mode") or ""),
        "harness": context["harness"],
        "prompt_hash": context["prompt_hash"],
        "priority": context["request_role"],
        "task_index": context["task_index"],
        "request_id": context["request_id"],
        "parent_run_id": context["parent_run_id"],
        "correlation_id": context["correlation_id"],
        "case_id": context["case_id"],
        "gap_id": context["gap_id"],
        "dynamo_agent_priority": context["request_role"],
        "sglang_priority": priority if priority is not None else "",
        "dynamo_hint_priority": priority if priority is not None else "",
        "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
        "harness_controller_signal": harness_controller_signal,
        "speculative_prefill": bool(meta.get("speculative_prefill")),
        "speculative_prefill_role": meta.get("speculative_prefill_role", ""),
        "speculative_prefill_strategy": meta.get("speculative_prefill_strategy", ""),
        **controller_fields,
        "parent_request_id": meta.get("parent_request_id", ""),
        "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
        "warmup_prompt_tokens": meta.get("warmup_prompt_tokens", ""),
        **_kv_storage_fields(meta),
        **priority_chain,
        **cache_chain,
    }
    speculative_prefill = {
        "speculative_prefill": bool(meta.get("speculative_prefill")),
        "role": meta.get("speculative_prefill_role", ""),
        "strategy": meta.get("speculative_prefill_strategy", ""),
        "parent_request_id": meta.get("parent_request_id", ""),
        "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
        "warmup_prompt_tokens": meta.get("warmup_prompt_tokens", ""),
    }
    native_cache_bridge = None
    if cache_chain["gateway_cache_lowered"] == "yes":
        native_cache_bridge = as_dict(json.loads(str(cache_chain["gateway_cache_translation"] or "{}")))
    cache_salt = str(cache_chain["gateway_cache_salt"]) if is_present(cache_chain.get("gateway_cache_salt")) else ""
    return BackendRequest(
        prompt_text=prompt_text,
        max_tokens=int(meta.get("max_tokens") or payload.get("max_tokens") or payload.get("max_output_tokens") or 8),
        priority=priority,
        cache_salt=cache_salt,
        temperature=0,
        stream=True,
        request_context=context,
        agent_hints=agent_hints,
        request_metadata=request_metadata,
        speculative_prefill=speculative_prefill,
        native_cache_bridge=native_cache_bridge,
    )
