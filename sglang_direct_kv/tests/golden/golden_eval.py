"""Evaluate gateway / driver functions over the golden matrices.

Shared by ``generate_golden.py`` (records fixtures) and
``test_golden_behavior.py`` (replays and compares).
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from golden_matrix import (
    DUES_MS,
    ENV_PROFILES,
    WAITS_MS,
    WINDOW_CASES,
    digest,
    env_profile,
    eviction_meta_cases,
    gateway_cases,
    outbound_cases,
)

FULL_SAMPLE_EVERY = 211  # keep a few full outputs for human-readable diffs


def evaluate_gateway(gw: Any) -> dict[str, Any]:
    """Run every gateway translation entry point over the gateway matrix."""

    out: dict[str, Any] = {"digests": {}, "samples": {}}
    for profile in ENV_PROFILES:
        with env_profile(profile):
            for index, case in enumerate(gateway_cases()):
                payload = copy.deepcopy(case["payload"])
                meta = copy.deepcopy(case["meta"])
                headers = dict(case["headers"])
                result = {
                    "sglang_priority": gw.sglang_priority(meta, payload),
                    "priority_translation_context": gw.priority_translation_context(meta, payload),
                    "cache_translation_context": gw.cache_translation_context(meta, payload, headers),
                    "emitted_priority_signal": gw.emitted_priority_signal(payload),
                    "emitted_cache_signal": gw.emitted_cache_signal(payload, headers),
                    "emitted_signal_is_urgent": gw.emitted_signal_is_urgent(payload),
                    "metadata_context": gw.metadata_context(meta, prompt_hash="ph"),
                    "build_sglang_payload": gw.build_sglang_payload(payload, meta, case["api_kind"], "golden-model"),
                }
                # Inputs must not be mutated by translation.
                result["inputs_unchanged"] = payload == case["payload"] and meta == case["meta"]
                key = f"{profile}|{case['case']}"
                out["digests"][key] = {name: digest(value) for name, value in result.items()}
                if index % FULL_SAMPLE_EVERY == 0:
                    out["samples"][key] = result
    return out


def evaluate_driver(drv: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for profile in ENV_PROFILES:
        with env_profile(profile):
            rows: dict[str, Any] = {}
            rows["controller_prepare_lead_ms"] = {str(w): drv.controller_prepare_lead_ms(w) for w in WAITS_MS}
            rows["controller_targeted_prefetch_lead_ms"] = {
                str(w): drv.controller_targeted_prefetch_lead_ms(w) for w in WAITS_MS
            }
            rows["controller_direct_load_estimate_ms"] = {
                str(t): drv.controller_direct_load_estimate_ms(t) for t in [-5, 0, 1, 1000, 12345]
            }
            rows["controller_direct_load_allowed_wait_classes"] = sorted(drv.controller_direct_load_allowed_wait_classes())
            rows["controller_direct_load_contract"] = drv.controller_direct_load_contract()
            rows["controller_direct_load_safety_margin_ms"] = drv.controller_direct_load_safety_margin_ms()
            rows["controller_direct_load_latest_finish_ms"] = {
                str(d): drv.controller_direct_load_latest_finish_ms(d) for d in DUES_MS
            }
            rows["controller_direct_load_window"] = [drv.controller_direct_load_window(**case) for case in WINDOW_CASES]
            rows["deadline_fair_priority_for_due"] = {str(d): drv.deadline_fair_priority_for_due(d) for d in DUES_MS}
            rows["value_aware_eviction_metadata"] = [
                drv.value_aware_eviction_metadata(meta) for meta in eviction_meta_cases()
            ]
            rows["outbound_priority_fields"] = [
                drv.outbound_priority_fields(meta, api_kind) for meta, api_kind in outbound_cases()
            ]
            out[profile] = rows
    return out


def compare(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Small structural diff that reports the first few mismatching paths."""

    problems: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            elif key not in expected:
                problems.append(f"{path}.{key}: unexpected")
            else:
                problems.extend(compare(expected[key], actual[key], f"{path}.{key}"))
            if len(problems) > 20:
                break
    elif isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for index, (left, right) in enumerate(zip(expected, actual)):
            problems.extend(compare(left, right, f"{path}[{index}]"))
    elif expected != actual:
        problems.append(f"{path}: expected {expected!r} got {actual!r}")
    return problems


Loader = Callable[[], Any]
