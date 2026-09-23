"""Controller timing policy: prepare lead times, direct-load windows, deadline ranks.

These functions decide *when* the controller should act relative to a tool
call's expected return (how early to prepare, whether a host->device prefix
load fits before the replay deadline, which scheduler priority a replay due
at time T gets).  They are backend-neutral: they return numbers and admission
records; lowering them into SGLang happens elsewhere.

Moved verbatim from ``sglang_direct_kv/scripts/run_multi_harness_replay_driver.py``
during the SGLang-portability refactor (the driver re-imports them).  They
still read their tuning knobs from environment variables at call time, exactly
as before, so existing run scripts keep working.
"""

from __future__ import annotations

import os
from typing import Any

AUTHORIZED_DIRECT_LOAD_MECHANISM = "prepared_prefix_control"


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def controller_prepare_lead_ms(wait_ms: int) -> int:
    if wait_ms < 500:
        return 0
    if wait_ms <= 3_000:
        return min(wait_ms, 500)
    if wait_ms <= 10_000:
        return min(wait_ms, 1_000)
    return min(wait_ms, 1_500)


def controller_targeted_prefetch_lead_ms(wait_ms: int) -> int:
    override = os.environ.get("CONTROLLER_TARGETED_PREFETCH_LEAD_MS")
    if override not in (None, ""):
        try:
            return max(0, int(float(override)))
        except ValueError:
            pass
    return controller_prepare_lead_ms(wait_ms)


def controller_direct_load_estimate_ms(prompt_tokens: int) -> int:
    fixed_ms = float(os.environ.get("CONTROLLER_DIRECT_LOAD_FIXED_MS", "1000") or "1000")
    ms_per_token = float(os.environ.get("CONTROLLER_DIRECT_LOAD_MS_PER_TOKEN", "2.0") or "2.0")
    multiplier = float(os.environ.get("CONTROLLER_DIRECT_LOAD_ESTIMATE_MULTIPLIER", "1.0") or "1.0")
    base_estimate_ms = fixed_ms + (max(0, prompt_tokens) * ms_per_token)
    return max(0, int(round(base_estimate_ms * max(0.0, multiplier))))


def controller_direct_load_allowed_wait_classes() -> set[str]:
    raw = os.environ.get("CONTROLLER_DIRECT_LOAD_ALLOWED_WAIT_CLASSES", "all")
    classes = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not classes or "all" in classes or "*" in classes:
        return set()
    return classes


def controller_direct_load_contract() -> str:
    if env_truthy("CONTROLLER_DIRECT_LOAD_REQUIRE_COMPLETION", default=False):
        return "requires_direct_load_completion_before_replay"
    return "best_effort_direct_load_before_replay"


def controller_direct_load_safety_margin_ms() -> int:
    return max(0, int(float(os.environ.get("CONTROLLER_DIRECT_LOAD_SAFETY_MARGIN_MS", "750") or "750")))


def controller_direct_load_latest_finish_ms(replay_due_ms: float) -> float:
    return replay_due_ms - controller_direct_load_safety_margin_ms()


def controller_direct_load_mechanism() -> str:
    mechanism = os.environ.get("CONTROLLER_DIRECT_LOAD_MECHANISM", AUTHORIZED_DIRECT_LOAD_MECHANISM).strip()
    mechanism = mechanism or AUTHORIZED_DIRECT_LOAD_MECHANISM
    if mechanism != AUTHORIZED_DIRECT_LOAD_MECHANISM:
        raise ValueError(
            "Unsupported controller direct-load mechanism "
            f"{mechanism!r}. The legacy synthetic-request KV warmup path has been removed; "
            f"use {AUTHORIZED_DIRECT_LOAD_MECHANISM!r}, which calls SGLang/HiCache prepare-prefix control."
        )
    return mechanism


def controller_direct_load_window(
    *,
    tool_start_ms: float,
    replay_due_ms: float,
    prompt_tokens: int,
    wait_class: str | None = None,
) -> dict[str, Any]:
    estimated_ms = controller_direct_load_estimate_ms(prompt_tokens)
    safety_margin_ms = controller_direct_load_safety_margin_ms()
    latest_finish_ms = replay_due_ms - safety_margin_ms
    available_slack_ms = latest_finish_ms - tool_start_ms
    allowed_wait_classes = controller_direct_load_allowed_wait_classes()
    normalized_wait_class = (wait_class or "").strip().lower()
    contract = controller_direct_load_contract()
    if allowed_wait_classes and normalized_wait_class not in allowed_wait_classes:
        return {
            "admitted": False,
            "reason": "skip_prefetch_wait_class_not_allowed_by_guarantee_contract",
            "contract": contract,
            "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)),
            "estimated_ms": estimated_ms,
            "safety_margin_ms": safety_margin_ms,
            "available_slack_ms": available_slack_ms,
            "start_ms": tool_start_ms,
            "latest_finish_ms": latest_finish_ms,
        }
    if latest_finish_ms <= tool_start_ms:
        return {
            "admitted": False,
            "reason": "skip_prefetch_no_safe_window_before_replay",
            "contract": contract,
            "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)) if allowed_wait_classes else "all",
            "estimated_ms": estimated_ms,
            "safety_margin_ms": safety_margin_ms,
            "available_slack_ms": available_slack_ms,
            "start_ms": tool_start_ms,
            "latest_finish_ms": latest_finish_ms,
        }
    if estimated_ms > available_slack_ms:
        return {
            "admitted": False,
            "reason": "skip_prefetch_not_enough_eta_slack",
            "contract": contract,
            "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)) if allowed_wait_classes else "all",
            "estimated_ms": estimated_ms,
            "safety_margin_ms": safety_margin_ms,
            "available_slack_ms": available_slack_ms,
            "start_ms": tool_start_ms,
            "latest_finish_ms": latest_finish_ms,
        }
    # Start at the latest safe point that still leaves the estimated load time
    # plus the replay safety margin. Starting too early can observe the prefix
    # before pressure evicts it, then falsely conclude no H2D work is needed.
    start_ms = max(tool_start_ms, latest_finish_ms - estimated_ms)
    return {
        "admitted": True,
        "reason": "admit_prefetch_latest_safe_start_before_replay",
        "contract": contract,
        "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)) if allowed_wait_classes else "all",
        "estimated_ms": estimated_ms,
        "safety_margin_ms": safety_margin_ms,
        "available_slack_ms": available_slack_ms,
        "start_ms": start_ms,
        "latest_finish_ms": latest_finish_ms,
    }


def deadline_fair_priority_for_due(replay_due_ms: float) -> int:
    base = int(os.environ.get("CONTROLLER_DEADLINE_FAIR_BASE_PRIORITY", "100000") or "100000")
    bucket_ms = max(1, int(os.environ.get("CONTROLLER_DEADLINE_FAIR_BUCKET_MS", "10") or "10"))
    due_bucket = int(max(0.0, replay_due_ms) // bucket_ms)
    return max(1, base - due_bucket)


__all__ = ["controller_prepare_lead_ms", "controller_targeted_prefetch_lead_ms", "controller_direct_load_estimate_ms", "controller_direct_load_allowed_wait_classes", "controller_direct_load_contract", "controller_direct_load_safety_margin_ms", "controller_direct_load_latest_finish_ms", "controller_direct_load_mechanism", "controller_direct_load_window", "deadline_fair_priority_for_due"]
