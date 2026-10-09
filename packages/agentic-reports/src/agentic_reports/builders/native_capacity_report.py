"""Presentation helpers for native SGLang versus capacity-safe admission."""

from __future__ import annotations


HEADERS = (
    "Seed / returns / policy", "Whole workload (s)", "Due → first token mean / p95 (ms)",
    "Backend TTFT mean / p95 (ms)", "Total due delay (s)", "Pre-submission wait (s)",
    "Proven host / GPU source replays", "Capacity violations",
)
COMPARISON_HEADERS = (
    "Seed / returns", "Native → capacity-safe workload", "Workload change",
    "Native → capacity-safe mean replay delay", "Replay-delay change",
    "Native → capacity-safe p95 replay delay",
)
NOTES = (
    "Lower time is better. Whole-workload change is capacity-safe relative to native SGLang. "
    "Due-to-first-token includes waiting outside SGLang plus backend TTFT."
)


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.3f}"


def table_rows(summary: dict) -> list[tuple[object, ...]]:
    rows = []
    for arm in summary.get("arms", []):
        violations = arm.get("capacity_violations")
        rows.append((
            f"{arm['seed']} / {arm['pattern']} / {arm['policy']}",
            _seconds(arm["workflow_duration_ms"]),
            f"{arm['mean_due_to_first_token_ms']:.1f} / {arm['p95_due_to_first_token_ms']:.1f}",
            f"{arm['mean_ttft_ms']:.1f} / {arm['p95_ttft_ms']:.1f}",
            _seconds(arm["total_due_to_first_token_ms"]),
            _seconds(arm["total_submission_delay_ms"]),
            f"{arm['native_host_hit_replays']} / {arm['native_gpu_hit_replays']}",
            len(violations or []),
        ))
    return rows


def comparison_rows(summary: dict) -> list[tuple[object, ...]]:
    return [(
        f"{row['seed']} / {row['pattern']}",
        f"{_seconds(row['native_workflow_ms'])} → {_seconds(row['capacity_safe_workflow_ms'])} s",
        f"{row['workflow_delta_ms'] / 1000:+.3f} s ({row['workflow_change_pct']:+.1f}%)",
        (f"{row['native_mean_due_to_first_token_ms']:.1f} → "
         f"{row['capacity_safe_mean_due_to_first_token_ms']:.1f} ms"),
        f"{row['mean_due_to_first_token_delta_ms']:+.1f} ms",
        (f"{row['native_p95_due_to_first_token_ms']:.1f} → "
         f"{row['capacity_safe_p95_due_to_first_token_ms']:.1f} ms"),
    ) for row in summary.get("comparisons", [])]


def finding(summary: dict) -> str:
    medians = {row["pattern"]: row for row in summary.get("pattern_medians", [])}
    spread = medians.get("spread")
    burst = medians.get("burst")
    if not spread or not burst:
        return "The paired native-versus-capacity-safe comparison is incomplete."
    return (
        "With identical workloads and backend memory limits, capacity-safe admission shortened "
        f"whole-workload time by {abs(spread['workflow_change_pct']):.1f}% for spread returns and "
        f"{abs(burst['workflow_change_pct']):.1f}% for burst returns. It also increased average "
        f"tool-due-to-first-token delay by {spread['mean_due_to_first_token_delta_ms']:.0f} ms and "
        f"{burst['mean_due_to_first_token_delta_ms']:.0f} ms respectively because requests waited "
        "outside SGLang for a safe slot and complete KV restoration."
    )


def setup(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    return (
        f"{workload.get('session_count')} equal-priority sessions, "
        f"{workload.get('turns_per_session')} tool returns each, "
        f"{workload.get('initial_tokens')} initial tokens, "
        f"{workload.get('decode_tokens')} output tokens per replay, and "
        f"{workload.get('tool_wait_ms')} ms tool waits. Native SGLang received every due request "
        "immediately; capacity-safe held requests outside SGLang, restored their complete CPU-tier "
        f"KV, and admitted at most {workload.get('max_active_capacity_safe')} at once. Both used "
        f"the same {workload.get('gpu_token_limit')}-token GPU KV limit, "
        f"{workload.get('host_cache_gb')} GiB host cache, CUDA graphs, overlap scheduling, fresh "
        "backends, and lean KV tracing. Two order-reversed seeds tested spread and 75 ms burst returns."
    )


def reproduction(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    settings = {
        "POLICY_COMPARE_RUN_ID": "new_unique_id",
        "POLICY_COMPARE_MODEL": manifest.get("model"),
        "POLICY_COMPARE_SEEDS": " ".join(map(str, workload.get("seeds") or [])),
        "POLICY_COMPARE_PATTERNS": " ".join(workload.get("return_patterns") or []),
        "POLICY_COMPARE_SESSIONS": workload.get("session_count"),
        "POLICY_COMPARE_TURNS": workload.get("turns_per_session"),
        "POLICY_COMPARE_INITIAL_TOKENS": workload.get("initial_tokens"),
        "POLICY_COMPARE_PRIME_TOKENS": workload.get("prime_tokens"),
        "POLICY_COMPARE_TOOL_WORDS": workload.get("tool_result_words"),
        "POLICY_COMPARE_DECODE_TOKENS": workload.get("decode_tokens"),
        "POLICY_COMPARE_WAIT_MS": workload.get("tool_wait_ms"),
        "POLICY_COMPARE_BURST_WINDOW_MS": workload.get("burst_window_ms"),
        "POLICY_COMPARE_SPREAD_WINDOW_MS": workload.get("spread_window_ms"),
        "POLICY_COMPARE_GPU_TOKENS": workload.get("gpu_token_limit"),
        "POLICY_COMPARE_HOST_GB": workload.get("host_cache_gb"),
        "POLICY_COMPARE_MAX_ACTIVE": workload.get("max_active_capacity_safe"),
    }
    prefix = " ".join(f"{key}='{value}'" for key, value in settings.items() if value is not None)
    return prefix + " bash infra/container/run_work_audit_native_vs_capacity_safe.sh"
