"""Presentation helpers for the GPU, CPU, and storage KV-tier study."""

from __future__ import annotations

from statistics import median
import shlex


SCHEMA = "agentic_work_audit.memory_tiers.summary.v1"
HEADERS = (
    "Seed / return pattern / mode",
    "Whole workload (s)",
    "Due to first token mean / p95 (ms)",
    "TTFT mean / p95 (ms)",
    "Total due delay / TTFT (s)",
    "Submission waiting (ms)",
    "Proven GPU / CPU / storage source requests",
    "Proven GPU / CPU / storage source tokens",
    "Active sessions max / limit",
    "Active KV estimate max / limit",
    "Slot wait / KV preparation mean (ms)",
)
COMPARISON_HEADERS = (
    "Seed / return pattern / tier",
    "Whole-workload change vs GPU",
    "Mean / p95 replay-delay increase (ms)",
    "Mean TTFT increase (ms)",
)
NOTES = (
    "Positive changes mean the lower tier was slower than the all-GPU reference. "
    "Due-to-first-token includes any delay before submission plus backend TTFT. "
    "Tier-source counts come from native SGLang evidence, not from the requested mode name. "
    "Capacity-safe arms use pre-admission residency; original pressure arms use replay cache matches. "
    "File-backed storage uses normal operating-system caching; the page cache was not flushed."
)


def table_rows(summary: dict) -> list[tuple[str, ...]]:
    rows = []
    for arm in summary.get("arms", []):
        rows.append((
            f"{arm['seed']} / {arm['pattern']} / {arm['mode']}",
            f"{arm['workflow_duration_ms'] / 1000:.3f}",
            f"{arm['mean_due_to_first_token_ms']:.1f} / {arm['p95_due_to_first_token_ms']:.1f}",
            f"{arm['mean_ttft_ms']:.1f} / {arm['p95_ttft_ms']:.1f}",
            f"{arm['total_due_to_first_token_ms'] / 1000:.3f} / {arm['total_ttft_ms'] / 1000:.3f}",
            f"{arm['total_submission_delay_ms']:.1f}",
            f"{arm['native_gpu_hit_replays']} / {arm['native_host_hit_replays']} / "
            f"{arm['native_storage_hit_replays']}",
            f"{arm['native_gpu_hit_tokens']:,} / {arm['native_host_hit_tokens']:,} / "
            f"{arm['native_storage_hit_tokens']:,}",
            (f"{arm['max_active_observed']} / {arm['max_active_allowed']}"
             if arm.get("max_active_observed") is not None else "not recorded"),
            (f"{arm['max_active_tokens_observed']:,} / {arm['active_token_limit']:,} tokens"
             if arm.get("max_active_tokens_observed") is not None else "not recorded"),
            (f"{arm['mean_slot_wait_ms']:.1f} / {arm['mean_kv_prepare_ms']:.1f}"
             if arm.get("mean_slot_wait_ms") is not None else "not recorded"),
        ))
    return rows


def comparison_rows(summary: dict) -> list[tuple[str, ...]]:
    return [(
        f"{row['seed']} / {row['pattern']} / {row['mode']}",
        f"{row['workload_delta_ms'] / 1000:+.3f}s ({row['workload_change_pct']:+.1f}%)",
        f"{row['mean_due_to_first_token_delta_ms']:+.1f} / "
        f"{row['p95_due_to_first_token_delta_ms']:+.1f}",
        f"{row['mean_ttft_delta_ms']:+.1f}",
    ) for row in summary.get("comparisons", [])]


def setup(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    runtime = manifest.get("backend_runtime_contract") or {}
    capacity_safe = summary.get("variant") == "capacity_safe_active_set"
    capacity_note = (
        f" At most {workload.get('max_active', '?')} sessions could run at once, and their "
        f"estimated combined active KV had to remain below {workload.get('active_token_limit', '?')} "
        "tokens. Waiting sessions were restored completely before admission."
        if capacity_safe else ""
    )
    return (
        f"{workload.get('session_count', '?')} equal-priority sessions; "
        f"{workload.get('turns_per_session', '?')} tool returns per session; "
        f"{workload.get('initial_tokens', '?')} initial prompt tokens; "
        f"{workload.get('decode_tokens', '?')} output tokens per replay; "
        f"{workload.get('tool_wait_ms', '?')} ms tool waits. Return patterns: "
        f"{', '.join(workload.get('return_patterns') or [])}; the burst spans "
        f"{workload.get('burst_window_ms', '?')} ms and the spread control spans "
        f"{workload.get('spread_window_ms', '?')} ms. Fresh backend per arm; no KV prefetch; "
        f"all-GPU cap {workload.get('resident_gpu_tokens', '?')} tokens; lower-tier GPU cap "
        f"{workload.get('restricted_gpu_tokens', '?')} tokens; CPU caches "
        f"{workload.get('resident_host_cache_gb', '?')} GiB allocated but unused for measured "
        "replays in all-GPU mode, "
        f"{workload.get('host_cache_gb', '?')} GiB for CPU mode and "
        f"{workload.get('storage_host_cache_gb', '?')} GiB for storage mode. "
        f"CUDA graphs {'on' if workload.get('cuda_graph') else 'off'}; overlap scheduling "
        f"{'on' if workload.get('overlap_schedule') else 'off'}; frontend priority equal; "
        f"{workload.get('trace_profile', 'kv_lifecycle_lean')} tracing. Model: "
        f"{manifest.get('model', 'not recorded')}; hardware: "
        f"{manifest.get('hardware_profile', 'not recorded')}; backend: "
        f"{manifest.get('backend_version') or runtime.get('backend_version', 'not recorded')}."
        + capacity_note
    )


def finding(summary: dict) -> str:
    if summary.get("status") != "complete":
        reasons = summary.get("issues") or summary.get("exposure_warnings") or []
        return "Evidence incomplete; no tier-performance conclusion. " + "; ".join(reasons[:2])
    pieces = []
    comparisons = summary.get("comparisons", [])
    for pattern in ("spread", "burst"):
        for mode, label in (("host", "CPU tier"), ("storage", "storage tier")):
            values = [row for row in comparisons if row["pattern"] == pattern and row["mode"] == mode]
            if not values:
                continue
            workload = median(row["workload_change_pct"] for row in values)
            delay = median(row["mean_due_to_first_token_delta_ms"] for row in values)
            pieces.append(f"{pattern} {label}: whole workload {workload:+.1f}% and mean replay delay {delay:+.1f} ms vs all-GPU.")
    return " ".join(pieces)


def reproduction(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    capacity_safe = summary.get("variant") == "capacity_safe_active_set"
    prefix = "CAPACITY_TIER" if capacity_safe else "TIER"
    values = {
        f"{prefix}_RUN_ID": str(summary.get("run_id", "memory_tiers")) + "_repeat",
        f"{prefix}_MODEL": manifest.get("model"),
        f"{prefix}_SEEDS": " ".join(map(str, workload.get("seeds") or [])),
        f"{prefix}_PATTERNS": " ".join(workload.get("return_patterns") or []),
        f"{prefix}_MODES": " ".join(workload.get("modes") or []),
        f"{prefix}_SESSIONS": workload.get("session_count"),
        f"{prefix}_TURNS": workload.get("turns_per_session"),
        f"{prefix}_INITIAL_TOKENS": workload.get("initial_tokens"),
        f"{prefix}_TOOL_WORDS": workload.get("tool_result_words"),
        f"{prefix}_DECODE_TOKENS": workload.get("decode_tokens"),
        f"{prefix}_WAIT_MS": workload.get("tool_wait_ms"),
        f"{prefix}_BURST_WINDOW_MS": workload.get("burst_window_ms"),
        f"{prefix}_SPREAD_WINDOW_MS": workload.get("spread_window_ms"),
        f"{prefix}_RESIDENT_GPU_TOKENS": workload.get("resident_gpu_tokens"),
        f"{prefix}_RESIDENT_HOST_GB": workload.get("resident_host_cache_gb"),
        f"{prefix}_RESTRICTED_GPU_TOKENS": workload.get("restricted_gpu_tokens"),
        f"{prefix}_HOST_GB": workload.get("host_cache_gb"),
        f"{prefix}_STORAGE_HOST_GB": workload.get("storage_host_cache_gb"),
    }
    if capacity_safe:
        values[f"{prefix}_MAX_ACTIVE"] = workload.get("max_active")
        values[f"{prefix}_ACTIVE_TOKEN_LIMIT"] = workload.get("active_token_limit")
    else:
        values[f"{prefix}_MAX_INFLIGHT"] = workload.get("max_inflight")
    command = " \\\n".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items() if value is not None
    )
    script = ("run_work_audit_capacity_safe_tiers.sh" if capacity_safe
              else "run_work_audit_memory_tiers.sh")
    return command + f" \\\nbash infra/container/{script}"
