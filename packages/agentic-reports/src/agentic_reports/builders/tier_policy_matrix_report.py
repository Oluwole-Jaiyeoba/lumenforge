"""Presentation helpers for the paired tier and admission-policy matrix."""

from __future__ import annotations

from statistics import median
import shlex


HEADERS = (
    "Seed / returns / tier / policy", "Whole workload (s)",
    "Due → first token mean / p95 (ms)", "Backend TTFT mean / p95 (ms)",
    "Pre-submission wait (s)", "Proven GPU / CPU / storage source replays",
    "GPU tokens / host GiB / storage", "Capacity violations",
)
POLICY_HEADERS = (
    "Seed / returns / tier", "Native → capacity-safe workload", "Workload change",
    "Native → capacity-safe mean replay delay", "Replay-delay change",
    "Native → capacity-safe p95 replay delay",
)
TIER_HEADERS = (
    "Seed / returns / policy / lower tier", "Workload change vs GPU resident",
    "Mean replay-delay change vs GPU resident",
)
NOTES = (
    "Within each tier, native SGLang and capacity-safe used the same model, requests, "
    "GPU KV capacity, host-cache capacity, storage configuration, CUDA graphs, overlap "
    "scheduling, and trace profile. Only admission and restore policy changed. Lower is better."
)


def table_rows(summary: dict) -> list[tuple[str, ...]]:
    rows = []
    for arm in summary.get("arms", []):
        backend = arm.get("backend_contract") or {}
        rows.append((
            f"{arm['seed']} / {arm['pattern']} / {arm['mode']} / {arm['policy']}",
            f"{arm['workflow_duration_ms'] / 1000:.3f}",
            f"{arm['mean_due_to_first_token_ms']:.1f} / {arm['p95_due_to_first_token_ms']:.1f}",
            f"{arm['mean_ttft_ms']:.1f} / {arm['p95_ttft_ms']:.1f}",
            f"{arm['total_submission_delay_ms'] / 1000:.3f}",
            f"{arm['native_gpu_hit_replays']} / {arm['native_host_hit_replays']} / "
            f"{arm['native_storage_hit_replays']}",
            f"{backend.get('gpu_token_limit', '?')} / {backend.get('host_cache_gb', '?')} / "
            f"{'file' if backend.get('storage_enabled') else 'off'}",
            str(len(arm.get("capacity_violations") or [])),
        ))
    return rows


def policy_rows(summary: dict) -> list[tuple[str, ...]]:
    return [(
        f"{row['seed']} / {row['pattern']} / {row['mode']}",
        f"{row['native_workflow_ms'] / 1000:.3f} → {row['capacity_safe_workflow_ms'] / 1000:.3f} s",
        f"{row['workflow_change_pct']:+.1f}%",
        f"{row['native_mean_due_to_first_token_ms']:.1f} → "
        f"{row['capacity_safe_mean_due_to_first_token_ms']:.1f} ms",
        f"{row['mean_due_to_first_token_delta_ms']:+.1f} ms",
        f"{row['native_p95_due_to_first_token_ms']:.1f} → "
        f"{row['capacity_safe_p95_due_to_first_token_ms']:.1f} ms",
    ) for row in summary.get("policy_comparisons", [])]


def tier_rows(summary: dict) -> list[tuple[str, ...]]:
    return [(
        f"{row['seed']} / {row['pattern']} / {row['policy']} / {row['mode']}",
        f"{row['workload_delta_ms'] / 1000:+.3f} s ({row['workload_change_pct']:+.1f}%)",
        f"{row['mean_due_to_first_token_delta_ms']:+.1f} ms",
    ) for row in summary.get("tier_comparisons", [])]


def setup(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    return (
        f"{workload.get('session_count', '?')} equal-priority sessions; "
        f"{workload.get('turns_per_session', '?')} tool returns each; "
        f"{workload.get('initial_tokens', '?')} initial prompt tokens; "
        f"{workload.get('decode_tokens', '?')} output tokens per replay; "
        f"{workload.get('tool_wait_ms', '?')} ms tool waits; return patterns "
        f"{', '.join(workload.get('return_patterns') or [])}. GPU-resident capacity: "
        f"{workload.get('resident_gpu_tokens', '?')} tokens and "
        f"{workload.get('resident_host_cache_gb', '?')} GiB host cache. CPU-tier capacity: "
        f"{workload.get('restricted_gpu_tokens', '?')} GPU tokens and "
        f"{workload.get('host_cache_gb', '?')} GiB host cache. Storage-tier capacity: "
        f"{workload.get('restricted_gpu_tokens', '?')} GPU tokens, "
        f"{workload.get('storage_host_cache_gb', '?')} GiB host cache, and file-backed storage. "
        "Within every tier, native and capacity-safe used exactly the same capacities and workload. "
        "CUDA graphs and overlap scheduling were on; frontend priority was equal; no KV was "
        f"prefetched before tool return. Model: {manifest.get('model', 'not recorded')}; "
        f"hardware: {manifest.get('hardware_profile', 'not recorded')}."
    )


def finding(summary: dict) -> str:
    if summary.get("status") != "complete":
        reasons = summary.get("issues") or summary.get("exposure_warnings") or []
        return "Evidence incomplete; no policy or tier conclusion. " + "; ".join(reasons[:2])
    pieces = []
    rows = summary.get("policy_comparisons", [])
    for pattern in ("spread", "burst"):
        for mode, label in (("resident", "GPU"), ("host", "CPU"), ("storage", "storage")):
            selected = [row for row in rows if row["pattern"] == pattern and row["mode"] == mode]
            if selected:
                workflow = median(row["workflow_change_pct"] for row in selected)
                replay = median(row["mean_due_to_first_token_delta_ms"] for row in selected)
                pieces.append(
                    f"{pattern} {label}: capacity-safe changed whole-workload time by "
                    f"{workflow:+.1f}% and mean replay delay by {replay:+.1f} ms."
                )
    return " ".join(pieces)


def reproduction(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    values = {
        "TIER_POLICY_RUN_ID": str(summary.get("run_id", "tier_policy_matrix")) + "_repeat",
        "TIER_POLICY_MODEL": manifest.get("model"),
        "TIER_POLICY_SEEDS": " ".join(map(str, workload.get("seeds") or [])),
        "TIER_POLICY_PATTERNS": " ".join(workload.get("return_patterns") or []),
        "TIER_POLICY_SESSIONS": workload.get("session_count"),
        "TIER_POLICY_TURNS": workload.get("turns_per_session"),
        "TIER_POLICY_INITIAL_TOKENS": workload.get("initial_tokens"),
        "TIER_POLICY_PRIME_TOKENS": workload.get("prime_tokens"),
        "TIER_POLICY_TOOL_WORDS": workload.get("tool_result_words"),
        "TIER_POLICY_DECODE_TOKENS": workload.get("decode_tokens"),
        "TIER_POLICY_WAIT_MS": workload.get("tool_wait_ms"),
        "TIER_POLICY_BURST_WINDOW_MS": workload.get("burst_window_ms"),
        "TIER_POLICY_SPREAD_WINDOW_MS": workload.get("spread_window_ms"),
        "TIER_POLICY_RESIDENT_GPU_TOKENS": workload.get("resident_gpu_tokens"),
        "TIER_POLICY_RESIDENT_HOST_GB": workload.get("resident_host_cache_gb"),
        "TIER_POLICY_RESIDENT_MAX_ACTIVE": workload.get("resident_max_active"),
        "TIER_POLICY_RESTRICTED_GPU_TOKENS": workload.get("restricted_gpu_tokens"),
        "TIER_POLICY_HOST_GB": workload.get("host_cache_gb"),
        "TIER_POLICY_STORAGE_HOST_GB": workload.get("storage_host_cache_gb"),
        "TIER_POLICY_RESTRICTED_MAX_ACTIVE": workload.get("restricted_max_active"),
    }
    prefix = " \\\n".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items() if value is not None
    )
    return prefix + " \\\nbash infra/container/run_work_audit_tier_policy_matrix.sh"
