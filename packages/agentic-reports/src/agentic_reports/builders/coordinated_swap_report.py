"""Presentation of the optimistic swap study, separate from its measurements."""

from statistics import median
import shlex

SCHEMA = "agentic_work_audit.coordinated_swap.summary.v1"
HEADERS = ("Trial / mode", "Whole workload (s)", "Due to first token mean (ms)",
           "Due to first token p95 (ms)", "Total TTFT (s)", "Total due delay (s)",
           "Submission waiting total (s)", "Scheduled slot padding sum (s)",
           "Explicit native load batches", "KV ready on time / late", "Minimum GPU prefix reuse")

COMPARISON_HEADERS = ("Trial / reference", "Whole-workload change", "Mean replay-delay change (ms)",
                      "Sessions finishing sooner / later")
SESSION_HEADERS = ("Trial / session", "Independent finish (s)", "Coordinated finish (s)",
                   "Resident finish (s)")
CONTROL_HEADERS = ("Trial / mode", "Excluded setup (s)", "Measured control calls",
                   "Total control wall time (s)", "Native CUDA-stream intervals / total (s)",
                   "Explicit restored / released KV tokens")
NOTES = ("Negative changes mean coordinated finished sooner or had less delay. Session finish times "
         "start at the measured workload start, not initial setup. TTFT, due delay, submission waiting "
         "and scheduled slot padding totals add time across requests; they are not elapsed workload time. "
         "Scheduled padding is the gap from each reply to its planned slot boundary, including the "
         "last slot's unused padding; whole-workload time stops at the actual last reply. "
         "Explicit load batches count prepare controls, not automatic loads in the independent mode. "
         "KV ready on time / late counts every coordinated replay in the session-pipeline setup; "
         "older barrier runs count only explicit restores. The modes deliberately use different "
         "residency policies; this is an optimistic comparison, not a production fairness test.")
CONTROL_NOTES = ("Control wall time includes waiting for the backend and checking its reply. "
                 "The CUDA-stream interval can include gaps between launching copies; it is not "
                 "a measurement of copy-engine busy time alone. These intervals can overlap control "
                 "wall time, so do not add them together. Initial priming and diagnostics are reported "
                 "as excluded setup, not hidden inside the workload duration.")


def control_rows(summary):
    rows = []
    for arm in summary.get("arms", []):
        wall = arm.get("control_wall_ms_by_action")
        counts = arm.get("control_count_by_action")
        cuda = arm.get("native_cuda_interval_total_ms")
        rows.append((f"{arm['trial']} / {arm['mode']}", f"{arm['setup_ms'] / 1000:.3f}",
                     str(sum(counts.values())) if counts is not None else "unavailable",
                     f"{sum(wall.values()) / 1000:.3f}" if wall is not None else "unavailable",
                     f"{arm['native_cuda_interval_count']} / {cuda / 1000:.3f}" if cuda is not None else "not measured",
                     f"{arm['measured_loaded_tokens']:,} / {arm['measured_released_tokens']:,}"))
    return rows


def comparison_rows(summary):
    return [(f"{c['trial']} / {c['reference']}", f"{c['workload_change_pct']:+.1f}%",
             f"{c['due_to_first_token_change_ms']:+.1f}",
             f"{c.get('sessions_finished_sooner', 'unavailable')} / {c.get('sessions_finished_later', 'unavailable')}")
            for c in summary.get("comparisons", [])]


def session_rows(summary):
    rows = []
    for trial in sorted({a["trial"] for a in summary.get("arms", [])}):
        modes = {a["mode"]: a.get("per_session_completion_ms", {})
                 for a in summary["arms"] if a["trial"] == trial}
        sessions = sorted({s for values in modes.values() for s in values})
        for session in sessions:
            values = [modes.get(mode, {}).get(session) for mode in ("independent", "coordinated", "resident")]
            rows.append((f"{trial} / {session}", *(f"{v / 1000:.3f}" if v is not None else "unavailable" for v in values)))
    return rows


def table_rows(summary):
    return [(f"{a['trial']} / {a['mode']}", f"{a['workload_ms'] / 1000:.3f}",
             f"{a['mean_due_to_first_token_ms']:.1f}", f"{a['p95_due_to_first_token_ms']:.1f}",
             f"{a['total_ttft_ms'] / 1000:.3f}", f"{a['total_due_to_first_token_ms'] / 1000:.3f}",
             f"{a['total_submission_delay_ms'] / 1000:.3f}", f"{a['total_alignment_padding_ms'] / 1000:.3f}",
             str(a['measured_load_count']),
             f"{a.get('kv_ready_before_due', a['restores_before_due'])} / "
             f"{a.get('kv_ready_after_due', a['restores_after_due'])}",
             f"{100 * a['minimum_gpu_prefix_fraction']:.1f}%" if a['minimum_gpu_prefix_fraction'] is not None else "unavailable")
            for a in summary.get("arms", [])]


def setup(summary):
    config = summary.get("configuration") or next((a["config"] for a in summary.get("arms", [])), {})
    manifest = summary.get("_manifest") or {}
    runtime = manifest.get("backend_runtime_contract") or {}
    schedule = config.get("schedule_style", "barrier")
    timing = ("individual session clocks; no group barrier; "
              f"{config.get('prefetch_lead_ms', 750)} ms prefetch lead; "
              f"{config.get('headroom_sessions', 2)} session headroom; "
              f"at most {config.get('max_inflight', 8)} requests in flight"
              if schedule == "session_pipeline" else "paired AB/CD group rotation")
    return (f"20 separate sessions; 4 labels of 5; {config.get('turns', '?')} tool rounds; "
            f"1,000 ms waits; {timing}; restricted GPU/8 GiB host KV; "
            f"{config.get('initial_tokens', 8192)} initial prompt words and "
            f"{config.get('tool_words', 16)} new tool words per round; "
            f"GPU token caps {config.get('restricted_gpu_tokens', 110592)} restricted / "
            f"{config.get('resident_gpu_tokens', 262144)} resident; "
            f"{config.get('decode_tokens', 32)} output tokens per replay; "
            f"native {config.get('io_backend', 'direct')} KV transfer; "
            f"{config.get('restore_style', 'serial')} restore submission; "
            f"{config.get('control_style', 'individual')} control calls; "
            f"{config.get('trace_profile', 'kv_lifecycle_lean')} tracing; "
            "no storage; CUDA graphs and overlap scheduling on. "
            f"Model: {config.get('model', 'not recorded')}; hardware: {manifest.get('hardware_profile', 'see manifest')}; "
            f"backend version: {runtime.get('backend_version', 'see runtime record')}. "
            "Initial setup is excluded. The restricted KV budget is imposed on the same GPU, "
            "not a claim that all its physical memory was exhausted.")


def finding(summary):
    if summary.get("status") != "complete":
        return "Evidence incomplete; no performance conclusion. " + "; ".join(summary.get("issues", [])[:2])
    pieces = []
    for reference in ("independent", "resident"):
        values = [c["workload_change_pct"] for c in summary.get("comparisons", []) if c["reference"] == reference]
        if values:
            value = median(values)
            pieces.append(f"Coordinated workload was {abs(value):.1f}% {'shorter' if value < 0 else 'longer'} "
                          f"than {reference} (median paired change).")
    late = sum(a.get("kv_ready_after_due", a["restores_after_due"])
               for a in summary.get("arms", []) if a["mode"] == "coordinated")
    pieces.append(f"{late} coordinated replays had KV become ready after their tool deadline.")
    return " ".join(pieces)


def reproduction(summary):
    config = summary.get("configuration") or next((a["config"] for a in summary.get("arms", [])), {})
    modes = list(dict.fromkeys(a["mode"] for a in summary.get("arms", [])))
    trials = sorted({a["trial"] for a in summary.get("arms", [])})
    values = {"SWAP_RUN_ID": summary.get("run_id", "coordinated_swap") + "_repeat",
              "SWAP_TURNS": config.get("turns", 40), "SWAP_DECODE_TOKENS": config.get("decode_tokens", 32),
              "SWAP_INITIAL_TOKENS": config.get("initial_tokens", 8192),
              "SWAP_GPU_TOKENS": config.get("restricted_gpu_tokens", 110592),
              "SWAP_RESIDENT_TOKENS": config.get("resident_gpu_tokens", 262144),
              "SWAP_TOOL_WORDS": config.get("tool_words", 16),
              "SWAP_IO_BACKEND": config.get("io_backend", "direct"),
              "SWAP_RESTORE_STYLE": config.get("restore_style", "serial"),
              "SWAP_CONTROL_STYLE": config.get("control_style", "individual"),
              "SWAP_SCHEDULE_STYLE": config.get("schedule_style", "barrier"),
              "SWAP_PREFETCH_LEAD_MS": config.get("prefetch_lead_ms", 750),
              "SWAP_HEADROOM_SESSIONS": config.get("headroom_sessions", 2),
              "SWAP_MAX_INFLIGHT": config.get("max_inflight", 8),
              "SWAP_TRACE_PROFILE": config.get("trace_profile", "kv_lifecycle_lean"),
              "SWAP_TRIALS": " ".join(map(str, trials or [1])),
              "SWAP_MODES": " ".join(modes or ["independent", "coordinated", "resident"])}
    return " \\\n".join(f"{k}={shlex.quote(str(v))}" for k, v in values.items()) + " \\\nbash infra/container/run_work_audit_coordinated_swap.sh"
