"""Manager-facing tables backed by frozen independent-controller results."""
from __future__ import annotations

from statistics import median

from agentic_backends.controller_audit import HOST_CACHE_ENV, TESTBED_DIRECTORY

HEADERS = (
    "Trial / mode", "Whole workload (s)", "All requests: total TTFT (s)",
    "Replays: total TTFT (s)", "Replays: total lateness (s)",
    "Replay TTFT median (ms)", "Replay delay median / p95 (ms)", "Requests / replays", "Evidence gate",
)
PAIR_HEADERS = ("Trial", "Whole workload: baseline → controller", "Change",
                "Total replay lateness: baseline → controller", "Change", "Comparison")
EXPOSURE_HEADERS = ("Trial / mode", "Ranked replays", "Accepted prepare loads",
                    "Ready before due / rejected prepare calls", "GPU eviction calls",
                    "Evicted token slots", "Admission decisions / holds")
NOTES = (
    "Lower is better. Total TTFT/lateness sum across requests and can exceed the "
    "whole-workload clock because sessions overlap. Whole-workload time includes initial "
    "requests and tool waits, but excludes backend startup and preflight. Each trial "
    "uses its own seed; trial 2 reverses arm order. All sessions have equal importance. "
    "Compare policies within a scenario: prompt sizes and tool-wait distributions differ across scenarios. "
    "CUDA graphs and overlap scheduling are on. Good/bad short-filler runtime admits "
    "are not classified by these tests; this is unavailable evidence, not zero bad admits. "
    "Admission/hold counts are shown for RTG; other modes do not use that admission policy. "
    "Ready before due means the prepare call reported completion before the replay deadline; "
    "it does not prove that the prefix remained resident or was used when replay arrived. "
    "Raw decisions remain available in the traces."
)


def number(value, divisor=1, places=3):
    return "unavailable" if value is None else f"{value / divisor:.{places}f}"


def table_rows(summary: dict) -> list[tuple[str, ...]]:
    rows = []
    for arm in summary.get("arms", []):
        m = arm["metrics"]
        label = "Baseline" if arm["mode"] == "no_prefetch" else summary["name"]
        rows.append((
            f"{arm['trial']} / {label}", number(m["workload_ms"], 1000),
            number(m["total_all_ttft_ms"], 1000), number(m["total_replay_ttft_ms"], 1000),
            number(m["total_replay_lateness_ms"], 1000),
            number(m.get("median_replay_ttft_ms"), places=1),
            number(m["median_replay_delay_ms"], places=1) + " / " + number(m["p95_replay_delay_ms"], places=1),
            f"{m['requests']} / {m['replays']}", "; ".join(arm["issues"]) or "passed",
        ))
    return rows


def pair_rows(summary: dict) -> list[tuple[str, ...]]:
    rows = []
    for pair in summary.get("pairs", []):
        work = pair["metrics"]["workload_ms"]
        late = pair["metrics"]["total_replay_lateness_ms"]
        rows.append((str(pair["trial"]),
                     f"{number(work['baseline'], 1000)} → {number(work['controller'], 1000)} s",
                     number(work["change_pct"], places=1) + "%" if pair["comparable"] else "withheld",
                     f"{number(late['baseline'], 1000)} → {number(late['controller'], 1000)} s",
                     number(late["change_pct"], places=1) + "%" if pair["comparable"] else "withheld",
                     "; ".join(pair["issues"]) or "matched"))
    return rows


def exposure_rows(summary: dict) -> list[tuple[str, ...]]:
    rows = []
    for arm in summary.get("arms", []):
        exposure = arm.get("exposure") or {}
        decisions = exposure.get("admission_decisions")
        rows.append((f"{arm['trial']} / {'Baseline' if arm['mode'] == 'no_prefetch' else summary['name']}",
                     *(str(exposure.get(key, "unavailable")) for key in
                       ("ranked_replays", "accepted_prepare_loads")),
                     f"{exposure.get('prepare_ready_before_due', 'unavailable')} / {exposure.get('rejected_prepares', 'unavailable')}",
                     *(str(exposure.get(key, "unavailable")) for key in
                       ("device_eviction_calls", "device_evicted_token_slots")),
                     f"{decisions} / {exposure['hold_decisions']}" if decisions is not None else "not applicable"))
    return rows


def finding(summary: dict) -> str:
    valid = [p for p in summary.get("pairs", []) if p["comparable"]]
    if len(valid) != 2:
        return "Comparison not established: workload/evidence checks did not pass for both trials."
    pieces = []
    for key, label in (("workload_ms", "Whole-workload time"), ("total_replay_lateness_ms", "Total replay lateness")):
        before = median(p["metrics"][key]["baseline"] for p in valid) / 1000
        after = median(p["metrics"][key]["controller"] for p in valid) / 1000
        change = 100 * (after / before - 1) if before else 0
        pieces.append(f"{label}: {before:.3f} → {after:.3f} s ({change:+.1f}%).")
    return " ".join(pieces) + " Medians of two trials; see each trial below."


def setup(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    spec = manifest.get("spec") or {}
    scenario = (spec.get("scenarios") or {}).get(summary["scenario"], {})
    env = {**spec.get("common_env", {}), **scenario.get("env", {})}
    return (
        f"16 equal-importance sessions, 2 tool returns each; {env.get('P3_HIGH_KNOBS', 'see frozen spec')}. "
        f"Tool-wait distribution: {env.get('TOOL_WAIT_PROFILE_SPEC', 'see spec')}. "
        f"Seeds: {spec.get('seeds', [])}; opposite policy order in trial 2. "
        f"Model: {manifest.get('model', 'see manifest')}; GPU KV limit: {env.get('MAX_TOTAL_TOKENS', '?')} tokens; "
        f"host cache: {env.get(HOST_CACHE_ENV, '?')} GiB; storage off. "
        "Main output limit 8 tokens, peer limit 2. CUDA graphs and overlap scheduling on. "
        f"Tracing: {env.get('TRACE_PROFILE', '?')}, count-only indices. "
        + summary.get("matching", "") + " " + summary.get("limits", "")
    )


def reproduction(summary: dict) -> str:
    manifest = summary.get("_manifest") or {}
    return (
        f"# Experiment source commit: {summary.get('source_revision', 'see manifest')}\n"
        "# Use a separate checkout at this commit; keep the archived evidence untouched.\n"
        f"source {TESTBED_DIRECTORY}/.venv/bin/activate\n"
        'export PYTHONPATH="$(printf \'%s:\' "$PWD"/packages/*/src)"\n'
        "python -m agentic_experiments.runners.run_controller_audit_pivots \\\n"
        f"  --run-id {summary['run_id']}_repeat_$(date +%Y%m%d_%H%M%S) \\\n"
        f"  --scenarios {summary['scenario']} \\\n"
        "  --spec configs/experiment_specs/controller_audit_pivots.json \\\n"
        '  --model-cache "$HOME/.cache/huggingface" \\\n'
        f"  --image {manifest.get('image', 'SEE_RECORDED_IMAGE')}\n"
        "# Exact image ID/digest, model snapshot, dependency lists and arm environments:\n"
        "# run_manifest.json, model_identity.json, *dependencies.txt in the evidence directory.\n"
        "# Source-synced host without git: append --source-archive PATH/source.tar.gz\n"
        f"# and --source-revision {summary.get('source_revision', 'FULL_COMMIT_SHA')}.\n"
        f"# Reporting source commit: {summary.get('analysis_revision', 'see summary.json')}.\n"
        "# Publish from that separate reporting checkout, using --run-root /absolute/path/to/new/run.\n"
        "# See docs/work_audit/CONTROLLER_PIVOT_REPRODUCTION.md for publication commands."
    )


def evidence_files(summary: dict) -> list[tuple[str, str]]:
    files = [(name, label) for name, label in (
        ("summary.json", "Summary and per-arm measurements"), ("run_manifest.json", "Exact commands and runtime"),
        ("experiment_spec.json", "Frozen workload settings"), ("source.tar.gz", "Source archive"),
        ("model_identity.json", "Model snapshot and hashes"), ("host_dependencies.txt", "Host dependencies"),
        ("container_dependencies.txt", "Container dependencies"), ("evidence_sha256.json", "Evidence hashes"),
        ("analysis_source.py", "Analyzer used for these results"),
    )]
    for arm in summary.get("arms", []):
        files.append((f"{arm['label']}.log", f"Trial {arm['trial']} {arm['mode']}: launcher log"))
        for filename in ("server_info.json", "live_sentinel_report.json", "m27_trace.jsonl.gz", "backend_trace.jsonl.gz"):
            files.append((f"{arm['case_path']}/{filename}", f"Trial {arm['trial']} {arm['mode']}: {filename}"))
        files.append((f"{arm['report_path']}/global_kv_readiness_by_mode.csv", f"Trial {arm['trial']} {arm['mode']}: every replay"))
    return files
