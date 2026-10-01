# Agentic Work Audit

This fourth research lane asks whether cache and GPU work occurs at a useful
time in an agentic tool-wait/replay lifecycle. It is separate from the hint
benchmark, controller comparison, and hardware bottleneck lanes.

## Boundaries

- `packages/agentic-work-audit` owns backend-neutral events and conservative
  analysis. It must not import SGLang, a harness SDK, or controller policy.
- `packages/agentic-backend-sglang/.../audit_v0510.py` translates the pinned
  backend's trace into neutral events. A new backend version needs its own
  translator and tests; unknown hooks must fail the gate.
- `packages/agentic-experiments/.../run_work_audit_validation.py` runs the
  small validation. `packages/agentic-reports/.../build_work_audit_report.py`
  builds the top-level `WORK_AUDIT.html` index from saved summaries.
- The imported `hicache_audit.zip` is reference material, not an installed
  plugin. Its proposed UnifiedRadixCache hooks are not assumed to exist in
  SGLang 0.5.10.post1.

## First Live Validation

Use the accessible A10G host with the existing pinned SGLang 0.5.10.post1
container and `v0510` adapter. Set `SGLANG_DOCKER_IMAGE` and
`AGENTIC_MODEL_CACHE` for that host, then run:

```bash
bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The launcher refuses a different backend version or adapter and checks hook
installation. It runs two sequential sessions: a warm-prefix control and a
host-backed prefix. In the host case, it explicitly evicts the device copy,
proves a host copy exists, requests native load-back, and joins load completion
and layer-copy evidence to the same session. Both cases contain an initial
model call, a 500 ms tool wait, and a replay. No frontend priority is assigned.
This is a controlled probe, not a production workload or a performance
comparison. A completed load before replay does **not** prove replay stalled
on that load.

The run stores `harness_events.jsonl`, `backend_trace.jsonl`,
`normalized_events.jsonl`, `summary.json`, `run_manifest.json`, and logs under
`sglang_direct_kv/artifacts/results/work_audit/<run_id>/`. If any required
identity, hook, or native-copy proof is missing, the analyzer exits nonzero.
Four opportunity measures remain explicitly unknown: backup reuse, avoidable
eviction, HBM occupancy, and idle stageable work. Those require further
block-identity, residency, and policy-constrained counterfactual evidence.

The first validated reference is
[`work_audit_a10g_20261001_final`](../reports/work_audit/work_audit_a10g_20261001_final/summary.json).
It observed a warm replay and a separate, explicitly host-backed replay with
no frontend priority. This validates the instrumentation path, not which
condition is faster.

## Next Phases

1. Validate trace identity and timing on the pinned host, and inspect raw
   evidence before accepting a summary.
2. Add block-identity and residency ledgers; count backup creation, re-use,
   eviction, and native load by session without equating a match to a hit.
3. Add a controlled policy comparison with equal work, arrival times, and
   backend settings. Report measured replay latency and work, alongside
   separate opportunity estimates with uncertainty and constraints.
4. Translate and validate hooks for another SGLang release only after the
   new adapter reproduces the same event contract and fails closed on drift.
