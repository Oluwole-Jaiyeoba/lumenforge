# Agentic Work Audit

This fourth research lane asks whether cache and GPU work occurs at a useful
time in an agentic tool-wait/replay lifecycle. It is separate from the hint
benchmark, controller comparison, and hardware bottleneck lanes.

## Boundaries

- `packages/agentic-work-audit` owns backend-neutral events and conservative
  analysis. It must not import SGLang, a harness SDK, or controller policy.
- `packages/agentic-backend-sglang/.../audit_v0510.py` translates the pinned
  backend's trace into neutral events and labels its native lifecycle hooks.
  The block-lifecycle view reuses `agentic_reports.block_ledger` rather than
  installing another SGLang plugin. A new backend version needs its own
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
`normalized_events.jsonl`, `summary.json`, `block_audit.json`,
`run_manifest.json`, and logs under
`sglang_direct_kv/artifacts/results/work_audit/<run_id>/`. If any required
identity, hook, or native-copy proof is missing, the analyzer exits nonzero.
The block audit fails closed if the selected host node, semantic load,
and per-layer copy signatures cannot be linked. A semantic load, its nested
load callback, and its per-layer copies are counted separately. Replay
prefix matches are limited to the request's submit-to-first-token interval;
they do not prove the exact loaded block was consumed. Backup reuse,
avoidable eviction, HBM occupancy, and idle stageable work remain unknown
without stronger residency and counterfactual evidence.

The first validated reference is
[`work_audit_a10g_20261001_final`](../reports/work_audit/work_audit_a10g_20261001_final/summary.json).
It observed a warm replay and a separate, explicitly host-backed replay with
no frontend priority. This validates the instrumentation path, not which
condition is faster.

To regenerate the lifecycle audit from a saved run without launching a model:

```bash
RUN_ID=work_audit_a10g_20261001_final
RUN="sglang_direct_kv/artifacts/results/work_audit/$RUN_ID"
PYTHONPATH="$(printf '%s:' packages/*/src)" python3 -m agentic_reports.audits.build_work_audit_block_audit \
  --trace "$RUN/backend_trace.jsonl" \
  --harness "$RUN/harness_events.jsonl" \
  --summary "$RUN/summary.json" \
  --out "$RUN/block_audit.json"
```

The first validated host-backed run has one semantic load of 4096 tokens,
one nested load observation, and 28 linked per-layer copies. The selected
host-resident node accounts for 2048 tokens; the larger native load total
must not be described as 28 separate loads or as proof of same-block replay
use. This is a lifecycle account, not a performance result.

The shared-instrumentation replay of that saved trace is stored separately in
[`instrumentation_analysis.json`](../reports/work_audit/work_audit_a10g_20261001_final/instrumentation_analysis.json),
so the original validation artifact is unchanged. It found 2046 loaded GPU
slots in the replay's prefix match after SGLang split the original node.
That is evidence of matched slot lineage, not direct evidence that model
kernels consumed those exact slots. See [`INSTRUMENTATION.md`](../../INSTRUMENTATION.md)
for the proof ladder and hook ownership rules.

## Two-Replay Lifecycle Audit

The opt-in lifecycle case adds a second 500 ms tool wait and replay to each
session. The reference pair above used `kv_lifecycle`. For new comparisons use
`WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean`: it keeps the cache hooks and a
control-only scheduler pump while disabling high-volume scheduler trace and
copy-detail telemetry. Do not use `full_debug` for latency comparison. Both sessions
remain equal importance, with no frontend priority. On a host with the pinned
image and model cache configured, run the two case orders separately:

```bash
WORK_AUDIT_SECOND_REPLAY=1 WORK_AUDIT_CASE_ORDER=warm-host \
  WORK_AUDIT_RUN_ID=work_audit_two_replays_20261002 \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
WORK_AUDIT_SECOND_REPLAY=1 WORK_AUDIT_CASE_ORDER=host-warm \
  WORK_AUDIT_RUN_ID=work_audit_two_replays_20261002_reverse \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The first A10G run passed both evidence gates. Warm replay TTFTs were 220.942
and 223.312 ms; host-backed replay TTFTs were 229.089 and 218.525 ms. Its one
planned load moved 4096 tokens before replay; 2046 loaded GPU slots appeared
in the first replay's matched prefix. The reversed-order run also passed after
the ledger classified a **separate** 41-token load during the first replay:
warm TTFTs were 217.073 and 215.682 ms, and host-backed TTFTs were 573.944
and 217.552 ms. The second replay had a cache match in both sessions, but the
exact originally loaded slots were not linked to that second match. The 41-token
replay-time load is observed work; its causal contribution to the 573.944 ms
TTFT is not established by these two runs.

Compact evidence, run manifests, and compressed raw traces are archived in
[`work_audit_two_replays_20261002`](../reports/work_audit/work_audit_two_replays_20261002/summary.json)
and [`work_audit_two_replays_20261002_reverse`](../reports/work_audit/work_audit_two_replays_20261002_reverse/summary.json).
These runs validate a lifecycle ledger, not a performance benefit. A stronger
speed claim needs more alternating-order pairs and workload-matched overhead checks.

The lean profile passed the same installed-hook and live lifecycle gates in two
more A10G runs. Set `WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean` with the two
commands above, using fresh run IDs. The warm-first
[`lean run`](../reports/work_audit/work_audit_two_replays_lean_pump_20261002/summary.json)
had warm/host first-replay TTFTs of 81.517/82.560 ms and second-replay TTFTs
of 81.419/81.372 ms. It observed one planned 4096-token load and no replay-time
load. The host-first
[`lean reverse run`](../reports/work_audit/work_audit_two_replays_lean_reverse_20261002/summary.json)
had first-replay TTFTs of 82.300/436.253 ms and second-replay TTFTs of
80.926/87.126 ms; it observed a separate 43-token load during the host-backed
first replay. Both runs linked 2046 or more loaded GPU slots to that first
replay's matched prefix, but neither proved exact model-kernel consumption or
same-slot reuse on replay two. They remain exploratory, not a causal timing
comparison.

The preliminary tracing-off/on calibration on the same A10G measured median
latency of 2149.7/10455.3 ms with `kv_lifecycle` (+386.36%), versus
2153.0/2199.4 ms with `kv_lifecycle_lean` and its control-only pump (+2.15%).
This calibration used a smaller 173-input-token, 64-output-token request and
only one off-then-on pair, so it does not precisely correct the long-prefix
audit TTFTs. The saved [legacy profile](../reports/instrumentation_overhead_kv_lifecycle_20261002.json)
and [lean pump](../reports/instrumentation_overhead_kv_lifecycle_lean_pump_20261002.json)
measurements contain every trial.

## Next Phases

1. Validate trace identity and timing on the pinned host, and inspect raw
   evidence before accepting a summary.
2. Reuse the existing logical-block ledger for session-linked writes,
   evictions, and native loads. The pinned trace supports this now; exact
   physical-block reuse and a complete residency timeline remain future work.
3. Add a controlled policy comparison with equal work, arrival times, and
   backend settings. Report measured replay latency and work, alongside
   separate opportunity estimates with uncertainty and constraints.
4. Translate and validate hooks for another SGLang release only after the
   new adapter reproduces the same event contract and fails closed on drift.
