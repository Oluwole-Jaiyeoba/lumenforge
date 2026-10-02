# KV Lifecycle Audit

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
  builds the top-level `KV_LIFECYCLE_AUDIT.html` index from saved summaries.
  It keeps one row per archived run, ordered by the first request's UTC date
  and time. Expand a row for setup, pair-level measurements, limits, a
  reconstructed command, and immutable evidence links. If the request time is
  missing, the page labels a manifest completion-time fallback rather than
  presenting it as the experiment start.
  New run manifests also record the effective prompt target, output cap,
  minimum host prefix, eviction attempts, host-cache size, and GPU memory
  fraction. Archived runs lacking these fields leave them unrecorded.
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

## Paired KV Timing Study

The timing study holds both sessions at equal importance. Each case primes a
long prefix, proves that its GPU copy was evicted while its host copy remains,
and then starts a 2-second tool wait. `early` requests and confirms a native
load during that wait. `late` requests the same kind of load after the tool
finishes and submits replay as soon as the command is accepted. Both cases
also have a second tool wait and replay. They use the same model and prompt
shape, though their session-specific text and GPU state may differ. One
discarded early/late pair warms the backend before measured pairs.

```bash
WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_CASE_ORDER=early-late WORK_AUDIT_WARMUP_PAIRS=1 \
  WORK_AUDIT_PAIRS=2 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=4096 \
  WORK_AUDIT_REQUIRE_SLOT_PROOF=1 \
  WORK_AUDIT_RUN_ID=work_audit_timing_early_late \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_CASE_ORDER=late-early WORK_AUDIT_WARMUP_PAIRS=1 \
  WORK_AUDIT_PAIRS=2 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=4096 \
  WORK_AUDIT_REQUIRE_SLOT_PROOF=1 \
  WORK_AUDIT_RUN_ID=work_audit_timing_late_early \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The primary latency is **tool completion to first replay token**. It includes
delay between tool completion and request submission; ordinary replay TTFT
does not. The summary also records native loaded tokens, when a completed
CUDA load was observed, task duration, second-replay TTFT, and exact loaded
GPU slots appearing in the first replay's prefix match. The backend reports
CUDA completion when polled, so a post-due observation is an upper bound on
the finish time, not proof that the transfer continued past the due point.
`comparable=false` suppresses a pair delta when evidence or initial-request
latency drifts. A late load is a timing *candidate*, not automatically a
mistake. Repeated order-balanced pairs, matched instrumentation overhead,
and competing-work costs are needed before calling it avoidable harm.

The normal lean profile samples large tensor indices. This small study raises
the exact-index limit only so the cache-slot lineage can be checked; it does
not enable broad scheduler or full-debug tracing. Measure its cost on the
long-prefix request shape before using absolute TTFT differences:

```bash
python3 scripts/measure_instrumentation_overhead.py \
  --image "$SGLANG_DOCKER_IMAGE" --model-cache "$AGENTIC_MODEL_CACHE" \
  --profile kv_lifecycle_lean --control-only-pump \
  --workload audit_long_prefix --prompt-tokens 4090 --max-tokens 16 \
  --wait-ms 2000 --max-exact-indices 4096 --repetitions 2 --order off-on \
  --mem-fraction-static 0.70 \
  --out-dir sglang_direct_kv/artifacts/results/work_audit_overhead/off_on
```

Repeat with `--order on-off` and a different output directory. This
calibration matches request size and two tool waits but does **not** perform
native host-KV eviction/load. Its measured latency excludes the fixed tool
waits. It cannot by itself bound tracing overhead on the explicit load path.
For lower-overhead timing runs, omit `WORK_AUDIT_EXACT_INDICES` and
`WORK_AUDIT_REQUIRE_SLOT_PROOF`. That keeps the default 256-index sample and
reports exact slot lineage as unknown, while still requiring session-linked
native CUDA completion, layer copies, and replay prefix matches. Do not
combine the sampled timing and exact-lineage runs as if they had identical
instrumentation overhead.

### A10G Timing Reference (2026-10-02)

Four order-balanced, warmed runs used the pinned `0.5.10.post1` container,
`v0510` adapter, 4090-word input shape, 16 output tokens, a 2-second first
tool wait, and two replays per case. All eight measured pairs passed the
first-replay evidence and initial-latency comparability gates. Both sessions
had equal importance; there were no frontend priority hints or competing
fillers. The first pair in each sampled run had a large second-replay TTFT
outlier, so its **full-task** delta is withheld even though its first-replay
delta remains reportable.

| Capture | Case order | Late minus early, tool-return to first token (ms) | Main source of difference |
| --- | --- | --- | --- |
| 4096 exact indices | early, late | +226.759, +236.500 | replay submission gap |
| 4096 exact indices | late, early | +228.737, +228.620 | replay submission gap |
| 256 sampled indices | early, late | +166.419, +178.118 | replay submission gap |
| 256 sampled indices | late, early | +170.088, +173.832 | replay submission gap |

In early cases, the controller requested and confirmed the native load during
the tool wait. In late cases, it requested the load only after the tool
returned, and the control call delayed replay submission by about 164–177 ms
with sampled tracing. Once replay was submitted, its TTFT was similar in both
conditions (sampled late-minus-early differences: -4.178, +0.988, +6.364,
+1.014 ms). Each measured case loaded 4096 tokens. Exact-index runs linked
2047–2048 loaded GPU slots to the first replay's matched prefix; sampled
runs report that slot lineage as **unknown**, not zero reuse. [The run
index](../../KV_LIFECYCLE_AUDIT.html) keeps each experiment in one row and
shows paired measurements and raw evidence when expanded.

The exact-index runs measured about 221–236 ms of load-control-call time and
about 205–216 ms of native CUDA load time, whereas sampled runs measured
about 164–185 ms and 151–160 ms. That capture setting changes the measured
path, so the exact-index and sampled numbers are separate views, not pooled
replicates. A matched three-request/two-wait **request-path-only** calibration
found `kv_lifecycle_lean` tracing added +4.40% and +4.61% in off/on and on/off
order at 4096 exact indices, and +4.12% and +4.36% at 256 sampled indices.
The [four calibration records](../reports/work_audit_overhead/) retain every
trial. They do **not** measure tracing overhead on native host-KV load-back.

This is evidence of a badly timed *controller command* in this serial replay
path: issuing load after tool return adds a submission gap. It is not proof
that SGLang inherently must block, that the model consumed those exact slots,
or that GPU memory-bandwidth interference caused the delay. A nonblocking
submission path and concurrent filler requests are separate future tests.

## Next Phases

1. Test a nonblocking late-load submission path to see how much of the
   observed delay is imposed by the current control-call ordering.
2. Add concurrent, equally important sessions to measure whether preparing
   one replay early delays other work or simply uses otherwise idle time.
3. Extend the logical-block ledger toward exact physical-block reuse and a
   complete residency timeline; matched GPU slots alone do not prove model
   consumption.
4. Translate and validate hooks for another SGLang release only after the
   new adapter reproduces the same event contract and fails closed on drift.
