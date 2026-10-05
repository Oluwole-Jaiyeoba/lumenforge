# KV Lifecycle Audit

This fourth research lane asks whether cache and GPU work occurs at a useful
time in an agentic tool-wait/replay lifecycle. It is separate from the hint
benchmark, controller comparison, and hardware bottleneck lanes.
Its guiding question is: given what the harness knew at the time, was the GPU
or memory system doing the wrong work at the wrong time? The top-level report
tracks five ledgers: host backups, GPU evictions, session resumes, HBM
occupancy, and GPU time. Current live evidence covers only part of the resume
timing ledger; the other ledgers must not be called wasted or avoidable without
linked evidence and a feasible comparison.

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
- `docs/work_audit/research_progress.json` records answered research questions,
  supported conclusions, remaining uncertainty, and the archived runs behind
  each conclusion. Each question has a stable `id` and short table label.
  `related_run_ids` maps older runs to the question they tested, while
  `evidence_run_ids` lists only runs supporting the stated answer. New run
  manifests record `workload.research_question_id`; the report rejects a
  conflicting archived mapping and leaves unknown IDs visibly unmapped.
  Set `WORK_AUDIT_RESEARCH_QUESTION_ID` when a new protocol studies a
  different question from its study's default. Prepend new milestones after
  checking their evidence; do
  not rewrite older conclusions without explicitly correcting them. Regenerate
  the top-level page from the repo root with:

```bash
PYTHONPATH="$(printf '%s:' packages/*/src)" python3 -m agentic_reports.builders.build_work_audit_report \
  --results-dir docs/reports/work_audit \
  --progress-file docs/work_audit/research_progress.json \
  --out KV_LIFECYCLE_AUDIT.html
```
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

In early cases, the audit client requested and confirmed the native load during
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

This is evidence of a potentially badly timed *load command* in this serial replay
path: issuing load after tool return adds a submission gap. It is not proof
that SGLang inherently must block, that the model consumed those exact slots,
or that GPU memory-bandwidth interference caused the delay.

## Nonblocking Late-Load Control

The next control used the same lean trace and one warmed, measured triple:
early load, late load that waits for the control response, and late load whose
replay is submitted without waiting for that response. All requests had equal
importance. Run it on the pinned host with its image and model cache set:

```bash
WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_CASE_ORDER=early-late-late_nonblocking \
  WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 \
  WORK_AUDIT_EXACT_INDICES=256 WORK_AUDIT_REQUIRE_SLOT_PROOF=0 \
  WORK_AUDIT_RUN_ID=work_audit_nonblocking_20261002_01 \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The [saved A10G run](../reports/work_audit/work_audit_nonblocking_20261002_01/summary.json)
observed these milliseconds from tool return:

| Condition | Replay submitted | First token | Replay TTFT |
| --- | ---: | ---: | ---: |
| Early | 0.117 | 88.159 | 88.042 |
| Late, response-gated | 170.450 | 253.339 | 82.890 |
| Late, nonblocking | 0.490 | 243.659 | 243.169 |

So the nonblocking call removed the client-side submission gap in this run,
but did **not** restore the early case's tool-return-to-token time. The delay
mostly appeared inside replay TTFT instead. The native load was accepted
*after* nonblocking replay submission, so the strict matched-reuse comparison
is withheld even though a request-linked cache match was observed. This is
one measured triple, not proof that all nonblocking scheduling behaves this
way. The second-replay TTFT drift also prevents a full-task comparison.

## Concurrent Session Timeline

The first hand-checkable overlap probe starts three equal-importance sessions
together. Two tool calls have different predicted return times; the third
session ends without replay. A test policy explicitly limits active GPU
prefixes to two and evicts the long-wait session's prefix to host memory.
This is **not** a natural allocator- or out-of-memory-triggered eviction. The
archived raw event says `resident_sessions=3`; that was the client scenario's
assumption, not a measurement of all three GPU prefixes. Only the long
session's device eviction and host residency were directly proved. New runs
record this as `candidate_sessions=3`. At
the long tool return, its host load is requested without gating replay on the
control response. The backend-neutral evidence gate checks the eviction,
host residency, native layer copy, load, replay, and second-replay prefix
match for that same session.

```bash
WORK_AUDIT_STUDY=multisession WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 \
  WORK_AUDIT_RUN_ID=work_audit_multisession_20261002_01 \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The [saved A10G timeline](../reports/work_audit/work_audit_multisession_20261002_01/summary.json)
passed its identity and ordering gates. The short tool returned after
901.497 ms and its replay first token followed 81.635 ms later. The long
tool returned after 2501.266 ms; its first token followed 282.426 ms later.
Its replay was submitted 0.431 ms after tool return; native load acceptance
and completion were observed at 177.544 and 179.370 ms after tool return.
Its second replay had a request-linked prefix match. These two latencies are from **different
sessions** and do not measure the causal effect of eviction. The observed
late load request is a plausible timing opportunity, not a proven mistake.
At the time of this run, avoidable work was unknown; RQ5 now supplies an
early-load comparison under the same logical capacity policy. Backend cache-match events are not a
GPU-utilization trace, and a matched prefix does not prove exact kernel use.

## Matched Concurrent Timing Comparison

RQ5 changes only when the long session's host-backed KV is loaded. Each case
starts three equally important sessions with 900 ms and 2500 ms tool waits.
The long prefix is evicted to host; the session that ends releases its device
prefix in **both** modes, preserving a logical two-prefix budget. In the
early mode the long prefix is reloaded 1200 ms after tool-wait start, while
the short replay can still be active. In late mode the load is launched at
the long tool return without waiting for its control response before replay.
One warmup pair is discarded, then two measured pairs run in alternating
order on the same pinned backend. The analysis rejects a measured pair if
its initial phases differ substantially.

```bash
WORK_AUDIT_STUDY=multisession_compare WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_CASE_ORDER=late_nonblocking-early WORK_AUDIT_PAIRS=2 \
  WORK_AUDIT_WARMUP_PAIRS=1 \
  WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 \
  WORK_AUDIT_EARLY_AT_MS=1200 \
  WORK_AUDIT_RUN_ID=work_audit_concurrent_compare_$(date +%Y%m%d_%H%M%S) \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The summary reports long replay due-to-first-token, short replay completion,
and total workflow time separately. A comparison is withheld if either
case lacks native load, host residency, released-slot, or replay cache-match
proof. The two-prefix rule is an explicit test policy, **not** proof of
physical GPU occupancy or natural memory pressure. Different cases use
equal-shape, session-specific prompts rather than byte-identical prefixes.

The [archived A10G comparison](../reports/work_audit/work_audit_concurrent_compare_20261002_02/summary.json)
discarded one warmup pair and validated both measured pairs. Early loading
reduced the long replay's tool-return-to-first-token time by 200.7 and
512.7 ms, and reduced whole-workflow completion by 211.9 and 502.8 ms.
The short session's completion was 178.2 and 194.5 ms slower, even though
its first-token timing changed little. In both early cases, the load overlapped
the short request's lifetime. This proves a controlled tradeoff, not whether
the short-session delay came from GPU bandwidth, backend scheduling, or both.
The result is too small and synthetic to establish a universal policy win.

## After-Short Load Window

RQ6 compares the same three-session, equal-importance workload with a third
load schedule. Late loading starts when the long tool returns; early loading
starts 1200 ms into that wait; **post-short** loading starts only after the
short replay actually finishes. The long tool wait is 2500 ms. If post-short
load-back does not complete before that return, the case fails instead of
quietly becoming a late load. The ended session releases its prefix in all
three modes. One warmup triplet is discarded; two measured triplets alternate
condition order. The pinned backend is SGLang `0.5.10.post1` with the lean KV
lifecycle trace.

```bash
SGLANG_DOCKER_IMAGE=agentic-sglang-standard:0.5.10.post1 \
  AGENTIC_MODEL_CACHE=/home/ec2-user/.cache/huggingface \
  WORK_AUDIT_STUDY=multisession_window \
  WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_CASE_ORDER=late_nonblocking-early-post_short \
  WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 \
  WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 \
  WORK_AUDIT_EARLY_AT_MS=1200 \
  WORK_AUDIT_RUN_ID=work_audit_post_short_20261002_01 \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The [archived A10G run](../reports/work_audit/work_audit_post_short_20261002_01/summary.json)
validated both measured triplets. Long tool-return-to-first-token time was
277.6/607.0 ms late, 82.4/81.8 ms early, and 89.4/81.4 ms post-short.
Short tool-return-to-completion time was 817.8/820.3 ms late,
1014.9/1040.4 ms early, and 811.2/831.8 ms post-short. Whole-workflow
durations were 8110.6/8463.7 ms late, 7901.8/7945.7 ms early, and
7902.5/7953.6 ms post-short. These are individual triplet results, not
averages across a production workload. The post-short trigger used the
**observed** completion of the short replay; a production controller would
need a reliable way to identify or predict such a window. The test does not
isolate GPU bandwidth from backend scheduling or demonstrate natural cache
capacity pressure.

## Controller-Chosen Load Window

RQ7 repeats the same three-session workload in four modes: late, early,
scripted post-short, and `controller_window`. The controller policy lives in
`packages/agentic-controller/src/agentic_controller/kv_prepare_window.py`.
The experiment runner feeds it observed short-replay completion, host
residency, ended-session slot release, and the harness's predicted long tool
return. It loads only if the remaining wait covers a fixed 250 ms load estimate
plus a 150 ms safety margin; otherwise it defers to the late, nonblocking path.
Every decision and outcome is recorded in `harness_events.jsonl` and checked
against backend load events. All tasks retain equal frontend importance.

```bash
SGLANG_DOCKER_IMAGE=agentic-sglang-standard:0.5.10.post1 \
  AGENTIC_MODEL_CACHE=/home/ec2-user/.cache/huggingface \
  WORK_AUDIT_STUDY=multisession_controller \
  WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean \
  WORK_AUDIT_CASE_ORDER=late_nonblocking-early-post_short-controller_window \
  WORK_AUDIT_PAIRS=4 WORK_AUDIT_WARMUP_PAIRS=1 \
  WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 \
  WORK_AUDIT_EARLY_AT_MS=1200 \
  WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 \
  WORK_AUDIT_RUN_ID=work_audit_controller_window_20261002_01 \
  bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The [archived A10G run](../reports/work_audit/work_audit_controller_window_20261002_01/summary.json)
passed all four measured comparisons. Median tool-return-to-first-token for
the long session was 351.7 ms late, 81.9 ms early, 82.0 ms scripted post-short,
and 85.6 ms controller-chosen. Median short return-to-finish was 864.8, 1105.3,
861.7, and 859.6 ms respectively. Median whole-workflow duration was 8.42,
8.04, 8.05, and 8.07 seconds respectively. The controller made four load
decisions with zero measured tool-return overshoots. These are small synthetic
samples; the policy is invoked by the audit runner, not yet deployed in the
production gateway, and the capacity eviction is explicit.

## Busy-Workload Comparison (RQ8)

`infra/container/run_work_audit_busy.sh` starts a fresh pinned backend for
each arm. Each seed runs 12 equally important sessions, three tool waits per
session, and growing 8192-token coding-task contexts. The baseline uses
ordinary SGLang cache handling. The controller arm checks host residency up
to three times per wait and requests native load-back when its 250 ms estimate
plus 150 ms margin fits before the expected tool return. It does not force
eviction or delay replay submission for the control response. Seed parity
reverses arm order. Both arms use `kv_lifecycle_lean` tracing.

```bash
SGLANG_DOCKER_IMAGE=agentic-sglang-standard:0.5.10.post1 \
  AGENTIC_MODEL_CACHE=/path/to/model/cache \
  WORK_AUDIT_RUN_ID=work_audit_busy_$(date +%Y%m%d_%H%M%S) \
  WORK_AUDIT_SEEDS="1 2" \
  bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The [archived two-seed comparison](../reports/work_audit/work_audit_busy_pair_20261002_01/summary.json)
includes each arm's metrics, harness timeline, hook gate, and compressed raw
trace. The one-seed pilot is archived separately and is not counted as an
independent seed. In both paired seeds the controller completed six loads
before tool return, but summed replay TTFT rose by 26.97 and 24.79 seconds,
and whole-workload completion rose by 2.02 and 0.33 seconds. All 12 sessions
had higher summed return-to-first-token time in both seeds. This is a result
about the controller implementation including its control checks, not a
measurement of early-copy cost alone or GPU bandwidth contention.

### KV-load attribution follow-up

Set `WORK_AUDIT_MODES` to add a check-only arm. It performs the same plan
checks and a plan-only placebo call when it would load, but does not move KV.
All three arms use the focused `kv_attribution` trace: request ingress,
cache lookup, and native load events, without per-batch scheduler logging.
Missing request-linked stages are reported as missing coverage.

```bash
SGLANG_DOCKER_IMAGE=agentic-sglang-standard:0.5.10.post1 \
  AGENTIC_MODEL_CACHE=/path/to/model/cache \
  WORK_AUDIT_RUN_ID=work_audit_kv_attribution_$(date +%Y%m%d_%H%M%S) \
  WORK_AUDIT_SEEDS="1 2" \
  WORK_AUDIT_MODES="baseline check_only controller" \
  bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The [archived three-arm run](../reports/work_audit/work_audit_kv_attribution_20261002_04/summary.json)
found variable check-only effects across seeds. The real-load arm added 11.04
and 19.42 seconds of summed replay TTFT versus check-only, with three and two
extra native loads. The added delay occurred before first token; multi-chunk
post-first-token generation was not slower. This is a workload-level
association from five early-load attempts, not a measured per-copy cost or
proof of HBM contention. A separate
[tracing-off/on microcheck](../reports/instrumentation_overhead_kv_attribution_20261003.json)
measured +3.05% median latency for a sequential long-prefix request path;
it does not bound overhead in the busy, naturally evicting workload.

The [one-seed load-phase repeat](../reports/work_audit/work_audit_load_phase_20261005_01/summary.json)
used the same three modes and records timestamps within each accepted native
load. Three completed loads spent 121, 308, and 1362 ms in
`ready_to_load_host_cache()` on the scheduler control path, while control-queue
waits were 2-12 ms and `load_back()` calls were 6-29 ms. CUDA-event durations
closely matched the ready-call wall times; these intervals overlap and must
not be summed. A fourth attempt was not admitted. Summed replay TTFT was
420.77 s without checks, 423.83 s with checks only, and 435.56 s with real
loads across 36 replays. The 12 multi-chunk replies did not show slower
generation after their first token. This locates a blocking backend call but
does not prove a specific memory-bandwidth collision or how much hardware
could remove. The [run manifest](../reports/work_audit/work_audit_load_phase_20261005_01/run_manifest.json)
and compressed traces preserve the settings and event timeline.
