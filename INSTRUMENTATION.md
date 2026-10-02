# Shared Instrumentation

This is the entry point for any lane that needs SGLang observations. Do not add
another lane-local SGLang hook. The backend package owns all SGLang-private
names, hook installation, version adapters, and raw-to-stable translation.

| Layer | Owner | Responsibility |
| --- | --- | --- |
| In-process hooks and version drift | `packages/agentic-backend-sglang/` | Install read-only hooks through the selected adapter; emit raw trace and installation report |
| Shared evidence contract | `packages/agentic-instrumentation/` | Stable signal IDs, nanosecond event schema, profiles, coverage checks, and conservative cache-slot proof |
| Controller experiment | `packages/agentic-controller/`, launchers | Decide policy; use backend profile and the existing fail-loud sentinel preflight |
| Work audit | `packages/agentic-reports/` | Count semantic loads separately from layer copies; validate required lifecycle evidence |
| Hardware experiment | `packages/agentic-hardware-probes/` | GPU-only measurements remain independent; any future SGLang trace joins use the shared evidence contract |
| Hint benchmark | `packages/agentic-harness-scenarios/` | Harness request hints remain separate from backend observations; use the shared backend boundary if testing translation |

## Profiles and Gates

The shared catalog defines `request_boundary`, `controller_queue`,
`kv_lifecycle`, `copy_timing`, and `full_debug`. SGLang launch flags are resolved only in
`agentic_backends.sglang.instrumentation_profiles`. Historical launcher names
(`minimal`, `deadline`, `controller_decision`, `idle_gap`, `cache_debug`,
`full_debug`) keep their existing flag defaults. Unknown names fail.

```bash
python3 -m agentic_instrumentation signals
python3 -m agentic_instrumentation profiles
python3 -m agentic_backends.sglang.instrumentation_profiles kv_lifecycle
```

To inspect every hook in a concrete run and validate its live evidence:

```bash
python3 -m agentic_backends.sglang.trace_contract \
  --adapter v0510 --profile controller_queue \
  --installation /path/to/hook_installation_report.json \
  --trace /path/to/live_sentinel_trace.jsonl \
  --out /path/to/instrumentation_audit.json
```

The command exits nonzero when the installation adapter differs, a required
hook is absent, a required event did not occur, or its proof fields are empty.
`--events-out` exports recognized events in the backend-neutral schema; use
repeatable `--export-signal` arguments to keep the file small. The
`--installation-only` option checks hook installation without requiring a
workload event, and must not be described as live behavior proof.
If the trace contains `trace.install.summary`, `--installation` can be omitted;
the audit will use that run's embedded installation record.
The inventory includes all adapter targets, even unobserved and raw-only
hooks. It records field *names* and counts, not prompt contents. A profile's
optional events can be absent when the small probe did not exercise them.
`full_debug` enables broad capture, but does not claim that every possible KV
transition must occur in every short run.

The new `controller_queue` profile turns off copy-detail telemetry to keep
queue experiments lean. The historical controller profile names retain their
previous copy-telemetry defaults so existing comparisons do not silently
change.

Before controller experiments, the existing
`agentic_backends.sglang.instrumentation_preflight` checks the runtime
contract, installed hooks, live sentinel, shared `controller_queue` evidence,
and controller action. The shared batch-completion signal proves a scheduler
batch finished; it does **not** prove that each request in that batch finished.
Request completion timing remains a separate gateway/client observation. Before a new
work-audit validation, its launcher checks the pinned backend runtime and
installed lifecycle hooks, then runs the shared `kv_lifecycle` trace gate;
the report builder requires the expected session-linked load and replay-match
evidence. These are distinct checks:
an installed hook alone is not evidence that the workload exercised it.

## Cache Proof Ladder

1. `kv.load_gpu`: a semantic host-to-device load call returned. Per-layer copy
   records are supporting detail, not additional loads. Where the backend
   enqueues asynchronously, a separate completion status is needed before
   claiming the transfer finished.
2. `kv.prefix_match`: the later replay matched cache slots. A matching device
   index fingerprint provides stronger evidence that some loaded slots appeared
   in the replay match, even if SGLang split/renamed the cache node. The audit
   reports the count, but an unobserved eviction or slot reuse remains possible.
3. Exact model-kernel consumption of those same slots is **not proven** by the
   current trace. Do not describe a prefix match as exact consumption.

The slot check requires exact values or a contiguous range verified against
its digest, matching session and replay request, correct time order, and no
observed intervening device eviction. Missing evidence yields `unknown`, not
zero reuse or a fabricated success.

## Adding a Hook or Porting a Version

1. Add the method/event mapping to the version adapter under
   `packages/agentic-backend-sglang/src/agentic_backends/sglang/versions/`.
2. Add a stable capability in `hook_registry.py` and map the raw event into a
   shared `EvidenceEvent` in the backend package. Never expose an SGLang
   private symbol from `agentic-instrumentation`.
3. Declare the signal and its fields in the shared catalog. Put it in a profile
   only if a lane needs to require it. Keep high-volume captures opt-in.
4. Add synthetic positive and negative tests, an installation check, and a
   saved-trace regression before claiming support for a new backend version.
5. Measure tracing overhead on the actual accelerator before calling a
   high-volume profile low-overhead. The paired microcheck is:

```bash
python3 scripts/measure_instrumentation_overhead.py \
  --image "$SGLANG_DOCKER_IMAGE" \
  --model-cache "$AGENTIC_MODEL_CACHE" \
  --out-dir sglang_direct_kv/artifacts/results/instrumentation_overhead
```

It starts the same model twice on an otherwise idle GPU, first with tracing
off, then with the full trace on; it reports individual request latencies,
token counts, and trace size. One ordered pair is only a smoke measurement,
not a statistically controlled claim. Run it only when no experiment is using
the GPU.

On the pinned A10G setup (2026-10-02), six measured 64-output-token requests
per case had median latency **2,149.6 ms off** and **10,774.1 ms with
full-debug tracing** (+401.2%). The traced case wrote 141.2 MB of backend
trace. The raw per-request latencies and token counts are archived in
[`docs/reports/instrumentation_overhead_20261002.json`](docs/reports/instrumentation_overhead_20261002.json).
This single off-then-on comparison is preliminary, but it is enough to show
that full-debug tracing is intrusive for this small serial workload. Keep it
for diagnosis, use the leanest profile that proves a claim, and measure each
profile before drawing conclusions from absolute latency. Both modes of the
Scenario 1 regression used the same full-debug setting, but that does not
remove the need to re-evaluate absolute timings with lighter tracing.

The pinned validated work-audit adapter is `v0510` for SGLang
`0.5.10.post1`. Other adapters may expose different hooks; a missing required
capability must stop the run. The existing controller reference protocol and
its results are unchanged by this refactor: rebuilding the saved equal-importance
Scenario 1 report before and after the shared KV mapping produced identical
`controlled_replay_report.json` SHA-256
`0bb4a166eebc50de6b34845ee6cd9c4ee81bc578bd005463385ffa96c659e892`.
The saved live sentinel passed the shared queue gate on the pinned adapter,
with 45 hook targets inventoried. These are regression checks, not a fresh
GPU performance result. A fresh pinned Scenario 1 run on 2026-10-02 then
passed strict preflight and the shared queue gate in both modes, and the
equal-importance validator confirmed 32 replays per mode with no frontend
priority. Its total replay TTFT was 99.79 s (no prefetch) versus 67.65 s
(controller), and total deadline debt was 231.05 s versus 162.22 s. These
remain run-to-run measurements, not byte-identical timing; see
`CONTROLLER_EXPERIMENTS.html` for the archived report and reproduction command.
The saved work-audit trace also passed the shared `kv_lifecycle` gate (three
semantic loads, 56 layer-copy rows, and 57 prefix matches). Re-analyzing that
trace produced the same validation summary after excluding only source-file
path spelling; no outcome metric changed.

The hardware probe lane's independent CUDA copy test has no SGLang hooks to
centralize. Its sustained-decode SGLang-correlated launcher now resolves the
`copy_timing` profile, checks installed hooks before each trial, gates live
copy evidence after reload trials, and writes normalized events plus
`backend_evidence_join.json` for each trial. The control checks installation
only. The join requires target acceptance and donor copy completion in the
expected client-decode interval. It does not prove GPU-kernel overlap or HBM
contention. Independent GPU profiler evidence remains separate.

The hint suite observes native harness requests. For a backend-facing run,
gate the trace with `request_boundary` and export `request.accepted` events:

```bash
python3 -m agentic_backends.sglang.trace_contract \
  --adapter v0510 --profile request_boundary \
  --trace /path/to/backend_trace.jsonl \
  --out /path/to/backend_audit.json \
  --events-out /path/to/backend_events.jsonl \
  --export-signal request.accepted
python3 -m agentic_experiments.runners.audit_hint_backend_evidence \
  --hint-observations /path/to/observed_hint_evidence.jsonl \
  --backend-audit /path/to/backend_audit.json \
  --backend-events /path/to/backend_events.jsonl \
  --request-map /path/to/request_map.json \
  --out /path/to/hint_backend_audit.json
```

The request map is an array of
`{"scenario_id":"...","payload_index":1,"backend_request_id":"...","correlation_id":"..."}`.
Only an ID independently present in the captured native request and backend
event proves same-request linkage. Without it, the output says `external_mapping_only` and
`same_request_proven=false`. Even a matched request does not prove a hint was
translated or affected scheduling/cache behavior; those need separate
downstream value and effect evidence. Native hint observations never come from
the backend trace. Architecture tests reject new hook tables outside the
backend package.
