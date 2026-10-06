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
`kv_lifecycle`, `kv_lifecycle_lean`, `kv_decode_overlap`, `kv_attribution`, `copy_timing`, and `full_debug`. SGLang launch flags are resolved only in
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
`kv_lifecycle_lean` keeps the same required cache load and prefix-match
signals as `kv_lifecycle`, but omits broad scheduler tracing. Work-audit runs
can enable its control-only scheduler pump with
`WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean`; that pump processes prepare
commands without writing scheduler events. Its installation and live evidence
must pass the same fail-loud gates. The original `kv_lifecycle` flag defaults
remain unchanged for archived runs.
`kv_decode_overlap` adds request-linked decode-batch timestamps to the lean
lifecycle evidence. Set `WORK_AUDIT_FORWARD_TRACE=1` to also time nested model
forward calls for the RQ10 audit. The gate then requires the additional worker
hook; with the knob off, earlier batch-only runs remain reproducible. Neither
variant enables full scheduler debug tracing.
For a small RQ10 kernel-attribution run, set `WORK_AUDIT_NSYS_ENABLE=1` as
well. The host needs `nsys`; the launcher mounts its CLI into the pinned
SGLang container and adds call-linked NVTX ranges without enabling full debug
logging. The resulting CUDA trace is expensive and must be interpreted
separately from an unprofiled timing run. If Nsight loses CUDA records, the
analyzer fails the full run; `--pair 1` can extract an explicitly labeled
`validated_subset` from a fully captured pair, never a full-run claim. On the
A10G host, two longer captures lost later CUDA events and a shorter reverse
capture produced no kernel events; this path is not yet a reliable full-run
profiler recipe.
The offline `analyze_work_audit_cuda_launch_gaps` runner can further divide
the captured short-forward kernel gaps into time before the next CPU CUDA
launch, time inside its launch API, and time after that API until its kernel
starts. Give it `--sqlite`, `--summary`, `--kernel-attribution`, and `--out`;
the kernel report cross-check rejects partial linkage. This does not identify
why the CPU waited or measure whether the whole GPU was idle.
`kv_attribution` combines request-ingress timestamps with KV load and
prefix-match events, but keeps per-batch scheduler logging, per-layer copy
detail, GPU sampling, and full debug off. The three-arm busy audit
(`WORK_AUDIT_MODES="baseline check_only controller"`)
uses it to separate control-check overhead from additional load-mode effects.
Control-to-confirmation windows are not physical CUDA copy intervals; stage
coverage and tracing overhead must be checked before drawing a hardware claim.

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

The 2026-10-02 A10G work-audit calibration found that `kv_lifecycle` was also
expensive for a small serial request: median 2149.691 ms tracing off versus
10455.326 ms on (+386.36%; 83,990,189 trace bytes). With
`kv_lifecycle_lean` and its control-only pump, the corresponding medians were
2153.017 and 2199.396 ms (+2.15%; 331,232 trace bytes). These are separate
off-then-on probes with three measured requests per case, not a matched
long-prefix audit or a statistically controlled overhead bound. See the
[original-profile](docs/reports/instrumentation_overhead_kv_lifecycle_20261002.json)
and [lean-pump](docs/reports/instrumentation_overhead_kv_lifecycle_lean_pump_20261002.json)
raw timing summaries.

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
`CONTROLLER_POLICY_RESULTS.html` for the archived report and reproduction command.
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

For NAT, the native `_DynamoTransport` capture can optionally forward each
captured request to a running SGLang chat endpoint. The benchmark forwarder
changes only the model name and adds `custom_params.agentic_kv` identity;
`nvext` hints are preserved, not synthesized. For example, while the pinned
backend is running:

```bash
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit --scenarios nat_priority_high \
  --nat-dynamo-transport-capture \
  --nat-backend-url http://127.0.0.1:30000/v1/chat/completions \
  --nat-backend-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --run-id nat_boundary_live --out-dir /path/to/nat_boundary_live
```

For a pinned, end-to-end reference run, set `SGLANG_DOCKER_IMAGE` and
`AGENTIC_MODEL_CACHE`, then use
`bash infra/container/run_nat_hint_boundary_reference.sh
Qwen/Qwen2.5-Coder-7B-Instruct`. It starts a clean container with the lean
`request_boundary` profile, gates installed hooks before the NAT request,
then requires live acceptance and a matched correlation before succeeding.

This writes `backend_request_map.json` beside the hint observations. The
correlation ID is explicitly marked `benchmark_forwarder` in the captured
record; it is **not** a native NAT hint. The backend must independently log
that ID and pass the live `request_boundary` gate before the audit can claim
linked acceptance. The request map is an array of
`{"scenario_id":"...","payload_index":1,"backend_request_id":"...","correlation_id":"..."}`.
Only an ID independently present in the capture-boundary record and backend
event proves the forwarded request's linkage. Without it, the output says `external_mapping_only` and
`same_request_proven=false`. Even a matched request does not prove a hint was
translated or affected scheduling/cache behavior; those need separate
downstream value and effect evidence. Native hint observations never come from
the backend trace. Architecture tests reject new hook tables outside the
backend package.

The pinned A10G NAT reference run on 2026-10-02 passed the live
`request_boundary` gate and the strict `--require-same-request` join. NAT emitted `priority=100`
plus five other request-planning/cache-identity fields. The benchmark-owned ID
matched the backend acceptance record; the audit still reports
`backend_hint_effect=not_proven`. Small reproducibility artifacts are archived
under [`docs/reports/hint_benchmark/nat_hint_boundary_live_20261002`](docs/reports/hint_benchmark/nat_hint_boundary_live_20261002),
including the [join audit](docs/reports/hint_benchmark/nat_hint_boundary_live_20261002/hint_backend_audit.json)
and [native hint observations](docs/reports/hint_benchmark/nat_hint_boundary_live_20261002/hint_run/observed_hint_evidence.jsonl).

The 2026-10-02 A10G `copy_timing` reference also passed its live reload gates:
the before-decode case had 56 donor layer-copy records before decode and none
during it; the direct-overlap case had 56 during the client-visible decode
window. Decode durations were 42.671 s (control), 43.349 s (reload before),
and 43.361 s (reload during), one trial each. The 12 ms difference between
reload timings is not a reliable slowdown estimate. The [hardware ledger](GPU_INTERFERENCE.html)
and [archived summary](docs/reports/hardware/shared_evidence_live_20261002/sustained_decode_kv_overlap_summary.json)
record the measurements and their proof limits.
