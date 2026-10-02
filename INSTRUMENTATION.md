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

The shared catalog defines `controller_queue`, `kv_lifecycle`, `copy_timing`,
and `full_debug`. SGLang launch flags are resolved only in
`agentic_backends.sglang.instrumentation_profiles`. Historical launcher names
(`minimal`, `deadline`, `controller_decision`, `idle_gap`, `cache_debug`,
`full_debug`) keep their existing flag defaults. Unknown names fail.

```bash
python3 -m agentic_instrumentation signals
python3 -m agentic_instrumentation profiles
python3 -m agentic_backends.sglang.instrumentation_profiles kv_lifecycle
```

The new `controller_queue` profile turns off copy-detail telemetry to keep
queue experiments lean. The historical controller profile names retain their
previous copy-telemetry defaults so existing comparisons do not silently
change.

Before controller experiments, the existing
`agentic_backends.sglang.instrumentation_preflight` checks the runtime
contract, installed hooks, live sentinel, and controller action. Before a new
work-audit validation, its launcher checks the pinned backend runtime and
installed lifecycle hooks; the report builder then requires the expected
session-linked load and replay-match evidence. These are distinct checks:
an installed hook alone is not evidence that the workload exercised it.

## Cache Proof Ladder

1. `kv.load_gpu`: a semantic host-to-device load completed. Per-layer copy
   records are supporting detail, not additional loads.
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
   high-volume profile low-overhead. That comparison has **not** been measured
   as part of this centralization.

The pinned validated work-audit adapter is `v0510` for SGLang
`0.5.10.post1`. Other adapters may expose different hooks; a missing required
capability must stop the run. The existing controller reference protocol and
its results are unchanged by this refactor; a fresh GPU regression is needed
before claiming identical performance.
