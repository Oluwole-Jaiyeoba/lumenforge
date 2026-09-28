# Hardware Bottleneck Characterization

This is the third research lane. It measures where GPU compute and memory
movement delay agentic replays before proposing hardware changes. It is not a
controller-mode comparison or evidence that a harness natively emits a hint.

The entry point is [`HARDWARE_EXPERIMENTS.html`](../../HARDWARE_EXPERIMENTS.html).
Its source of truth is `configs/hardware_experiment_registry.json`; rebuild the
page with `python3 scripts/build_hardware_experiments.py`. Planned cases contain
no measured outcome. A measured entry requires a reproducible command, a run
ID, a report, a manifest, and an evidence file that can be opened from the
checkout. The manifest must use `agentic_run_manifest.v2`, match the displayed
run ID, and record a finished status. Use `--check` to detect a stale page.

## First two cases

1. **KV movement interference:** Pair identical replay/decode samples with and
   without competing host-to-GPU KV movement. Measure replay TTFT and lateness,
   decode inter-token latency, copy duration, and overlap. Use backend KV events
   for software-visible timing; use profiler or hardware counters before
   claiming physical transfer-engine or memory-bandwidth saturation.
2. **Active compute blocking:** Make a replay ready during prefill and vary only
   the prefill chunk size. Align replay ready/submit/admit/first-token times
   with active GPU work. Separate a busy GPU from a gateway or queue gap.

Use lightweight traces for latency comparisons and a separate profiler pass
for physical attribution. Profiling can change timing. Hold model, backend,
arrival schedule, inputs, cache state, seed, and request population fixed in
each comparison. Keep semantic frontend priority out of these first cases.
The accessible standard-NVIDIA remote GPU is the initial test platform; GH200
and other hardware need their own measured runs and runtime manifests.

The portable `agentic-hardware-probes` package pairs sample IDs and rejects
comparisons with different workload, model, backend, hardware profile, seed,
or instrumentation profile. Given collected JSON files with the package's
`ProbeRun` fields, run:

```json
{
  "run_id": "example_control",
  "condition": "control",
  "hardware_profile": "recorded_gpu_profile",
  "backend_version": "recorded_backend_version",
  "model": "recorded_model",
  "workload_id": "fixed_workload_id",
  "seed": 7,
  "instrumentation_profile": "lightweight",
  "samples": [
    {"sample_id": "replay_1", "metrics_ms": {"replay_ttft_ms": 100.0}}
  ]
}
```

This is a format example, not an observed run. An interference run uses the
same contract and sample IDs, a distinct run ID, and
`"condition": "interference"`. Then compare:

```bash
python3 -m agentic_hardware_probes control.json interference.json --metric replay_ttft_ms
```

A positive change means the interference case took longer. This comparison
does **not** establish why it took longer. Physical claims require separate
evidence and review. SGLang-specific capture belongs in
`agentic-backend-sglang`, machine profiler wrappers under `infra/`, experiment
orchestration in `agentic-experiments`, and reports in `agentic-reports`.

Existing SGLang-visible H2D analysis is described in
[`KV_H2D_BANDWIDTH_PRESSURE.md`](../sglang_direct_kv/KV_H2D_BANDWIDTH_PRESSURE.md).
Its physical-evidence limitation is recorded in
[`INSTRUMENTATION_AUDIT.md`](../sglang_direct_kv/INSTRUMENTATION_AUDIT.md).
