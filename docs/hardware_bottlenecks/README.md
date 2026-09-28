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

## First controlled run

The first runnable case is deliberately narrow. It sends no frontend priority
or VIP labels. Both conditions use the same logical prompt/sample set, and
every individual sample starts from a fresh containerized backend. The
interference condition must first prove that a donor prefix is host-resident,
then request its load-back immediately before target replay. The command fails
rather than silently downgrading to a no-movement comparison.

Run this from the repository root after setting the runtime image and host
model cache for the selected machine:

```bash
export SGLANG_DOCKER_IMAGE='your-pinned-sglang-image'
export AGENTIC_MODEL_CACHE='/absolute/path/to/model-cache'
export BACKEND_RUNTIME_PROFILE='nvidia_standard'
export REPORT_LABEL="hardware_kv_movement_$(date +%Y%m%d_%H%M%S)"
bash infra/container/run_kv_movement_interference_reference.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Useful small-run knobs are `TRIALS=4`, `SEED=7`, `HICACHE_SIZE_GB=8`,
`MEM_FRACTION_STATIC=0.70`, and `DONOR_PROMPT_TOKENS=4090`. The donor default
is deliberately aligned with the pinned runtime's prefill chunking so its
evicted KV segment is large enough for SGLang to load back. The command writes
its report under
`sglang_direct_kv/artifacts/results/hardware/$REPORT_LABEL/`. Do not promote a
result to `HARDWARE_EXPERIMENTS.html` until its report shows accepted prepared
loads and the paired run manifest is complete.

The first mechanism pass sets `MIN_LOAD_TOKENS=1` explicitly. The pinned
SGLang runtime normally skips its tiny final response leaf below a 10-token
throughput threshold. This override still invokes SGLang's own `load_back`
and host-to-GPU copy path, but it must be reported as a small-copy validation,
not as evidence of memory-bandwidth saturation.
