# GPU Interference

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
`MEM_FRACTION_STATIC=0.70`, `DONOR_PROMPT_TOKENS=4090`, and
`MINIMUM_HOST_TOKENS=512`. The probe follows the donor cache path and selects
the largest evicted, host-backed segment. It fails rather than falling back to
a tiny leaf when that segment is below `MINIMUM_HOST_TOKENS`. The command writes
its report under
`sglang_direct_kv/artifacts/results/hardware/$REPORT_LABEL/`. Do not promote a
result to `HARDWARE_EXPERIMENTS.html` until its report shows accepted prepared
loads and the paired run manifest is complete.

`MIN_LOAD_TOKENS=1` is retained only for a diagnostic small-copy mechanism
pass. The normal experiment leaves it unset and uses SGLang's native threshold.
The earlier six-token result is documented as a small-copy validation, not as
evidence of memory-bandwidth saturation.

## Sustained decode overlap

The next probe tests the collision directly. It keeps all requests equal at
the frontend and compares three conditions from a fresh backend each time:

1. `decode_control`: stage a host-resident donor prefix, but do not reload it.
2. `non_overlap_reload`: finish reloading the donor before target decode starts.
3. `direct_overlap_reload`: start target decode, wait for streamed output, then
   reload the donor while decode is still producing tokens.

Run it with:

```bash
export SGLANG_DOCKER_IMAGE='your-pinned-sglang-image'
export AGENTIC_MODEL_CACHE='/absolute/path/to/model-cache'
export BACKEND_RUNTIME_PROFILE='nvidia_standard'
export REPORT_LABEL="sustained_decode_kv_overlap_$(date +%Y%m%d_%H%M%S)"
bash infra/container/run_sustained_decode_kv_overlap_reference.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The default is a one-trial sanity run. Use `TRIALS=6` or more only after it
passes. Other useful knobs are `DECODE_TOKENS=256`, `WARMUP_CHUNKS=24`,
`MINIMUM_OVERLAP_INTERVALS=1`, `DONOR_PROMPT_TOKENS=4090`, and
`MINIMUM_HOST_TOKENS=512`. This uses the approximately 4K-token native load
path already proven by the earlier movement experiment. `DEVICE_FREE_TOKENS`
defaults to zero. A positive value applies the same optional native
device-cache eviction setup in all three conditions; it is never part of the
measured condition difference.

Each trial uses the shared `copy_timing` instrumentation profile and writes
`instrumentation_audit.json`, `normalized_backend_evidence.jsonl`, and
`backend_evidence_join.json` beside its raw backend trace. Missing required
hooks or donor copy evidence fails the trial. The control checks hook
installation without requiring a reload. These joins establish when backend
copy hooks completed relative to the client-visible decode interval; they do
not by themselves establish HBM bandwidth contention.

The direct-overlap case fails unless SGLang's own per-load CUDA events show a
completed reload with positive physical duration between decode start and
decode finish, with at least the configured number of streamed output
intervals intersecting the observed load envelope. A fast load may finish
before the HTTP poller observes an intermediate `active` state; that does not
invalidate its CUDA-event duration.
The report compares client-visible stream-update intervals before, during, and
after reload. It does not assume that every visible update equals exactly one
generated token. This is lightweight timing evidence. Run the profiler separately before making a
claim about transfer-engine or memory-controller bandwidth, and do not pool
profiled timing with the lightweight run.

## Sustained KV pressure

The stronger follow-up keeps the earlier single-reload evidence intact and
adds a separate `sustained_kv_pressure` workload. Every condition stages the
same pool of eight distinct, host-resident donor prefixes. Requests have equal
frontend semantics and no frontend priority.

The three conditions are:

1. `decode_control`: load no donors during decode.
2. `single_reload`: load one donor after 24 visible output updates.
3. `sustained_reload`: load all eight donors sequentially after the same
   warmup point.

Run the one-trial proof pass with:

```bash
export SGLANG_DOCKER_IMAGE='your-pinned-sglang-image'
export AGENTIC_MODEL_CACHE='/absolute/path/to/model-cache'
export BACKEND_RUNTIME_PROFILE='nvidia_standard'
export REPORT_LABEL="sustained_kv_pressure_sanity_$(date +%Y%m%d_%H%M%S)"
TRIALS=1 bash infra/container/run_sustained_kv_pressure_reference.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

This pressure workload defaults to `MEM_FRACTION_STATIC=0.80` and
`DEVICE_FREE_TOKENS=40000`, both held constant across all conditions. The
common precondition clears enough device-cache capacity before measurement for
the complete eight-prefix train. This avoids mixing the intended host-to-device
traffic with emergency cache eviction during decode. The run still fails
loudly if the backend cannot allocate any individual load.

Only after that passes, use `TRIALS=6` for paired evidence. The script rotates
condition order for each sample and starts a clean backend for every
sample/condition. The sustained condition fails unless all eight native loads
finish inside decode, each has positive CUDA duration, every physical reload
window intersects a client-visible decode interval, their total CUDA load time
reaches `MINIMUM_TOTAL_CUDA_LOAD_MS`, and their largest observed gap remains
below `MAXIMUM_INTER_LOAD_GAP_MS`. Reloads can share one long output gap; the
gate checks every reload window rather than requiring an arbitrary number of
distinct streamed updates.

The HTML report includes complete-decode and pressure-window timelines,
paired decode-time changes, reload counts, moved tokens, CUDA duration,
pressure duty cycle, and visible output timing before, during, between, and
after reloads. This remains lightweight timing evidence; use a separate
profiler pass for physical memory-bandwidth attribution.

## KV pressure duty sweep

`sustained_kv_pressure` proves that a short train of reloads can overlap with
decode. The next controlled step is `kv_pressure_duty_sweep`: the same target
decode is run under four independent conditions with no frontend priority
labels:

1. `decode_control`: no native reloads.
2. `pressure_low`: a small native-reload budget.
3. `pressure_medium`: a larger native-reload budget.
4. `pressure_high`: the largest native-reload budget.

The runner stages each donor by asking SGLang's native control path to evict
its device copy, then requires a plan-only query to prove that the donor
remains host-backed and eligible for a later native reload. It uses the same
native evict-and-prove sequence before a donor is reused. If either proof
fails, the experiment stops rather than pretending it created more
host-to-GPU traffic. This avoids adding large unmeasured prompt work merely to
stage the donor pool.

Run the first complete sweep with:

```bash
export SGLANG_DOCKER_IMAGE='your-pinned-sglang-image'
export AGENTIC_MODEL_CACHE='/absolute/path/to/model-cache'
export BACKEND_RUNTIME_PROFILE='nvidia_standard'
export REPORT_LABEL="kv_pressure_duty_sweep_$(date +%Y%m%d_%H%M%S)"
TRIALS=3 bash infra/container/run_kv_pressure_duty_sweep_reference.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The default probe uses 384 decode tokens, eight host-backed donors, and reload
budgets of 20, 60, and 100 for low, medium, and high pressure. The knobs
`LOW_LOAD_COUNT`, `MEDIUM_LOAD_COUNT`, `HIGH_LOAD_COUNT`,
`LOW_MIN_CUDA_SHARE_PCT`, `MEDIUM_MIN_CUDA_SHARE_PCT`, and
`HIGH_MIN_CUDA_SHARE_PCT` allow calibration on another machine. The report
treats measured CUDA-load share as authoritative: low, medium, and high are
requested budgets, not an assumption that a certain amount of physical
contention occurred.

The sweep rejects any run where a requested load lacks positive CUDA duration,
falls outside active decode, does not intersect a client-visible decode
interval, or cannot be reconciled with the requested count. It records total
decode, TTFT, total KV moved, native CUDA copy time, active-copy share of
decode, pressure-envelope share, and the number of recycle proofs.

## Natural multi-agent KV pressure

The controlled duty sweep deliberately injects native reloads to establish a
reference curve. `natural_multi_agent_kv_pressure` is the separate realism
baseline: equal-priority synthetic agent sessions build context, wait for
tools, and resume through ordinary requests. It never calls the prepared-prefix
control API and never injects a reload. SGLang may naturally reload an evicted,
host-backed prefix when one of those sessions resumes.

The report records each replay decode, native `HiRadixCache.load_back` events
that overlap its post-first-token decode period, and the session that caused
each observed reload. It distinguishes all reloads from reloads caused by a
different session. It maps observed reload counts to the controlled reference
buckets, while clearly marking that count-based mapping as provisional until a
profiler pass measures natural CUDA-copy share directly.

Run the first observation-only baseline with:

```bash
export SGLANG_DOCKER_IMAGE='your-pinned-sglang-image'
export AGENTIC_MODEL_CACHE='/absolute/path/to/model-cache'
export BACKEND_RUNTIME_PROFILE='nvidia_standard'
export REPORT_LABEL="natural_multi_agent_kv_pressure_$(date +%Y%m%d_%H%M%S)"
SESSION_COUNT=8 TOOL_WAITS=3 SESSION_PREFIX_TOKENS=8192 REPLAY_TOKENS=384 \
  bash infra/container/run_natural_multi_agent_kv_pressure_reference.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```
