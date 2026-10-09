# Controller Scenarios in the KV Lifecycle Audit

These are independent controller experiments, reported alongside the memory-tier
comparison in the KV Lifecycle Audit. The controller and backend packages still
own their policies and hooks. Do not combine the three policies in these runs.

## Frozen Definition

The executable specification is
`configs/experiment_specs/controller_audit_pivots.json`. Each scenario has a native
`no_prefetch` baseline and one controller mode. Each pair runs twice, with seeds
20260916 and 20260917 and reversed baseline/controller order. There are 12 arms.

| Scenario | Change from its baseline | What stays off |
| --- | --- | --- |
| 1. Replay scheduling | Existing ready-time GPU-backfill controller orders replays using expected tool-return time | Direct KV preparation and retention ranking |
| 2. Early KV preparation | Existing direct prepare hook requests useful host-resident KV during long tool waits | Queue ranking and retention ranking |
| 3. Reuse-time retention | Native priority radix eviction receives controller-derived next-wait insertion ranks | Queue-priority scheduling and direct preparation |

All sessions have equal application importance. Scenario 3 is an updated study,
not a repetition of the historical high/normal/low-value experiment. Its native
cache keeps maximum historical insertion priority on an existing node: assigning
zero to a terminal request does not demote its previously inserted prefix.

## Workload and Measurement

- 16 independent sessions, two tool returns each, 48 model calls including initial
  turns, and 32 measured replays per arm. Hatcher/DeepAgents synthetic driver.
- Qwen2.5-Coder-7B-Instruct, A10G 24 GB, pinned SGLang 0.5.10.post1/v0510 adapter.
- 24,576 GPU KV tokens and 8 GiB host cache in every arm; no storage tier.
- Scenario 1: 4,096-token main prompt, 1,536-token peer prompts. Scenarios 2/3:
  4,096-token prompts for all sessions. Prompts are template token targets, not
  guaranteed tokenizer counts. Output limits: main 8 tokens, peers 2 tokens.
- Maximum submission concurrency six. Scenario 2 uses 60-120 second tool waits;
  scenarios 1/3 use the exact mixed wait distribution in the JSON specification.
- Same model-prompt hashes, output limits, waits, seed and capacities within each pair.
  Pairing uses `harness_controller_signal.cache.conversation_prefix_hash`, not the
  transport-body hash that also contains mode-specific instrumentation metadata.
  Tool waits begin after each session's previous reply, so actual arrival times
  can shift when the policy changes completion times. This is a closed-loop
  synthetic agent workload, not a fixed open-loop arrival replay.
- Whole-workload time starts at the driver's workload clock and includes initial
  requests, tool waits and all replies. It excludes backend startup and preflight.
  Replay TTFT starts at gateway submission; deadline debt is summed positive
  lateness against the recorded replay deadline. These are different measurements.
- CUDA graphs and overlap scheduling must both be enabled. Runtime checks reject
  wrong settings. Shared `controller_queue` tracing, count-only indices, no full
  debug, no per-copy CUDA timing, and no idle-gap/blocker deep dives.

## Run on the GPU Host

Use the repository revision recorded in the selected experiment's manifest.
Do not overwrite an old run ID. The backend ports and GPU must be idle. The host
needs Docker with NVIDIA support, the recorded image, cached model snapshot,
the existing `sglang_direct_kv/.venv`, and the native harness dependencies.

```bash
cd /path/to/agentic_hardware
source sglang_direct_kv/.venv/bin/activate
export PYTHONPATH="$(printf '%s:' "$PWD"/packages/*/src)"
python -m agentic_experiments.runners.run_controller_audit_pivots \
  --run-id controller_pivots_repeat_$(date +%Y%m%d_%H%M%S) \
  --spec configs/experiment_specs/controller_audit_pivots.json \
  --model-cache "$HOME/.cache/huggingface" \
  --image agentic-sglang-standard:0.5.10.post1
```

To repeat just one scenario, append `--scenarios 1` (or `2` or `3`). Both trials
and both arms still run. On a source-synced host without `.git`, supply
`--source-archive /path/to/source.tar.gz --source-revision FULL_COMMIT_SHA` using
the archived bundle from that run. The runner verifies every archived source
file against the deployed files before executing the experiment. If replaying
an old source bundle, unpack it into a separate checkout, not over current work.

## Evidence Kept for Every Run

Under `sglang_direct_kv/artifacts/results/work_audit/RUN_ID`:

- `experiment_spec.json`: complete settings, scenario purpose and limitations.
- `run_manifest.json`: full revision, source archive hash, GPU/driver, exact image
  identity, per-arm commands and effective experiment environment (no credentials).
- `source.tar.gz`: tracked packages, launchers, scripts and configurations.
- `model_identity.json`: model snapshot paths and configuration/tokenizer hashes.
- `host_dependencies.txt` and `container_dependencies.txt`: installed versions.
- `status.json`: progress or explicit failure; unfinished runs are not evidence
  of a policy win.
- `raw/runs/controlled/ARM`: actual server configuration, installation/live gates,
  logs, raw gateway/backend traces (compressed after reporting), request metrics.
- `raw/reports/ARM`: per-request tables and the existing controller report.
- Each arm's launcher log and the report analyzer's source revision/code are
  preserved with the published evidence, separately from the measured source.

The audit must show each trial's workload duration, total/median replay TTFT,
total/median replay lateness, request coverage and policy exposure. Pairing gates
must compare request identities, prompt hashes, output budgets and sampled waits.
Missing events or zero real KV preparation/eviction exposure must be visible.
No throughput or cache advantage should be inferred solely from configured flags.
All prior results remain archived, even when a rerun does not reproduce a win.

## Publish Completed Pairs

The experiment revision and report-analysis revision can differ. The first runs
used frozen experiment source while the new report was being implemented.
Repeat the workload from `source_revision`; publish from a separate checkout at
`analysis_revision` (both are in the published `summary.json`), or a later
compatible reporting revision. Do not replace measured source files during a run.
Give the publisher an absolute path to the copied run directory when it is outside
that reporting checkout.

The analyzer skips scenarios until all four arms finish. It checks workload
identity and runtime isolation before adding a key pivot. Invalid pairs remain
visible as supporting evidence, without a percentage improvement claim.

```bash
python -m agentic_reports.analysis.analyze_controller_audit_pivots \
  --run-root sglang_direct_kv/artifacts/results/work_audit/YOUR_RUN_ID \
  --publish-dir docs/reports/work_audit \
  --progress-file docs/work_audit/research_progress.json
python -m agentic_reports.builders.build_work_audit_report \
  --results-dir docs/reports/work_audit \
  --progress-file docs/work_audit/research_progress.json \
  --out KV_LIFECYCLE_AUDIT.html \
  --markdown-out KV_LIFECYCLE_AUDIT.md
```

Keep the recorded container image. A tag is not an immutable image: compare its
ID with `image_inspect` in the manifest before repeating. The archived
`infra/container/build_sglang_runtime.sh` and `Dockerfile.sglang` describe the build,
but rebuilding with unconstrained dependency downloads is not guaranteed to
recreate the same image. Use the saved dependency inventories to identify drift;
a changed runtime must be labeled a new-runtime reproduction. Do not commit
multi-gigabyte model weights or Docker image exports to the repository.
