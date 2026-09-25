# Controller Instrumentation Validation Gate

Controller experiments are valid only when the SGLang instrumentation needed by
that experiment is installed and observed on a live request. This prevents an
adapter mismatch from silently producing misleading controller results.

## Phases

1. **Contract foundation** - Define stable capability IDs and evidence each
   experiment requires in `configs/controller_instrumentation_contracts/`.
2. **Static preflight** - Verify the runtime capability handshake and selected
   adapter before the server starts.
3. **Installation check** - Require the in-server tracer to publish a hook
   installation report after it starts.
4. **Live sentinel check** - Send one excluded, tiny request through the normal
   gateway and require the expected trace events.
5. **Experiment gate** - Block controller measurements when a required check
   fails. `observe_only` records the failure but labels the run invalid.
6. **Reporting** - Preserve all preflight artifacts with the run so reports can
   state whether their measurements are valid for controller claims.

## Current behavior

`controller_replay_v1.json` is the default contract for controller modes.
Scenario 1 uses `scenario1_ready_time_reference.json`. The runner writes these
artifacts inside every controller case:

- `instrumentation_contract.json`
- `hook_installation_report.json`
- `live_sentinel_report.json`
- `live_sentinel_trace.jsonl`
- `controller_preflight_probe.jsonl`
- `backend_trace.jsonl`

The sentinel request is labeled `instrumentation_sentinel` and is excluded from
workload metrics. Contracts contain only stable hook IDs; the selected SGLang
adapter resolves them to its private targets. Optional hooks are warnings;
required hooks and required live events are blocking in the default `strict`
policy.

Gateway events and in-process backend events are deliberately written to
separate JSONL files. The gate validates gateway receipt from `m27_trace.jsonl`
and SGLang hook activity from `backend_trace.jsonl`, then records their joined
sentinel slice in `live_sentinel_trace.jsonl`. This prevents one writer from
hiding the other evidence stream.

## Operator controls

- `CONTROLLER_INSTRUMENTATION_GATE=auto|on|off` selects whether the gate runs.
  `auto` enables it whenever a controller mode is present.
- `CONTROLLER_INSTRUMENTATION_POLICY=strict|observe_only` controls failure
  handling. Use `strict` for experiments that make controller claims.
- `CONTROLLER_INSTRUMENTATION_CONTRACT=/absolute/path/to/contract.json`
  selects an experiment-specific contract.

Do not use `observe_only` results as evidence that controller instrumentation
worked. It is for diagnosing a new backend adapter only.
