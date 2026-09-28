# Portable Architecture Map

## Purpose

This document is the source of truth for separating the research ideas in this
repository from SGLang, individual harnesses, machine setup, and report code.
The portable package promotion is now complete. Existing commands and experiment
paths remain available through compatibility imports while later phases isolate
the SGLang backend and split experiment/report ownership.

The target is a testbed where:

- harness integrations can be developed without importing the controller;
- controller policies can be tested without Claude Code, NAT, or SGLang;
- SGLang can be upgraded or replaced through a versioned backend adapter;
- experiments compose packages instead of containing their implementation;
- reports consume stable, normalized records rather than raw backend events;
- remote A10G host, GH200, Docker, and local setup differ through configuration and launch
  scripts, not through controller or harness logic.

## Status (2026-09-23)

Phases 1-4 are implemented on branch `refactor/sglang-portability`; Phase 5-7
are partially done. Details, decisions and every behavior change are in
[`docs/architecture/README_RESTRUCTURING.md`](README_RESTRUCTURING.md). Summary:

| Phase | Status |
| --- | --- |
| 1-3.5 Contracts, controller/harness split, package promotion | done (before this branch) |
| 4 Consolidate SGLang integration | **done**: `packages/agentic-backend-sglang` (`agentic_backends.sglang`) owns all SGLang imports, hook tables, raw event maps, request lowering, launch flags and patch bootstrap; versioned adapters v0510-v0520; static surface check; explicit bootstrap via `agentic_backends.sglang.trace.install()` |
| 5 Split experiments and reports | **done structurally** (testbed split): `packages/agentic-experiments` (runners, replay driver, gateway, workloads) and `packages/agentic-reports` (builders, audits, block ledger); milestone-only scripts frozen in `sglang_direct_kv/scripts/legacy/`. Still open: decomposing the driver's `main_async` and moving reports onto normalized observations |
| 6 Deployment portability | partial: SGLang pinned, Docker paths mount `packages/`, CI without SGLang |
| 7 Remove compatibility shims | not started (old `agentic_kv` paths and `sglang_direct_kv/scripts/*.py` are thin aliases/wrappers) |

The package table below now also includes `agentic_gateway` (backend-neutral
request translation; may import core + harnesses). Its placement rationale is
decision D12 in the restructuring README.

## Architectural Decision

The agentic harnesses and the agentic controller are separate packages. They
communicate only through shared contracts.

```text
actual harness clients
        |
        v
agentic_harnesses -----> agentic_core <----- agentic_controller
                              ^                       |
                              |                       v
                     agentic_backend_api <---- BackendCommand
                              |
                              v
                  agentic_backends.sglang
                              |
                              v
                           SGLang

agentic_experiments composes the packages above.
agentic_reports reads their normalized output records.
```

`agentic_harnesses` answers, "What did the harness expose?"
`agentic_controller` answers, "What should the system do with that information?"
`agentic_backends.sglang` answers, "How is that decision implemented on this
specific SGLang version?"

## Target Packages

| Package | Owns | Must not own |
| --- | --- | --- |
| `agentic_core` | Stable schemas and identifiers: harness signals, request/session identity, controller decisions, backend commands, observations, scheduler state, KV residency, and hardware telemetry. | Harness SDKs, SGLang imports, experiment orchestration, HTML reports, machine setup. |
| `agentic_harnesses` | Claude Code, NAT, Codex, OpenCode, and other real-client adapters; native signal capture; source provenance; normalization to `HarnessSignal`. | Scheduling policy, SGLang lowering, controller decisions, backend mutation. |
| `agentic_controller` | Deadline scheduling, admission, demotion, SJF, runtime estimation, KV preparation, eviction, and policy state. | Direct harness invocation, raw provider fields, direct SGLang imports, HTML generation. |
| `agentic_backend_api` | Backend capability discovery and the abstract interface for priority, admission, KV prepare/demote/release, cancellation, and telemetry. | SGLang module paths or version-specific event names. |
| `agentic_backends.sglang` | SGLang request lowering, HiCache/RadixCache integration, trace hooks, raw-event translation, and per-version compatibility adapters. | Harness semantics and controller policy. |
| `agentic_experiments` | Workload manifests, deterministic timelines, mode composition, run metadata, and orchestration. | Policy implementation and backend internals. |
| `agentic_reports` | Normalized evidence tables, deadline/cost accounting, audits, plots, and report rendering. | Live backend mutation and raw SGLang interpretation. |
| `agentic_prompt_codec` | Lossless shorthand rules, codecs, tokenizer accounting, validation, and codec reports. | Controller and SGLang behavior. |

The backend-neutral packages now live under top-level `packages/`. SGLang-bound
code remains in `sglang_direct_kv` until the versioned backend adapter is
extracted in Phase 4.

## Allowed Dependencies

The dependency direction is one-way:

| Package | May import |
| --- | --- |
| `agentic_core` | Python standard library and generic serialization libraries only. |
| `agentic_harnesses` | `agentic_core`; optional harness/provider SDKs behind extras. |
| `agentic_controller` | `agentic_core`; `agentic_backend_api` types when required. |
| `agentic_backend_api` | `agentic_core`. |
| `agentic_backends.sglang` | `agentic_core`, `agentic_backend_api`, and supported SGLang versions. |
| `agentic_experiments` | Public interfaces from core, harnesses, controller, backend API, and selected adapters. |
| `agentic_reports` | `agentic_core` schemas and normalized experiment records. |
| `agentic_prompt_codec` | Its own public interfaces; optionally `agentic_core` for shared request envelopes. |

Forbidden directions:

- `agentic_core` must not import any project package.
- `agentic_harnesses` and `agentic_controller` must not import each other.
- No package except `agentic_backends.sglang` may import `sglang` or reference
  private `sglang.srt.*` APIs.
- Reports must not infer semantics directly from raw SGLang event names.
- Machine launch scripts must not define controller policy.
- Experiment runners must not contain a second implementation of policy logic.

## Backend-Neutral Contracts

The first extraction should stabilize these records before moving behavior:

| Contract | Minimum responsibility |
| --- | --- |
| `HarnessSignal` | Native field, normalized meaning, source harness, provenance lane, attachment level, client/session/task/request IDs, timestamp, confidence, and raw evidence reference. |
| `RequestEnvelope` | Request identity, model, token shape, session/task relationship, phase, ready time, deadline, priority class, reusable-prefix identity, and workload role. |
| `ControllerDecision` | Inputs considered, chosen action, rejected alternatives, reason, policy version, and decision timestamp. |
| `BackendCommand` | Backend-neutral action such as set priority, admit/hold, prepare/demote/release KV, cancel, or observe. |
| `BackendCapabilities` | Which commands and telemetry fields the selected backend/version can actually support. |
| `BackendObservation` | Queue, running batch, scheduler, KV, request lifecycle, and command-result observations in normalized form. |
| `SchedulerState` | Ready, waiting, running, and held work plus capacity and batching state. |
| `KVResidency` | Prefix/block identity, tier, size, reuse state, movement state, and retention value. |
| `HardwareTelemetry` | Timestamped utilization, memory, bandwidth, throttling, and device identity. |
| `RunManifest` | Versions, commit, model, hardware, configuration, seed, enabled instrumentation, and artifact locations. |

All contracts require a schema version. Raw provider and SGLang fields should be
preserved as evidence but must not become required controller inputs.

## Current Repository Audit

### Top-Level Areas

Updated 2026-09-23 after the top-level cleanup: root keeps only the landing
README, the active HANDOFF, and code/deployment directories.

| Current path | Current role | Target owner | Disposition |
| --- | --- | --- | --- |
| `README.md` | Short repository landing page. | Repository documentation. | Keep at root; substantive documentation lives under `docs/`. |
| `HANDOFF.md` | Active hand-off for the next agent/engineer. | Repository documentation. | Kept at root on purpose (owner decision). |
| `docs/` | All current documentation; see `docs/index.md`. Architecture (`docs/architecture/`), hint benchmark (`docs/hint_benchmark/`), testbeds, proposals, reports (`docs/reports/`), presentations, and research PDFs. | Repository documentation. | Canonical home. |
| External historical archive | Preserved report copies and build scratch outside the repository. | Historical material. | Never import from it. |
| `packages/` | All packages: core, backend API, controller, harnesses, gateway, prompt codec, harness scenarios, hardware probes, `agentic-backend-sglang`, `agentic-experiments`, `agentic-reports`. | Package owners. | See Python Packages below. |
| `sglang_direct_kv/` | The SGLang testbed: shell entry points, thin script wrappers, `scripts/legacy/`, configs, `sitecustomize`, `agentic_kv` compatibility layer, tests. | Testbed. | Keep thin; new code goes into packages. |
| `scripts/`, `tests/`, `.github/` | Workspace install, portability checks, architecture tests, CI. | Repository tooling. | Keep. |
| `infra/remote/`, `infra/accelerator/gh200/` | Provider-neutral remote-host and NVIDIA GH200 helpers. | `deployment/*`. | Keep operational path until wrappers replace it. |
| `.codex_external/` | External datasets and repositories used for analysis (git-ignored). | External inputs. | Never make package imports depend on this path. |

### Python Packages

| Current path | Contents | Target owner | Portability assessment |
| --- | --- | --- | --- |
| `packages/agentic-core/` | Versioned harness, request, controller, observation, scheduler, KV, hardware, and run-manifest contracts. | `agentic_core`. | Standalone package; standard-library-only and protected by import-boundary tests. |
| `packages/agentic-backend-api/` | Backend capabilities, action results, and the backend adapter protocol. | `agentic_backend_api`. | Standalone package; depends only on `agentic_core`. |
| `packages/agentic-controller/` | Controller policy, lifecycle state, SJF, modes, workload profiles, and runtime estimation/calibration. | `agentic_controller`. | Standalone package; imports neither harnesses nor SGLang. |
| `packages/agentic-harnesses/` | Harness signal normalization and hint-benchmark implementation. | `agentic_harnesses`. | Standalone package; imports neither the controller nor SGLang. |
| `packages/agentic-hardware-probes/` | Paired-sample comparison contracts for controlled GPU bottleneck experiments. | `agentic_hardware_probes`. | Standalone standard-library package; no backend, harness, or controller imports. |
| `sglang_direct_kv/src/agentic_kv/controller/` | Compatibility imports plus gateway and targeted-prefetch backend adapters. | Compatibility layer; adapters move to the backend package in Phase 4. | Existing experiment imports remain valid. |
| `sglang_direct_kv/src/agentic_kv/hint_benchmark/` | Compatibility imports for the extracted hint benchmark. | Compatibility layer. | Existing CLI and test imports remain valid. |
| `sglang_direct_kv/src/agentic_kv/harness_scenarios/` | Synthetic and real scenario execution, policies, adapters, metrics, and reports. | Runner pieces to `agentic_experiments`; signal adapters to `agentic_harnesses`; backend adapter to backend package; reports to `agentic_reports`. | Mixed ownership and duplicated at top level. Do not extend both copies. |
| `agentic_harness_scenarios/src/agentic_harness_scenarios/` | Smaller portable copy of the harness-scenario framework. | Temporary migration source for `agentic_experiments`. | Overlaps the in-tree copy and lacks its `real_runner.py`; consolidate behind one canonical package. |
| `sglang_direct_kv/src/agentic_kv/sglang_adapters/` | Version selection, hook targets, raw-event mappings, and capability inspection. | `agentic_backends.sglang`. | Correct architectural idea; broaden into the only SGLang integration boundary. |
| `sglang_direct_kv/src/agentic_kv/sglang_compat.py` | Private SGLang/RadixCache compatibility mutations. | `agentic_backends.sglang.compat`. | High version-coupling; isolate and test per supported version. |
| `sglang_direct_kv/src/agentic_kv/sglang_trace_patch.py` | Runtime hook installation and SGLang telemetry interception. | `agentic_backends.sglang.telemetry`. | High version-coupling. Raw events must be normalized before leaving the adapter. |
| `sglang_direct_kv/src/agentic_kv/sglang_client.py` | SGLang request client. | Generic backend HTTP client plus SGLang request serializer in `agentic_backends.sglang`. | Split transport from SGLang payload semantics. |
| `sglang_direct_kv/src/agentic_kv/hints.py` | Hint representation and handling. | `agentic_core` contracts plus harness-specific normalization in `agentic_harnesses`. | Review field provenance before extraction. |
| `sglang_direct_kv/src/agentic_kv/agent_trace.py`, `policies.py`, `metrics.py` | Workload trace loading, legacy experiment policy, and metrics output. | `agentic_experiments` and `agentic_reports`. | Experiment support, not core controller behavior. |
| `sglang_direct_kv/src/agentic_kv/runtime_telemetry.py`, `instrumentation.py`, `nvtx.py`, `torch_cuda_profiler.py` | Telemetry transport and optional profiling. | Generic pieces in backend API/experiments; CUDA and SGLang-specific pieces in backend adapter. | Separate normalized telemetry from collection mechanism. |
| `sglang_direct_kv/src/agentic_kv/block_ledger/`, `evidence_schema.py`, `evidence_audit.py` | KV-event normalization, ledger construction, and evidence validation. | Normalized schemas in `agentic_core`; transformations and rendering in `agentic_reports`; raw maps in SGLang adapter. | Valuable boundary, but `normalizer.py` currently imports SGLang raw-event maps. |
| `packages/agentic-prompt-codec/` | Codec interfaces, rules, tokenizers, validation, and reporting. | Standalone `agentic_prompt_codec`. | Independent optional package with tokenizer extras. |
| `sglang_direct_kv/src/sitecustomize.py` | Automatic SGLang patch installation. | `agentic_backends.sglang.bootstrap`. | Deployment-sensitive and implicit; replace with explicit opt-in bootstrap after compatibility coverage exists. |

### Direct SGLang Coupling Sites

> **Resolved 2026-09-23.** All files below were moved into
> `packages/agentic-backend-sglang`. The only remaining testbed file that
> imports SGLang is `scripts/smoke_priority_radix_eviction.py` (tracked by
> `tests/architecture/legacy_sglang_internal_users.json`). The text below is
> the original audit, kept for history.

The current direct `sglang`/`sglang.srt` imports are concentrated in:

- `sglang_direct_kv/src/agentic_kv/sglang_compat.py`
- `sglang_direct_kv/src/agentic_kv/sglang_adapters/capabilities.py`
- `sglang_direct_kv/src/agentic_kv/sglang_adapters/v0510.py`
- `sglang_direct_kv/scripts/probe_sglang_kv_paths.py`
- `sglang_direct_kv/scripts/smoke_priority_radix_eviction.py`

This is encouraging: the coupling is not spread uniformly across the project.
These files become the initial contents or tests of `agentic_backends.sglang`.
Other files that refer to raw SGLang event names, filesystem paths, HTTP fields,
or shell launch flags are also adapter consumers even when they do not import
the Python package directly.

### Script Families

| Current scripts | Target owner | Rule |
| --- | --- | --- |
| `run_*realistic.sh`, `run_harness_*`, `run_multi_harness_replay_driver.py`, `run_*workload.py` | `agentic_experiments` | Become thin CLI wrappers around reusable experiment APIs. |
| `run_sglang_server.sh`, `run_sglang_hicache_server.sh`, `probe_sglang_*`, `smoke_*radix*` | `agentic_backends.sglang` and deployment | May contain SGLang flags and internals; must not define controller behavior. |
| `harness_sglang_gateway.py`, `nemo_agent_toolkit_wrapper.py`, `openai_proxy_logger.py`, `run_hint_benchmark.py` | Split between `agentic_harnesses`, backend adapters, and experiment CLI. | Capture native fields before normalization and keep provenance explicit. |
| `build_*report.py`, `build_*audit.py`, `summarize_*`, `plot_design_space.py`, `analyze_hint_outcomes.py` | `agentic_reports` | Read normalized artifacts; backend-specific parsing happens before this layer. |
| `collect_run_environment.py`, `sample_gpu_utilization.py`, profiler correlation scripts | Experiment instrumentation. | Emit normalized observations and declare instrumentation cost in the run manifest. |
| `run_milestone*.sh` and milestone-specific report scripts | Legacy/reproducibility wrappers. | Freeze for reproducibility; do not use as foundations for new APIs. |
| Prompt codec and trajectory scripts | `agentic_prompt_codec` tools. | Keep independent from controller and backend packages. |
| `setup_nvidia_a10g_24gb.sh`, `setup_nvidia_gh200_96gb.sh` | Deployment. | Parameterize package/backend versions; no experiment policy. |

### Configuration Ownership

| Current config | Target owner |
| --- | --- |
| `configs/hint_benchmark/*_hints.json`, `*_knobs.json`, `*_scenarios.json` | `agentic_harnesses` benchmark data. |
| `configs/harness_scenarios/` | `agentic_experiments`. |
| `configs/hardware/*.env`, `g5_2xlarge_smoke.yaml` | Deployment profiles and experiment manifests. |
| `configs/prompt_codecs/` | `agentic_prompt_codec`. |

Machine profiles may select values such as endpoint, GPU name, model, concurrency,
container image, and backend version. They must not select different Python
implementations of controller policy.

## Known Boundary Problems

> Status 2026-09-23: 3 (duplicate harness scenarios), 4 (gateway adapters
> under the controller) and 6 (implicit patch install) are resolved; 2 (driver)
> and 5 (reports reading raw event names) are partially resolved -- see
> docs/architecture/README_RESTRUCTURING.md section 9.

1. `agentic_kv` is currently an umbrella package containing controller,
   harness, backend, experiment, telemetry, and reporting concerns.
2. `run_multi_harness_replay_driver.py` combines workload construction,
   controller modes, harness behavior, backend requests, trace collection, and
   artifact output. It is the highest-value orchestration split.
3. Harness scenarios exist in both `agentic_harness_scenarios/` and
   `agentic_kv/harness_scenarios/`. New functionality can silently diverge.
4. Gateway adapters live under the controller even though lowering decisions is
   a backend responsibility.
5. Some report and ledger code understands raw SGLang/HiCache event names.
6. `sitecustomize.py` installs patches implicitly, making runtime behavior depend
   on environment/path setup.
7. Milestone scripts are useful historical reproductions but are difficult to
   treat as stable public interfaces.
8. Generated artifacts and external datasets sit near source code; package code
   must never rely on their local presence.

## SGLang Version Isolation

Each supported SGLang release gets an adapter selected through capability
probing, not scattered version checks. An adapter owns:

- request-field lowering;
- supported scheduler and KV actions;
- hook targets and private API paths;
- raw-to-normalized event mappings;
- launch flags and required environment variables;
- capability limitations;
- compatibility tests and recorded fixtures.

The adapter must fail clearly when a required capability is absent. It must not
silently claim an action occurred when it only recorded intent. Observe-only,
gateway-level, and internal-backend actions remain distinct in command results.

Upgrading SGLang should therefore require:

1. add or update one adapter;
2. run adapter contract tests against the new version;
3. verify normalized traces are unchanged or deliberately schema-versioned;
4. run a small reference experiment;
5. leave controller and harness packages untouched.

## Compatibility Test Gates

Before moving runtime files, add these gates:

- core schema serialization and backward-compatibility tests;
- controller tests using a fake backend and synthetic normalized signals;
- harness-adapter fixtures proving native field, provenance, and attachment
  level without a controller or SGLang process;
- backend-adapter contract tests for every declared capability;
- raw SGLang trace fixtures that normalize to stable observations;
- experiment golden manifests proving identical modes, seeds, and workloads;
- report tests using only normalized fixture data;
- one end-to-end remote A10G host reference run preserving current report metrics.

## Migration Sequence

### Phase 1: Architecture map and ownership freeze

- Maintain this inventory and dependency policy.
- Do not add new SGLang imports outside the adapter boundary.
- Do not add new functionality to both harness-scenario copies.
- Record ambiguous ownership here before moving code.

Status: complete when this document is reviewed and linked from the repository
README. No runtime behavior changes in this phase.

### Phase 2: Extract stable contracts

- Create `agentic_core` and `agentic_backend_api`.
- Move or adapt schemas without changing serialized output.
- Add schema and fake-backend contract tests.
- Provide temporary compatibility imports from `agentic_kv`.

Status: implemented. Existing `agentic_kv.controller` imports re-export the new
contract classes, preserving class identity and serialized controller schema
`agentic_controller.v1`. Current experiment runners have not been migrated to
the new normalized harness/request/observation contracts yet; that adoption
belongs to later phases.

### Phase 3: Separate controller and harness packages

- Move controller policy/estimation/state to `agentic_controller`.
- Move native harness capture and hint benchmarking to `agentic_harnesses`.
- Prove neither package imports the other or SGLang.
- Preserve current CLI commands through wrappers.

Status: implemented. Controller policy/state/estimation code now lives in
`agentic_controller`; harness signal normalization and hint-benchmark code now
live in `agentic_harnesses`. Dependency-boundary tests enforce their separation,
and the old `agentic_kv` paths remain compatibility wrappers. Concrete gateway
and SGLang-targeted adapters intentionally remain in the compatibility package
until Phase 4.

### Phase 3.5: Promote portable packages and centralize documentation

- Move backend-neutral packages to top-level `packages/` directories.
- Give every portable package independent build metadata.
- Install packages in dependency order through `scripts/install_workspace.sh`.
- Preserve historical commands through `agentic_kv` compatibility imports.
- Move substantive README content and project PDFs under `docs/`.

Status: implemented. The root README is intentionally a small landing page;
`docs/index.md` is the documentation directory. remote A10G host and GH200 setup scripts use
the workspace installer, so deployment does not rely on the old nested source
layout.

### Phase 4: Consolidate SGLang integration

- Move all SGLang imports, raw event maps, request lowering, and patch bootstrap
  into `agentic_backends.sglang`.
- Add adapter versions and contract tests.
- Replace implicit patch activation with an explicit backend bootstrap while
  retaining a compatibility wrapper for existing commands.

### Phase 5: Split experiments and reports

- Decompose the multi-harness driver into workload, harness, controller,
  backend, trace, and artifact services.
- Move report builders onto normalized schemas.
- Freeze milestone scripts as historical wrappers.

### Phase 6: Deployment portability

- Make local, remote A10G host, GH200, and Docker launch paths consume the same run manifest.
- Pin tested dependency/backend combinations without embedding them in core.
- Add setup validation and a portable minimal reference run.

### Phase 7: Remove compatibility shims

- Remove old `agentic_kv` import paths only after all maintained commands and
  reports use the new packages.
- Archive or mark superseded scripts.
- Publish the supported backend/harness/version matrix.

## Migration Guardrails

- existing experiment commands remain authoritative through compatibility
  wrappers;
- generated artifacts are not rewritten by package moves;
- SGLang launch behavior changes only inside deployment/backend boundaries;
- unrelated worktree changes remain untouched;
- compatibility shims remain until maintained commands use public package APIs.

## Definition of Portable

The architecture is portable when all of the following are true:

- controller unit tests run without SGLang or harness clients installed;
- harness capture tests run without SGLang installed;
- report tests run from normalized fixtures without a live backend;
- changing SGLang versions changes only an SGLang adapter and deployment pin;
- the same experiment manifest can run on remote A10G host or GH200 with a different machine
  profile;
- unsupported capabilities are reported as unsupported, not simulated silently;
- actual, injected, normalized, and simulated signals retain distinct provenance;
- old experiment commands continue to work through compatibility wrappers during
  migration.
