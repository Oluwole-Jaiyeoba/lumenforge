# Agentic Hardware Architecture

Read this file before changing package boundaries, deployment topology,
controller behavior, harness integrations, SGLang adapters, or experiment
artifacts. It is the concise architectural contract for engineers and agents
working on this repository.

Detailed restructuring history lives in
[`docs/architecture/README_RESTRUCTURING.md`](docs/architecture/README_RESTRUCTURING.md).
The full package inventory lives in
[`docs/architecture/ARCHITECTURE_MAP.md`](docs/architecture/ARCHITECTURE_MAP.md).

## Main Workstreams

This repository contains two related but distinct research workstreams:

1. **Controller experiments** study how harness knowledge and backend
   observations can improve scheduling, replay deadlines, KV-cache management,
   TTFT, and total workload duration. The human-readable record is
   [`CONTROLLER_EXPERIMENTS.html`](CONTROLLER_EXPERIMENTS.html).
2. **Hint benchmarking** determines which signals real harnesses expose, when
   they appear, where they attach, and how to reproduce them. The operator
   entry point is [`HINT_BENCHMARK_RUNBOOK.html`](HINT_BENCHMARK_RUNBOOK.html).

Hint-emission evidence is not controller-performance evidence. Keep their
claims, runs, and reports separate even when they share harness adapters.

## Implemented Package Boundaries

The portable package split is implemented and enforced by architecture tests:

| Package | Responsibility |
| --- | --- |
| `agentic-core` | Stable schemas, identities, signals, commands, and observations. |
| `agentic-harnesses` | Native harness capture, hint benchmarking, provenance, and normalization. |
| `agentic-controller` | Backend-neutral scheduling, admission, estimation, KV, and eviction policy. |
| `agentic-gateway` | Backend-neutral request translation between harnesses and backends. |
| `agentic-backend-api` | Abstract backend capabilities and command interfaces. |
| `agentic-backend-sglang` | The only package allowed to know SGLang internals or version drift. |
| `agentic-experiments` | Composition and experiment orchestration. |
| `agentic-reports` | Normalized analysis, audits, and report generation. |

The dependency direction is:

```text
Harness clients -> agentic-harnesses -> agentic-core
                                      -> agentic-gateway

Harness facts + backend observations -> agentic-controller
Controller commands -> agentic-backend-api -> agentic-backend-sglang -> SGLang

agentic-experiments composes the path.
agentic-reports reads normalized artifacts from the path.
```

Rules:

- Controller code must not import a harness SDK or SGLang.
- Harness code must not import controller policy or SGLang.
- SGLang imports, private APIs, hook targets, flags, and raw event names belong
  only in `agentic-backend-sglang` or deployment code.
- Reports should consume stable normalized records rather than private SGLang
  structures.
- `sglang_direct_kv/` is a testbed and compatibility surface, not the owner of
  new portable policy.

## Approved Host And Container Topology

The following is the approved target for GPU machines. The package separation,
SGLang adapter boundary, runtime profiles, normalized capability handshake,
and immutable run-manifest contract are implemented. GPU support is only
claimed after that profile passes the validation gates below.

```text
Host
  real harness clients and their credentials
  agentic-harnesses
  agentic-controller
  agentic-gateway
  experiment orchestration
  report generation
             |
             | versioned HTTP/JSON contracts
             v
GPU container
  pinned SGLang checkout or image
  vendor GPU runtime
  agentic-backend-sglang adapter
  backend-side trace hooks and scheduler/KV instrumentation
             |
             v
Shared read-only/write mounts
  model cache / run configuration / artifacts
```

### Controller experiments

The controller remains on the host. It reasons from normalized harness facts
and backend observations, then sends backend-neutral commands through the
gateway. SGLang and all instrumentation that observes SGLang internals live in
the GPU container. Backend traces are written to the mounted run artifact
directory for host-side reporting.

### Hint benchmark suite

Real harness clients remain on the host so their native runtimes,
authentication, and request behavior are preserved. The container-side
benchmark boundary captures and validates the request near the backend and may
forward it to SGLang when the scenario requires execution. Fixture or injected
signals must never be reported as native harness emissions.

Credentials stay on the host. Do not bake credentials, user configuration, or
machine-specific paths into an image.

## Backend Runtime Profiles

Do not build one universal SGLang image. Maintain independently pinned runtime
profiles for materially different GPU stacks, for example:

```text
nvidia-gh200
nvidia-standard
amd-rocm
```

Each runtime profile must record:

- GPU vendor, architecture, and host architecture;
- container image or Dockerfile and immutable image identity (a registry
  digest for published images, or a Docker image ID during local bring-up);
- exact SGLang version or commit;
- backend adapter selection;
- model-cache and artifact mounts;
- launch flags and environment requirements;
- supported and unsupported backend capabilities.

Hardware profiles may choose values and capabilities. They must not choose a
different Python implementation of controller policy.

Runtime profiles live in `configs/backend_runtimes/`. The common container
preflight is `infra/container/probe_sglang_runtime.sh`. It emits
`agentic_backend_runtime.v1`, including both normalized capabilities and the
raw backend-specific probe for debugging. The runtime record distinguishes a
published-image digest from a local Docker image ID, so an agent can validate
a newly built image before it is published.

## Stable Boundary And Capability Handshake

Host/container communication must use versioned project-owned schemas. The
minimum contract should cover:

- request, session, task, and replay identity;
- harness facts and their provenance;
- expected replay-ready time and deadline;
- controller commands and decision identifiers;
- backend action acknowledgements and effect level;
- scheduler, KV, timing, and hardware observations.

At startup, the backend must expose a health and capability record containing
its runtime, version, adapter, and supported actions. The host must fail early
when a requested experiment needs an unsupported capability. Recording intent
is not proof that the backend acted.

## Reproducible Run Contract

Every reference experiment should produce one immutable run manifest containing:

- source commit;
- controller and schema versions;
- container image digest;
- SGLang version or commit and selected adapter;
- hardware/runtime profile;
- model and launch configuration;
- harnesses, modes, seed, and workload parameters;
- enabled instrumentation and its expected cost;
- declared backend capabilities;
- artifact paths and completion status.

The host and container write into one run-specific mounted artifact directory.
Model caches are mounted separately and are not copied into run artifacts.

`scripts/create_run_manifest.py` writes `agentic_run_manifest.v2`. The GH200
host/container runner performs the runtime preflight and writes this manifest
before it starts the experiment matrix.

For development, source may be bind-mounted read-only for rapid iteration. A
reference experiment should use pinned package and image identities.

## SGLang Version Changes

To support another SGLang release:

1. Probe the release with the existing static compatibility checker.
2. Add or update one version adapter under `agentic-backend-sglang`.
3. Update the runtime profile and capability record.
4. Run adapter, architecture, golden, and report tests.
5. Run one minimal GPU sentinel.
6. Reproduce one validated controller reference experiment.

Do not add scattered SGLang version checks to controller, harness, experiment,
or report code.

## Required Validation Gates

Before treating a deployment as supported, require:

```bash
bash scripts/install_workspace.sh
bash scripts/check_portability.sh
```

Then run, in order:

1. backend health and capability probe;
2. host-to-container contract smoke test;
3. native harness boundary smoke test;
4. minimal GPU sentinel;
5. one reference controller experiment;
6. report and artifact validation.

For the GH200 deployment path, begin with
[`docs/deployment/nvidia_gh200_96gb.md`](docs/deployment/nvidia_gh200_96gb.md)
and follow the explicit
[`GH200 agent handoff checklist`](docs/deployment/GH200_AGENT_HANDOFF.md).

## Change Checklist

Before committing an architectural change, confirm:

- Does it preserve the two workstreams and their evidence boundaries?
- Is portable logic outside the SGLang adapter and deployment layers?
- Is machine/vendor/version variation represented as configuration or an
  adapter rather than policy duplication?
- Are unsupported capabilities explicit?
- Can the run be reconstructed from its manifest?
- Are native, provider/config, injected, fixture, and backend-confirmed
  evidence still distinguishable?
- Did the portability tests and the smallest relevant end-to-end test pass?
