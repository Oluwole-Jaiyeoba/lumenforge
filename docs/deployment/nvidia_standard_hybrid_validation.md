# Standard NVIDIA Hybrid Validation

Use this first on a conventional NVIDIA CUDA machine before moving the same
hybrid setup to GH200. The host runs the harness, controller, gateway, and
report builder. Docker runs only SGLang and its GPU-specific dependencies.

## What This Proves

This validation runs the known Scenario 1 reference without application-level
priority classes:

```text
harness: Hatcher / DeepAgents
pressure: P3
modes: no_prefetch, controller_ready_time_gpu_backfill
controller: ready-time scheduling only
```

All replays have equal importance. The controller can order replay work by
expected readiness time; it does not receive a user-priority class.

## Prepare The Image

First probe an existing explicit image. If it reports SGLang `0.5.10.post1`
and adapter `v0510`, reuse it. Otherwise build the backend-only image:

```bash
cd /path/to/agentic_hardware

BACKEND_RUNTIME_PROFILE=nvidia_standard \
SGLANG_RUNTIME_TAG=agentic-sglang-standard:0.5.10.post1 \
bash infra/container/build_sglang_runtime.sh
```

## Run The Reference

Set the host model cache and the explicit image tag. The cache is mounted into
the SGLang container; it is not copied into the image or artifact directory.

```bash
cd /path/to/agentic_hardware

export AGENTIC_MODEL_CACHE=/path/to/model_cache
export SGLANG_DOCKER_IMAGE=agentic-sglang-standard:0.5.10.post1

bash infra/container/run_scenario1_hybrid_reference.sh
```

The launcher first writes `runtime/backend_runtime.json` and
`runtime/run_manifest.json`, then starts the normal host-side Scenario 1
driver. SGLang starts through Docker with host networking, so the host gateway
reaches it through the ordinary `127.0.0.1:30000` backend endpoint.

## Acceptance Criteria

- Runtime contract reports `probe_ok: true`, SGLang `0.5.10.post1`, and `v0510`.
- The server log says SGLang launched in the configured Docker image.
- The report contains both Scenario 1 modes and retains the equal-importance
  priority contract.
- The artifact directory contains the runtime contract, run manifest, logs,
  traces, metrics, and report.

Do not require exact equality with historical TTFT or lateness numbers. The
goal is a clean completed hybrid run with a plausible comparison under the
same workload contract.
