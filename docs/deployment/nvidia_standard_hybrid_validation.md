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

## Validated Reference

The first completed standard-NVIDIA hybrid run used:

```text
run label: scenario1_hybrid_reference_20260925_r6
SGLang image: agentic-sglang-standard:0.5.10.post1
adapter: v0510
model: Qwen/Qwen2.5-Coder-7B-Instruct
```

It completed both equal-importance Scenario 1 modes with 32 replay requests:

| Mode | Average replay TTFT | Average replay lateness | Total TTFT | Total replay debt |
| --- | ---: | ---: | ---: | ---: |
| `no_prefetch` | 2.70 s | 6.77 s | 86.34 s | 216.67 s |
| `controller_ready_time_gpu_backfill` | 2.03 s | 3.10 s | 65.12 s | 99.33 s |

The controller improvement is therefore not an application-priority result:
all replays used the same importance class. It came from ordering work with
replay-ready-time information and filling GPU gaps accordingly.

The container launcher also enforces three practical boundary rules proven by
this run:

- Container server processes use the image's `python3`, never the host
  virtual-environment interpreter.
- PyTorch and FlashInfer use a writable temporary cache inside the container.
- Files written into the mounted run directory are writable by the host user
  by way of a container `umask 000`, so the host driver can append trace
  records created by the container.

The labeled `master_report.html`, `evidence_tables.html`, runtime contract,
and run manifest are the evidence for this reference. Keep later validation
runs in a new label; do not overwrite this one.
