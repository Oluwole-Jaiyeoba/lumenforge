# GH200 Agent Handoff Checklist

Use this checklist when bringing the repository to an NVIDIA GH200 machine.
Do not change controller policy, harness behavior, or experiment parameters to
solve a deployment problem. First make the backend runtime contract pass.
The source must be the published, up-to-date `main` commit (or a deliberate
source transfer); check `git log -1 --oneline` before proceeding. The current
GH200 path has scaffolding but has not yet been validated on that GPU.

## Architecture To Preserve

```text
GH200 host
  harness CLIs, credentials, controller, gateway, orchestration, reports
      |
      | HTTP/JSON plus mounted artifacts
      v
SGLang Docker container
  CUDA runtime, SGLang 0.5.10.post1, SGLang adapter, trace hooks
```

The host owns credentials and real harness behavior. The container owns GPU
libraries and all SGLang-version-specific code. The `packages/` source mount
is read-only; the testbed mount remains writable for run artifacts. The model
cache is a separate mount.

## Bring-Up Steps

1. Confirm the machine and Docker GPU runtime:

   ```bash
   uname -m                 # must report aarch64 or arm64
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

2. Install host-side dependencies. This prepares real harness CLIs and the
   controller. The legacy testbed environment may include SGLang Python
   packages for compatibility tooling, but the hybrid deployment must never
   launch an SGLang server from that host environment:

   ```bash
   cd /path/to/agentic_hardware
   INSTALL_SYSTEM_DEPS=0 bash sglang_direct_kv/scripts/setup_nvidia_gh200_96gb.sh
   ```

3. Build the backend-only ARM64 image. Set `SGLANG_BASE_IMAGE` if the site has
   an approved GH200 CUDA base image:

   ```bash
   cd /path/to/agentic_hardware
   BACKEND_RUNTIME_PROFILE=nvidia_gh200 \
   SGLANG_RUNTIME_TAG=agentic-sglang-gh200:0.5.10.post1 \
   bash infra/container/build_sglang_runtime.sh
   ```

4. Probe the image before attempting an experiment. This must report SGLang
   `0.5.10.post1`, adapter `v0510`, and `probe_ok: true`:

   ```bash
   BACKEND_RUNTIME_PROFILE=nvidia_gh200 \
   SGLANG_DOCKER_IMAGE=agentic-sglang-gh200:0.5.10.post1 \
   BACKEND_RUNTIME_CONTRACT_OUT="$PWD/sglang_direct_kv/artifacts/gh200_runtime.json" \
   bash infra/container/probe_sglang_runtime.sh
   ```

5. Read the generated runtime contract. If its version, adapter, or required
   capability is wrong, fix the image or adapter before running a workload.

6. Set the host model-cache location and run the locked equal-importance
   Scenario 1 sentinel. It compares only `no_prefetch` with
   `controller_ready_time_gpu_backfill` at P3; all sessions have equal
   application importance:

   ```bash
   export AGENTIC_MODEL_CACHE=/path/to/model_cache
   export SGLANG_DOCKER_IMAGE=agentic-sglang-gh200:0.5.10.post1
   ./infra/accelerator/gh200/run_sentinel.sh
   ```

7. Require a successful live instrumentation preflight and the final
   equal-importance validator: 32 replays per mode, no frontend priority, and
   controller-derived queue ranks on every RTG replay. Preserve the run's
   `runtime/backend_runtime.json`, `runtime/run_manifest.json`,
   `all_replay_summary.csv`, and `master_report.html` together. These prove
   which runtime and request contract produced the result.

## If Something Fails

- Image build failure: record the base image, CUDA version, full build log,
  and target architecture. Do not silently change the SGLang pin.
- Capability-probe failure: inspect the generated probe output and update only
  `agentic-backend-sglang` after verifying the target SGLang source surface.
- Host harness failure: repair the host CLI or its credentials; do not move
  that client into the container.
- Sentinel failure after a successful handshake: keep the runtime contract and
  compare its launch settings and artifacts against the known reference run.
- Do not launch the retired apples-to-apples, scaled-pressure, or combined-mode
  wrappers to diagnose the active Scenario 1 protocol; they intentionally stop.
