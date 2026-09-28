# GH200 Setup And Run Guide

This is the current bring-up path for the equal-importance Scenario 1 controller
experiment. Read [HANDOFF.md](../../HANDOFF.md) and
[GH200_AGENT_HANDOFF.md](GH200_AGENT_HANDOFF.md) first. The older signal-family
and combined-mode commands are [archived](archive/nvidia_gh200_96gb_legacy.md);
they are not current reproduction instructions.

## Runtime Boundary

- GH200 host: DeepAgents harness, controller, gateway, credentials, orchestration,
  and reports.
- Docker container: pinned SGLang `0.5.10.post1`, CUDA runtime, the `v0510`
  adapter, and backend trace hooks.
- Shared mounts: model cache and run artifacts. Do not copy a virtual
  environment or a GPU image built on another architecture.

The active comparison is `no_prefetch` versus
`controller_ready_time_gpu_backfill` on the same seeded P3 workload. Every
session has equal application importance; the controller may derive queue
ranks from expected tool-return times. The experiment does not accept
front-end high/low priority classes.

## 1. Get The Source

Clone `main` on the GH200 after it has been published to the intended GitHub
repository. The local working copy is not a substitute for a published commit.
Confirm the checkout before any run:

```bash
git status --short
git branch --show-current
git log -1 --oneline
```

If transferring a local checkout instead, use
`infra/accelerator/gh200/sync_to_gh200.sh` from the source machine. Do not
transfer `.venv`, `.venvs`, caches, or old artifact trees.

## 2. Check The Host

Run from the repository root on GH200:

```bash
uname -m
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The architecture must be `aarch64` or `arm64`, and the Docker GPU check must
see the accelerator. Do not stop unrelated containers or GPU jobs.

Install the host dependencies if they are not already present:

```bash
INSTALL_SYSTEM_DEPS=0 bash sglang_direct_kv/scripts/setup_nvidia_gh200_96gb.sh
```

The host Python environment is `sglang_direct_kv/.venv`. It may contain
compatibility packages, but the serving process must run only in Docker.

## 3. Build And Probe SGLang

Build the ARM64 backend image on GH200. If the site requires a different
approved CUDA base, set `SGLANG_BASE_IMAGE` before building; keep the SGLang
version pinned.

```bash
BACKEND_RUNTIME_PROFILE=nvidia_gh200 \
SGLANG_RUNTIME_TAG=agentic-sglang-gh200:0.5.10.post1 \
bash infra/container/build_sglang_runtime.sh

export SGLANG_DOCKER_IMAGE=agentic-sglang-gh200:0.5.10.post1
BACKEND_RUNTIME_PROFILE=nvidia_gh200 \
BACKEND_RUNTIME_CONTRACT_OUT="$PWD/sglang_direct_kv/artifacts/gh200_runtime.json" \
bash infra/container/probe_sglang_runtime.sh
```

The probe must report `probe_ok: true`, SGLang `0.5.10.post1`, adapter
`v0510`, and an image identity. A probe failure is a stop condition; do not
change controller policy or workload parameters to work around it.

## 4. Run The Locked Scenario

Set the host model cache to a directory containing the model snapshot. The
sentinel also accepts the older `AGENTIC_GH200_MODEL_CACHE` name, but
`AGENTIC_MODEL_CACHE` is the common hybrid-run variable.

```bash
export AGENTIC_MODEL_CACHE="$HOME/dynamo_model_cache"
export SGLANG_DOCKER_IMAGE=agentic-sglang-gh200:0.5.10.post1
./infra/accelerator/gh200/run_sentinel.sh
```

The sentinel selects `nvidia_gh200`, runs DeepAgents at P3, and compares only
baseline with ready-time GPU backfill. Its strict instrumentation gate checks
the adapter hooks and a live model request before measuring the workload. The
post-run validator requires 32 deadline-bearing replays per mode, complete
timing samples, peer/normal harness signals, no frontend priority, and
controller-derived backend ranks on all RTG replays. A failed gate or validator
means the run is not evidence of controller performance.

The runner prints a `REPORT_LABEL`. Find the per-run report and proof files at:

```text
sglang_direct_kv/artifacts/results/reports/<REPORT_LABEL>/master_report.html
sglang_direct_kv/artifacts/results/reports/<REPORT_LABEL>/all_replay_summary.csv
sglang_direct_kv/artifacts/results/reports/<REPORT_LABEL>/instrumentation_preflight_summary.json
sglang_direct_kv/artifacts/results/runs/controlled/<REPORT_LABEL>/runtime/backend_runtime.json
sglang_direct_kv/artifacts/results/runs/controlled/<REPORT_LABEL>/runtime/run_manifest.json
```

Compare the shape and measurement coverage with the
[validated standard-NVIDIA reference](../reports/scenario1_equal_importance_20260928.html).
Do not expect identical latency numbers on a different GPU. GH200 performance
remains unverified until this run completes on that machine.
