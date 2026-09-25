# Infrastructure Boundaries

`infra/remote/` contains provider-neutral transfer and host-readiness helpers.
Connection details belong in `~/.config/agentic_hardware/remote.env`, outside
the repository.

`infra/accelerator/gh200/` contains optional setup and launch helpers for an
NVIDIA GH200 system. These helpers select a capability profile; controller and
harness packages do not import them.

Hardware behavior is selected through `HARDWARE_PROFILE`, not host identity.
See `sglang_direct_kv/configs/hardware/`.

## Portable Backend Containers

`infra/container/probe_sglang_runtime.sh` is the common preflight for GPU
containers. It reads a profile from `configs/backend_runtimes/`, runs the
SGLang capability probe inside that image, and writes a versioned runtime
handshake for the host-side experiment driver.

The controller, harness clients, credentials, orchestration, and reports stay
on the host. SGLang, its vendor GPU runtime, and SGLang-specific hooks stay in
the container. A reference run stops before loading a model when the observed
SGLang version, adapter, image identity, or required capabilities do not match
the selected runtime profile.

Current profiles:

| Profile | Status | Purpose |
| --- | --- | --- |
| `nvidia_gh200` | reference | NVIDIA GH200, ARM64 host |
| `nvidia_standard` | supported | Conventional NVIDIA CUDA hosts |
| `amd_rocm` | experimental | ROCm container boundary; image must be supplied explicitly |

`infra/container/Dockerfile.sglang` and
`infra/container/build_sglang_runtime.sh` provide the backend-only image
scaffold. Build the image natively on the target machine, then use the probe
to record its local Docker image ID or published registry digest before an
experiment is allowed to proceed.

Example preflight:

```bash
BACKEND_RUNTIME_PROFILE=nvidia_gh200 \
SGLANG_DOCKER_IMAGE='repository/image@sha256:digest' \
BACKEND_RUNTIME_CONTRACT_OUT="$PWD/artifacts/backend_runtime.json" \
bash infra/container/probe_sglang_runtime.sh
```
