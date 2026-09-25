# Infrastructure Boundaries

`infra/remote/` contains provider-neutral transfer and host-readiness helpers.
Connection details belong in `~/.config/agentic_hardware/remote.env`, outside
the repository.

`infra/accelerator/gh200/` contains optional setup and launch helpers for an
NVIDIA GH200 system. These helpers select a capability profile; controller and
harness packages do not import them.

Hardware behavior is selected through `HARDWARE_PROFILE`, not host identity.
See `sglang_direct_kv/configs/hardware/`.
