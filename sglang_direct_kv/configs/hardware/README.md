# Hardware Capability Profiles

Each profile is a small environment file consumed by the experiment launcher.
It describes capacity and pressure scale only. Controller policy, harness
signals, and SGLang instrumentation remain independent of the profile.

| Profile | Intended accelerator | Runtime | Memory |
| --- | --- | --- | --- |
| `nvidia_a10g_24gb` | NVIDIA A10G class | CUDA | 24 GB |
| `nvidia_gh200_96gb` | NVIDIA GH200 | CUDA | 96 GB |
| `amd_mi300x_192gb` | AMD MI300X class | ROCm | 192 GB |

Select one with:

```bash
HARDWARE_PROFILE=nvidia_a10g_24gb \
bash sglang_direct_kv/scripts/run_harness_deadline_pressure.sh
```

Profiles can be overridden without editing the repository:

```bash
HARDWARE_PROFILE=none \
HARDWARE_PROFILE_PATH=/absolute/path/to/custom-profile.env \
bash sglang_direct_kv/scripts/run_harness_deadline_pressure.sh
```

Validate a new profile against a reference workload before treating its
pressure scale as comparable to another device.
