# Agentic Hardware

Portable infrastructure for harness-aware scheduling, KV-cache control, and
agentic inference experiments.

## Four Research Lanes

- [Controller Policy Results](CONTROLLER_POLICY_RESULTS.html): controller scheduling,
  KV-cache, replay deadline, TTFT, and workload-duration results.
- [Harness Signal Benchmark](HARNESS_SIGNAL_BENCHMARK.html): reproducible harness
  signal-emission experiments and the commands used to observe them.
- [GPU Interference](GPU_INTERFERENCE.html): controlled GPU
  compute and memory-movement measurements, with measured and planned cases
  clearly separated; see [the lane protocol](docs/hardware_bottlenecks/README.md).
- [KV Lifecycle Audit](KV_LIFECYCLE_AUDIT.html): when cache work happens, whether a
  replay reuses it, and which opportunities remain unproven; see
  [the validation protocol](docs/work_audit/README.md).

The active Scenario 1 controller experiment gives every request equal application
importance. Harnesses expose expected tool-return times; the controller derives
temporary queue ranks from those times. Earlier front-end priority experiments
are historical evidence, not part of this protocol. Use
`infra/container/run_scenario1_hybrid_reference.sh` for its locked reproduction.
For GH200, start with [the handoff](HANDOFF.md) and
[the GH200 guide](docs/deployment/nvidia_gh200_96gb.md); its first live run is
`infra/accelerator/gh200/run_sentinel.sh`. The ARM64 image and live GPU run
still need validation on that machine. Confirm that the intended GitHub
repository's `main` includes these changes before cloning it there.

- [Architecture contract](ARCHITECTURE.md)
- [Shared SGLang instrumentation](INSTRUMENTATION.md)
- [Current handoff](HANDOFF.md)
- [Project overview](docs/project/overview.md)
- [SGLang-portability restructuring (start here if you change code)](docs/architecture/README_RESTRUCTURING.md)
- [Architecture map](docs/architecture/ARCHITECTURE_MAP.md)
- [SGLang compatibility matrix](packages/agentic-backend-sglang/COMPATIBILITY.md)
- [Documentation index](docs/index.md)
- [Remote host setup](docs/deployment/remote_host.md)
- [NVIDIA GH200 setup](docs/deployment/nvidia_gh200_96gb.md)

Install the complete development workspace with:

```bash
bash scripts/install_workspace.sh
```

Run every check that needs no GPU or SGLang with:

```bash
bash scripts/check_portability.sh
```

## Repository layout

```text
HANDOFF.md          active hand-off for the next agent/engineer
docs/               all other documentation (index: docs/index.md), reports, and presentations
packages/           all code: portable packages, agentic-experiments, agentic-reports;
                    hardware probes; SGLang-specific code only in agentic-backend-sglang
sglang_direct_kv/   SGLang testbed: shell entry points, script wrappers, configs, tests
scripts/            workspace install and portability checks
tests/              architecture (boundary) tests
infra/              remote-host and accelerator deployment helpers
```
