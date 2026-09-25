# Agentic Hardware

Portable infrastructure for harness-aware scheduling, KV-cache control, and
agentic inference experiments.

- [Current handoff](HANDOFF.md)
- [Milestone experiment registry](MILESTONE_EXPERIMENTS.html)
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
                    SGLang-specific code only in agentic-backend-sglang
sglang_direct_kv/   SGLang testbed: shell entry points, script wrappers, configs, tests
scripts/            workspace install and portability checks
tests/              architecture (boundary) tests
infra/              remote-host and accelerator deployment helpers
```
