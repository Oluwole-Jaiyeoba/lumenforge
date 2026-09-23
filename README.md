# Agentic Hardware

Portable infrastructure for harness-aware scheduling, KV-cache control, and
agentic inference experiments.

- [Project overview](docs/project/overview.md)
- [SGLang-portability restructuring (start here if you change code)](README_RESTRUCTURING.md)
- [Architecture map](ARCHITECTURE_MAP.md)
- [SGLang compatibility matrix](packages/agentic-backend-sglang/COMPATIBILITY.md)
- [Documentation index](docs/index.md)
- [EC2 setup](docs/deployment/aws.md)
- [GH200 setup](docs/deployment/gh200.md)

Install the complete development workspace with:

```bash
bash scripts/install_workspace.sh
```

Run every check that needs no GPU or SGLang with:

```bash
bash scripts/check_portability.sh
```
