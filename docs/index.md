# Documentation Index

The repository root contains the active architecture, handoff, and two
human-readable workstream entry points. Supporting documentation lives under
`docs/`.

## Start here

- [Current handoff](../HANDOFF.md) (repository root)
- [Architecture contract](../ARCHITECTURE.md) (repository root)
- [Project overview](project/overview.md)
- [SGLang-portability restructuring](architecture/README_RESTRUCTURING.md) -- read before changing code
- [Architecture map](architecture/ARCHITECTURE_MAP.md)

## Architecture (`docs/architecture/`)

- [Top-level architecture contract](../ARCHITECTURE.md)
- [SGLang-portability restructuring](architecture/README_RESTRUCTURING.md)
- [Architecture map](architecture/ARCHITECTURE_MAP.md)
- [SGLang backend package](../packages/agentic-backend-sglang/README.md) and its [compatibility matrix](../packages/agentic-backend-sglang/COMPATIBILITY.md)
- [SGLang testbed layout](../sglang_direct_kv/README.md) and [where each script's code lives](../sglang_direct_kv/scripts/README.md)

## Deployment (`docs/deployment/`)

- [Remote host](deployment/remote_host.md)
- [NVIDIA GH200](deployment/nvidia_gh200_96gb.md)
- [GH200 agent handoff checklist](deployment/GH200_AGENT_HANDOFF.md)
- [Standard NVIDIA hybrid validation](deployment/nvidia_standard_hybrid_validation.md)

## Testbeds (`docs/testbeds/`)

- [SGLang direct KV testbed](testbeds/sglang_direct_kv.md)
- [Harness-aware scenarios (package)](testbeds/agentic_harness_scenarios.md)
- [Harness-aware scenarios (experiment design)](testbeds/HARNESS_AWARE_SCENARIOS.md)

## Hint benchmark (`docs/hint_benchmark/`)

- [Hint benchmarking suite](hint_benchmark/HINT_BENCHMARKING_SUITE.md)
- [Hint benchmark runbook source](hint_benchmark/HINT_BENCHMARK_RUNBOOK.md)
- [Hint benchmark runbook HTML](../HINT_BENCHMARK_RUNBOOK.html)
- [Hint signal findings](hint_benchmark/HINT_SIGNAL_FINDINGS.md)

## Packages (`docs/packages/`, `docs/compatibility/`)

- [Agentic controller](packages/agentic_controller.md)
- [Agentic harnesses](packages/agentic_harnesses.md)
- [Agentic prompt codec](packages/agentic_prompt_codec.md)
- Compatibility notes for the old `agentic_kv` paths: [controller](compatibility/agentic_kv_controller.md), [harness scenarios](compatibility/agentic_kv_harness_scenarios.md)

## Proposals (`docs/proposals/`)

- [Hardware proposal](proposals/HARDWARE_PROPOSAL.md)
- [Deadline/priority-aware migration engine](proposals/DEADLINE_PRIORITY_AWARE_MIGRATION_ENGINE.md)
- [KV page tagging](proposals/KV_PAGE_TAGGING.md)
- [Replay-path instrumentation](proposals/REPLAY_PATH_INSTRUMENTATION_PROPOSAL.md)
- [Hardware emulation environment](proposals/HARDWARE_EMULATION_ENVIRONMENT.md)

## SGLang research notes (`docs/sglang_direct_kv/`, `docs/research/`)

- [Effective runtime estimation](sglang_direct_kv/EFFECTIVE_RUNTIME_ESTIMATION.md)
- [Filler timing calibration](sglang_direct_kv/FILLER_TIMING_CALIBRATION.md)
- [Prompt codec](sglang_direct_kv/prompt_codec.md)
- [KV block ledger](sglang_direct_kv/KV_BLOCK_LEDGER.md), [exact KV movement attribution](sglang_direct_kv/KV_EXACT_MOVEMENT_ATTRIBUTION.md), [H2D bandwidth pressure](sglang_direct_kv/KV_H2D_BANDWIDTH_PRESSURE.md)
- [Replay delay breakdown](sglang_direct_kv/REPLAY_DELAY_BREAKDOWN.md), [deep replay instrumentation](sglang_direct_kv/REPLAY_DELAY_DEEP_INSTRUMENTATION.md), [instrumentation audit](sglang_direct_kv/INSTRUMENTATION_AUDIT.md)
- Research PDFs: `docs/research/`

## Reports and presentations

- `docs/reports/`: [latest master report](reports/latest_master_report.html),
  [harness-aware scenario tracker](reports/harness_aware_scenario_tracker.html),
  [replay friction deep dive](reports/replay_friction_deep_dive.html).
  `infra/accelerator/gh200/download.sh` and `run_harness_deadline_pressure.sh` refresh these copies.
- `docs/presentations/`: slide decks and their rendered assets.

## Historical artifacts

Historical report backups are preserved outside the repository so source and
current documentation stay portable. Nothing in the workspace imports them.
