# Documentation Index

Everything except the landing page (`README.md`) and the active hand-off
(`HANDOFF.md`) lives under `docs/`.

## Start here

- [Current handoff](../HANDOFF.md) (repository root)
- [Project overview](project/overview.md)
- [SGLang-portability restructuring](architecture/README_RESTRUCTURING.md) -- read before changing code
- [Architecture map](architecture/ARCHITECTURE_MAP.md)

## Architecture (`docs/architecture/`)

- [SGLang-portability restructuring](architecture/README_RESTRUCTURING.md)
- [Architecture map](architecture/ARCHITECTURE_MAP.md)
- [SGLang backend package](../packages/agentic-backend-sglang/README.md) and its [compatibility matrix](../packages/agentic-backend-sglang/COMPATIBILITY.md)

## Deployment (`docs/deployment/`)

- [EC2](deployment/aws.md)
- [GH200](deployment/gh200.md)

## Testbeds (`docs/testbeds/`)

- [SGLang direct KV testbed](testbeds/sglang_direct_kv.md)
- [Harness-aware scenarios (package)](testbeds/agentic_harness_scenarios.md)
- [Harness-aware scenarios (experiment design)](testbeds/HARNESS_AWARE_SCENARIOS.md)

## Hint benchmark (`docs/hint_benchmark/`)

- [Hint benchmarking suite](hint_benchmark/HINT_BENCHMARKING_SUITE.md)
- [Hint benchmark runbook](hint_benchmark/HINT_BENCHMARK_RUNBOOK.md)
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
  `gh200/download.sh` and `run_harness_deadline_pressure.sh` refresh these copies.
- `docs/presentations/`: slide decks and their rendered assets.

## Archive (`docs/archive/`)

Kept in git for history; nothing imports from here.

- `backups/`: older copies of the master report and scenario tracker (formerly top-level `backups/`)
- `codex-build/`, `codex_build/`: Codex slide/report build scratch (formerly `.codex-build/`, `.codex_build/`)
- `last_conversation_verbatim.pdf`
