# `sglang_direct_kv/scripts`

Every command documented anywhere in the repository still works from
`sglang_direct_kv/`, e.g. `python scripts/run_multi_harness_replay_driver.py ...`
or `bash scripts/run_harness_deadline_pressure.sh <model>`.

What lives here now:

- **Shell entry points** (`*.sh`): experiment orchestration, SGLang server
  launch, remote A10G host/GH200 setup. These are the real code for the testbed.
- **Thin Python wrappers** (`*.py`, 57 files): the code moved into
  `packages/` on 2026-09-23. Each wrapper calls
  `_moved.run_or_alias(__name__, "<package.module>")`, which runs the package
  module as the program (identical `--help`, arguments and exit codes) or, when
  imported, makes the old module name an alias of the package module.
- **SGLang helpers kept as scripts**: `probe_sglang_capabilities.py`,
  `sglang_preflight.py` (already thin wrappers over `agentic_backends.sglang`).
- **`legacy/`**: frozen milestone scripts that no active entry point uses
  (see `legacy/README.md`).

New code goes into the packages, not here. Where each script's code lives:

## Experiment runners -> `agentic_experiments.runners`
`run_agentbench_sglang_preflight.py`, `run_agentbench_sglang_task.py`, `run_agentic_traffic_workload.py`, `run_harness_aware_scenarios.py`, `run_harness_scenario_claim.py`, `run_hint_benchmark.py`, `run_multi_harness_replay_driver.py`, `run_multi_session_agentic_replay.py`, `run_nemo_nat_service_priority_probe.py`, `run_priority_queue_jump_workload.py`, `run_real_client_wireability_probe.py`, `run_real_prompt_controlled_replay.py`, `run_workload.py`, `smoke_agentic_controller.py`, `smoke_multi_harness_wireability.py`

## Gateway and client-side processes -> `agentic_experiments.gateway`
`harness_sglang_gateway.py`, `live_prefetch_controller.py`, `nemo_agent_toolkit_wrapper.py`, `openai_proxy_logger.py`

## Workloads and timing models -> `agentic_experiments.workloads`
`evaluate_aiconfigurator_timing.py`, `extract_agentbench_trace_replay_workload.py`, `fit_filler_timing_model.py`, `generate_synthetic_replay_workload.py`

## Run environment capture -> `agentic_experiments.environment`
`collect_run_environment.py`, `sample_gpu_utilization.py`

## Prompt-codec evaluation -> `agentic_experiments.prompt_codec_eval`
`evaluate_prompt_codec.py`, `run_prompt_encoding_matrix.py`, `scan_prompt_shorthand_trajectories.py`

## Report builders -> `agentic_reports.builders`
`build_agentic_prefetch_timeline.py`, `build_controller_idle_gap_audit.py`, `build_filler_runtime_truth.py`, `build_filler_timing_calibration.py`, `build_live_agentbench_tool_gap_report.py`, `build_live_direct_kv_load_report.py`, `build_live_paired_agentbench_report.py`, `build_memory_admission_audit.py`, `build_milestone27_controlled_replay_report.py`, `build_multi_harness_deadline_summary.py`, `build_proactive_kv_correctness_audit.py`, `build_replay_friction_deep_dive.py`, `build_value_aware_eviction_audit.py`

## Summaries -> `agentic_reports.summaries`
`summarize_agentic_traffic_results.py`, `summarize_kv_trace.py`, `summarize_milestone12_paired_evidence.py`, `summarize_priority_queue_sanity.py`, `summarize_torch_cuda_profiles.py`

## Audits and validators -> `agentic_reports.audits`
`analyze_hint_outcomes.py`, `audit_master_report_evidence.py`, `inspect_backend_submit_gaps.py`, `validate_kv_block_ledger.py`, `validate_replay_path_classifier.py`

## Shared analysis -> `agentic_reports.analysis`
`correlate_torch_profile_with_agent_trace.py`, `replay_path_classifier.py`

## SGLang-internal tools -> `agentic_backends.sglang.tools`
`extract_sglang_kv_targets.py`, `investigate_hicache_reuse.py`, `probe_sglang_kv_paths.py`, `smoke_priority_radix_eviction.py`
