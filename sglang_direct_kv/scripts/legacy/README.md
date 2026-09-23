# Legacy experiment scripts (frozen)

These scripts reproduce earlier milestones (milestones 4-40 of the direct-KV
testbed). Nothing in the current workflow calls them: they were moved here
because no active entry point (`scripts/*.sh` without "milestone" in the name,
or any script not referenced by another script) reaches them.

Rules:

- Run them from `sglang_direct_kv/`, e.g. `bash scripts/legacy/run_milestone6_design_space.sh <model>`,
  exactly as before (paths inside were updated to `scripts/legacy/`).
- Do not build new work on them; do not add new files here.
- Milestone-named scripts that are still used by the active master-report
  pipeline (`run_milestone11/12/21/22/23/24/26/27/36/9_agentic_traffic`,
  `build_milestone27_controlled_replay_report.py`,
  `summarize_milestone12_paired_evidence.py`) stay in `scripts/`.

See `docs/architecture/README_RESTRUCTURING.md` (testbed split) for how the
active/legacy split was computed.
