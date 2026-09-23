# SGLang testbed (`sglang_direct_kv`)

The lab that runs agentic replay-deadline experiments against a real SGLang
server. After the 2026-09-23 restructuring it holds only testbed-specific
material; reusable code lives in `packages/`.

| Path | What it is |
| --- | --- |
| `scripts/*.sh` | Experiment entry points (`run_harness_deadline_pressure.sh`, `run_*_realistic.sh`, master-report pipeline), SGLang server launch, EC2/GH200 setup |
| `scripts/*.py` | Thin wrappers that keep `python scripts/<name>.py` working; the code is in `agentic_experiments`, `agentic_reports` or `agentic_backends.sglang.tools` (see `scripts/README.md`) |
| `scripts/legacy/` | Frozen milestone scripts, kept for reproducing old results |
| `configs/` | Experiment settings: hardware profiles, hint-benchmark scenarios, harness scenarios, prompt codecs |
| `src/sitecustomize.py` | Installs the SGLang trace hooks when the server starts |
| `src/agentic_kv/` | Compatibility layer only: every module is an alias of (or re-exports) its new package location |
| `tests/` | Testbed, behavior-preservation (golden) and backward-compatibility tests |
| `artifacts/` | Run outputs (git-ignored, large) |

Where the code went:

| Package | Contents |
| --- | --- |
| `packages/agentic-experiments` | replay driver, runners, harness gateway, workloads, run-environment capture, prompt-codec evaluation |
| `packages/agentic-reports` | report builders, summaries, audits, KV block ledger, evidence audit |
| `packages/agentic-backend-sglang` | everything that touches SGLang internals (adapters, trace hooks, `tools/`) |

Documentation: `docs/testbeds/sglang_direct_kv.md` (long-form history),
`docs/sglang_direct_kv/` (investigation notes),
`docs/architecture/README_RESTRUCTURING.md` (why and how the code moved).
