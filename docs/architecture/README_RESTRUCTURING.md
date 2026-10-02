# SGLang-Portability Restructuring (September 2026)

Current shared instrumentation entry point: [`INSTRUMENTATION.md`](../../INSTRUMENTATION.md).
The backend package alone owns SGLang hooks; the new neutral
`agentic-instrumentation` package defines evidence contracts for all lanes.

This document is the hand-off for the restructuring that made the codebase
independent of any single SGLang release. It is written for an engineer or AI
agent who was not part of the work. Read it before changing anything under
`packages/`, `sglang_direct_kv/src/`, the gateway, or the launch scripts.

- Branch: `refactor/sglang-portability`, fast-forward merged into `main` on 2026-09-23 (push to origin pending from the owner's machine; the session had no GitHub credentials)
- Base: `317b9a1` (main) + `491aea5` (checkpoint of the uncommitted Phase 3.5 work that was in the tree)
- Backup of the pre-refactor tree (code only, no artifacts):
  `~/Documents/GitHub/backup/agentic_hardware_pre_refactor_20260923/`

---

## 1. Goal and result

**Goal (from the project owner):** a highly modular, portable codebase that
does not depend on the SGLang underneath, so moving from one SGLang version to
another does not break things.

**Result in one paragraph:** every piece of code that knows SGLang internals
(module paths, class/method/attribute names, CLI flags, request wire fields)
now lives in one package, `packages/agentic-backend-sglang`
(`agentic_backends.sglang`). Per-release knowledge is plain data in
`versions/vXXXX.py`. A torch-free static checker compares any SGLang release
against that data, and it was run against every release from 0.5.10.post1 to
0.5.20. The controller, harness and gateway packages are proven
SGLang-free by tests that fail the build if that changes. Gateway and driver
logic that moved out of scripts is proven **byte-identical** to the old code
by golden tests. Upgrading SGLang is now: run the checker, add or adjust one
adapter file, run the checks, run one GPU reference experiment.

What was **not** done (and why) is in section 9. The original GPU verification
item is now complete; section 8 records the reference run and outcome.

---

## 2. Where things are now

**Top-level cleanup (2026-09-23, after the refactor):** the repository root
keeps only active entry-point documents/reports and directories. The current
entry points are `README.md`, `HANDOFF.md`, `ARCHITECTURE.md`,
`CONTROLLER_EXPERIMENTS.html`, and `HINT_BENCHMARK_RUNBOOK.html`. This file
moved to `docs/architecture/` together with
`ARCHITECTURE_MAP.md`; the hint-benchmark docs moved to `docs/hint_benchmark/`,
`HARNESS_AWARE_SCENARIOS.md` to `docs/testbeds/`, the five proposal documents
to `docs/proposals/`, the three top-level HTML reports to `docs/reports/`,
`presentation/` to `docs/presentations/`. Historical backups and build scratch
were later moved outside the repository to preserve clean portability. A stray
screenshot was deleted. `infra/accelerator/gh200/download.sh` now copies the master report to
`docs/reports/`, and `run_harness_deadline_pressure.sh` writes the
replay-friction reader copy there. All markdown links were rewritten and
checked. The full table of contents is `docs/index.md`.

**Testbed split (2026-09-23, third pass; decisions D16-D20):**
`sglang_direct_kv` was reduced to the SGLang testbed itself. Its Python
scripts moved into two new packages, `agentic-experiments` (runners, replay
driver, harness gateway, workloads) and `agentic-reports` (report builders,
summaries, audits, KV block ledger, evidence audit); four SGLang-internal
tools moved to `agentic_backends.sglang.tools`. The remaining real code in
`agentic_kv` moved too, so `agentic_kv` is now a pure compatibility layer.
29 unreachable milestone scripts were frozen in `sglang_direct_kv/scripts/legacy/`,
and the testbed's loose notes moved to `docs/sglang_direct_kv/`. Every
`python scripts/<name>.py` command still works through a thin wrapper.


```text
packages/
  agentic-core/              contracts only (+ NEW BackendRequest)
  agentic-backend-api/       backend protocols (+ NEW RequestLowering, TelemetryNormalizer,
                             LaunchPlanner, CompatibilityProbe, EffectLevel, CompatibilityReport)
  agentic-controller/        policy (+ NEW lead_times.py, eviction_value.py moved from the driver)
  agentic-harnesses/         harness signals (+ NEW request_hints.py, emission_emulation.py)
  agentic-gateway/           NEW: backend-neutral request translation (client_api, translation)
  agentic-backend-sglang/    NEW: the ONLY SGLang-aware code
    src/agentic_backends/sglang/
      versions/              v0510 v0511 v0513 v0516 v0520 adapter data + base.py
      selection.py           version range -> static probe -> loud fallback
      surface.py             static AST compatibility checker (CLI)
      lowering.py            BackendRequest -> SGLang /v1/chat/completions JSON
      launch.py              launch_spec() + flag preflight (CLI)
      telemetry.py           raw trace rows -> BackendObservation
      adapters.py            controller-command adapters (moved; + effect_level)
      trace/patch.py         in-server tracing + prepare-prefix control (moved verbatim + install changes)
      instrumentation/       nvtx, runtime_telemetry, torch_cuda_profiler (moved verbatim)
      compat.py              radix-eviction "priority" registration (moved verbatim)
      capabilities.py        runtime capability probe (moved verbatim)
      hooks.py               pre-refactor hook-table API for the shims
    COMPATIBILITY.md         per-release check results 0.5.10.post1 .. 0.5.20
      tools/                 SGLang-internal probes/analysis moved from sglang_direct_kv/scripts (testbed split)
  agentic-harness-scenarios/ MOVED from repo root (was duplicated in agentic_kv)
  agentic-experiments/       NEW (testbed split): runners/, gateway/, workloads/, environment/,
                             prompt_codec_eval/, basic_workload/, real_runner.py, paths.py
  agentic-reports/           NEW (testbed split): builders/, summaries/, audits/, analysis/,
                             block_ledger/, evidence_audit.py, evidence_schema.py
sglang_direct_kv/            the SGLang testbed: shell entry points, thin script wrappers, scripts/legacy/,
                             configs, sitecustomize, agentic_kv (compatibility aliases only), tests
tests/architecture/          NEW: boundary rules enforced by tests
scripts/check_portability.sh NEW: every GPU-free check in one command
.github/workflows/portability.yml  NEW: CI (no SGLang) + surface check vs SGLang wheels
```

Dependency direction (enforced by `tests/architecture`):

```text
agentic_core  <-  agentic_backend_api  <-  agentic_controller
agentic_core  <-  agentic_harnesses    <-  agentic_gateway
agentic_core, agentic_backend_api      <-  agentic_backends.sglang   (only one allowed to import sglang)
core, backends.sglang, controller, prompt_codec          <-  agentic_reports
every portable package + backends.sglang (not reports)   <-  agentic_experiments   (composition root)
sglang_direct_kv (testbed) wires all of the above to a real SGLang server
```

Request path after the refactor:

```text
harness client --HTTP--> scripts/harness_sglang_gateway.py (wrapper for agentic_experiments.gateway.harness_sglang_gateway)
    agentic_harnesses.request_hints   what hints did the client send?
    agentic_gateway.translation       what should this request carry?  -> agentic_core.BackendRequest
    agentic_backends.sglang.lowering  how does SGLang X.Y spell that?    -> JSON body
--HTTP--> SGLang  (trace hooks from agentic_backends.sglang.trace installed via sitecustomize)
```

---

## 3. Decisions and why

Each decision lists the alternative that was rejected.

### D1. One backend package, adapters as data
All SGLang knowledge sits in `agentic_backends.sglang`; each release line is a
`versions/vXXXX.py` module containing only data (`AdapterSpec`: hook table,
raw-event map, surface requirements, version range, verification level).
Newer adapters are derived from older ones with `replace_hook_targets(...)` so
the diff between releases is readable.
*Rejected:* version `if` checks inside the trace code (what existed before,
and why upgrades broke silently).

### D2. `agentic_backends` is a namespace package
Import path `agentic_backends.sglang` leaves room for `agentic_backends.vllm`
etc. without a shared `__init__`. Distribution name `agentic-backend-sglang`.

### D3. Static (AST) surface check instead of importing SGLang
Importing `sglang.srt` needs torch + CUDA. Parsing the source answers "does
this class/method/attribute/flag/field still exist?" on a laptop or CI in
under a second, and works on a downloaded wheel without installing it.
It understands inheritance through imports, ignores `TYPE_CHECKING` imports,
and recognizes CLI flags declared either as literals or as dataclass fields
(SGLang 0.5.2x generates flags from fields).
*Limitation:* it checks names, not signatures or semantics. Attributes set
dynamically are invisible; mark those `required=False` with a note.

### D4. Selection: range, then probe, then loud fallback
`selection.select_adapter` order: `AGENTIC_SGLANG_ADAPTER` override, then
version range, then (unknown version) static probe of every adapter newest
first, then fallback with status `unverified` and a warning. With
`AGENTIC_SGLANG_STRICT=1` anything untrusted raises.
*Before:* any unparsable/unknown version silently became `v0510`, anything
>= 0.5.11 became `v0511`, which was an alias of `v0510`.

### D5. Required vs optional surface
Hooks for alternate code paths (MLA/NSA/Mamba/hybrid host pools, dLLM, split
prefill, embedding, overlap loop) are optional (`OPTIONAL_HOOKS` in v0510).
The MHA (Qwen/Llama) path, HiCache/HiRadix/Radix hooks, scheduler core,
prepare-prefix control, request fields and launch flags are required.

### D6. Event names are a project-owned contract
When SGLang moves a method, the new adapter hooks the new location but keeps
the old **event name** (e.g. v0513 hooks
`SchedulerBatchResultProcessor.process_batch_result_prefill` and still emits
`scheduler.process_batch_result_prefill`). Reports that match on event names
therefore keep working across SGLang releases. Rule: never rename an event in
an adapter; add a new one if the meaning changes.

### D7. Move verbatim, prove with golden tests
Gateway translation (`harness_sglang_gateway.py`) and driver helpers
(`run_multi_harness_replay_driver.py`) were moved with an AST tool that copies
exact source text; only `build_sglang_payload` was split into
`translate_request` (neutral) + `lower_translation` (SGLang). Before moving,
`sglang_direct_kv/tests/golden/` recorded outputs from the old code: 5,472
gateway cases x 2 environment profiles (every mode x phase x metadata variant,
OpenAI/Anthropic/Responses payloads, cache headers) and every driver helper.
The tests replay them through the scripts **and** through the new packages;
both must match exactly.

### D8. Compatibility aliases, not copies
Old module paths (`agentic_kv.sglang_trace_patch`, `agentic_kv.nvtx`,
`agentic_kv.controller.backend`, `agentic_kv.sglang_adapters.*`,
`agentic_kv.harness_scenarios.*` ...) are 5-line modules calling
`agentic_kv._compat_alias.alias(__name__, "<new module>")`, which puts the
new module object in `sys.modules` under the old name. Private names and
module globals (e.g. the trace registries) are therefore shared. Scripts,
`sitecustomize` and historical commands keep working unchanged.
*Rejected:* `from new import *` (drops private names, duplicates globals).
*Exception (fixed in the testbed split):* a compatibility **package** must not
be an alias. If `agentic_kv.harness_scenarios.policies` is an alias, Python
resolves `agentic_kv.harness_scenarios.policies.baseline` through the real
package's `__path__` and loads a *second copy* of `baseline.py` (duplicate
classes and state). The first pass had this latent bug for
`harness_scenarios/policies` and `harness_scenarios/adapters`; those package
`__init__` files, and the new `agentic_kv/block_ledger/__init__.py`, now
re-export (`from new import *`) while each submodule file stays an alias.
`test_refactor_compat.py` checks the submodule identities.

### D9. No mandatory reinstall on the server
`agentic_kv/__init__.py` already put `<repo>/packages/*/src` on `sys.path`.
New packages were placed under `packages/` so the SGLang server (started with
`PYTHONPATH=sglang_direct_kv/src`) finds them without `pip install`. A test
starts a fresh interpreter exactly like the launch scripts and checks that the
hooks install. The two Docker launch paths were fixed to mount `packages/`
(they previously mounted only `sglang_direct_kv`). Re-running
`bash scripts/install_workspace.sh` is still recommended.

### D10. Fail loudly, opt into failing hard
Missing hooks, unverified adapters and unknown flags now print warnings to
stderr and are recorded in the trace (`trace.install.summary`,
`trace.adapter.selected.selection`). `AGENTIC_SGLANG_STRICT=1` turns them
into errors (sitecustomize raises `SystemExit`, because `site.py` swallows
ordinary exceptions). Default is non-strict so existing runs do not stop.

### D11. `acted` kept, `effect_level` added
`BackendActionResult.acted` historically meant "accepted for lowering into a
request field", not "SGLang did it". Changing its meaning would change
report columns, so it was kept, and `effect_level` was added:
`recorded_only`, `lowered_at_request_boundary`, `dispatched_to_backend`,
`backend_confirmed`, `unsupported`. It is only serialized when set.

### D12. Gateway translation is its own package
Translation depends on harness hint parsing, so it cannot live in the
controller (controller must not import harnesses), and it is backend-neutral,
so it cannot live in the backend. Hence `agentic_gateway` (core + harnesses).

### D13. Data keys containing "sglang" were not renamed
Keys like `controller_sglang_priority`, `sglang_priority`,
`sglang_cache_salt`, mode names `storage_hicache_*` are recorded artifact
columns and CLI identifiers. Renaming them would break report comparability.
They are listed with reasons in `tests/architecture/vocabulary_allowlist.json`;
the test allows the count per file to go down, never up.

### D14. SGLang pinned to 0.5.10.post1
`sglang_direct_kv/requirements.txt` had `sglang[all]` unpinned, so a fresh
remote A10G host setup would install the newest release (0.5.20 as of this writing), which
is 10 releases past what any result used. In the local artifacts, 151
recorded `sglang_version` entries say 0.5.10.post1 and 3 say 0.5.11.

### D15. Trace patch moved, not split
`trace/patch.py` (3.5k lines, module-level registries shared across ~100
functions) was moved verbatim; only the installer (`_wrap_method`,
`_try_patch`, `install_sglang_kv_trace`) changed. Splitting it without a GPU
to verify the extracted records would risk silent data changes. The safety
net for now is the adapter surface (every attribute the patch relies on) plus
tests that run the real installer against a generated fake SGLang tree.
See section 9 for the split plan.

### D16. Split the testbed by role, keep script names as module names
Scripts became modules in `agentic_experiments.{runners,gateway,workloads,environment,prompt_codec_eval}`,
`agentic_reports.{builders,summaries,audits,analysis}` or
`agentic_backends.sglang.tools`, **with their file names unchanged**, so a
module is found by the same name used in commands, docs and logs, and
argparse `--help` output is byte-identical. The full mapping is
`sglang_direct_kv/scripts/README.md`. Shell entry points (`*.sh`) stayed in
the testbed: they launch the SGLang server and wire processes together.
*Rejected:* renaming modules (breaks every runbook) and moving the `.sh`
files (they are the testbed).

### D17. Thin wrappers via `runpy`, not copies or `python -m`
`scripts/<name>.py` is an 8-line wrapper calling `_moved.run_or_alias`. As a
program it runs the package module with
`runpy.run_module(..., run_name="__main__", alter_sys=True)`; imported, it
aliases the package module. `_moved.py` also puts `src/` on `sys.path` and
imports `agentic_kv`, so wrappers work without `pip install` (same as D9).
Proof: `--help` output and exit code of all 59 moved scripts compared before
and after: identical; the harness gateway still starts as a subprocess from
its script path. *Rejected:* rewriting ~40 shell scripts to `python -m`
(needs installed packages everywhere, breaks documented commands).

### D18. Only import lines and `__file__` changed in moved code
An AST-based tool moved the files and rewrote exactly: sibling-script
imports (`from build_x import` -> `from agentic_reports.builders.build_x import`),
`agentic_kv.*` imports (to the canonical module each name is defined in,
found by object identity), and seven `Path(__file__).parents[1]` uses (to
`agentic_experiments.paths.testbed_root()`, which returns the same
`sglang_direct_kv` directory; override with `AGENTIC_TESTBED_ROOT`). Three
`SRC_ROOT` `sys.path` hacks were removed. The full diff against the originals
is 177 lines across 72 files, all of those kinds.

### D19. Legacy = unreachable, computed, not guessed
A reference graph (script file names mentioned in scripts, plus sibling
imports) was walked from every non-milestone shell script and every Python
script no other script references. The 29 files not reached went to
`scripts/legacy/`. Milestone-named scripts still used by the master-report
pipeline (e.g. `run_milestone27_*`, `build_milestone27_controlled_replay_report.py`)
stayed. Four legacy shell scripts compute their root from their own location;
their `SCRIPT_DIR/..` became `SCRIPT_DIR/../..`. All `scripts/...` paths
referenced by any shell or Python script were checked to exist.

### D20. New packages get boundary rules and a vocabulary ratchet
`tests/architecture` now covers `agentic_experiments` (composition root: may
import everything except `agentic_reports` and `agentic_kv`) and
`agentic_reports` (core, controller, prompt codec, and `agentic_backends` for
the raw-event map). Both still contain many SGLang-shaped data keys (about
1,200 occurrences); their counts were recorded in
`vocabulary_allowlist.json` and may only go down.
`legacy_sglang_internal_users.json` is now empty: no testbed file imports
SGLang any more.

---

## 4. What the SGLang compatibility check found

Full table: `packages/agentic-backend-sglang/COMPATIBILITY.md`. Summary:

| SGLang | Adapter | What the pre-refactor code would have done |
| --- | --- | --- |
| 0.5.10.post1 | v0510 (runtime-verified) | fine |
| 0.5.11 - 0.5.12.post1 | v0511 | silently lost NSA host-pool hooks (optional) |
| 0.5.13 - 0.5.15.post1 | v0513 | silently lost scheduler prefill/decode result hooks (moved to `SchedulerBatchResultProcessor`) |
| 0.5.16 - 0.5.19 | v0516 | also silently lost MHA host-pool load/backup hooks (moved to `mem_cache/pool_host/`), i.e. the core KV load/write trace on Qwen |
| 0.5.20 | v0520 | also: server refuses to start, `--disable-piecewise-cuda-graph` no longer exists (default flag in `run_harness_deadline_pressure.sh`) |

All adapters have zero missing *required* surface on their releases. Only
v0510 is runtime-verified; v0511 was probed in the real Docker image; the
rest are static-only.

---

## 5. Behavior changes (everything that is NOT byte-identical)

1. **Adapter choice for SGLang >= 0.5.13** uses the new hook tables (before:
   v0510 tables that silently missed hooks). For the pinned 0.5.10.post1 the
   hook table and every event name are unchanged.
2. **Unknown future versions** map to the newest adapter (was v0511) after a
   static probe, with a warning.
3. **Trace file** gains one row `trace.install.summary` and a `selection`
   object inside `trace.adapter.selected`. Existing rows are unchanged.
4. **Server startup** statically parses a few SGLang source files (well under
   a second) to attach a surface report. Disable with
   `AGENTIC_SGLANG_SURFACE_CHECK=0`.
5. **stderr warnings** when the adapter is not runtime-verified, hooks are
   missing, or flags are unknown. Silent before.
6. **sitecustomize** skips trace installation in processes where `sglang` is
   not importable (before: it ran, found nothing, and wrote only
   `trace.install.start/end` rows). Install failures always print now.
7. **Controller action records** from the `Gateway*` / targeted-prefetch
   adapters gain an `effect_level` key.
8. **SGLang pin** `==0.5.10.post1` (was unpinned).
9. **Launch scripts** (`run_sglang_server.sh`, `run_sglang_hicache_server.sh`)
   run a flag preflight before starting (warning only; `AGENTIC_SGLANG_PREFLIGHT=0`
   disables it; strict mode makes it fatal). Skipped for Docker launches.
10. **Docker:** `probe_sglang_capabilities_docker.sh` mounts the repository
    root at `/workspace` with workdir `/workspace/sglang_direct_kv`
    (output paths unchanged); `run_sglang_hicache_server.sh` Docker mode also
    mounts `packages/` read-only.
11. `agentic_harness_scenarios/` moved to `packages/agentic-harness-scenarios/`.
12. `install_workspace.sh` also installs `agentic-gateway`,
    `agentic-backend-sglang`, `agentic-harness-scenarios`.
13. Top-level cleanup: `infra/accelerator/gh200/download.sh` copies `latest_master_report.html`
    to `docs/reports/` (was the repo root); `run_harness_deadline_pressure.sh`
    passes `--top-level-copy-dir <repo>/docs/reports` to the replay-friction
    analyzer (was the repo root).
14. Testbed split: `sglang_direct_kv/scripts/*.py` are wrappers; code lives in
    `packages/agentic-experiments`, `packages/agentic-reports`,
    `agentic_backends.sglang.tools`. `install_workspace.sh` installs the two
    new packages; `agentic-kv` depends on them.
15. `agentic_experiments` code locates the testbed with `testbed_root()`
    instead of its own file location (same result; `AGENTIC_TESTBED_ROOT`
    overrides).
16. 29 milestone scripts moved to `sglang_direct_kv/scripts/legacy/`; run
    them as `scripts/legacy/<name>`.
17. Bug fix: importing `agentic_kv.harness_scenarios.policies.*` or
    `.adapters.*` no longer creates duplicate module copies (see D8).

**Unchanged (proven by golden tests):** every gateway request body and
translation context, every driver timing/eviction/emulation helper output,
all mode names, all report column names.

---

## 6. Environment variables added

| Variable | Default | Effect |
| --- | --- | --- |
| `AGENTIC_SGLANG_ADAPTER` | unset | Force an adapter (`v0510`, `v0511`, `v0513`, `v0516`, `v0520`). |
| `AGENTIC_SGLANG_STRICT` | `0` | Untrusted adapter, missing required hook/surface or unknown flag becomes fatal. |
| `AGENTIC_SGLANG_SURFACE_CHECK` | `1` | Attach a static surface report at trace install. |
| `AGENTIC_SGLANG_PREFLIGHT` | `1` | Launch scripts check server flags before starting. |

---

## 7. How to verify (no GPU needed)

```bash
bash scripts/install_workspace.sh          # editable installs of all packages + testbed
bash scripts/check_portability.sh          # architecture + all package tests + testbed tests (incl. golden)
python tests/architecture/boundaries.py    # human-readable boundary report
python -m agentic_backends.sglang.surface  # check the INSTALLED SGLang (does not import it)
```

Check any SGLang release without installing it:

```bash
pip download "sglang==0.5.20" --no-deps -d /tmp/w && python -m zipfile -e /tmp/w/*.whl /tmp/sgl
python -m agentic_backends.sglang.surface --sglang-src /tmp/sgl          # adapter chosen by version
python -m agentic_backends.sglang.surface --sglang-src /tmp/sgl --all    # every adapter
python sglang_direct_kv/scripts/sglang_preflight.py -- --disable-piecewise-cuda-graph   # flag check (installed SGLang)
```

Test inventory added by this work:

| Where | What it proves |
| --- | --- |
| `sglang_direct_kv/tests/golden/` | moved gateway/driver code is byte-identical (script and package paths) |
| `sglang_direct_kv/tests/test_refactor_compat.py` | all 70 scripts import; old module paths are aliases; a fresh interpreter started like the SGLang server installs hooks via sitecustomize |
| `packages/agentic-backend-sglang/tests/` | surface checker (positive/negative, inheritance, TYPE_CHECKING, dataclass flags), selection rules, lowering, effect levels, telemetry, launch preflight, and the **real** trace installer run against generated fake SGLang trees (reference, broken, strict, 0.5.20 layout) |
| `packages/*/tests/` | small contract tests per package |
| `packages/agentic-experiments/tests`, `packages/agentic-reports/tests` | every moved module imports; `testbed_root()` resolves to the former script parent; block ledger builds (testbed split) |
| `tests/architecture/` | dependency direction, "only agentic_backends touches sglang", vocabulary ratchet, no new testbed SGLang users |

Result at hand-off: all checks pass on Python 3.10 and 3.11 in a fresh venv
**without SGLang installed** (112 testbed tests, 26 backend-package tests,
5 scenario tests, 12 other package tests, 4 architecture tests). Tests were verified to fail when a boundary is
violated.

---

## 8. GPU reference verification

The focused Scenario 1 reference completed after restructuring with SGLang
`0.5.10.post1`, the `v0510` adapter, the `nvidia_a10g_24gb` hardware profile,
and the exact reference workload/seed. Run label:
`scenario1_portability_reference_20260925_171817`.

Compared with `no_prefetch`, `controller_ready_time_gpu_backfill` produced:

- total replay TTFT: `82.35 s -> 71.99 s` (`12.6%` better);
- total replay deadline debt: `205.23 s -> 167.07 s` (`18.6%` lower);
- full workload duration: `176.25 s -> 167.53 s` (`4.9%` shorter).

The run exercised the end-to-end controller, gateway, SGLang adapter,
instrumentation, artifact, and report path. See
[`CONTROLLER_EXPERIMENTS.html`](../../CONTROLLER_EXPERIMENTS.html).

---

## 9. Remaining work, in priority order

1. **Common host/container deployment contract:** implement the approved
   topology and run-manifest/capability handshake in `ARCHITECTURE.md`.
2. **Upgrade to a newer SGLang (optional, when wanted):** make
   `--disable-piecewise-cuda-graph` conditional in
   `run_harness_deadline_pressure.sh:61` and
   `run_milestone22_live_agentbench_bridge.sh:36` (only for < 0.5.20), change
   the pin, run section 8 on the new version, set that adapter's
   `verification="runtime"` and `tested_versions`. Watch the two v0513 notes:
   scheduler-result events now fire on `SchedulerBatchResultProcessor`, so
   their scheduler-queue fields may be thinner.
3. **Split `trace/patch.py`** into: `trace/extract.py` (per-object readers for
   Req / ScheduleBatch / TreeNode / KV pools; the only place with SGLang
   attribute names, ideally driven by the adapter), `trace/registry.py`
   (agent-context, priority and preparable-prefix registries),
   `trace/control.py` (prepare-prefix HTTP server + `_execute_prepare_prefix_command`),
   `trace/events.py` (event writing, runtime telemetry, NVTX). Record a golden
   trace from a GPU run first and compare after the split.
4. **Reports on normalized observations:** the report code now lives in
   `agentic_reports` (testbed split), but `block_ledger/normalizer.py`,
   `evidence_audit.py` and the builders still match raw event names
   (`grep -rlE '"(hiradix|hicache|hostpool|radix)\.' packages/agentic-reports`).
   Thanks to D6 this no longer breaks on SGLang upgrades; moving them onto
   `agentic_backends.sglang.telemetry.SGLangTelemetryNormalizer` is the clean
   end state and would let most of the vocabulary allowlist go to zero.
5. **Decompose the replay driver.** `main_async` in
   `run_multi_harness_replay_driver.py` is ~3,500 lines. Suggested cut:
   harness CLI command builders (`codex_command` ... `hermes_agent_command`)
   -> `agentic_harnesses.clients`; remaining emulation helpers
   (`attach_*_priority_*`, `nat_inferred_*`) -> `agentic_harnesses.emission_emulation`;
   workload/timeline construction and the event loop -> separate modules
   inside `agentic_experiments` (the driver itself already moved there in the
   testbed split). Extend the golden tests before each move.
6. **Resolve gateway/controller mode divergence** (found during the move, not
   changed): the gateway's `PRIORITY_ENABLED_MODES` does not include
   `controller_priority_demotion_admission_shorthand`,
   `controller_oracle_timeline`, or the `controller_harness_aware_*` modes, and
   its `CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES` differs from
   `agentic_controller.modes`. In shorthand mode the gateway therefore never
   lowers a priority. Decide whether that is intended, then make
   `agentic_gateway.translation` import the controller's mode sets.
7. **Second backend** to prove independence: implement the four protocols in
   `agentic_backend_api` for a mock OpenAI-compatible server or vLLM and run
   the gateway golden matrix through its lowering.
8. **Remove compatibility aliases** (ARCHITECTURE_MAP Phase 7): once nothing
   imports `agentic_kv.*` or the script wrappers by module name (tests,
   `sitecustomize`, historical commands), delete them; then burn down
   `vocabulary_allowlist.json`.
9. **Move package tests out of the testbed:** `sglang_direct_kv/tests` still
   holds controller, harness, hint-benchmark and prompt-codec tests that use
   testbed configs. Move each next to its package with the config fixtures it
   needs.

---

## 10. Rules for future changes

- New SGLang-specific code goes in `packages/agentic-backend-sglang` only.
  The architecture tests fail otherwise.
- Never rename a trace event name in an adapter (D6).
- Never lower a vocabulary allowlist count by renaming an artifact column
  without migrating the reports that read it.
- Moving code out of a script: extend `tests/golden` first, move verbatim,
  keep the old name importable from the script.
- Adding an adapter: derive from the newest one with `replace_hook_targets`,
  set `version_range` and `verification`, register in `versions/__init__.py`,
  add the release to the CI matrix, regenerate `COMPATIBILITY.md`.
- Keep `scripts/check_portability.sh` green before every commit.

---

## 11. Commits on the branch

| Commit | Content |
| --- | --- |
| `491aea5` | Checkpoint of the uncommitted pre-refactor working tree (package pyprojects, install script, docs reorganization) |
| `0e4c860` | Golden fixtures recorded from the pre-refactor code |
| `0d9b23c` | `agentic-backend-sglang` package, adapters, surface checker, selection, moved trace/compat/instrumentation, aliases, Docker mounts, preflight |
| `ee53e2f` | Gateway translation and driver helpers moved into packages |
| `88d3869` | Architecture tests, package tests, CI, check script, SGLang pin |
| `a9b1512` | Single copy of the harness-scenario framework |
| `1ddcaaa` | v0520 adapter, launch-flag surface, compatibility matrix |
| (last) | This README and doc updates |
| (testbed split) | `Move sglang_direct_kv notes into docs/sglang_direct_kv`; `Freeze unreachable milestone scripts in scripts/legacy` (+ docs follow-up); `Split sglang_direct_kv into agentic-experiments and agentic-reports packages` -- see `git log` |
