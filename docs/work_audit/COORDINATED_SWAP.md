# Coordinated CPU/GPU KV Swapping

## Question

Can perfectly coordinated tool waits let twenty independent sessions behave
nearly like GPU-resident sessions, even though their combined KV exceeds the
GPU cache budget? This is an optimistic proof of concept in the KV Lifecycle
Audit lane, not a new lane and not a shared-context/composite agent.

## Fixed Setup

- Twenty separate contexts, four groups of five: A/B and C/D alternate.
  `swap-s00` through `swap-s04` are A, `s05` through `s09` are B,
  `s10` through `s14` are C, and `s15` through `s19` are D.
  The runner records pair 0 for A/B and pair 1 for C/D; contexts remain separate.
- Every tool wait lasts 1,000 ms. A paired service slot is at least 1,000 ms.
  Faster replies wait until the slot boundary; this padding is recorded.
  The workload ends at the last reply, without waiting out its final slot.
  The raw scheduled-padding total includes that final slot's unused padding;
  it is a schedule diagnostic, not an additional elapsed-time measurement.
- Forty replays per session; first validate three rounds using all twenty.
- Start near 8,192 tokens per session; append sixteen synthetic tool-result
  words each round; generate exactly eight tokens per replay in the final run. Actual token counts
  are recorded. Generated answers do not determine the next synthetic prompt.
- Restricted GPU cache: 110,592 tokens. Resident reference: 262,144 tokens.
  Each pair must fit the restricted cap; all twenty must exceed it.
- Host cache: 8 GiB; native write-through backups; no storage tier.
- Pinned SGLang 0.5.10.post1/v0510 in Docker, A10G, Qwen2.5-1.5B-Instruct.
  CUDA graphs and overlap scheduling are both enabled. Count-only lifecycle tracing;
  no full debug, profiler, or experimental copy worker.

The restricted limit is an imposed KV-pool budget, not all physical GPU memory.
The resident control deliberately raises that budget on the same GPU. This
tests hiding CPU spill under controlled capacity, not physical out-of-memory
behavior of a larger production model.

The initial 32-output-token, direct-transfer pilot exceeded the one-second
service window even in the resident control. Keep it as calibration evidence,
not as the final ideal handover setup. A second pilot uses eight output tokens
and `SWAP_IO_BACKEND=kernel`, SGLang's existing native kernel-copy path. Neither
change extends the tool wait or increases the restricted GPU cache budget.
Record the selected transfer path and output size for every comparison.
The first setup attempt stopped because both diagnostic replies were empty;
this was not evidence of unequal KV tensors. The diagnostic now forces eight
output tokens and requires nonempty, identical text before measurement.
Initial pilots used `kv_lifecycle_lean`, which still read and hashed tensor
index arrays. The final timing profile is `kv_lifecycle_counts`: dimensions and
counts are logged without GPU-to-CPU tensor-value reads. It does not prove
exact slot identity. Keep the earlier pilots separate from count-only results;
their CUDA-event intervals include tracing-induced gaps between copy launches.
The final run also combines release and residency checks per ten-session pair.
The previous count-only pilot still made those checks individually. Preserve
these as different configurations, not interchangeable repeats.

## Modes

| Mode | What Happens |
| --- | --- |
| Independent | Stagger initial readiness within one second. Each session subsequently waits 1,000 ms after its own reply. SGLang manages the restricted cache normally. |
| Coordinated | Prime separate host-backed prefixes, prepare the first pair, then alternate pairs. After a pair finishes useful work, release its unlocked, host-backed KV and restore the next pair through native HiCache. Transfers can use the remaining slot time. |
| Resident | Same ideal paired schedule, with enough GPU KV capacity for all histories. Check residency before and after measurement and prefix reuse on every replay. |

Initial priming and arranging the pairs are excluded from the measured window,
as an explicit optimistic assumption; their duration is saved separately.
All control calls, transfers, late submissions and slot padding during the
measured window count. Original tool deadlines are never replaced by restore
completion times. This is not a causal comparison of grouping alone: arrival
patterns, concurrency and residency policy change together.

## Safety and Evidence

SGLang internals stay in the backend package. The opt-in
`AGENTIC_KV_COORDINATED_AUDIT_ENABLE` control supports the pinned single-rank
HiRadixCache only. It rejects shared prefix anchors, active/locked nodes and
unfinished host backups before releasing any nodes. It does not use the older
global eviction control. Restore commands refresh the cache-tree match and
load the entire missing prefix chain, not just the largest individual node.
The current group path reserves the ten prefixes and submits SGLang's native
merged load queue once. Earlier calibration pilots restored one prefix at a
time; those remain separately labeled. A failed partial group reservation is
still submitted before the experiment aborts, so it does not strand reserved
GPU slots without a copy operation.

Before each coordinated replay, the runner waits for native copy completion
and confirms full GPU residency. If this is late, the lateness remains in the
results. A setup diagnostic compares eight generated tokens before and after
one explicit release/restore. This is not a tensor-by-tensor proof of every
transfer. Native trace joins must show every replay and at least 90% GPU-prefix
reuse in coordinated/resident modes; incomplete evidence blocks conclusions.

Record workload duration, due-to-first-token mean/p95/total, total TTFT,
submission waiting, alignment padding, per-session completion, restored and
released tokens, and restores before/after their due times. Compare all three
trials individually. A coordinated workload within 5% of the resident reference
is the initial timing target, not a claim of equivalent latency or production
performance. Report first-token delays and late restores separately.

## Run on the GPU Host

```bash
SWAP_RUN_ID=coordinated_swap_pilot \
SWAP_TURNS=3 SWAP_TRIALS=1 \
SWAP_IO_BACKEND=kernel SWAP_DECODE_TOKENS=8 \
bash infra/container/run_work_audit_coordinated_swap.sh
```

Only after the pilot passes:

```bash
SWAP_RUN_ID=coordinated_swap_full \
SWAP_TURNS=40 SWAP_TRIALS="1 2 3" \
SWAP_IO_BACKEND=kernel SWAP_DECODE_TOKENS=8 \
bash infra/container/run_work_audit_coordinated_swap.sh
```

The launcher rotates mode order and starts a fresh backend for every arm.
It refuses to disturb an existing backend. Results are under
`sglang_direct_kv/artifacts/results/work_audit/$SWAP_RUN_ID`.
Archive the summary, per-arm case results, compressed raw traces, hook gates,
runtime contract, launch settings and source fingerprint. Keep failed pilots
as failures; do not merge their numbers into the completed comparison.
The initial manifest keeps its launch-time `created` status; `summary.json`
holds the final validation status. Source archives identify synced working-tree
code even when the remote checkout has no Git commit. Server-selected random
seeds are recorded per arm; these are rotated repeats, not matched random-seed
pairs. The task text and forced output length are fixed across modes.

Runner: `packages/agentic-experiments/src/agentic_experiments/runners/run_work_audit_coordinated_swap.py`.
Analyzer: sibling `analyze_work_audit_coordinated_swap.py`.
Neutral schedule: `packages/agentic-controller/src/agentic_controller/coordinated_swap.py`.
Backend control: `packages/agentic-backend-sglang/src/agentic_backends/sglang/versions/v0510_coordinated_kv.py`.

Calibration changes must create a new run ID and be recorded. Never silently
shorten output, increase tool waits, or enlarge the restricted GPU cap to make
the coordinated case appear successful.

## Completed Reference

`coordinated_swap_count_full_20261008` completed all nine arms, 800 replays each.
The native prefix checks passed, with at least 98.9% GPU prefix reuse in the
coordinated and resident modes. The paired handovers were not hidden: each
coordinated arm had 790 session restores observed ready after its deadline.
All twenty sessions finished later than their independent counterparts in
every trial. The workload was 24.2-24.3% slower than independent operation.
This is a negative timing result, not a reason to discard the experiment.

Exact per-trial results, every session's finish time, control wall time and
CUDA-stream intervals are in the RQ21 details of `KV_LIFECYCLE_AUDIT.md` and
`KV_LIFECYCLE_AUDIT.html`. The initial and final analysis snapshots are archived
alongside raw traces. A later source cleanup moved runtime-field validation and
native-event parsing into the backend package without changing their logic;
the run's source bundle retains the code actually executed.
When restoring source archives, skip `._*` files: these are incidental macOS
metadata, not Python modules or experiment evidence.

Remote regression command:

```bash
export PYTHONPATH="$(printf '%s:' "$PWD"/packages/*/src)"
sglang_direct_kv/.venv/bin/python -m pytest --import-mode=importlib -q \
  packages/agentic-backend-sglang/tests packages/agentic-reports/tests \
  packages/agentic-experiments/tests/test_work_audit_coordinated_swap.py \
  packages/agentic-instrumentation/tests tests/architecture
```

Result: 154 passed, with three pre-existing architecture failures. The same
three failures were reproduced from unmodified commit
`3b93cdb6f8dcb1c20827b9ba4755b9e3788ab148`: vocabulary allowlist drift, the
existing attribution analyzer's instrumentation import, and the navigation
test expecting an HTML link where README now links the Markdown audit.
Comparing the boundary scans found no new import violations, internal-backend
violations, or vocabulary growth from this change. These unrelated failures
were not hidden by relaxing the architecture tests.
