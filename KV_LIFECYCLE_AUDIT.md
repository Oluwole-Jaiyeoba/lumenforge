# KV Lifecycle Audit

**Research question:** Given what the harness knew, was KV moved, kept, or rebuilt at the wrong time?

This audit compares observed cache work with replay timing and whole-workload outcomes. A faster control call is not automatically a faster agent task. All times below are from saved runs; experimental and hypothetical claims are kept separate.

## Research progress

### RQ9: Does off-scheduler KV loading help?

**Question.** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**What the evidence says.** An opt-in prototype for pinned SGLang 0.5.10.post1 passed a live exact host-versus-device K/V equality check across all model layers before publishing a 4096-token prefix; the trace recorded 56 K/V copy events. In one warmed, comparable three-session pair per execution mode, early-load replay delay was 89.5 ms with scheduler copies and 89.9 ms with worker copies. For a load requested at tool return, the worker control response fell from about 200 ms to 16 ms, but the replay arrived before commit, recomputed its prefix, and its first-token delay rose from 276 to 1131 ms. In separate one-seed 12-session busy runs, summed replay TTFT was 13.16 seconds above check-only for scheduler copies and 22.40 seconds above check-only for worker copies; the runs admitted three and two loads respectively, so this is not a matched causal difference. Moving copy submission off the scheduler is feasible, but these runs do not show an end-to-end performance win.

**Not yet proved.** The prototype only handles explicit prepare-control loads, direct host I/O, and one tensor-parallel rank. A replay reaching the backend before commit may recompute rather than wait for the prefix. One warmed pair and one busy seed per mode cannot establish general performance. The busy admissions differed. Dedicated-stream copy time also changed under overlap, so scheduler relief is not an isolated hardware-speed measurement. Production replay gating, multi-rank safety, and native automatic load paths remain unimplemented.

### RQ8: Does timed KV preparation help a busy system?

**Question.** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**What the evidence says.** The original two-seed A10G comparison found higher summed replay TTFT and slower workflows with timed preparation. A three-arm comparison separated no checks, plan-only checks, and real early loads: the load arm added +11.04 and +19.42 seconds of summed replay TTFT versus checks-only across 36 replays per seed, while sustained generation did not slow. In a one-seed phase-timing repeat, three completed early loads spent 121, 308, and 1362 ms inside SGLang's ready-to-load call on the scheduler path; control-queue waits were only 2-12 ms. A separate diagnostic profile of one 2048-token direct load found 117.44 MB moved in 3920 small copies: 10.884 ms of active host-to-device copies spread across 156.859 ms. A one-seed control using SGLang's supported kernel I/O backend also held the scheduler for 106, 314, and 1314 ms on its three accepted loads and added 17.61 seconds summed replay TTFT versus checks-only. Neither I/O backend moved the load off the scheduler path; post-first-token generation did not slow in these runs.

**Not yet proved.** The profiler diagnostic perturbs timing and is not a performance comparison; gaps between copy events cannot be assumed recoverable. The phase-timing and kernel controls each have only one seed and three accepted early loads. CUDA-event duration overlaps wall-clock ready-call time and must not be added to it. Matched-arm differences locate affected stages but do not prove per-copy causality, HBM contention, or how much a hardware offload would save; concurrent batch trajectories can diverge. An opt-in off-scheduler prototype is evaluated separately in RQ9. A separate sequential tracing-off/on microcheck measured +3.05% median latency but does not bound overhead in the busy workload. Prompts and tool waits remain synthetic.

### RQ7: Can a controller choose the quiet load window?

**Question.** Using observed short-replay completion, host residency, slot release, and the long tool-return estimate, can a controller policy decide when to load KV without frontend importance ranks?

**What the evidence says.** Yes in this controlled A10G run. All four warmed, order-balanced four-mode trials passed the evidence gates. The controller chose load four times, each native load completed before long tool return, and no overshoot was recorded. Median long return-to-first-token was 85.6 ms with the controller versus 351.7 ms with late loading; median short return-to-finish was 859.6 ms versus 1105.3 ms with early loading. The controller's outcome was close to scripted post-short loading.

**Not yet proved.** This policy was invoked by the experiment harness, not yet by the production gateway. The load-time estimate was a fixed 250 ms plus a 150 ms margin; this run did not test the defer path under real load or naturally arising cache pressure. Four synthetic trials do not establish a universal win or identify a hardware bottleneck.

### RQ6: Can we load after the short replay finishes?

**Question.** Under the same equal-importance, logical two-prefix workload, can loading just after the short replay completes preserve the long replay benefit while avoiding the short-session penalty?

**What the evidence says.** In two warmed, order-balanced A10G triplets, post-short loading saved 188.2 and 525.5 ms in long replay tool-return-to-first-token time versus nonblocking late loading. The short session finished 203.7 and 208.6 ms sooner than with early loading; its completion was close to the late control. Whole workflows finished 208.1 and 510.1 ms sooner than late. All three-mode evidence gates passed.

**Not yet proved.** The post-short trigger used the observed client completion event, not a prediction available to a production controller. The two-prefix budget was explicit, not natural capacity pressure. Two synthetic triplets do not establish a general policy win or identify GPU bandwidth versus backend scheduling as the cause.

### RQ5: Does early loading help the whole shared system?

**Question.** With equal-importance sessions and the same logical two-prefix budget, does loading during the wait help the returning session without delaying another session or the whole workflow?

**What the evidence says.** In two warmed, order-balanced A10G pairs, early loading moved the long replay first token forward by 201 and 513 ms and shortened workflow completion by 212 and 503 ms. The short session finished 178 and 195 ms later. This is a measured tradeoff, not a no-cost win.

**Not yet proved.** The short-session delay is not attributed to GPU bandwidth versus backend scheduling. The two-prefix cap was an explicit logical policy, not measured physical occupancy or natural capacity pressure. Two synthetic pairs do not establish a production-wide benefit.

### RQ4: What happens when equal-importance sessions overlap?

**Question.** Can a small, equal-importance, concurrent workload link tool waits, one controlled host eviction/load, replays, and an ending session?

**What the evidence says.** Yes. The three-session A10G trace passed the identity and ordering gates. The 900 ms and 2500 ms tool waits overlapped; the long session was explicitly evicted to host, loaded, and replayed; the third session ended without replay.

**Not yet proved.** The eviction enforced a synthetic two-prefix policy, not natural capacity pressure. Different-session TTFTs do not prove a causal eviction penalty; RQ5 supplies the separate matched timing comparison.

### RQ3: Does nonblocking late loading remove the delay?

**Question.** Does not waiting for the late-load control response remove the observed tool-return-to-first-token penalty?

**What the evidence says.** In one warmed triple, nonblocking submission removed the roughly 170 ms client submission gap, but first token still arrived 243.659 ms after tool return versus 88.159 ms with early loading. The delay appeared in replay TTFT instead.

**Not yet proved.** The native load was accepted after nonblocking replay submission, so the strict comparable delta was withheld. More order-balanced runs are needed; the full-task comparison was also withheld because second-replay TTFT drifted.

### RQ2: Does loading during the tool wait help?

**Question.** In a controlled replay, does requesting host-KV load during a tool wait shorten time from tool return to first token compared with requesting it after the wait?

**What the evidence says.** In these sampled runs, yes: late preparation added 166-178 ms across four comparable pairs. Most of that difference occurred before replay submission.

**Not yet proved.** The sequential runs did not test competing-session cost; RQ5 now measures that separately under explicit logical capacity. Neither result establishes a production or hardware benefit.

### RQ1: Can we trace one session's KV lifecycle reliably?

**Question.** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**What the evidence says.** Yes for these controlled A10G validations: the host-backed runs linked eviction, host residency, native load, layer copies, and replay to a session. Two-replay probes also checked a later replay's cache match.

**Not yet proved.** A matched prefix does not prove exact model-kernel consumption. These sequential validations do not show a policy speedup, natural capacity pressure, or avoidable work.

## Experiment index

Newest first. Select a run to see its setup, measurements, limits, and reproduction command.

| UTC date / time | Experiment | Question | Setup | Main result | Gate |
| --- | --- | --- | --- | --- | --- |
| 2026-10-05 18:28:34 | [Concurrent early vs late · worker load](#run-work_audit_async_fixed_worker_20261005_01) | RQ9 | 3 equal-importance sessions · 1 measured pair · 900 / 2500 ms waits · worker KV load | Long replay first token after tool: early 89.9 ms; late 1131.2 ms | validated |
| 2026-10-05 18:26:11 | [Concurrent early vs late · scheduler load](#run-work_audit_async_fixed_scheduler_20261005_02) | RQ9 | 3 equal-importance sessions · 1 measured pair · 900 / 2500 ms waits · scheduler KV load | Long replay first token after tool: early 89.5 ms; late 276.0 ms | validated |
| 2026-10-05 18:12:20 | [Busy workload · KV-load attribution · scheduler load](#run-work_audit_async_scheduler_busy_20261005_01) | RQ9 | 12 sessions × 3 tool waits; 1 paired seeds; natural capacity pressure | 5 early-load attempts · load arm +13.2 s summed replay TTFT vs checks-only | complete |
| 2026-10-05 18:03:46 | [Busy workload · KV-load attribution · worker load](#run-work_audit_async_worker_busy_20261005_01) | RQ9 | 12 sessions × 3 tool waits; 1 paired seeds; natural capacity pressure | 2 early-load attempts · load arm +22.4 s summed replay TTFT vs checks-only | complete |
| 2026-10-05 18:00:42 | [Lifecycle validation · worker load](#run-work_audit_async_worker_verify_20261005_01) | RQ9 | host → warm · 2 replay(s)/case · 500 ms waits | Host load validated · warm replay 222.7 ms; host-backed replay 899.5 ms | validated |
| 2026-10-05 16:35:35 | [Busy workload · KV-load attribution](#run-work_audit_load_kernel_20261005_01) | RQ8 | 12 sessions × 3 tool waits; 1 paired seeds; natural capacity pressure | 3 early-load attempts · load arm +17.6 s summed replay TTFT vs checks-only | complete |
| 2026-10-05 14:31:03 | [Busy workload · KV-load attribution](#run-work_audit_load_phase_20261005_01) | RQ8 | 12 sessions × 3 tool waits; 1 paired seeds; natural capacity pressure | 4 early-load attempts · load arm +11.7 s summed replay TTFT vs checks-only | complete |
| 2026-10-03 01:19:35 | [Busy workload · KV-load attribution](#run-work_audit_kv_attribution_20261002_04) | RQ8 | 12 sessions × 3 tool waits; 2 paired seeds; natural capacity pressure | 5 early-load attempts · load arm +11.0 s / +19.4 s summed replay TTFT vs checks-only | complete |
| 2026-10-02 22:14:42 | [Busy workload · controller KV timing](#run-work_audit_busy_pair_20261002_01) | RQ8 | 12 sessions × 3 tool waits; 2 paired seeds; natural capacity pressure | 2 paired seeds · workflow 1175.4 ms slower; total replay TTFT 25878.9 ms higher | validated |
| 2026-10-02 22:08:50 | [Busy workload · controller KV timing](#run-work_audit_busy_pilot_20261002) | RQ8 | 12 sessions × 3 tool waits; 1 paired seeds; natural capacity pressure | 1 paired seeds · workflow 149.3 ms slower; total replay TTFT 20687.8 ms higher | validated |
| 2026-10-02 21:16:50 | [Controller-chosen load window](#run-work_audit_controller_window_20261002_01) | RQ7 | 3 equal-importance sessions · 4 measured trials · 900 / 2500 ms waits | 4/4 matched four-mode trials · controller long replay 266.3 ms faster vs late; workflow 283.6 ms faster vs late | validated |
| 2026-10-02 20:34:20 | [Three concurrent load windows](#run-work_audit_post_short_20261002_01) | RQ6 | 3 equal-importance sessions · 2 measured trials · 900 / 2500 ms waits | 2/2 matched triplets · post-short long replay 356.9 ms faster vs late; short completion 206.1 ms faster vs early | validated |
| 2026-10-02 19:48:50 | [Concurrent early vs late](#run-work_audit_concurrent_compare_20261002_02) | RQ5 | 3 equal-importance sessions · 2 measured pairs · 900 / 2500 ms waits | 2/2 matched pairs · long replay 356.7 ms faster; short completion 186.4 ms slower; workflow 357.3 ms faster | validated |
| 2026-10-02 18:42:21 | [Concurrent timeline](#run-work_audit_multisession_20261002_01) | RQ4 | 3 concurrent sessions · 900 / 2500 ms tool waits | Overlapping waits · short first token 81.6 ms; long first token 282.4 ms after tool return | validated |
| 2026-10-02 18:33:37 | [Early vs late](#run-work_audit_nonblocking_20261002_01) | RQ3 | early → late → late_nonblocking · 1 measured pair(s) · 2000 ms waits | Nonblocking late load: 0/1 comparable pair(s) | validated |
| 2026-10-02 16:41:32 | [Early vs late](#run-work_audit_timing_sampled_late_early_20261002) | RQ2 | late → early · 2 measured pair(s) · 2000 ms waits | Late +170.1 ms / +173.8 ms to first token | validated |
| 2026-10-02 16:38:54 | [Early vs late](#run-work_audit_timing_sampled_early_late_20261002) | RQ2 | early → late · 2 measured pair(s) · 2000 ms waits | Late +166.4 ms / +178.1 ms to first token | validated |
| 2026-10-02 16:25:50 | [Early vs late](#run-work_audit_timing_exact_late_early_20261002) | RQ2 | late → early · 2 measured pair(s) · 2000 ms waits | Late +228.7 ms / +228.6 ms to first token | validated |
| 2026-10-02 16:22:26 | [Early vs late](#run-work_audit_timing_exact_early_late_20261002) | RQ2 | early → late · 2 measured pair(s) · 2000 ms waits | Late +226.8 ms / +236.5 ms to first token | validated |
| 2026-10-02 15:34:04 | [Lifecycle validation](#run-work_audit_two_replays_lean_reverse_20261002) | RQ1 | host → warm · 2 replay(s)/case · not recorded ms waits | Host load validated · warm replay 82.3 ms; host-backed replay 436.3 ms | validated |
| 2026-10-02 15:26:59 | [Lifecycle validation](#run-work_audit_two_replays_lean_pump_20261002) | RQ1 | warm → host · 2 replay(s)/case · not recorded ms waits | Host load validated · warm replay 81.5 ms; host-backed replay 82.6 ms | validated |
| 2026-10-02 14:54:21 | [Lifecycle validation](#run-work_audit_two_replays_20261002_reverse) | RQ1 | host → warm · 2 replay(s)/case · not recorded ms waits | Host load validated · warm replay 217.1 ms; host-backed replay 573.9 ms | validated |
| 2026-10-02 14:50:49 | [Lifecycle validation](#run-work_audit_two_replays_20261002) | RQ1 | warm → host · 2 replay(s)/case · not recorded ms waits | Host load validated · warm replay 220.9 ms; host-backed replay 229.1 ms | validated |
| 2026-10-01 22:42:31 | [Lifecycle validation](#run-work_audit_a10g_20261001_final) | RQ1 | warm → host · not recorded replay(s)/case · not recorded ms waits | Host load validated · warm replay 227.1 ms; host-backed replay 230.3 ms | validated |

## Experiment details

<a id="run-work_audit_async_fixed_worker_20261005_01"></a>
<details>
<summary><strong>2026-10-05 18:28:34 UTC · Concurrent early vs late · worker load</strong> · work_audit_async_fixed_worker_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** Early loading sped the long replay and whole workflow, but delayed the short session.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution worker. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. 1 warmup pair were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 89.9 | 1006.4 | 7900.2 | -959.3 |
| Trial 1 · late_nonblocking | 1131.2 | 808.5 | 8953.5 | 523.8 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- The two-prefix cap is a test policy enforced by explicit evictions, not measured physical occupancy.
- Session-specific prompts have equal shape but are not byte-identical.
- The comparison is concurrent but synthetic and does not establish a production effect.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_async_fixed_worker_20261005_01 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ9 WORK_AUDIT_STUDY=multisession_compare WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late_nonblocking-early WORK_AUDIT_PAIRS=1 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_async_fixed_worker_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_async_fixed_worker_20261005_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_async_fixed_worker_20261005_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_async_fixed_worker_20261005_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_async_fixed_worker_20261005_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_async_fixed_scheduler_20261005_02"></a>
<details>
<summary><strong>2026-10-05 18:26:11 UTC · Concurrent early vs late · scheduler load</strong> · work_audit_async_fixed_scheduler_20261005_02</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** Early loading sped the long replay and whole workflow, but delayed the short session.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution scheduler. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. 1 warmup pair were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 89.5 | 991.1 | 7896.4 | -1078.5 |
| Trial 1 · late_nonblocking | 276.0 | 810.7 | 8089.1 | 201.7 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- The two-prefix cap is a test policy enforced by explicit evictions, not measured physical occupancy.
- Session-specific prompts have equal shape but are not byte-identical.
- The comparison is concurrent but synthetic and does not establish a production effect.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_async_fixed_scheduler_20261005_02 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ9 WORK_AUDIT_STUDY=multisession_compare WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late_nonblocking-early WORK_AUDIT_PAIRS=1 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=0 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_async_fixed_scheduler_20261005_02/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_async_fixed_scheduler_20261005_02/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_async_fixed_scheduler_20261005_02/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_async_fixed_scheduler_20261005_02/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_async_fixed_scheduler_20261005_02/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_async_scheduler_busy_20261005_01"></a>
<details>
<summary><strong>2026-10-05 18:12:20 UTC · Busy workload · KV-load attribution · scheduler load</strong> · work_audit_async_scheduler_busy_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 421934.4 | 90257.4 | 0 |
| Seed 1 · check_only | 36 | 423615.9 | 90480.8 | 0 |
| Seed 1 · controller | 36 | 436774.1 | 91272.0 | 3 |

**Evidence gate.** complete. Timestamp: First request (UTC).

**Limits**

- A load control-to-confirmation window is not the physical CUDA copy interval. The CUDA-event duration can overlap CPU calls and must not be added to their wall times. Stage differences isolate check-only from check-plus-load policy, but concurrent batch trajectories can diverge; they do not prove HBM bandwidth contention. Missing request-linked stages are reported as missing coverage, never zero delay.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_async_scheduler_busy_20261005_01' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ9' WORK_AUDIT_SEEDS='1' WORK_AUDIT_MODES='baseline check_only controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' WORK_AUDIT_LOAD_EXECUTION='scheduler' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_async_scheduler_busy_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_async_scheduler_busy_20261005_01/run_manifest.json)

</details>

<a id="run-work_audit_async_worker_busy_20261005_01"></a>
<details>
<summary><strong>2026-10-05 18:03:46 UTC · Busy workload · KV-load attribution · worker load</strong> · work_audit_async_worker_busy_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution worker, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 423918.7 | 92364.6 | 0 |
| Seed 1 · check_only | 36 | 420453.7 | 92069.0 | 0 |
| Seed 1 · controller | 36 | 442849.6 | 91877.7 | 2 |

**Evidence gate.** complete. Timestamp: First request (UTC).

**Limits**

- A load control-to-confirmation window is not the physical CUDA copy interval. The CUDA-event duration can overlap CPU calls and must not be added to their wall times. Stage differences isolate check-only from check-plus-load policy, but concurrent batch trajectories can diverge; they do not prove HBM bandwidth contention. Missing request-linked stages are reported as missing coverage, never zero delay.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_async_worker_busy_20261005_01' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ9' WORK_AUDIT_SEEDS='1' WORK_AUDIT_MODES='baseline check_only controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' WORK_AUDIT_LOAD_EXECUTION='worker' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_async_worker_busy_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_async_worker_busy_20261005_01/run_manifest.json)

</details>

<a id="run-work_audit_async_worker_verify_20261005_01"></a>
<details>
<summary><strong>2026-10-05 18:00:42 UTC · Lifecycle validation · worker load</strong> · work_audit_async_worker_verify_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: host → warm; tool wait: 500 ms; replays/case: 2; measured pairs: 1; warmup pairs: 0; trace: kv_lifecycle; exact-index limit: 256; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load. Prompt target: 4090 words; output cap: 16 tokens; minimum host prefix: 512 tokens; eviction attempts: 4; host cache: 8 GB; GPU memory fraction: 0.7.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 222.7 | 0 | 0 | 4163 |
| host_backed | 899.5 | 2048 | 4096 | 4162 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_async_worker_verify_20261005_01 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ9 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle WORK_AUDIT_CASE_ORDER=host-warm WORK_AUDIT_SECOND_REPLAY=1 WORK_AUDIT_WAIT_MS=500 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_load_kernel_20261005_01"></a>
<details>
<summary><strong>2026-10-05 16:35:35 UTC · Busy workload · KV-load attribution</strong> · work_audit_load_kernel_20261005_01</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend kernel, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 437550.8 | 94228.2 | 0 |
| Seed 1 · check_only | 36 | 435266.4 | 94086.8 | 0 |
| Seed 1 · controller | 36 | 452880.2 | 95508.3 | 3 |

**Evidence gate.** complete. Timestamp: First request (UTC).

**Limits**

- A load control-to-confirmation window is not the physical CUDA copy interval. The CUDA-event duration can overlap CPU calls and must not be added to their wall times. Stage differences isolate check-only from check-plus-load policy, but concurrent batch trajectories can diverge; they do not prove HBM bandwidth contention. Missing request-linked stages are reported as missing coverage, never zero delay.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_load_kernel_20261005_01' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ8' WORK_AUDIT_SEEDS='1' WORK_AUDIT_MODES='baseline check_only controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_load_kernel_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_load_kernel_20261005_01/run_manifest.json)

</details>

<a id="run-work_audit_load_phase_20261005_01"></a>
<details>
<summary><strong>2026-10-05 14:31:03 UTC · Busy workload · KV-load attribution</strong> · work_audit_load_phase_20261005_01</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 420768.6 | 94256.0 | 0 |
| Seed 1 · check_only | 36 | 423830.5 | 92192.5 | 0 |
| Seed 1 · controller | 36 | 435562.2 | 92621.3 | 3 |

**Evidence gate.** complete. Timestamp: First request (UTC).

**Limits**

- A load control-to-confirmation window is not the physical CUDA copy interval. The CUDA-event duration can overlap CPU calls and must not be added to their wall times. Stage differences isolate check-only from check-plus-load policy, but concurrent batch trajectories can diverge; they do not prove HBM bandwidth contention. Missing request-linked stages are reported as missing coverage, never zero delay.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_load_phase_20261005_01' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ8' WORK_AUDIT_SEEDS='1' WORK_AUDIT_MODES='baseline check_only controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_load_phase_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_load_phase_20261005_01/run_manifest.json)

</details>

<a id="run-work_audit_kv_attribution_20261002_04"></a>
<details>
<summary><strong>2026-10-03 01:19:35 UTC · Busy workload · KV-load attribution</strong> · work_audit_kv_attribution_20261002_04</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 443656.2 | 92901.0 | 0 |
| Seed 1 · check_only | 36 | 452951.0 | 93159.9 | 0 |
| Seed 1 · controller | 36 | 463992.0 | 95750.0 | 0 |
| Seed 2 · baseline | 36 | 444470.7 | 94304.5 | 0 |
| Seed 2 · check_only | 36 | 442884.2 | 92321.8 | 0 |
| Seed 2 · controller | 36 | 462304.9 | 95437.8 | 0 |

**Evidence gate.** complete. Timestamp: First request (UTC).

**Limits**

- A load control-to-confirmation window is not the physical CUDA copy interval. Stage differences isolate check-only from check-plus-load policy, but concurrent batch trajectories can diverge; they do not prove HBM bandwidth contention. Missing request-linked stages are reported as missing coverage, never zero delay.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_kv_attribution_20261002_04' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ8' WORK_AUDIT_SEEDS='1 2' WORK_AUDIT_MODES='baseline check_only controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_kv_attribution_20261002_04/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_kv_attribution_20261002_04/run_manifest.json)

</details>

<a id="run-work_audit_busy_pair_20261002_01"></a>
<details>
<summary><strong>2026-10-02 22:14:42 UTC · Busy workload · controller KV timing</strong> · work_audit_busy_pair_20261002_01</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** Controller-timed KV preparation worsened workflow and total replay TTFT in every paired seed.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; lean KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Native loads |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 335410.0 | 78052.2 | 35 |
| Seed 1 · controller | 36 | 362379.4 | 80071.5 | 43 |
| Seed 2 · baseline | 36 | 334372.1 | 79541.3 | 36 |
| Seed 2 · controller | 36 | 359160.5 | 79872.8 | 47 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- Synthetic coding-task prompts and tool waits, not measured production trajectories.
- Arms use fresh backends and identical recipes, but concurrent batch and eviction paths may diverge.
- A changed latency is not by itself proof of HBM contention or avoidable KV movement.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_busy_pair_20261002_01' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ8' WORK_AUDIT_SEEDS='1 2' WORK_AUDIT_MODES='baseline controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_busy_pair_20261002_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_busy_pair_20261002_01/run_manifest.json)

</details>

<a id="run-work_audit_busy_pilot_20261002"></a>
<details>
<summary><strong>2026-10-02 22:08:50 UTC · Busy workload · controller KV timing</strong> · work_audit_busy_pilot_20261002</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** Controller-timed KV preparation worsened workflow and total replay TTFT in every paired seed.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; lean KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Native loads |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 335038.7 | 79697.6 | 35 |
| Seed 1 · controller | 36 | 355726.5 | 79846.9 | 43 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- Synthetic coding-task prompts and tool waits, not measured production trajectories.
- Arms use fresh backends and identical recipes, but concurrent batch and eviction paths may diverge.
- A changed latency is not by itself proof of HBM contention or avoidable KV movement.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='work_audit_busy_pilot_20261002' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ8' WORK_AUDIT_SEEDS='1' WORK_AUDIT_MODES='baseline controller' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_TOOL_WAITS='3' WORK_AUDIT_PREFIX_TOKENS='8192' WORK_AUDIT_REPLAY_TOKENS='64' WORK_AUDIT_WAIT_MIN_MS='800' WORK_AUDIT_WAIT_MAX_MS='3500' WORK_AUDIT_ESTIMATED_LOAD_MS='250.0' WORK_AUDIT_MARGIN_MS='150.0' WORK_AUDIT_MINIMUM_HOST_TOKENS='512' HICACHE_SIZE_GB='8.0' MEM_FRACTION_STATIC='0.8' bash infra/container/run_work_audit_busy.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_busy_pilot_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_busy_pilot_20261002/run_manifest.json)

</details>

<a id="run-work_audit_controller_window_20261002_01"></a>
<details>
<summary><strong>2026-10-02 21:16:50 UTC · Controller-chosen load window</strong> · work_audit_controller_window_20261002_01</summary>

**Question (RQ7).** Using observed short-replay completion, host residency, slot release, and the long tool-return estimate, can a controller policy decide when to load KV without frontend importance ranks?

**Finding.** The controller improved long replay and workflow time versus late loading in every matched trial.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution scheduler. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. The third mode requested load immediately after observing the short replay finish, before long tool return. The fourth mode let the controller decide using a 250 ms load estimate and 150 ms margin. 1 warmup trial were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · controller_window | 89.2 | 816.3 | 7908.3 | -584.1 |
| Trial 1 · post_short | 82.0 | 815.9 | 7913.9 | -571.3 |
| Trial 1 · early | 82.4 | 1028.3 | 7936.1 | -1076.2 |
| Trial 1 · late_nonblocking | 291.6 | 830.0 | 8157.4 | 218.0 |
| Trial 2 · late_nonblocking | 633.0 | 845.8 | 8516.7 | 241.9 |
| Trial 2 · early | 81.7 | 1069.3 | 7978.3 | -1034.2 |
| Trial 2 · post_short | 82.3 | 851.7 | 7998.6 | -511.6 |
| Trial 2 · controller_window | 82.2 | 859.7 | 8062.1 | -500.0 |
| Trial 3 · controller_window | 89.0 | 859.5 | 8085.2 | -488.1 |
| Trial 3 · post_short | 82.0 | 871.8 | 8101.0 | -471.0 |
| Trial 3 · early | 81.8 | 1141.3 | 8109.9 | -1008.4 |
| Trial 3 · late_nonblocking | 350.0 | 883.7 | 8403.2 | 276.6 |
| Trial 4 · late_nonblocking | 353.5 | 885.7 | 8434.7 | 279.9 |
| Trial 4 · early | 82.0 | 1180.9 | 8179.4 | -968.2 |
| Trial 4 · post_short | 81.5 | 911.1 | 8179.8 | -397.2 |
| Trial 4 · controller_window | 81.7 | 920.2 | 8201.7 | -385.2 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- Controller used a fixed predeclared load-time estimate, not exact future runtime.
- The after-short policy uses an observed client completion event, not a production prediction.
- The two-prefix cap is enforced by explicit evictions, not measured physical occupancy.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_controller_window_20261002_01 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ7 WORK_AUDIT_STUDY=multisession_controller WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late_nonblocking-early-post_short-controller_window WORK_AUDIT_PAIRS=4 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_controller_window_20261002_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_controller_window_20261002_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_controller_window_20261002_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_controller_window_20261002_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_controller_window_20261002_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_post_short_20261002_01"></a>
<details>
<summary><strong>2026-10-02 20:34:20 UTC · Three concurrent load windows</strong> · work_audit_post_short_20261002_01</summary>

**Question (RQ6).** Under the same equal-importance, logical two-prefix workload, can loading just after the short replay completes preserve the long replay benefit while avoiding the short-session penalty?

**Finding.** Post-short loading improved long replay and workflow time versus late loading, while sparing the short session versus early loading.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution scheduler. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. The third mode requested load immediately after observing the short replay finish, before long tool return. 1 warmup trial were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · post_short | 89.4 | 811.2 | 7902.5 | -590.0 |
| Trial 1 · early | 82.4 | 1014.9 | 7901.8 | -1089.7 |
| Trial 1 · late_nonblocking | 277.6 | 817.8 | 8110.6 | 203.8 |
| Trial 2 · late_nonblocking | 607.0 | 820.3 | 8463.7 | 214.7 |
| Trial 2 · early | 81.8 | 1040.4 | 7945.7 | -1063.6 |
| Trial 2 · post_short | 81.4 | 831.8 | 7953.6 | -545.7 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- The after-short policy uses an observed client completion event, not a production prediction.
- The two-prefix cap is enforced by explicit evictions, not measured physical occupancy.
- Session-specific prompts have equal shape but are not byte-identical.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_post_short_20261002_01 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ6 WORK_AUDIT_STUDY=multisession_window WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late_nonblocking-early-post_short WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_post_short_20261002_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_post_short_20261002_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_post_short_20261002_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_post_short_20261002_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_post_short_20261002_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_concurrent_compare_20261002_02"></a>
<details>
<summary><strong>2026-10-02 19:48:50 UTC · Concurrent early vs late</strong> · work_audit_concurrent_compare_20261002_02</summary>

**Question (RQ5).** With equal-importance sessions and the same logical two-prefix budget, does loading during the wait help the returning session without delaying another session or the whole workflow?

**Finding.** Early loading sped the long replay and whole workflow, but delayed the short session.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution scheduler. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. 1 warmup pair were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 89.6 | 990.3 | 7896.1 | -1079.7 |
| Trial 1 · late_nonblocking | 290.3 | 812.0 | 8108.1 | 214.8 |
| Trial 2 · late_nonblocking | 595.0 | 817.6 | 8417.9 | 201.9 |
| Trial 2 · early | 82.3 | 1012.1 | 7915.1 | -1092.3 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- The two-prefix cap is a test policy enforced by explicit evictions, not measured physical occupancy.
- Session-specific prompts have equal shape but are not byte-identical.
- The comparison is concurrent but synthetic and does not establish a production effect.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_concurrent_compare_20261002_02 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ5 WORK_AUDIT_STUDY=multisession_compare WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late_nonblocking-early WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_concurrent_compare_20261002_02/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_concurrent_compare_20261002_02/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_concurrent_compare_20261002_02/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_concurrent_compare_20261002_02/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_concurrent_compare_20261002_02/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_multisession_20261002_01"></a>
<details>
<summary><strong>2026-10-02 18:42:21 UTC · Concurrent timeline</strong> · work_audit_multisession_20261002_01</summary>

**Question (RQ4).** Can a small, equal-importance, concurrent workload link tool waits, one controlled host eviction/load, replays, and an ending session?

**Finding.** The trace linked overlapping tool waits, host-KV movement, and replays across separate sessions.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; trace kv_lifecycle_lean. Three equal-importance sessions began together. Two tool waits overlapped; the third session ended without replay. An explicit control command evicted the long-wait session's GPU prefix under a synthetic two-prefix budget, then the client proved host residency. Its load was requested when the tool returned without gating replay on the control response. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. No frontend task had higher semantic priority.

**Key measurements**

| Session | Tool wait (ms) | First token after tool (ms) | Replay TTFT (ms) | Cached prefix tokens |
| --- | --- | --- | --- | --- |
| short | 901.5 | 81.6 | 81.5 | 4177 |
| long | 2501.3 | 282.4 | 282.0 | 4177 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- Explicit test-budget eviction is not evidence of organic backend capacity pressure.
- The three initial sessions are candidates; only the long session has direct eviction and host-residency proof.
- A cache match is not proof that model kernels consumed those exact KV slots.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_multisession_20261002_01 WORK_AUDIT_STUDY=multisession WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=2500 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_multisession_20261002_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_multisession_20261002_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_multisession_20261002_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_multisession_20261002_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_multisession_20261002_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_nonblocking_20261002_01"></a>
<details>
<summary><strong>2026-10-02 18:33:37 UTC · Early vs late</strong> · work_audit_nonblocking_20261002_01</summary>

**Question (RQ3).** Does not waiting for the late-load control response remove the observed tool-return-to-first-token penalty?

**Finding.** Nonblocking submission shortened the client gap, but the strict nonblocking comparison was withheld.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: early → late → late_nonblocking; tool wait: 2000 ms; replays/case: 2; measured pairs: 1; warmup pairs: 1; trace: kv_lifecycle_lean; exact-index limit: 256; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load. Prompt target: 4090 words; output cap: 16 tokens; minimum host prefix: 512 tokens; eviction attempts: 4; host cache: 8 GB; GPU memory fraction: 0.7.

**Key measurements**

| Trial / mode | Submit after due (ms) | First token after due (ms) | Replay TTFT (ms) | Task duration (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 0.1 | 88.2 | 88.0 | 7455.2 |
| Trial 1 · late | 170.4 | 253.3 | 82.9 | 7014.9 |
| Trial 1 · late_nonblocking | 0.5 | 243.7 | 243.2 | 7265.2 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- work_audit_nonblocking_20261002_01-pair01-early: exact loaded-slot lineage was not proved
- work_audit_nonblocking_20261002_01-pair01-late: exact loaded-slot lineage was not proved
- work_audit_nonblocking_20261002_01-pair01-late_nonblocking: exact loaded-slot lineage was not proved

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_nonblocking_20261002_01 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=early-late-late_nonblocking WORK_AUDIT_PAIRS=1 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 WORK_AUDIT_EXACT_INDICES=256 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_nonblocking_20261002_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_nonblocking_20261002_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_nonblocking_20261002_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_nonblocking_20261002_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_nonblocking_20261002_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_timing_sampled_late_early_20261002"></a>
<details>
<summary><strong>2026-10-02 16:41:32 UTC · Early vs late</strong> · work_audit_timing_sampled_late_early_20261002</summary>

**Question (RQ2).** In a controlled replay, does requesting host-KV load during a tool wait shorten time from tool return to first token compared with requesting it after the wait?

**Finding.** Loading during the tool wait brought the first token sooner in every matched trial.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: late → early; tool wait: 2000 ms; replays/case: 2; measured pairs: 2; warmup pairs: 1; trace: kv_lifecycle_lean; exact-index limit: 256; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Trial / mode | Submit after due (ms) | First token after due (ms) | Replay TTFT (ms) | Task duration (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · late | 163.9 | 253.6 | 89.7 | 7898.8 |
| Trial 1 · early | 0.2 | 83.5 | 83.3 | 7130.8 |
| Trial 2 · late | 172.9 | 256.5 | 83.6 | 7270.7 |
| Trial 2 · early | 0.1 | 82.7 | 82.6 | 7095.5 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- work_audit_timing_sampled_late_early_20261002-pair01-late: exact loaded-slot lineage was not proved
- work_audit_timing_sampled_late_early_20261002-pair01-early: exact loaded-slot lineage was not proved
- work_audit_timing_sampled_late_early_20261002-pair02-late: exact loaded-slot lineage was not proved

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_timing_sampled_late_early_20261002 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late-early WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=256 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_timing_sampled_late_early_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_timing_sampled_late_early_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_timing_sampled_late_early_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_timing_sampled_late_early_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_timing_sampled_late_early_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_timing_sampled_early_late_20261002"></a>
<details>
<summary><strong>2026-10-02 16:38:54 UTC · Early vs late</strong> · work_audit_timing_sampled_early_late_20261002</summary>

**Question (RQ2).** In a controlled replay, does requesting host-KV load during a tool wait shorten time from tool return to first token compared with requesting it after the wait?

**Finding.** Loading during the tool wait brought the first token sooner in every matched trial.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: early → late; tool wait: 2000 ms; replays/case: 2; measured pairs: 2; warmup pairs: 1; trace: kv_lifecycle_lean; exact-index limit: 256; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Trial / mode | Submit after due (ms) | First token after due (ms) | Replay TTFT (ms) | Task duration (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 0.1 | 88.8 | 88.7 | 7728.9 |
| Trial 1 · late | 170.7 | 255.2 | 84.5 | 7275.1 |
| Trial 2 · early | 0.1 | 82.8 | 82.7 | 7101.3 |
| Trial 2 · late | 177.2 | 260.9 | 83.7 | 7295.7 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Limits**

- work_audit_timing_sampled_early_late_20261002-pair01-early: exact loaded-slot lineage was not proved
- work_audit_timing_sampled_early_late_20261002-pair01-late: exact loaded-slot lineage was not proved
- work_audit_timing_sampled_early_late_20261002-pair02-early: exact loaded-slot lineage was not proved

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_timing_sampled_early_late_20261002 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=early-late WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=256 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_timing_sampled_early_late_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_timing_sampled_early_late_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_timing_sampled_early_late_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_timing_sampled_early_late_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_timing_sampled_early_late_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_timing_exact_late_early_20261002"></a>
<details>
<summary><strong>2026-10-02 16:25:50 UTC · Early vs late</strong> · work_audit_timing_exact_late_early_20261002</summary>

**Question (RQ2).** In a controlled replay, does requesting host-KV load during a tool wait shorten time from tool return to first token compared with requesting it after the wait?

**Finding.** Loading during the tool wait brought the first token sooner in every matched trial.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: late → early; tool wait: 2000 ms; replays/case: 2; measured pairs: 2; warmup pairs: 1; trace: kv_lifecycle_lean; exact-index limit: 4096; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Trial / mode | Submit after due (ms) | First token after due (ms) | Replay TTFT (ms) | Task duration (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · late | 220.9 | 312.3 | 91.4 | 7333.8 |
| Trial 1 · early | 0.1 | 83.6 | 83.5 | 7107.0 |
| Trial 2 · late | 228.8 | 312.4 | 83.7 | 7330.6 |
| Trial 2 · early | 0.1 | 83.8 | 83.7 | 7130.1 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_timing_exact_late_early_20261002 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late-early WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=4096 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_timing_exact_early_late_20261002"></a>
<details>
<summary><strong>2026-10-02 16:22:26 UTC · Early vs late</strong> · work_audit_timing_exact_early_late_20261002</summary>

**Question (RQ2).** In a controlled replay, does requesting host-KV load during a tool wait shorten time from tool return to first token compared with requesting it after the wait?

**Finding.** Loading during the tool wait brought the first token sooner in every matched trial.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: early → late; tool wait: 2000 ms; replays/case: 2; measured pairs: 2; warmup pairs: 1; trace: kv_lifecycle_lean; exact-index limit: 4096; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Trial / mode | Submit after due (ms) | First token after due (ms) | Replay TTFT (ms) | Task duration (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 0.1 | 89.9 | 89.8 | 7158.3 |
| Trial 1 · late | 231.1 | 316.7 | 85.6 | 7336.8 |
| Trial 2 · early | 0.2 | 83.6 | 83.4 | 7113.2 |
| Trial 2 · late | 236.1 | 320.1 | 84.0 | 7338.8 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_timing_exact_early_late_20261002 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=early-late WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=4096 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_lean_reverse_20261002"></a>
<details>
<summary><strong>2026-10-02 15:34:04 UTC · Lifecycle validation</strong> · work_audit_two_replays_lean_reverse_20261002</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: host → warm; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle_lean; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 82.3 | 0 | 0 | 4178 |
| host_backed | 436.3 | 2048 | 4096 | 4177 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_lean_reverse_20261002 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=host-warm WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_lean_pump_20261002"></a>
<details>
<summary><strong>2026-10-02 15:26:59 UTC · Lifecycle validation</strong> · work_audit_two_replays_lean_pump_20261002</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: warm → host; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle_lean; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 81.5 | 0 | 0 | 4179 |
| host_backed | 82.6 | 2048 | 4096 | 4178 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_lean_pump_20261002 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=warm-host WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_20261002_reverse"></a>
<details>
<summary><strong>2026-10-02 14:54:21 UTC · Lifecycle validation</strong> · work_audit_two_replays_20261002_reverse</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: host → warm; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 217.1 | 0 | 0 | 4161 |
| host_backed | 573.9 | 2048 | 4096 | 4160 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_20261002_reverse WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle WORK_AUDIT_CASE_ORDER=host-warm WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_20261002"></a>
<details>
<summary><strong>2026-10-02 14:50:49 UTC · Lifecycle validation</strong> · work_audit_two_replays_20261002</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: warm → host; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 220.9 | 0 | 0 | 4160 |
| host_backed | 229.1 | 2048 | 4096 | 4159 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_20261002 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle WORK_AUDIT_CASE_ORDER=warm-host WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_a10g_20261001_final"></a>
<details>
<summary><strong>2026-10-01 22:42:31 UTC · Lifecycle validation</strong> · work_audit_a10g_20261001_final</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: warm → host; tool wait: not recorded ms; replays/case: not recorded; measured pairs: not recorded; warmup pairs: not recorded; trace: not recorded; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 227.1 | 0 | 0 | 4162 |
| host_backed | 230.3 | 2048 | 4096 | 4161 |

**Evidence gate.** validated. Timestamp: First request (UTC).

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_a10g_20261001_final WORK_AUDIT_STUDY=validation WORK_AUDIT_CASE_ORDER=warm-host bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_a10g_20261001_final/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_a10g_20261001_final/run_manifest.json)

</details>
