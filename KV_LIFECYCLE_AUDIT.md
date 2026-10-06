# KV Lifecycle Audit

**Research question:** Given what the harness knew, was KV moved, kept, or rebuilt at the wrong time?

This audit compares observed cache work with replay timing and whole-workload outcomes. A faster control call is not automatically a faster agent task. All times below are from saved runs; experimental and hypothetical claims are kept separate.

## Research progress

### RQ13: Can existing backend features absorb KV-load overlap?

**Question.** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**What the evidence says.** In matched six-session, 2048-word, 96-output-token A10G runs, four worker KV loads added 475 and 446 ms to target tool-return-to-completion time with both features off (13.5% and 12.6%). With CUDA graphs and overlap scheduling both enabled, the added time was 6.6 and 1.2 ms (0.18% and 0.03%) across the same two seeds. Single-seed checks found +142 ms (4.1%) with graphs alone and +458 ms (11.4%) with overlap scheduling alone. First-token timing was essentially unchanged within each mode. One separate profiler run with both features on verified four physical host-to-GPU loads during target decode: 37.145 ms of copy activity in the decode interval, but only 0.044 ms concurrent with recorded decode kernels. Thus existing software features largely removed the measured incremental overlap penalty in this small workload, without proving a hardware bandwidth effect or hardware-offload benefit.

**Working hypothesis.** The original slowdown depends strongly on backend work-submission behavior; CUDA graphs together with overlap scheduling may keep decode work moving despite concurrent KV-load orchestration.

**Not yet proved.** Two seeds cover only the both-off and both-on comparison; single-feature arms have one seed. The combined mode raised no-load target first-token time from about 65-66 ms to 385-387 ms and no-load completion from about 3.53 to 3.65 seconds, so minimal added overlap cost is not an unconditional win. Worker-window overlap is a proxy in the unprofiled comparisons; the separate Nsight run verifies physical copy overlap but barely any copy/kernel concurrency and must not be used for latency estimates. The 2048-word prompts were chosen because a 4090-word graph-enabled four-load attempt hit KV-capacity admission limits. The exact reason the combined features remove the incremental penalty, production prevalence, and any hardware benefit remain unproven. The first-seed summaries retain their reused RQ11 driver label; their run manifests and this milestone identify them as RQ13, and later summaries emit RQ13 directly.

### RQ12: Where does the overlap slowdown occur?

**Question.** When native host-to-GPU KV copies physically overlap another session's decode, does added time appear inside decode kernels or in the host's cadence of submitting them?

**What the evidence says.** Two order-balanced six-session A10G profile pairs each captured zero versus four physical KV-load overlaps while the target had the same 94 model forwards and 32,712 linked kernels. Four loads contributed about 73 ms of physical copy overlap in each overlap arm. Summed decode-kernel execution changed by only +1.1 and +1.4 ms, while time inside forwards before the CPU began the next CUDA launch grew by 481 and 546 ms; gaps between forwards grew by another 135 and 138 ms. Earlier unprofiled six-session runs found 21-23% longer target completion with four worker-window overlaps. The profile supports delayed host-side launch cadence, not slower decode kernels, as the main measured locus of the delay.

**Working hypothesis.** KV-load orchestration or competition for host-side scheduler/launch resources delays submission of subsequent decode work. Physical copies do overlap kernels, but their direct bandwidth effect is not shown to explain the large completion penalty.

**Not yet proved.** The profiler locates time before CUDA launches but does not show why the host waited: Python/CPU contention, SGLang scheduling, batching, copy-launch overhead, or a mixture remain possible. Nsight perturbs timing, so profiled completion differences are not the clean slowdown estimate. This is synthetic traffic on one A10G and does not establish a hardware-offload speedup or a production-wide frequency. The first reverse-order pair had a post-run manifest recovery and a corrected research-question label; raw traces were unchanged.

### RQ11: Does more KV-load overlap delay other decoders?

**Question.** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**What the evidence says.** On the pinned A10G setup, both six- and twelve-session pilots realized 0, 1, 2, and 4 worker-window overlaps as scheduled. Target tool-return-to-finish time rose with dose: 3.631 to 4.399 seconds in the six-session first seed (+21.2%) and 3.895 to 4.640 seconds in the twelve-session first seed (+19.1%). Zero-versus-four endpoints repeated in a second seed: 3.630 to 4.475 seconds (+23.3%) with six sessions and 3.896 to 4.683 seconds (+20.2%) with twelve. Peer decoders finished later in the high-dose arms too. First-token timing did not rise consistently, so the extra time was mainly after first token. All doses executed the same four native donor loads; only timing changed within each series.

**Working hypothesis.** The added decode time may come primarily from delayed host-side submission of GPU work while KV loads are active, rather than slower decode kernels or direct HBM-bandwidth contention. The partial RQ10 captures suggested this; the RQ11 sweep itself did not capture enough physical CUDA activity to test it. RQ12 later tests this hypothesis in a separate six-session profile.

**Not yet proved.** Worker start-to-commit windows are proxies, not verified physical host-to-GPU copy overlap. A profiled repeat recorded all four NVTX load ranges but no CUDA kernel or memcpy activity, so the physical-overlap gate failed. The cause could be host launch cadence, scheduler/batch effects, GPU copies, or a mixture; this does not isolate HBM bandwidth or prove a hardware fix. The six-session series used 4090-word active prompts and 20- or 40-second donor waits, while the twelve-session series used 512-word active prompts and 40-second donor waits to fit capacity; compare doses within each series, not absolute times across series. These synthetic runs are small, and only zero/four endpoints have a second seed. Excluded capacity and timeout attempts appear in the experiment details.

### RQ10: Which stage slows another decoder?

**Question.** When an early worker KV load overlaps a different session's decode, is that session delayed before its first token, between backend batches, or inside model forward?

**What the evidence says.** In three warmed, order-balanced A10G pairs, early loading made the short session finish 128-182 ms later than loading after that session finished; first-token timing was nearly unchanged. Backend batch time increased 173-225 ms while gaps between batches decreased. A follow-up two-pair run placed 185-190 ms of added time inside model forward. In two later Nsight captures, the first pair in each had complete CUDA linkage: both modes launched 4872 short-replay kernels, total kernel execution changed by only 0.3-0.6 ms, but between-kernel gaps grew by 126 and 174 ms. Of those added gaps, 122 and 167 ms occurred before the CPU began the next CUDA launch; launch-call and after-launch time changed much less. Recorded stream-wait activity was about 1 ms per arm, and no blocking CUDA synchronization API was observed in those forward calls. No host-to-device copy was recorded during the short-forward kernel spans. The captured evidence points to delayed host-side launch cadence, not slower kernels or direct concurrent copy-bandwidth contention.

**Not yet proved.** Nsight lost CUDA activity for later cases in both two-pair profiling runs; the launch-gap result is a validated subset, not a complete order-balanced profiler experiment. A reverse-order run had no CUDA kernel capture and is excluded. The trace does not show why the host delayed its next launch: CPU scheduling, Python work, other software waits, or competition from the worker remain candidates. Gaps between this request's kernels are not global GPU-idle measurements. The workload used explicit host eviction, a synthetic logical two-prefix budget, and a 10-second long tool wait; hardware offload benefits and production frequency remain unproven.

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

Newest first. Each arrow goes from the named control to the changed case in the **Compared** column; lower times are better. Three-session rows show the long replay's first token and the short session's finish after tool return. Busy rows show summed replay TTFT across all replays. Workflow is total elapsed time. Rows without a validated comparison have no arrow. For multi-trial runs, arrows compare each mode's median time, which can differ from the median trial-by-trial improvement. Select an experiment for exact trial values and limits.

| Central date / time | Experiment | Question | Setup | Compared | Replay / long session | Other session | Whole workflow | Plain-English finding | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:12:14&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_both_d4_profile_p2048_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | Observed 4 verified copies; no profiled control | Profiled&nbsp;mechanism;&nbsp;no&nbsp;latency&nbsp;claim | See&nbsp;target&nbsp;launch&nbsp;timing&nbsp;in&nbsp;details | Not&nbsp;used&nbsp;for&nbsp;speed&nbsp;comparison | 4&nbsp;of&nbsp;4&nbsp;native&nbsp;loads&nbsp;physically&nbsp;overlapped&nbsp;the&nbsp;target&nbsp;decode&nbsp;window.&nbsp;This&nbsp;profiled&nbsp;run&nbsp;establishes&nbsp;overlap,&nbsp;not&nbsp;an&nbsp;unprofiled&nbsp;performance&nbsp;effect&nbsp;or&nbsp;hardware&nbsp;cause. | physical copy verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:07:50&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_baseline_d4_p2048_w40_seed2_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3537.5&nbsp;ms&nbsp;→&nbsp;3983.3&nbsp;ms | 3538.0&nbsp;ms&nbsp;→&nbsp;3983.5&nbsp;ms | 72.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;446&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;445&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:05:16&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_baseline_d0_p2048_w40_seed2_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3537.5&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3538.0&nbsp;ms | 74.5&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:02:18&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_both_d4_p2048_w40_seed2_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3654.2&nbsp;ms&nbsp;→&nbsp;3655.3&nbsp;ms | 3654.0&nbsp;ms&nbsp;→&nbsp;3655.2&nbsp;ms | 61.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;1&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;1&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:59:36&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_both_d0_p2048_w40_seed2_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3654.2&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3654.0&nbsp;ms | 61.7&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:56:24&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_baseline_d0_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3530.0&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3530.5&nbsp;ms | 72.6&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:53:50&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_baseline_d4_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3530.0&nbsp;ms&nbsp;→&nbsp;4005.5&nbsp;ms | 3530.5&nbsp;ms&nbsp;→&nbsp;4005.4&nbsp;ms | 72.7&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;475&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;475&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:51:15&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_both_d0_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3653.3&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3653.2&nbsp;ms | 62.0&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:48:33&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_both_d4_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3653.3&nbsp;ms&nbsp;→&nbsp;3659.9&nbsp;ms | 3653.2&nbsp;ms&nbsp;→&nbsp;3660.1&nbsp;ms | 61.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;7&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;7&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:45:31&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_overlap_d0_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;4002.5&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;4003.0&nbsp;ms | 80.6&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:42:55&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_overlap_d4_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 4002.5&nbsp;ms&nbsp;→&nbsp;4460.3&nbsp;ms | 4003.0&nbsp;ms&nbsp;→&nbsp;4460.2&nbsp;ms | 80.6&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;458&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;457&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:40:00&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_graph_d0_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3452.4&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3452.3&nbsp;ms | 55.7&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:37:09&nbsp;a.m.&nbsp;CDT | [Backend scheduling and KV overlap](#run-rq13_graph_d4_p2048_w40_20261006) | RQ13 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3452.4&nbsp;ms&nbsp;→&nbsp;3594.9&nbsp;ms | 3452.3&nbsp;ms&nbsp;→&nbsp;3594.8&nbsp;ms | 55.4&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;142&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;142&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;6,&nbsp;2026,&nbsp;12:21:23&nbsp;a.m.&nbsp;CDT | [Decode slowdown attribution](#run-rq12_s6_d4_profile_repeat_20261006) | RQ12 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 verified copies | Profiled&nbsp;mechanism;&nbsp;no&nbsp;latency&nbsp;claim | See&nbsp;target&nbsp;launch&nbsp;timing&nbsp;in&nbsp;details | Not&nbsp;used&nbsp;for&nbsp;speed&nbsp;comparison | 4&nbsp;physical&nbsp;KV&nbsp;loads&nbsp;overlapped&nbsp;decode.&nbsp;Versus&nbsp;the&nbsp;paired&nbsp;profiled&nbsp;control,&nbsp;CPU-before-launch&nbsp;gaps&nbsp;grew&nbsp;546&nbsp;ms&nbsp;while&nbsp;summed&nbsp;kernel&nbsp;execution&nbsp;changed&nbsp;+1.4&nbsp;ms.&nbsp;This&nbsp;locates&nbsp;delay&nbsp;in&nbsp;launch&nbsp;cadence;&nbsp;it&nbsp;does&nbsp;not&nbsp;identify&nbsp;why&nbsp;the&nbsp;host&nbsp;waited&nbsp;or&nbsp;measure&nbsp;an&nbsp;unprofiled&nbsp;speedup. | physical copy verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;12:17:50&nbsp;a.m.&nbsp;CDT | [Decode slowdown attribution](#run-rq12_s6_d0_profile_repeat_20261006) | RQ12 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Observed 0 verified copies; no profiled control | Profiled&nbsp;mechanism;&nbsp;no&nbsp;latency&nbsp;claim | See&nbsp;target&nbsp;launch&nbsp;timing&nbsp;in&nbsp;details | Not&nbsp;used&nbsp;for&nbsp;speed&nbsp;comparison | Profiled&nbsp;zero-overlap&nbsp;control:&nbsp;no&nbsp;KV&nbsp;copies&nbsp;ran&nbsp;during&nbsp;target&nbsp;decode.&nbsp;It&nbsp;captured&nbsp;94&nbsp;forwards&nbsp;and&nbsp;32712&nbsp;linked&nbsp;kernels&nbsp;for&nbsp;the&nbsp;paired&nbsp;launch&nbsp;comparison. | physical copy verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;12:13:04&nbsp;a.m.&nbsp;CDT | [Decode slowdown attribution](#run-rq12_s6_d0_profile_seed1_20261006) | RQ12 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Observed 0 verified copies; no profiled control | Profiled&nbsp;mechanism;&nbsp;no&nbsp;latency&nbsp;claim | See&nbsp;target&nbsp;launch&nbsp;timing&nbsp;in&nbsp;details | Not&nbsp;used&nbsp;for&nbsp;speed&nbsp;comparison | Profiled&nbsp;zero-overlap&nbsp;control:&nbsp;no&nbsp;KV&nbsp;copies&nbsp;ran&nbsp;during&nbsp;target&nbsp;decode.&nbsp;It&nbsp;captured&nbsp;94&nbsp;forwards&nbsp;and&nbsp;32712&nbsp;linked&nbsp;kernels&nbsp;for&nbsp;the&nbsp;paired&nbsp;launch&nbsp;comparison. | physical copy verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;12:04:41&nbsp;a.m.&nbsp;CDT | [Decode slowdown attribution](#run-rq12_s6_d4_profile_live_20261006) | RQ12 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 verified copies | Profiled&nbsp;mechanism;&nbsp;no&nbsp;latency&nbsp;claim | See&nbsp;target&nbsp;launch&nbsp;timing&nbsp;in&nbsp;details | Not&nbsp;used&nbsp;for&nbsp;speed&nbsp;comparison | 4&nbsp;physical&nbsp;KV&nbsp;loads&nbsp;overlapped&nbsp;decode.&nbsp;Versus&nbsp;the&nbsp;paired&nbsp;profiled&nbsp;control,&nbsp;CPU-before-launch&nbsp;gaps&nbsp;grew&nbsp;481&nbsp;ms&nbsp;while&nbsp;summed&nbsp;kernel&nbsp;execution&nbsp;changed&nbsp;+1.1&nbsp;ms.&nbsp;This&nbsp;locates&nbsp;delay&nbsp;in&nbsp;launch&nbsp;cadence;&nbsp;it&nbsp;does&nbsp;not&nbsp;identify&nbsp;why&nbsp;the&nbsp;host&nbsp;waited&nbsp;or&nbsp;measure&nbsp;an&nbsp;unprofiled&nbsp;speedup. | physical copy verified |
| Oct&nbsp;5,&nbsp;2026,&nbsp;9:09:29&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d4_shortactive_seed2_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3896.5&nbsp;ms&nbsp;→&nbsp;4682.6&nbsp;ms | 3896.4&nbsp;ms&nbsp;→&nbsp;4683.0&nbsp;ms | 79.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;786&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;787&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;9:06:47&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d0_shortactive_seed2_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3896.5&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3896.4&nbsp;ms | 79.0&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;9:03:55&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d4_seed2_longwait_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3630.4&nbsp;ms&nbsp;→&nbsp;4475.4&nbsp;ms | 3630.2&nbsp;ms&nbsp;→&nbsp;4475.4&nbsp;ms | 80.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;845&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;845&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;9:01:17&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d0_seed2_longwait_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3630.4&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3630.2&nbsp;ms | 78.1&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:58:45&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d0_seed2_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Excluded diagnostic | No&nbsp;comparable&nbsp;replay | No&nbsp;comparable&nbsp;peer | Not&nbsp;measured | Excluded&nbsp;diagnostic:&nbsp;The&nbsp;four&nbsp;post-decode&nbsp;native&nbsp;loads&nbsp;overshot&nbsp;the&nbsp;20-second&nbsp;donor&nbsp;tool-return&nbsp;window.&nbsp;A&nbsp;paired&nbsp;40-second&nbsp;control&nbsp;and&nbsp;four-load&nbsp;arm&nbsp;replaced&nbsp;this&nbsp;attempt. | excluded |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:56:00&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d1_shortactive_seed1_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;1&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 1 worker-window proxies | 3895.4&nbsp;ms&nbsp;→&nbsp;4050.3&nbsp;ms | 3895.5&nbsp;ms&nbsp;→&nbsp;4051.0&nbsp;ms | 79.0&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;155&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;155&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:53:28&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d2_shortactive_seed1_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;2&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 2 worker-window proxies | 3895.4&nbsp;ms&nbsp;→&nbsp;4239.7&nbsp;ms | 3895.5&nbsp;ms&nbsp;→&nbsp;4239.8&nbsp;ms | 79.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;344&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;344&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:50:48&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d4_shortactive_seed1_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3895.4&nbsp;ms&nbsp;→&nbsp;4640.1&nbsp;ms | 3895.5&nbsp;ms&nbsp;→&nbsp;4640.1&nbsp;ms | 79.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;745&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;745&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:48:17&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d0_shortactive_seed1_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3895.4&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3895.5&nbsp;ms | 79.0&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:45:36&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s12_d0_seed1_20261005) | RQ11 | 12&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Excluded diagnostic | No&nbsp;comparable&nbsp;replay | No&nbsp;comparable&nbsp;peer | Not&nbsp;measured | Excluded&nbsp;diagnostic:&nbsp;At&nbsp;eight&nbsp;long&nbsp;active&nbsp;prompts,&nbsp;SGLang&nbsp;refused&nbsp;donor&nbsp;2's&nbsp;native&nbsp;load&nbsp;because&nbsp;of&nbsp;load-back&nbsp;threshold,&nbsp;quota,&nbsp;or&nbsp;memory&nbsp;pressure.&nbsp;The&nbsp;same&nbsp;four-load&nbsp;control&nbsp;could&nbsp;not&nbsp;be&nbsp;completed. | excluded |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:41:17&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d4_profile_seed1_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 4 → observed 4 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;4462.8&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;4463.0&nbsp;ms | 82.0&nbsp;s | Nsight&nbsp;recorded&nbsp;the&nbsp;worker&nbsp;ranges&nbsp;but&nbsp;no&nbsp;CUDA&nbsp;kernels&nbsp;or&nbsp;copies.&nbsp;Physical&nbsp;overlap&nbsp;is&nbsp;unverified;&nbsp;this&nbsp;profiled&nbsp;run&nbsp;is&nbsp;excluded&nbsp;from&nbsp;the&nbsp;clean&nbsp;timing&nbsp;comparison. | profiler incomplete |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:38:20&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d4_seed1_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;4&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 4 worker-window proxies | 3630.5&nbsp;ms&nbsp;→&nbsp;4399.3&nbsp;ms | 3630.4&nbsp;ms&nbsp;→&nbsp;4399.2&nbsp;ms | 60.7&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;769&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;769&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:36:04&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d2_seed1_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;2&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 2 worker-window proxies | 3630.5&nbsp;ms&nbsp;→&nbsp;3989.7&nbsp;ms | 3630.4&nbsp;ms&nbsp;→&nbsp;3990.1&nbsp;ms | 60.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;359&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;360&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:33:33&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d1_seed1_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;1&nbsp;planned&nbsp;load&nbsp;overlaps | 0 → 1 worker-window proxies | 3630.5&nbsp;ms&nbsp;→&nbsp;3831.5&nbsp;ms | 3630.4&nbsp;ms&nbsp;→&nbsp;3831.4&nbsp;ms | 58.8&nbsp;s | Compared&nbsp;with&nbsp;its&nbsp;matched&nbsp;zero-overlap&nbsp;control,&nbsp;the&nbsp;target&nbsp;finished&nbsp;201&nbsp;ms&nbsp;later&nbsp;and&nbsp;peer&nbsp;decoders&nbsp;finished&nbsp;a&nbsp;median&nbsp;201&nbsp;ms&nbsp;later.&nbsp;Worker&nbsp;windows&nbsp;were&nbsp;observed;&nbsp;physical&nbsp;copy&nbsp;overlap&nbsp;and&nbsp;the&nbsp;precise&nbsp;cause&nbsp;remain&nbsp;unverified. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;8:31:13&nbsp;p.m.&nbsp;CDT | [KV-load overlap pressure](#run-rq11_s6_d0_seed1_20261005) | RQ11 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;4&nbsp;host-resident&nbsp;donor&nbsp;prefixes&nbsp;·&nbsp;0&nbsp;planned&nbsp;load&nbsp;overlaps | Planned 0 → observed 0 worker-window proxies; single dose | Target&nbsp;finish:&nbsp;3630.5&nbsp;ms | Other&nbsp;active&nbsp;decoders:&nbsp;median&nbsp;3630.4&nbsp;ms | 59.5&nbsp;s | Zero-overlap&nbsp;control:&nbsp;the&nbsp;same&nbsp;four&nbsp;donor&nbsp;loads&nbsp;ran&nbsp;only&nbsp;after&nbsp;target&nbsp;decode.&nbsp;This&nbsp;is&nbsp;the&nbsp;reference&nbsp;for&nbsp;other&nbsp;doses&nbsp;with&nbsp;the&nbsp;same&nbsp;seed&nbsp;and&nbsp;workload. | worker-window proxy only |
| Oct&nbsp;5,&nbsp;2026,&nbsp;4:54:36&nbsp;p.m.&nbsp;CDT | [Decode overlap attribution](#run-work_audit_cuda_kernels_graceful_20261005) | RQ10 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;early&nbsp;vs&nbsp;after-short&nbsp;worker&nbsp;load | GPU mechanism check; profiled timing not used | See&nbsp;clean&nbsp;RQ10&nbsp;run | See&nbsp;captured&nbsp;kernel&nbsp;table | No&nbsp;profiled&nbsp;workflow&nbsp;claim | In&nbsp;the&nbsp;captured&nbsp;pair,&nbsp;GPU&nbsp;kernels&nbsp;ran&nbsp;for&nbsp;about&nbsp;the&nbsp;same&nbsp;time;&nbsp;pauses&nbsp;between&nbsp;them&nbsp;grew.&nbsp;No&nbsp;H-to-D&nbsp;copy&nbsp;overlapped&nbsp;those&nbsp;kernel&nbsp;spans.&nbsp;Later&nbsp;profiler&nbsp;data&nbsp;was&nbsp;incomplete,&nbsp;so&nbsp;hardware&nbsp;attribution&nbsp;remains&nbsp;provisional. | validated timing; partial CUDA capture |
| Oct&nbsp;5,&nbsp;2026,&nbsp;4:45:44&nbsp;p.m.&nbsp;CDT | [Decode overlap attribution](#run-work_audit_cuda_kernels_20261005) | RQ10 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;early&nbsp;vs&nbsp;after-short&nbsp;worker&nbsp;load | GPU mechanism check; profiled timing not used | See&nbsp;clean&nbsp;RQ10&nbsp;run | See&nbsp;captured&nbsp;kernel&nbsp;table | No&nbsp;profiled&nbsp;workflow&nbsp;claim | In&nbsp;the&nbsp;captured&nbsp;pair,&nbsp;GPU&nbsp;kernels&nbsp;ran&nbsp;for&nbsp;about&nbsp;the&nbsp;same&nbsp;time;&nbsp;pauses&nbsp;between&nbsp;them&nbsp;grew.&nbsp;No&nbsp;H-to-D&nbsp;copy&nbsp;overlapped&nbsp;those&nbsp;kernel&nbsp;spans.&nbsp;Later&nbsp;profiler&nbsp;data&nbsp;was&nbsp;incomplete,&nbsp;so&nbsp;hardware&nbsp;attribution&nbsp;remains&nbsp;provisional. | validated timing; partial CUDA capture |
| Oct&nbsp;5,&nbsp;2026,&nbsp;4:05:08&nbsp;p.m.&nbsp;CDT | [Decode overlap attribution](#run-work_audit_decode_forward_repeated_20261005) | RQ10 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;early&nbsp;vs&nbsp;after-short&nbsp;worker&nbsp;load | After-short → early worker load, both before tool return | 121&nbsp;→&nbsp;281&nbsp;ms&nbsp;(160&nbsp;ms&nbsp;slower) | 819&nbsp;→&nbsp;972&nbsp;ms&nbsp;(153&nbsp;ms&nbsp;slower) | 15.4&nbsp;→&nbsp;15.6&nbsp;s&nbsp;(0.2&nbsp;s&nbsp;slower) | Early&nbsp;worker&nbsp;loading&nbsp;delayed&nbsp;the&nbsp;short&nbsp;response&nbsp;in&nbsp;every&nbsp;pair;&nbsp;the&nbsp;added&nbsp;batch&nbsp;time&nbsp;was&nbsp;in&nbsp;model&nbsp;forward,&nbsp;not&nbsp;queue&nbsp;gaps.&nbsp;HBM&nbsp;contention&nbsp;is&nbsp;not&nbsp;established. | validated |
| Oct&nbsp;5,&nbsp;2026,&nbsp;3:52:46&nbsp;p.m.&nbsp;CDT | [Decode overlap attribution](#run-work_audit_decode_overlap_repeated_20261005) | RQ10 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;3&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;early&nbsp;vs&nbsp;after-short&nbsp;worker&nbsp;load | After-short → early worker load, both before tool return | 123&nbsp;→&nbsp;118&nbsp;ms&nbsp;(6&nbsp;ms&nbsp;faster) | 826&nbsp;→&nbsp;955&nbsp;ms&nbsp;(130&nbsp;ms&nbsp;slower) | 15.4&nbsp;→&nbsp;15.5&nbsp;s&nbsp;(0.0&nbsp;s&nbsp;slower) | Early&nbsp;worker&nbsp;loading&nbsp;delayed&nbsp;the&nbsp;short&nbsp;response&nbsp;in&nbsp;every&nbsp;pair.&nbsp;The&nbsp;trace&nbsp;places&nbsp;the&nbsp;added&nbsp;time&nbsp;inside&nbsp;backend&nbsp;batches,&nbsp;not&nbsp;queue&nbsp;gaps. | validated |
| Oct&nbsp;5,&nbsp;2026,&nbsp;1:28:34&nbsp;p.m.&nbsp;CDT | [Concurrent early vs late · worker load](#run-work_audit_async_fixed_worker_20261005_01) | RQ9 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;1&nbsp;measured&nbsp;pair&nbsp;·&nbsp;900&nbsp;/&nbsp;2500&nbsp;ms&nbsp;waits&nbsp;·&nbsp;worker&nbsp;KV&nbsp;load | Late loading → early loading (1 pair) | 1,131&nbsp;→&nbsp;90&nbsp;ms&nbsp;(1,041&nbsp;ms&nbsp;faster) | 809&nbsp;→&nbsp;1,006&nbsp;ms&nbsp;(198&nbsp;ms&nbsp;later) | 8,953&nbsp;→&nbsp;7,900&nbsp;ms&nbsp;(1,053&nbsp;ms&nbsp;sooner) | Replay&nbsp;sooner;&nbsp;short&nbsp;session&nbsp;finished&nbsp;later;&nbsp;workflow&nbsp;sooner&nbsp;(1&nbsp;pair). | validated; 1 pair |
| Oct&nbsp;5,&nbsp;2026,&nbsp;1:26:11&nbsp;p.m.&nbsp;CDT | [Concurrent early vs late · scheduler load](#run-work_audit_async_fixed_scheduler_20261005_02) | RQ9 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;1&nbsp;measured&nbsp;pair&nbsp;·&nbsp;900&nbsp;/&nbsp;2500&nbsp;ms&nbsp;waits&nbsp;·&nbsp;scheduler&nbsp;KV&nbsp;load | Late loading → early loading (1 pair) | 276&nbsp;→&nbsp;90&nbsp;ms&nbsp;(186&nbsp;ms&nbsp;faster) | 811&nbsp;→&nbsp;991&nbsp;ms&nbsp;(180&nbsp;ms&nbsp;later) | 8,089&nbsp;→&nbsp;7,896&nbsp;ms&nbsp;(193&nbsp;ms&nbsp;sooner) | Replay&nbsp;sooner;&nbsp;short&nbsp;session&nbsp;finished&nbsp;later;&nbsp;workflow&nbsp;sooner&nbsp;(1&nbsp;pair). | validated; 1 pair |
| Oct&nbsp;5,&nbsp;2026,&nbsp;1:12:20&nbsp;p.m.&nbsp;CDT | [Busy workload · KV-load attribution · scheduler load](#run-work_audit_async_scheduler_busy_20261005_01) | RQ9 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;1&nbsp;paired&nbsp;seed;&nbsp;natural&nbsp;capacity&nbsp;pressure | Checks only → checks + early loads (1 seed); 36 replays/seed | 423.6&nbsp;→&nbsp;436.8&nbsp;s&nbsp;(13.2&nbsp;s&nbsp;higher) | Other-session&nbsp;effect&nbsp;not&nbsp;isolated | 90.5&nbsp;→&nbsp;91.3&nbsp;s&nbsp;(0.8&nbsp;s&nbsp;later) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;and&nbsp;total&nbsp;workload&nbsp;time&nbsp;both&nbsp;increased&nbsp;in&nbsp;this&nbsp;sample. | complete; 1 seed |
| Oct&nbsp;5,&nbsp;2026,&nbsp;1:03:46&nbsp;p.m.&nbsp;CDT | [Busy workload · KV-load attribution · worker load](#run-work_audit_async_worker_busy_20261005_01) | RQ9 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;1&nbsp;paired&nbsp;seed;&nbsp;natural&nbsp;capacity&nbsp;pressure | Checks only → checks + early loads (1 seed); 36 replays/seed | 420.5&nbsp;→&nbsp;442.8&nbsp;s&nbsp;(22.4&nbsp;s&nbsp;higher) | Other-session&nbsp;effect&nbsp;not&nbsp;isolated | 92.1&nbsp;→&nbsp;91.9&nbsp;s&nbsp;(0.2&nbsp;s&nbsp;sooner) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;increased,&nbsp;although&nbsp;the&nbsp;workload&nbsp;finished&nbsp;sooner. | complete; 1 seed |
| Oct&nbsp;5,&nbsp;2026,&nbsp;1:00:42&nbsp;p.m.&nbsp;CDT | [Lifecycle validation · worker load](#run-work_audit_async_worker_verify_20261005_01) | RQ9 | Case&nbsp;order:&nbsp;host&nbsp;→&nbsp;warm&nbsp;·&nbsp;2&nbsp;replays/case&nbsp;·&nbsp;500&nbsp;ms&nbsp;waits | Observation only; no policy comparison | Host-backed&nbsp;replay&nbsp;TTFT:&nbsp;899.5&nbsp;ms | No&nbsp;other&nbsp;session | Not&nbsp;measured | Linked&nbsp;host-backed&nbsp;KV&nbsp;movement&nbsp;to&nbsp;replay;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |
| Oct&nbsp;5,&nbsp;2026,&nbsp;11:35:35&nbsp;a.m.&nbsp;CDT | [Busy workload · KV-load attribution](#run-work_audit_load_kernel_20261005_01) | RQ8 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;1&nbsp;paired&nbsp;seed;&nbsp;natural&nbsp;capacity&nbsp;pressure | Checks only → checks + early loads (1 seed); 36 replays/seed | 435.3&nbsp;→&nbsp;452.9&nbsp;s&nbsp;(17.6&nbsp;s&nbsp;higher) | Other-session&nbsp;effect&nbsp;not&nbsp;isolated | 94.1&nbsp;→&nbsp;95.5&nbsp;s&nbsp;(1.4&nbsp;s&nbsp;later) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;and&nbsp;total&nbsp;workload&nbsp;time&nbsp;both&nbsp;increased&nbsp;in&nbsp;this&nbsp;sample. | complete; 1 seed |
| Oct&nbsp;5,&nbsp;2026,&nbsp;9:31:03&nbsp;a.m.&nbsp;CDT | [Busy workload · KV-load attribution](#run-work_audit_load_phase_20261005_01) | RQ8 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;1&nbsp;paired&nbsp;seed;&nbsp;natural&nbsp;capacity&nbsp;pressure | Checks only → checks + early loads (1 seed); 36 replays/seed | 423.8&nbsp;→&nbsp;435.6&nbsp;s&nbsp;(11.7&nbsp;s&nbsp;higher) | Other-session&nbsp;effect&nbsp;not&nbsp;isolated | 92.2&nbsp;→&nbsp;92.6&nbsp;s&nbsp;(0.4&nbsp;s&nbsp;later) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;and&nbsp;total&nbsp;workload&nbsp;time&nbsp;both&nbsp;increased&nbsp;in&nbsp;this&nbsp;sample. | complete; 1 seed |
| Oct&nbsp;2,&nbsp;2026,&nbsp;8:19:35&nbsp;p.m.&nbsp;CDT | [Busy workload · KV-load attribution](#run-work_audit_kv_attribution_20261002_04) | RQ8 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;2&nbsp;paired&nbsp;seeds;&nbsp;natural&nbsp;capacity&nbsp;pressure | Checks only → checks + early loads (per-arm median, 2 seeds); 36 replays/seed | 447.9&nbsp;→&nbsp;463.1&nbsp;s&nbsp;(15.2&nbsp;s&nbsp;higher) | Other-session&nbsp;effect&nbsp;not&nbsp;isolated | 92.7&nbsp;→&nbsp;95.6&nbsp;s&nbsp;(2.9&nbsp;s&nbsp;later) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;and&nbsp;total&nbsp;workload&nbsp;time&nbsp;both&nbsp;increased&nbsp;in&nbsp;this&nbsp;sample. | complete; 2 seeds |
| Oct&nbsp;2,&nbsp;2026,&nbsp;5:14:42&nbsp;p.m.&nbsp;CDT | [Busy workload · controller KV timing](#run-work_audit_busy_pair_20261002_01) | RQ8 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;2&nbsp;paired&nbsp;seeds;&nbsp;natural&nbsp;capacity&nbsp;pressure | Ordinary replay → controller-timed loads (per-arm median, 2 seeds); 36 replays/seed | 334.9&nbsp;→&nbsp;360.8&nbsp;s&nbsp;(25.9&nbsp;s&nbsp;higher) | Per&nbsp;seed:&nbsp;0&nbsp;helped;&nbsp;12&nbsp;harmed | 78.8&nbsp;→&nbsp;80.0&nbsp;s&nbsp;(1.2&nbsp;s&nbsp;later) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;and&nbsp;total&nbsp;workload&nbsp;time&nbsp;both&nbsp;increased&nbsp;in&nbsp;this&nbsp;sample. | validated; 2 seeds |
| Oct&nbsp;2,&nbsp;2026,&nbsp;5:08:50&nbsp;p.m.&nbsp;CDT | [Busy workload · controller KV timing](#run-work_audit_busy_pilot_20261002) | RQ8 | 12&nbsp;sessions&nbsp;×&nbsp;3&nbsp;tool&nbsp;waits;&nbsp;1&nbsp;paired&nbsp;seed;&nbsp;natural&nbsp;capacity&nbsp;pressure | Ordinary replay → controller-timed loads (1 seed); 36 replays/seed | 335.0&nbsp;→&nbsp;355.7&nbsp;s&nbsp;(20.7&nbsp;s&nbsp;higher) | Per&nbsp;seed:&nbsp;2&nbsp;helped;&nbsp;10&nbsp;harmed | 79.7&nbsp;→&nbsp;79.8&nbsp;s&nbsp;(0.1&nbsp;s&nbsp;later) | Combined&nbsp;replay&nbsp;first-token&nbsp;time&nbsp;and&nbsp;total&nbsp;workload&nbsp;time&nbsp;both&nbsp;increased&nbsp;in&nbsp;this&nbsp;sample. | validated; 1 seed |
| Oct&nbsp;2,&nbsp;2026,&nbsp;4:16:50&nbsp;p.m.&nbsp;CDT | [Controller-chosen load window](#run-work_audit_controller_window_20261002_01) | RQ7 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;4&nbsp;measured&nbsp;trials&nbsp;·&nbsp;900&nbsp;/&nbsp;2500&nbsp;ms&nbsp;waits | Late loading → controller-timed loading (per-mode median, 4 pairs) | 352&nbsp;→&nbsp;86&nbsp;ms&nbsp;(266&nbsp;ms&nbsp;faster) | 865&nbsp;→&nbsp;860&nbsp;ms&nbsp;(5&nbsp;ms&nbsp;earlier) | 8,419&nbsp;→&nbsp;8,074&nbsp;ms&nbsp;(345&nbsp;ms&nbsp;sooner) | Replay&nbsp;sooner;&nbsp;short-session&nbsp;effect&nbsp;varied;&nbsp;workflow&nbsp;sooner&nbsp;(4&nbsp;pairs). | validated; 4 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;3:34:20&nbsp;p.m.&nbsp;CDT | [Three concurrent load windows](#run-work_audit_post_short_20261002_01) | RQ6 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;2&nbsp;measured&nbsp;trials&nbsp;·&nbsp;900&nbsp;/&nbsp;2500&nbsp;ms&nbsp;waits | Late loading → post-short loading (per-mode median, 2 pairs) | 442&nbsp;→&nbsp;85&nbsp;ms&nbsp;(357&nbsp;ms&nbsp;faster) | 819&nbsp;→&nbsp;822&nbsp;ms&nbsp;(2&nbsp;ms&nbsp;later) | 8,287&nbsp;→&nbsp;7,928&nbsp;ms&nbsp;(359&nbsp;ms&nbsp;sooner) | Replay&nbsp;sooner;&nbsp;short-session&nbsp;effect&nbsp;varied;&nbsp;workflow&nbsp;sooner&nbsp;(2&nbsp;pairs). | validated; 2 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;2:48:50&nbsp;p.m.&nbsp;CDT | [Concurrent early vs late](#run-work_audit_concurrent_compare_20261002_02) | RQ5 | 3&nbsp;equal-importance&nbsp;sessions&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;900&nbsp;/&nbsp;2500&nbsp;ms&nbsp;waits | Late loading → early loading (per-mode median, 2 pairs) | 443&nbsp;→&nbsp;86&nbsp;ms&nbsp;(357&nbsp;ms&nbsp;faster) | 815&nbsp;→&nbsp;1,001&nbsp;ms&nbsp;(186&nbsp;ms&nbsp;later) | 8,263&nbsp;→&nbsp;7,906&nbsp;ms&nbsp;(357&nbsp;ms&nbsp;sooner) | Replay&nbsp;sooner;&nbsp;short&nbsp;session&nbsp;finished&nbsp;later;&nbsp;workflow&nbsp;sooner&nbsp;(2&nbsp;pairs). | validated; 2 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;1:42:21&nbsp;p.m.&nbsp;CDT | [Concurrent timeline](#run-work_audit_multisession_20261002_01) | RQ4 | 3&nbsp;concurrent&nbsp;sessions&nbsp;·&nbsp;900&nbsp;/&nbsp;2500&nbsp;ms&nbsp;tool&nbsp;waits | Observation only; no policy comparison | Long&nbsp;first&nbsp;token:&nbsp;282.4&nbsp;ms | Short&nbsp;first&nbsp;token:&nbsp;81.6&nbsp;ms | Not&nbsp;measured | Linked&nbsp;tool&nbsp;waits,&nbsp;KV&nbsp;movement,&nbsp;and&nbsp;replay&nbsp;across&nbsp;sessions;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |
| Oct&nbsp;2,&nbsp;2026,&nbsp;1:33:37&nbsp;p.m.&nbsp;CDT | [Early vs late](#run-work_audit_nonblocking_20261002_01) | RQ3 | Case&nbsp;order:&nbsp;early&nbsp;→&nbsp;late&nbsp;→&nbsp;late_nonblocking&nbsp;·&nbsp;1&nbsp;measured&nbsp;pair&nbsp;·&nbsp;2000&nbsp;ms&nbsp;waits | Late blocking → late nonblocking (comparison withheld) | No&nbsp;validated&nbsp;replay-speed&nbsp;delta | No&nbsp;other&nbsp;session | Full-task&nbsp;comparison&nbsp;withheld | Submission&nbsp;was&nbsp;faster,&nbsp;but&nbsp;the&nbsp;measured&nbsp;first-token&nbsp;delay&nbsp;remained;&nbsp;the&nbsp;strict&nbsp;comparison&nbsp;was&nbsp;withheld. | validated |
| Oct&nbsp;2,&nbsp;2026,&nbsp;11:41:32&nbsp;a.m.&nbsp;CDT | [Early vs late](#run-work_audit_timing_sampled_late_early_20261002) | RQ2 | Case&nbsp;order:&nbsp;late&nbsp;→&nbsp;early&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;2000&nbsp;ms&nbsp;waits | Late → early loading (per-mode median, 2 pairs) | 255&nbsp;→&nbsp;83&nbsp;ms&nbsp;(172&nbsp;ms&nbsp;faster) | No&nbsp;other&nbsp;session | Full-task&nbsp;effect&nbsp;not&nbsp;established | Loading&nbsp;during&nbsp;the&nbsp;tool&nbsp;wait&nbsp;brought&nbsp;the&nbsp;first&nbsp;token&nbsp;sooner&nbsp;in&nbsp;the&nbsp;matched&nbsp;replays. | validated; 2 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;11:38:54&nbsp;a.m.&nbsp;CDT | [Early vs late](#run-work_audit_timing_sampled_early_late_20261002) | RQ2 | Case&nbsp;order:&nbsp;early&nbsp;→&nbsp;late&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;2000&nbsp;ms&nbsp;waits | Late → early loading (per-mode median, 2 pairs) | 258&nbsp;→&nbsp;86&nbsp;ms&nbsp;(172&nbsp;ms&nbsp;faster) | No&nbsp;other&nbsp;session | Full-task&nbsp;effect&nbsp;not&nbsp;established | Loading&nbsp;during&nbsp;the&nbsp;tool&nbsp;wait&nbsp;brought&nbsp;the&nbsp;first&nbsp;token&nbsp;sooner&nbsp;in&nbsp;the&nbsp;matched&nbsp;replays. | validated; 2 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;11:25:50&nbsp;a.m.&nbsp;CDT | [Early vs late](#run-work_audit_timing_exact_late_early_20261002) | RQ2 | Case&nbsp;order:&nbsp;late&nbsp;→&nbsp;early&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;2000&nbsp;ms&nbsp;waits | Late → early loading (per-mode median, 2 pairs) | 312&nbsp;→&nbsp;84&nbsp;ms&nbsp;(229&nbsp;ms&nbsp;faster) | No&nbsp;other&nbsp;session | Full-task&nbsp;effect&nbsp;not&nbsp;established | Loading&nbsp;during&nbsp;the&nbsp;tool&nbsp;wait&nbsp;brought&nbsp;the&nbsp;first&nbsp;token&nbsp;sooner&nbsp;in&nbsp;the&nbsp;matched&nbsp;replays. | validated; 2 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;11:22:26&nbsp;a.m.&nbsp;CDT | [Early vs late](#run-work_audit_timing_exact_early_late_20261002) | RQ2 | Case&nbsp;order:&nbsp;early&nbsp;→&nbsp;late&nbsp;·&nbsp;2&nbsp;measured&nbsp;pairs&nbsp;·&nbsp;2000&nbsp;ms&nbsp;waits | Late → early loading (per-mode median, 2 pairs) | 318&nbsp;→&nbsp;87&nbsp;ms&nbsp;(232&nbsp;ms&nbsp;faster) | No&nbsp;other&nbsp;session | Full-task&nbsp;effect&nbsp;not&nbsp;established | Loading&nbsp;during&nbsp;the&nbsp;tool&nbsp;wait&nbsp;brought&nbsp;the&nbsp;first&nbsp;token&nbsp;sooner&nbsp;in&nbsp;the&nbsp;matched&nbsp;replays. | validated; 2 pairs |
| Oct&nbsp;2,&nbsp;2026,&nbsp;10:34:04&nbsp;a.m.&nbsp;CDT | [Lifecycle validation](#run-work_audit_two_replays_lean_reverse_20261002) | RQ1 | Case&nbsp;order:&nbsp;host&nbsp;→&nbsp;warm&nbsp;·&nbsp;2&nbsp;replays/case&nbsp;·&nbsp;wait&nbsp;not&nbsp;recorded | Observation only; no policy comparison | Host-backed&nbsp;replay&nbsp;TTFT:&nbsp;436.3&nbsp;ms | No&nbsp;other&nbsp;session | Not&nbsp;measured | Linked&nbsp;host-backed&nbsp;KV&nbsp;movement&nbsp;to&nbsp;replay;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |
| Oct&nbsp;2,&nbsp;2026,&nbsp;10:26:59&nbsp;a.m.&nbsp;CDT | [Lifecycle validation](#run-work_audit_two_replays_lean_pump_20261002) | RQ1 | Case&nbsp;order:&nbsp;warm&nbsp;→&nbsp;host&nbsp;·&nbsp;2&nbsp;replays/case&nbsp;·&nbsp;wait&nbsp;not&nbsp;recorded | Observation only; no policy comparison | Host-backed&nbsp;replay&nbsp;TTFT:&nbsp;82.6&nbsp;ms | No&nbsp;other&nbsp;session | Not&nbsp;measured | Linked&nbsp;host-backed&nbsp;KV&nbsp;movement&nbsp;to&nbsp;replay;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |
| Oct&nbsp;2,&nbsp;2026,&nbsp;9:54:21&nbsp;a.m.&nbsp;CDT | [Lifecycle validation](#run-work_audit_two_replays_20261002_reverse) | RQ1 | Case&nbsp;order:&nbsp;host&nbsp;→&nbsp;warm&nbsp;·&nbsp;2&nbsp;replays/case&nbsp;·&nbsp;wait&nbsp;not&nbsp;recorded | Observation only; no policy comparison | Host-backed&nbsp;replay&nbsp;TTFT:&nbsp;573.9&nbsp;ms | No&nbsp;other&nbsp;session | Not&nbsp;measured | Linked&nbsp;host-backed&nbsp;KV&nbsp;movement&nbsp;to&nbsp;replay;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |
| Oct&nbsp;2,&nbsp;2026,&nbsp;9:50:49&nbsp;a.m.&nbsp;CDT | [Lifecycle validation](#run-work_audit_two_replays_20261002) | RQ1 | Case&nbsp;order:&nbsp;warm&nbsp;→&nbsp;host&nbsp;·&nbsp;2&nbsp;replays/case&nbsp;·&nbsp;wait&nbsp;not&nbsp;recorded | Observation only; no policy comparison | Host-backed&nbsp;replay&nbsp;TTFT:&nbsp;229.1&nbsp;ms | No&nbsp;other&nbsp;session | Not&nbsp;measured | Linked&nbsp;host-backed&nbsp;KV&nbsp;movement&nbsp;to&nbsp;replay;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |
| Oct&nbsp;1,&nbsp;2026,&nbsp;5:42:31&nbsp;p.m.&nbsp;CDT | [Lifecycle validation](#run-work_audit_a10g_20261001_final) | RQ1 | Case&nbsp;order:&nbsp;warm&nbsp;→&nbsp;host&nbsp;·&nbsp;not&nbsp;recorded&nbsp;replays/case&nbsp;·&nbsp;wait&nbsp;not&nbsp;recorded | Observation only; no policy comparison | Host-backed&nbsp;replay&nbsp;TTFT:&nbsp;230.3&nbsp;ms | No&nbsp;other&nbsp;session | Not&nbsp;measured | Linked&nbsp;host-backed&nbsp;KV&nbsp;movement&nbsp;to&nbsp;replay;&nbsp;no&nbsp;speed&nbsp;win&nbsp;tested. | validated |

## Experiment details

<a id="run-rq13_both_d4_profile_p2048_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:12:14 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_both_d4_profile_p2048_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** 4 of 4 native loads physically overlapped the target decode window. This profiled run establishes overlap, not an unprofiled performance effect or hardware cause.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 80000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | 4 |
| Physical copy overlap (ms) | 37.1 |
| Copy concurrent with decode kernels (ms) | 0.0 |
| Profiler status | CUDA capture verified |
| Target first token after tool return (ms) | 1261.1 |
| Target finish after tool return (ms) | 4573.0 |
| Whole workload (ms) | 116028.9 |
| Linked decode forwards | 96 |
| Linked decode kernels | 1152 |
| Summed kernel execution (ms) | 3.6 |
| Inside-forward CPU-before-launch gaps (ms) | 5.7 |
| Between-forward gaps (ms) | 3010.9 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 1261.1 | 4573.0 |
| Peer 1 | 96 | 1289.4 | 4572.8 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 185.0 | 205.5 | 8.8 | 0.0 | 698.6 |
| 2 | 2048 | 144.7 | 151.9 | 8.9 | 0.0 | 734.9 |
| 3 | 2048 | 129.8 | 140.0 | 9.5 | 0.0 | 733.9 |
| 4 | 2048 | 116.2 | 141.3 | 9.9 | 0.0 | 734.0 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** physical copy and decode launches verified. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_both_d4_profile_p2048_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='both_p2048_profile' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='80000' WORK_AUDIT_FORWARD_TRACE='1' WORK_AUDIT_NSYS_ENABLE='1' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/runtime/backend_features.json) · [Physical copy overlap](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/nsys/physical_overlap.json) · [Decode launch timing](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/nsys/decode_submission.json) · [Nsight capture](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/nsys/backend.nsys-rep) · [Nsight SQLite trace](docs/reports/work_audit/rq13_both_d4_profile_p2048_20261006/nsys/backend.sqlite.gz)

</details>

<a id="run-rq13_baseline_d4_p2048_w40_seed2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:07:50 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_baseline_d4_p2048_w40_seed2_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Compared with its matched zero-overlap control, the target finished 446 ms later and peer decoders finished a median 445 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs off; overlap scheduling off. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 65.7 |
| Target finish after tool return (ms) | 3983.3 |
| Whole workload (ms) | 72824.1 |
| Matched control target finish (ms) | 3537.5 |
| Target finish change vs control (ms) | 445.8 |
| Peer median finish change vs control (ms) | 445.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 65.7 | 3983.3 |
| Peer 1 | 96 | 120.7 | 3983.5 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 158.2 | 218.2 | not captured | not captured | 102.4 |
| 2 | 2048 | 169.6 | 177.4 | not captured | not captured | 210.4 |
| 3 | 2048 | 160.4 | 211.5 | not captured | not captured | 210.8 |
| 4 | 2048 | 145.7 | 181.5 | not captured | not captured | 210.3 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_baseline_d4_p2048_w40_seed2_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='baseline_p2048_w40_seed2' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_seed2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_seed2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_seed2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_seed2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_seed2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_seed2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_baseline_d0_p2048_w40_seed2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:05:16 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_baseline_d0_p2048_w40_seed2_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs off; overlap scheduling off. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 66.2 |
| Target finish after tool return (ms) | 3537.5 |
| Whole workload (ms) | 74466.6 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 66.2 | 3537.5 |
| Peer 1 | 96 | 121.5 | 3538.0 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 2186.4 | 0.0 | not captured | not captured | 102.6 |
| 2 | 2048 | 4453.1 | 0.0 | not captured | not captured | 212.9 |
| 3 | 2048 | 4116.8 | 0.0 | not captured | not captured | 213.1 |
| 4 | 2048 | 3203.2 | 0.0 | not captured | not captured | 213.1 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_baseline_d0_p2048_w40_seed2_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='baseline_p2048_w40_seed2' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_seed2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_seed2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_seed2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_seed2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_seed2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_seed2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_both_d4_p2048_w40_seed2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:02:18 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_both_d4_p2048_w40_seed2_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Compared with its matched zero-overlap control, the target finished 1 ms later and peer decoders finished a median 1 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 385.3 |
| Target finish after tool return (ms) | 3655.3 |
| Whole workload (ms) | 61809.4 |
| Matched control target finish (ms) | 3654.2 |
| Target finish change vs control (ms) | 1.2 |
| Peer median finish change vs control (ms) | 1.2 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 385.3 | 3655.3 |
| Peer 1 | 96 | 422.7 | 3655.2 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 178.3 | 191.0 | not captured | not captured | 397.2 |
| 2 | 2048 | 127.2 | 145.3 | not captured | not captured | 428.7 |
| 3 | 2048 | 114.6 | 117.3 | not captured | not captured | 427.8 |
| 4 | 2048 | 114.8 | 133.8 | not captured | not captured | 427.9 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_both_d4_p2048_w40_seed2_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='both_p2048_w40_seed2' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_both_d4_p2048_w40_seed2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_both_d4_p2048_w40_seed2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_both_d4_p2048_w40_seed2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_both_d4_p2048_w40_seed2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_both_d4_p2048_w40_seed2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_both_d4_p2048_w40_seed2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_both_d0_p2048_w40_seed2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:59:36 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_both_d0_p2048_w40_seed2_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 386.6 |
| Target finish after tool return (ms) | 3654.2 |
| Whole workload (ms) | 61687.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 386.6 | 3654.2 |
| Peer 1 | 96 | 422.7 | 3654.0 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 5559.2 | 0.0 | not captured | not captured | 402.0 |
| 2 | 2048 | 5986.3 | 0.0 | not captured | not captured | 434.6 |
| 3 | 2048 | 3384.4 | 0.0 | not captured | not captured | 434.1 |
| 4 | 2048 | 5302.2 | 0.0 | not captured | not captured | 433.5 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_both_d0_p2048_w40_seed2_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='both_p2048_w40_seed2' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_both_d0_p2048_w40_seed2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_both_d0_p2048_w40_seed2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_both_d0_p2048_w40_seed2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_both_d0_p2048_w40_seed2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_both_d0_p2048_w40_seed2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_both_d0_p2048_w40_seed2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_baseline_d0_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:56:24 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_baseline_d0_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs off; overlap scheduling off. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 65.6 |
| Target finish after tool return (ms) | 3530.0 |
| Whole workload (ms) | 72558.7 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 65.6 | 3530.0 |
| Peer 1 | 96 | 120.2 | 3530.5 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 3027.8 | 0.0 | not captured | not captured | 101.0 |
| 2 | 2048 | 4124.7 | 0.0 | not captured | not captured | 209.0 |
| 3 | 2048 | 4934.2 | 0.0 | not captured | not captured | 209.2 |
| 4 | 2048 | 2898.6 | 0.0 | not captured | not captured | 209.2 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_baseline_d0_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='baseline_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_baseline_d0_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_baseline_d4_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:53:50 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_baseline_d4_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Compared with its matched zero-overlap control, the target finished 475 ms later and peer decoders finished a median 475 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs off; overlap scheduling off. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 65.4 |
| Target finish after tool return (ms) | 4005.5 |
| Whole workload (ms) | 72691.7 |
| Matched control target finish (ms) | 3530.0 |
| Target finish change vs control (ms) | 475.5 |
| Peer median finish change vs control (ms) | 474.9 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 65.4 | 4005.5 |
| Peer 1 | 96 | 120.3 | 4005.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 169.0 | 216.8 | not captured | not captured | 101.6 |
| 2 | 2048 | 190.7 | 278.5 | not captured | not captured | 210.1 |
| 3 | 2048 | 177.5 | 264.6 | not captured | not captured | 209.5 |
| 4 | 2048 | 150.4 | 161.3 | not captured | not captured | 209.9 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_baseline_d4_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='baseline_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_baseline_d4_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_both_d0_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:51:15 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_both_d0_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 385.3 |
| Target finish after tool return (ms) | 3653.3 |
| Whole workload (ms) | 62028.8 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 385.3 | 3653.3 |
| Peer 1 | 96 | 421.2 | 3653.2 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 2811.1 | 0.0 | not captured | not captured | 402.1 |
| 2 | 2048 | 5354.9 | 0.0 | not captured | not captured | 434.3 |
| 3 | 2048 | 5994.5 | 0.0 | not captured | not captured | 433.7 |
| 4 | 2048 | 2456.8 | 0.0 | not captured | not captured | 433.1 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_both_d0_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='both_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_both_d0_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_both_d0_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_both_d0_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_both_d0_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_both_d0_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_both_d0_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_both_d4_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:48:33 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_both_d4_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Compared with its matched zero-overlap control, the target finished 7 ms later and peer decoders finished a median 7 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 390.7 |
| Target finish after tool return (ms) | 3659.9 |
| Whole workload (ms) | 61763.1 |
| Matched control target finish (ms) | 3653.3 |
| Target finish change vs control (ms) | 6.6 |
| Peer median finish change vs control (ms) | 6.9 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 390.7 | 3659.9 |
| Peer 1 | 96 | 428.0 | 3660.1 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 177.7 | 191.9 | not captured | not captured | 403.3 |
| 2 | 2048 | 127.0 | 145.6 | not captured | not captured | 434.6 |
| 3 | 2048 | 133.4 | 151.4 | not captured | not captured | 435.0 |
| 4 | 2048 | 107.3 | 116.0 | not captured | not captured | 434.5 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_both_d4_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='both_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_both_d4_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_both_d4_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_both_d4_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_both_d4_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_both_d4_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_both_d4_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_overlap_d0_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:45:31 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_overlap_d0_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs off; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 697.0 |
| Target finish after tool return (ms) | 4002.5 |
| Whole workload (ms) | 80589.0 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 697.0 | 4002.5 |
| Peer 1 | 96 | 750.6 | 4003.0 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 3361.1 | 0.0 | not captured | not captured | 131.0 |
| 2 | 2048 | 4308.6 | 0.0 | not captured | not captured | 182.9 |
| 3 | 2048 | 3303.8 | 0.0 | not captured | not captured | 183.0 |
| 4 | 2048 | 3087.1 | 0.0 | not captured | not captured | 182.0 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_overlap_d0_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='overlap_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_overlap_d0_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_overlap_d0_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_overlap_d0_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_overlap_d0_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_overlap_d0_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_overlap_d0_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_overlap_d4_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:42:55 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_overlap_d4_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Compared with its matched zero-overlap control, the target finished 458 ms later and peer decoders finished a median 457 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs off; overlap scheduling on. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 697.2 |
| Target finish after tool return (ms) | 4460.3 |
| Whole workload (ms) | 80584.4 |
| Matched control target finish (ms) | 4002.5 |
| Target finish change vs control (ms) | 457.8 |
| Peer median finish change vs control (ms) | 457.2 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 697.2 | 4460.3 |
| Peer 1 | 96 | 750.4 | 4460.2 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 163.8 | 253.9 | not captured | not captured | 131.1 |
| 2 | 2048 | 172.9 | 190.6 | not captured | not captured | 182.5 |
| 3 | 2048 | 162.9 | 230.4 | not captured | not captured | 181.9 |
| 4 | 2048 | 155.6 | 156.3 | not captured | not captured | 182.3 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_overlap_d4_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='overlap_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_overlap_d4_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_overlap_d4_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_overlap_d4_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_overlap_d4_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_overlap_d4_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_overlap_d4_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_graph_d0_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:40:00 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_graph_d0_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling off. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 65.5 |
| Target finish after tool return (ms) | 3452.4 |
| Whole workload (ms) | 55661.0 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 65.5 | 3452.4 |
| Peer 1 | 96 | 120.1 | 3452.3 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 3156.7 | 0.0 | not captured | not captured | 103.4 |
| 2 | 2048 | 5984.9 | 0.0 | not captured | not captured | 453.9 |
| 3 | 2048 | 5154.2 | 0.0 | not captured | not captured | 453.0 |
| 4 | 2048 | 3591.7 | 0.0 | not captured | not captured | 452.9 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_graph_d0_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='graph_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_graph_d0_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_graph_d0_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_graph_d0_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_graph_d0_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_graph_d0_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_graph_d0_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq13_graph_d4_p2048_w40_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:37:09 a.m. CDT · Backend scheduling and KV overlap</strong> · rq13_graph_d4_p2048_w40_20261006</summary>

**Question (RQ13).** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**Finding.** Compared with its matched zero-overlap control, the target finished 142 ms later and peer decoders finished a median 142 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 2048 / 2048 words. CUDA graphs on; overlap scheduling off. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 65.4 |
| Target finish after tool return (ms) | 3594.9 |
| Whole workload (ms) | 55427.3 |
| Matched control target finish (ms) | 3452.4 |
| Target finish change vs control (ms) | 142.4 |
| Peer median finish change vs control (ms) | 142.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 65.4 | 3594.9 |
| Peer 1 | 96 | 120.1 | 3594.8 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012 | 156.4 | 157.6 | not captured | not captured | 101.6 |
| 2 | 2048 | 136.0 | 138.7 | not captured | not captured | 449.7 |
| 3 | 2048 | 117.9 | 120.0 | not captured | not captured | 448.7 |
| 4 | 2048 | 116.8 | 118.0 | not captured | not captured | 448.9 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq13_graph_d4_p2048_w40_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ13' WORK_AUDIT_PAIR_ID='graph_p2048_w40' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='2048' WORK_AUDIT_DONOR_PROMPT_WORDS='2048' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq13_graph_d4_p2048_w40_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq13_graph_d4_p2048_w40_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq13_graph_d4_p2048_w40_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq13_graph_d4_p2048_w40_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq13_graph_d4_p2048_w40_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq13_graph_d4_p2048_w40_20261006/runtime/backend_features.json)

</details>

<a id="run-rq12_s6_d4_profile_repeat_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 12:21:23 a.m. CDT · Decode slowdown attribution</strong> · rq12_s6_d4_profile_repeat_20261006</summary>

**Question (RQ12).** When native host-to-GPU KV copies physically overlap another session's decode, does added time appear inside decode kernels or in the host's cadence of submitting them?

**Finding.** 4 physical KV loads overlapped decode. Versus the paired profiled control, CPU-before-launch gaps grew 546 ms while summed kernel execution changed +1.4 ms. This locates delay in launch cadence; it does not identify why the host waited or measure an unprofiled speedup.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | 4 |
| Physical copy overlap (ms) | 73.2 |
| Copy concurrent with decode kernels (ms) | 35.4 |
| Profiler status | CUDA capture verified |
| Target first token after tool return (ms) | 191.5 |
| Target finish after tool return (ms) | 4510.4 |
| Whole workload (ms) | 80781.0 |
| Linked decode forwards | 94 |
| Linked decode kernels | 32712 |
| Summed kernel execution (ms) | 3192.4 |
| Inside-forward CPU-before-launch gaps (ms) | 613.9 |
| Between-forward gaps (ms) | 1725.0 |
| Kernel execution change vs profiled control (ms) | 1.4 |
| CPU-before-launch change vs profiled control (ms) | 545.9 |
| Matched control target finish (ms) | 3726.0 |
| Target finish change vs control (ms) | 784.4 |
| Peer median finish change vs control (ms) | 784.2 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 191.5 | 4510.4 |
| Peer 1 | 96 | 191.3 | 4510.3 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 234.1 | 324.1 | 18.0 | 9.1 | 255.0 |
| 2 | 4096 | 239.9 | 320.8 | 18.0 | 6.7 | 956.6 |
| 3 | 4096 | 265.5 | 281.4 | 18.4 | 9.4 | 956.9 |
| 4 | 4096 | 238.7 | 249.1 | 18.8 | 10.2 | 956.3 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** physical copy and decode launches verified. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq12_s6_d4_profile_repeat_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ12' WORK_AUDIT_PAIR_ID='profile_forward_order' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='1' WORK_AUDIT_NSYS_ENABLE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/backend_trace.jsonl.gz) · [Physical copy overlap](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/nsys/physical_overlap.json) · [Decode launch timing](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/nsys/decode_submission.json) · [Nsight capture](docs/reports/work_audit/rq12_s6_d4_profile_repeat_20261006/nsys/backend.nsys-rep)

</details>

<a id="run-rq12_s6_d0_profile_repeat_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 12:17:50 a.m. CDT · Decode slowdown attribution</strong> · rq12_s6_d0_profile_repeat_20261006</summary>

**Question (RQ12).** When native host-to-GPU KV copies physically overlap another session's decode, does added time appear inside decode kernels or in the host's cadence of submitting them?

**Finding.** Profiled zero-overlap control: no KV copies ran during target decode. It captured 94 forwards and 32712 linked kernels for the paired launch comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | 0 |
| Physical copy overlap (ms) | 0.0 |
| Copy concurrent with decode kernels (ms) | 0.0 |
| Profiler status | CUDA capture verified |
| Target first token after tool return (ms) | 190.7 |
| Target finish after tool return (ms) | 3726.0 |
| Whole workload (ms) | 81097.8 |
| Linked decode forwards | 94 |
| Linked decode kernels | 32712 |
| Summed kernel execution (ms) | 3191.0 |
| Inside-forward CPU-before-launch gaps (ms) | 68.0 |
| Between-forward gaps (ms) | 1587.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 190.7 | 3726.0 |
| Peer 1 | 96 | 190.8 | 3726.1 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 1584.9 | 0.0 | 0.0 | 0.0 | 122.3 |
| 2 | 4096 | 3554.2 | 0.0 | 0.0 | 0.0 | 253.1 |
| 3 | 4096 | 2844.9 | 0.0 | 0.0 | 0.0 | 253.5 |
| 4 | 4096 | 1813.1 | 0.0 | 0.0 | 0.0 | 252.9 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** physical copy and decode launches verified. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq12_s6_d0_profile_repeat_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ12' WORK_AUDIT_PAIR_ID='profile_forward_order' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='1' WORK_AUDIT_NSYS_ENABLE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/backend_trace.jsonl.gz) · [Physical copy overlap](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/nsys/physical_overlap.json) · [Decode launch timing](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/nsys/decode_submission.json) · [Nsight capture](docs/reports/work_audit/rq12_s6_d0_profile_repeat_20261006/nsys/backend.nsys-rep)

</details>

<a id="run-rq12_s6_d0_profile_seed1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 12:13:04 a.m. CDT · Decode slowdown attribution</strong> · rq12_s6_d0_profile_seed1_20261006</summary>

**Question (RQ12).** When native host-to-GPU KV copies physically overlap another session's decode, does added time appear inside decode kernels or in the host's cadence of submitting them?

**Finding.** Profiled zero-overlap control: no KV copies ran during target decode. It captured 94 forwards and 32712 linked kernels for the paired launch comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | 0 |
| Physical copy overlap (ms) | 0.0 |
| Copy concurrent with decode kernels (ms) | 0.0 |
| Profiler status | CUDA capture verified |
| Target first token after tool return (ms) | 190.1 |
| Target finish after tool return (ms) | 3726.9 |
| Whole workload (ms) | 79695.3 |
| Linked decode forwards | 94 |
| Linked decode kernels | 32712 |
| Summed kernel execution (ms) | 3190.9 |
| Inside-forward CPU-before-launch gaps (ms) | 68.4 |
| Between-forward gaps (ms) | 1578.7 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 190.1 | 3726.9 |
| Peer 1 | 96 | 190.3 | 3727.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 1132.2 | 0.0 | 0.0 | 0.0 | 123.3 |
| 2 | 4096 | 2542.5 | 0.0 | 0.0 | 0.0 | 253.1 |
| 3 | 4096 | 2165.4 | 0.0 | 0.0 | 0.0 | 253.0 |
| 4 | 4096 | 1712.2 | 0.0 | 0.0 | 0.0 | 252.8 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** physical copy and decode launches verified. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq12_s6_d0_profile_seed1_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ12' WORK_AUDIT_PAIR_ID='profile_reverse_order' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='1' WORK_AUDIT_NSYS_ENABLE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/backend_trace.jsonl.gz) · [Physical copy overlap](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/nsys/physical_overlap.json) · [Decode launch timing](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/nsys/decode_submission.json) · [Nsight capture](docs/reports/work_audit/rq12_s6_d0_profile_seed1_20261006/nsys/backend.nsys-rep)

</details>

<a id="run-rq12_s6_d4_profile_live_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 12:04:41 a.m. CDT · Decode slowdown attribution</strong> · rq12_s6_d4_profile_live_20261006</summary>

**Question (RQ12).** When native host-to-GPU KV copies physically overlap another session's decode, does added time appear inside decode kernels or in the host's cadence of submitting them?

**Finding.** 4 physical KV loads overlapped decode. Versus the paired profiled control, CPU-before-launch gaps grew 481 ms while summed kernel execution changed +1.1 ms. This locates delay in launch cadence; it does not identify why the host waited or measure an unprofiled speedup.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | 4 |
| Physical copy overlap (ms) | 73.3 |
| Copy concurrent with decode kernels (ms) | 31.7 |
| Profiler status | CUDA capture verified |
| Target first token after tool return (ms) | 203.3 |
| Target finish after tool return (ms) | 4452.5 |
| Whole workload (ms) | 80811.5 |
| Linked decode forwards | 94 |
| Linked decode kernels | 32712 |
| Summed kernel execution (ms) | 3192.0 |
| Inside-forward CPU-before-launch gaps (ms) | 549.2 |
| Between-forward gaps (ms) | 1713.6 |
| Kernel execution change vs profiled control (ms) | 1.1 |
| CPU-before-launch change vs profiled control (ms) | 480.8 |
| Matched control target finish (ms) | 3726.9 |
| Target finish change vs control (ms) | 725.7 |
| Peer median finish change vs control (ms) | 725.0 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 203.3 | 4452.5 |
| Peer 1 | 96 | 203.1 | 4452.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 232.9 | 317.2 | 17.9 | 6.6 | 253.7 |
| 2 | 4096 | 226.2 | 305.6 | 18.1 | 6.5 | 953.3 |
| 3 | 4096 | 229.6 | 256.0 | 18.3 | 7.9 | 953.6 |
| 4 | 4096 | 212.6 | 229.8 | 18.9 | 10.7 | 953.0 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** physical copy and decode launches verified. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq12_s6_d4_profile_live_20261006' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ12' WORK_AUDIT_PAIR_ID='profile_reverse_order' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='1' WORK_AUDIT_NSYS_ENABLE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/backend_trace.jsonl.gz) · [Physical copy overlap](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/nsys/physical_overlap.json) · [Decode launch timing](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/nsys/decode_submission.json) · [Nsight capture](docs/reports/work_audit/rq12_s6_d4_profile_live_20261006/nsys/backend.nsys-rep)

</details>

<a id="run-rq11_s12_d4_shortactive_seed2_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 9:09:29 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d4_shortactive_seed2_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 786 ms later and peer decoders finished a median 787 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 512 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 62.3 |
| Target finish after tool return (ms) | 4682.6 |
| Whole workload (ms) | 79841.3 |
| Matched control target finish (ms) | 3896.5 |
| Target finish change vs control (ms) | 786.1 |
| Peer median finish change vs control (ms) | 786.7 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 62.3 | 4682.6 |
| Peer 1 | 96 | 191.8 | 4683.1 |
| Peer 2 | 96 | 191.4 | 4683.0 |
| Peer 3 | 96 | 190.0 | 4683.0 |
| Peer 4 | 96 | 190.5 | 4682.9 |
| Peer 5 | 96 | 191.1 | 4683.2 |
| Peer 6 | 96 | 190.8 | 4682.9 |
| Peer 7 | 96 | 192.1 | 4683.1 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 244.5 | 280.9 | not captured | not captured | 253.4 |
| 2 | 4096 | 256.5 | 465.4 | not captured | not captured | 1119.7 |
| 3 | 4096 | 249.9 | 363.1 | not captured | not captured | 1119.0 |
| 4 | 4096 | 223.2 | 228.0 | not captured | not captured | 1119.3 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d4_shortactive_seed2_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='512' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d4_shortactive_seed2_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d4_shortactive_seed2_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s12_d4_shortactive_seed2_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d4_shortactive_seed2_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d4_shortactive_seed2_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s12_d0_shortactive_seed2_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 9:06:47 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d0_shortactive_seed2_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 512 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 187.7 |
| Target finish after tool return (ms) | 3896.5 |
| Whole workload (ms) | 79006.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 187.7 | 3896.5 |
| Peer 1 | 96 | 60.0 | 3896.0 |
| Peer 2 | 96 | 188.1 | 3896.5 |
| Peer 3 | 96 | 188.4 | 3896.6 |
| Peer 4 | 96 | 187.1 | 3896.4 |
| Peer 5 | 96 | 186.3 | 3896.3 |
| Peer 6 | 96 | 186.8 | 3896.3 |
| Peer 7 | 96 | 187.4 | 3896.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 264.8 | 0.0 | not captured | not captured | 121.4 |
| 2 | 4096 | 8147.3 | 0.0 | not captured | not captured | 251.7 |
| 3 | 4096 | 6129.0 | 0.0 | not captured | not captured | 251.0 |
| 4 | 4096 | 6063.5 | 0.0 | not captured | not captured | 251.3 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d0_shortactive_seed2_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='512' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d0_shortactive_seed2_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d0_shortactive_seed2_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s12_d0_shortactive_seed2_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d0_shortactive_seed2_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d0_shortactive_seed2_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d4_seed2_longwait_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 9:03:55 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d4_seed2_longwait_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 845 ms later and peer decoders finished a median 845 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 189.8 |
| Target finish after tool return (ms) | 4475.4 |
| Whole workload (ms) | 80843.6 |
| Matched control target finish (ms) | 3630.4 |
| Target finish change vs control (ms) | 845.0 |
| Peer median finish change vs control (ms) | 845.2 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 189.8 | 4475.4 |
| Peer 1 | 96 | 190.0 | 4475.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 274.6 | 307.0 | not captured | not captured | 259.1 |
| 2 | 4096 | 266.8 | 271.4 | not captured | not captured | 934.1 |
| 3 | 4096 | 271.9 | 322.4 | not captured | not captured | 933.1 |
| 4 | 4096 | 245.4 | 249.1 | not captured | not captured | 933.1 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d4_seed2_longwait_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d4_seed2_longwait_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d4_seed2_longwait_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d4_seed2_longwait_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d4_seed2_longwait_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d4_seed2_longwait_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d0_seed2_longwait_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 9:01:17 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d0_seed2_longwait_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 187.5 |
| Target finish after tool return (ms) | 3630.4 |
| Whole workload (ms) | 78122.6 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 187.5 | 3630.4 |
| Peer 1 | 96 | 187.3 | 3630.2 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 4213.6 | 0.0 | not captured | not captured | 119.9 |
| 2 | 4096 | 5820.1 | 0.0 | not captured | not captured | 249.2 |
| 3 | 4096 | 4547.5 | 0.0 | not captured | not captured | 248.5 |
| 4 | 4096 | 5140.7 | 0.0 | not captured | not captured | 248.9 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d0_seed2_longwait_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d0_seed2_longwait_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d0_seed2_longwait_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d0_seed2_longwait_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d0_seed2_longwait_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d0_seed2_longwait_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d0_seed2_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:58:45 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d0_seed2_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Excluded diagnostic: The four post-decode native loads overshot the 20-second donor tool-return window. A paired 40-second control and four-load arm replaced this attempt.

**Setup.** nvidia-standard; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Failure | The four post-decode native loads overshot the 20-second donor tool-return window. A paired 40-second control and four-load arm replaced this attempt. |
| Comparable timing | unavailable |

**Evidence gate.** excluded. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- No target or workflow timing from this attempt is included in the dose comparison.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d0_seed2_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='2' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d0_seed2_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d0_seed2_20261005/run_manifest.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d0_seed2_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d0_seed2_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s12_d1_shortactive_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:56:00 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d1_shortactive_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 155 ms later and peer decoders finished a median 155 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 512 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 1 |
| Worker-window overlaps (proxy) | 1 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 184.5 |
| Target finish after tool return (ms) | 4050.3 |
| Whole workload (ms) | 79049.9 |
| Matched control target finish (ms) | 3895.4 |
| Target finish change vs control (ms) | 154.9 |
| Peer median finish change vs control (ms) | 155.5 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 184.5 | 4050.3 |
| Peer 1 | 96 | 58.8 | 4050.9 |
| Peer 2 | 96 | 186.7 | 4051.1 |
| Peer 3 | 96 | 185.3 | 4050.9 |
| Peer 4 | 96 | 185.0 | 4050.4 |
| Peer 5 | 96 | 185.6 | 4051.0 |
| Peer 6 | 96 | 186.4 | 4051.1 |
| Peer 7 | 96 | 186.0 | 4051.0 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 205.0 | 214.8 | not captured | not captured | 471.3 |
| 2 | 4096 | 283.6 | 0.0 | not captured | not captured | 599.6 |
| 3 | 4096 | 3194.1 | 0.0 | not captured | not captured | 599.7 |
| 4 | 4096 | 3337.2 | 0.0 | not captured | not captured | 599.6 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d1_shortactive_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='1' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='512' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d1_shortactive_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d1_shortactive_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s12_d1_shortactive_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d1_shortactive_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d1_shortactive_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s12_d2_shortactive_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:53:28 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d2_shortactive_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 344 ms later and peer decoders finished a median 344 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 512 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 2 |
| Worker-window overlaps (proxy) | 2 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 185.9 |
| Target finish after tool return (ms) | 4239.7 |
| Whole workload (ms) | 79823.7 |
| Matched control target finish (ms) | 3895.4 |
| Target finish change vs control (ms) | 344.3 |
| Peer median finish change vs control (ms) | 344.3 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 185.9 | 4239.7 |
| Peer 1 | 96 | 59.3 | 4239.7 |
| Peer 2 | 96 | 186.9 | 4239.9 |
| Peer 3 | 96 | 186.3 | 4239.8 |
| Peer 4 | 96 | 185.6 | 4239.4 |
| Peer 5 | 96 | 186.6 | 4239.8 |
| Peer 6 | 96 | 185.1 | 4239.4 |
| Peer 7 | 96 | 187.3 | 4239.9 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 225.7 | 250.4 | not captured | not captured | 256.9 |
| 2 | 4096 | 239.2 | 254.1 | not captured | not captured | 1139.6 |
| 3 | 4096 | 3361.7 | 0.0 | not captured | not captured | 1138.6 |
| 4 | 4096 | 3274.6 | 0.0 | not captured | not captured | 1138.7 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d2_shortactive_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='2' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='512' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d2_shortactive_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d2_shortactive_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s12_d2_shortactive_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d2_shortactive_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d2_shortactive_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s12_d4_shortactive_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:50:48 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d4_shortactive_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 745 ms later and peer decoders finished a median 745 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 512 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 186.4 |
| Target finish after tool return (ms) | 4640.1 |
| Whole workload (ms) | 79777.4 |
| Matched control target finish (ms) | 3895.4 |
| Target finish change vs control (ms) | 744.7 |
| Peer median finish change vs control (ms) | 744.7 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 186.4 | 4640.1 |
| Peer 1 | 96 | 59.6 | 4640.1 |
| Peer 2 | 96 | 185.6 | 4639.7 |
| Peer 3 | 96 | 186.7 | 4640.1 |
| Peer 4 | 96 | 187.7 | 4640.3 |
| Peer 5 | 96 | 187.4 | 4640.2 |
| Peer 6 | 96 | 186.1 | 4640.0 |
| Peer 7 | 96 | 187.0 | 4640.2 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 232.3 | 247.9 | not captured | not captured | 258.6 |
| 2 | 4096 | 244.2 | 450.0 | not captured | not captured | 1134.5 |
| 3 | 4096 | 249.9 | 317.6 | not captured | not captured | 1133.8 |
| 4 | 4096 | 236.1 | 253.7 | not captured | not captured | 1134.1 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d4_shortactive_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='512' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d4_shortactive_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d4_shortactive_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s12_d4_shortactive_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d4_shortactive_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d4_shortactive_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s12_d0_shortactive_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:48:17 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d0_shortactive_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 512 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 186.1 |
| Target finish after tool return (ms) | 3895.4 |
| Whole workload (ms) | 78998.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 186.1 | 3895.4 |
| Peer 1 | 96 | 59.6 | 3895.6 |
| Peer 2 | 96 | 185.8 | 3895.2 |
| Peer 3 | 96 | 187.4 | 3895.6 |
| Peer 4 | 96 | 187.1 | 3895.5 |
| Peer 5 | 96 | 186.4 | 3895.4 |
| Peer 6 | 96 | 185.3 | 3895.1 |
| Peer 7 | 96 | 186.7 | 3895.5 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 403.9 | 0.0 | not captured | not captured | 121.4 |
| 2 | 4096 | 4652.9 | 0.0 | not captured | not captured | 251.3 |
| 3 | 4096 | 4168.1 | 0.0 | not captured | not captured | 251.3 |
| 4 | 4096 | 3431.1 | 0.0 | not captured | not captured | 251.2 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d0_shortactive_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='512' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d0_shortactive_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d0_shortactive_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s12_d0_shortactive_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d0_shortactive_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d0_shortactive_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s12_d0_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:45:36 p.m. CDT · KV-load overlap pressure</strong> · rq11_s12_d0_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Excluded diagnostic: At eight long active prompts, SGLang refused donor 2's native load because of load-back threshold, quota, or memory pressure. The same four-load control could not be completed.

**Setup.** nvidia-standard; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 7 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 40000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 12 |
| Planned overlapping loads | 0 |
| Failure | At eight long active prompts, SGLang refused donor 2's native load because of load-back threshold, quota, or memory pressure. The same four-load control could not be completed. |
| Comparable timing | unavailable |

**Evidence gate.** excluded. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- No target or workflow timing from this attempt is included in the dose comparison.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s12_d0_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='12' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='40000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s12_d0_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s12_d0_seed1_20261005/run_manifest.json) · [Harness timeline](docs/reports/work_audit/rq11_s12_d0_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s12_d0_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d4_profile_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:41:17 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d4_profile_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Nsight recorded the worker ranges but no CUDA kernels or copies. Physical overlap is unverified; this profiled run is excluded from the clean timing comparison.

**Setup.** nvidia-standard; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | capture_incomplete |
| Target first token after tool return (ms) | 190.6 |
| Target finish after tool return (ms) | 4462.8 |
| Whole workload (ms) | 81978.0 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 190.6 | 4462.8 |
| Peer 1 | 96 | 190.8 | 4463.0 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 235.5 | 314.5 | not captured | not captured | 253.9 |
| 2 | 4096 | 226.0 | 317.1 | not captured | not captured | 983.7 |
| 3 | 4096 | 228.6 | 280.4 | not captured | not captured | 984.0 |
| 4 | 4096 | 215.4 | 240.5 | not captured | not captured | 983.4 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d4_profile_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='1' WORK_AUDIT_NSYS_ENABLE='1' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/backend_trace.jsonl.gz) · [Nsight capture](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/nsys/backend.nsys-rep) · [Nsight SQLite trace](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/nsys/backend.sqlite.gz) · [Profiler status](docs/reports/work_audit/rq11_s6_d4_profile_seed1_20261005/nsys/profile_status.json)

</details>

<a id="run-rq11_s6_d4_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:38:20 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d4_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 769 ms later and peer decoders finished a median 769 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 4 |
| Worker-window overlaps (proxy) | 4 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 190.0 |
| Target finish after tool return (ms) | 4399.3 |
| Whole workload (ms) | 60693.3 |
| Matched control target finish (ms) | 3630.5 |
| Target finish change vs control (ms) | 768.8 |
| Peer median finish change vs control (ms) | 768.8 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 190.0 | 4399.3 |
| Peer 1 | 96 | 189.9 | 4399.2 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 251.9 | 282.9 | not captured | not captured | 258.3 |
| 2 | 4096 | 238.8 | 241.9 | not captured | not captured | 929.1 |
| 3 | 4096 | 252.8 | 291.3 | not captured | not captured | 928.8 |
| 4 | 4096 | 227.8 | 242.5 | not captured | not captured | 927.6 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d4_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='4' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d4_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d4_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d4_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d4_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d4_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d2_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:36:04 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d2_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 359 ms later and peer decoders finished a median 360 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 2 |
| Worker-window overlaps (proxy) | 2 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 189.3 |
| Target finish after tool return (ms) | 3989.7 |
| Whole workload (ms) | 60844.8 |
| Matched control target finish (ms) | 3630.5 |
| Target finish change vs control (ms) | 359.2 |
| Peer median finish change vs control (ms) | 359.7 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 189.3 | 3989.7 |
| Peer 1 | 96 | 189.4 | 3990.1 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 252.4 | 275.7 | not captured | not captured | 297.6 |
| 2 | 4096 | 215.6 | 227.0 | not captured | not captured | 927.4 |
| 3 | 4096 | 2378.9 | 0.0 | not captured | not captured | 927.3 |
| 4 | 4096 | 5259.7 | 0.0 | not captured | not captured | 926.4 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d2_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='2' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d2_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d2_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d2_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d2_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d2_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d1_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:33:33 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d1_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Compared with its matched zero-overlap control, the target finished 201 ms later and peer decoders finished a median 201 ms later. Worker windows were observed; physical copy overlap and the precise cause remain unverified.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 1 |
| Worker-window overlaps (proxy) | 1 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 187.8 |
| Target finish after tool return (ms) | 3831.5 |
| Whole workload (ms) | 58760.4 |
| Matched control target finish (ms) | 3630.5 |
| Target finish change vs control (ms) | 201.0 |
| Peer median finish change vs control (ms) | 201.0 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 187.8 | 3831.5 |
| Peer 1 | 96 | 187.6 | 3831.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 250.0 | 255.7 | not captured | not captured | 251.6 |
| 2 | 4096 | 2171.2 | 0.0 | not captured | not captured | 910.9 |
| 3 | 4096 | 6086.8 | 0.0 | not captured | not captured | 911.2 |
| 4 | 4096 | 3437.1 | 0.0 | not captured | not captured | 910.7 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d1_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='1' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d1_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d1_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d1_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d1_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d1_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-rq11_s6_d0_seed1_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 8:31:13 p.m. CDT · KV-load overlap pressure</strong> · rq11_s6_d0_seed1_20261005</summary>

**Question (RQ11).** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**Finding.** Zero-overlap control: the same four donor loads ran only after target decode. This is the reference for other doses with the same seed and workload.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. One target and 1 other active decoders resumed after a tool wait. The donor prefixes were explicitly host-resident; the same number of native worker loads ran in every dose, with their timing shifted around target decode. Target output cap 96 tokens; tool waits 900 / 20000 ms; active/donor prompts 4090 / 4090 words. Fresh backend for each dose; no frontend importance ranks.

**Key measurements**

| Measurement | Value |
| --- | --- |
| Sessions | 6 |
| Planned overlapping loads | 0 |
| Worker-window overlaps (proxy) | 0 |
| Physical H-to-D overlaps | not captured |
| Physical copy overlap (ms) | not captured |
| Copy concurrent with decode kernels (ms) | not captured |
| Profiler status | not requested |
| Target first token after tool return (ms) | 187.9 |
| Target finish after tool return (ms) | 3630.5 |
| Whole workload (ms) | 59458.4 |

| Active replay | Output tokens | First token after tool ms | Finish after tool ms |
| --- | --- | --- | --- |
| Target | 96 | 187.9 | 3630.5 |
| Peer 1 | 96 | 187.7 | 3630.4 |

| Donor | KV tokens | CUDA event ms | Worker window ms | Physical copy in decode ms | Copy with target kernels ms | Donor replay TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4060 | 1405.0 | 0.0 | not captured | not captured | 119.1 |
| 2 | 4096 | 2812.3 | 0.0 | not captured | not captured | 248.2 |
| 3 | 4096 | 3075.2 | 0.0 | not captured | not captured | 247.5 |
| 4 | 4096 | 2300.8 | 0.0 | not captured | not captured | 247.8 |

Worker windows are timing proxies, not proof of physical copy overlap. Profiled timing must not be used as an unprofiled slowdown estimate.

**Evidence gate.** worker_window_only. Timestamp: Target replay or staging; displayed in Central Time.

**Limits**

- Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.
- Physical copy overlap requires the separate Nsight evidence gate.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq11_s6_d0_seed1_20261005' WORK_AUDIT_STUDY='overlap_dose' WORK_AUDIT_RESEARCH_QUESTION_ID='RQ11' WORK_AUDIT_SESSION_COUNT='6' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_PLANNED_OVERLAP='0' WORK_AUDIT_SEED='1' WORK_AUDIT_DECODE_TOKENS='96' WORK_AUDIT_ACTIVE_PROMPT_WORDS='4090' WORK_AUDIT_DONOR_PROMPT_WORDS='4090' WORK_AUDIT_SHORT_WAIT_MS='900' WORK_AUDIT_DONOR_WAIT_MS='20000' WORK_AUDIT_FORWARD_TRACE='0' WORK_AUDIT_NSYS_ENABLE='0' AGENTIC_KV_PREPARE_LOAD_WORKER='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq11_s6_d0_seed1_20261005/summary.json) · [Run manifest](docs/reports/work_audit/rq11_s6_d0_seed1_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq11_s6_d0_seed1_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq11_s6_d0_seed1_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq11_s6_d0_seed1_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_cuda_kernels_graceful_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 4:54:36 p.m. CDT · Decode overlap attribution</strong> · work_audit_cuda_kernels_graceful_20261005</summary>

**Question (RQ10).** When an early worker KV load overlaps a different session's decode, is that session delayed before its first token, between backend batches, or inside model forward?

**Finding.** In the captured pair, GPU kernels ran for about the same time; pauses between them grew. No H-to-D copy overlapped those kernel spans. Later profiler data was incomplete, so hardware attribution remains provisional.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; trace kv_decode_overlap; model-forward trace on. Each case began three equal-importance sessions: a short tool wait, a long tool wait, and one session that ended. The long prefix was explicitly evicted to host. Its worker load began either during short decode or after short completion; both loads had to finish before long tool return. Tool waits: 900 / 10000 ms; early load at 1200 ms. 1 warmup pairs excluded; order reversed by pair. Prompt target 4090 words, output cap 16 tokens, host cache 8 GB, GPU memory fraction 0.7.

**Key measurements**

| Trial / mode | Short first token (ms) | Short finish (ms) | Batch time (ms) | Model forward (ms) | Other batch time (ms) | Between batches (ms) | Load overlap (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trial 1 · post_short | 319.2 | 838.7 | 296.4 | 292.0 | 4.4 | 190.9 | 0.0 |
| Trial 1 · early | 320.7 | 1045.8 | 514.0 | 506.5 | 7.5 | 166.6 | 339.1 |
| Trial 2 · early | 323.7 | 998.8 | 482.2 | 477.0 | 5.1 | 150.8 | 288.2 |
| Trial 2 · post_short | 348.5 | 867.6 | 294.8 | 290.5 | 4.3 | 191.8 | 0.0 |

**Nsight GPU check (captured pair only).** The full profiler run lost later CUDA data; these rows have complete kernel linkage. Profiled time is mechanism evidence, not the clean performance estimate.

| Captured case | Kernels | Kernel execution (ms) | Between-kernel gaps (ms) | H-to-D overlap (ms) |
| --- | --- | --- | --- | --- |
| Pair 1 · post_short | 4872 | 469.7 | 17.6 | 0.0 |
| Pair 1 · early | 4872 | 469.9 | 191.8 | 0.0 |

**CUDA launch check (same captured pair).** Most added gap time passed before the CPU started the next launch. This does not identify why the host waited or measure global GPU idle time.

| Captured case | Total gaps (ms) | Before CPU launch (ms) | During launch API (ms) | After launch API (ms) |
| --- | --- | --- | --- | --- |
| post_short | 17.6 | 10.9 | 1.6 | 5.1 |
| early | 191.8 | 178.3 | 8.7 | 4.7 |

Recorded stream-wait event activity was 0.886 ms after-short and 1.028 ms early; blocking CUDA synchronization API time was 0.000 ms and 0.000 ms. These are not additive to the gap categories.

**Evidence gate.** validated timing; partial CUDA capture. Timestamp: First request; displayed in Central Time.

**Limits**

- Worker start-to-commit is an upper bound on GPU-copy activity, not exact HBM overlap.
- Client stream content chunks need not equal model tokens; backend decode steps are separate evidence.
- Batch wall time includes CPU submission and synchronization; it is not GPU kernel time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_cuda_kernels_graceful_20261005 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ10 WORK_AUDIT_STUDY=multisession_overlap WORK_AUDIT_TRACE_PROFILE=kv_decode_overlap WORK_AUDIT_FORWARD_TRACE=1 WORK_AUDIT_NSYS_ENABLE=1 WORK_AUDIT_CASE_ORDER=early-post_short WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=10000 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/backend_trace.jsonl.gz) · [Captured GPU kernels](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/nsys/kernel_attribution_pair01.json) · [CUDA launch gaps](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/nsys/launch_gap_attribution_pair01.json) · [Nsight SQLite trace](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/nsys/backend.sqlite.gz) · [Nsight SQLite trace](docs/reports/work_audit/work_audit_cuda_kernels_graceful_20261005/nsys/backend.sqlite.gz)

</details>

<a id="run-work_audit_cuda_kernels_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 4:45:44 p.m. CDT · Decode overlap attribution</strong> · work_audit_cuda_kernels_20261005</summary>

**Question (RQ10).** When an early worker KV load overlaps a different session's decode, is that session delayed before its first token, between backend batches, or inside model forward?

**Finding.** In the captured pair, GPU kernels ran for about the same time; pauses between them grew. No H-to-D copy overlapped those kernel spans. Later profiler data was incomplete, so hardware attribution remains provisional.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; trace kv_decode_overlap; model-forward trace on. Each case began three equal-importance sessions: a short tool wait, a long tool wait, and one session that ended. The long prefix was explicitly evicted to host. Its worker load began either during short decode or after short completion; both loads had to finish before long tool return. Tool waits: 900 / 10000 ms; early load at 1200 ms. 1 warmup pairs excluded; order reversed by pair. Prompt target 4090 words, output cap 16 tokens, host cache 8 GB, GPU memory fraction 0.7.

**Key measurements**

| Trial / mode | Short first token (ms) | Short finish (ms) | Batch time (ms) | Model forward (ms) | Other batch time (ms) | Between batches (ms) | Load overlap (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trial 1 · post_short | 311.0 | 828.3 | 285.7 | 281.5 | 4.3 | 199.5 | 0.0 |
| Trial 1 · early | 320.6 | 994.1 | 459.3 | 452.8 | 6.5 | 168.3 | 287.4 |
| Trial 2 · early | 320.5 | 989.5 | 473.5 | 466.5 | 7.1 | 150.8 | 319.0 |
| Trial 2 · post_short | 343.3 | 861.3 | 298.8 | 294.3 | 4.5 | 188.3 | 0.0 |

**Nsight GPU check (captured pair only).** The full profiler run lost later CUDA data; these rows have complete kernel linkage. Profiled time is mechanism evidence, not the clean performance estimate.

| Captured case | Kernels | Kernel execution (ms) | Between-kernel gaps (ms) | H-to-D overlap (ms) |
| --- | --- | --- | --- | --- |
| Pair 1 · post_short | 4872 | 469.5 | 17.1 | 0.0 |
| Pair 1 · early | 4872 | 470.1 | 143.3 | 0.0 |

**CUDA launch check (same captured pair).** Most added gap time passed before the CPU started the next launch. This does not identify why the host waited or measure global GPU idle time.

| Captured case | Total gaps (ms) | Before CPU launch (ms) | During launch API (ms) | After launch API (ms) |
| --- | --- | --- | --- | --- |
| post_short | 17.1 | 10.5 | 1.5 | 5.0 |
| early | 143.3 | 132.6 | 5.6 | 5.2 |

Recorded stream-wait event activity was 0.892 ms after-short and 0.937 ms early; blocking CUDA synchronization API time was 0.000 ms and 0.000 ms. These are not additive to the gap categories.

**Evidence gate.** validated timing; partial CUDA capture. Timestamp: First request; displayed in Central Time.

**Limits**

- Worker start-to-commit is an upper bound on GPU-copy activity, not exact HBM overlap.
- Client stream content chunks need not equal model tokens; backend decode steps are separate evidence.
- Batch wall time includes CPU submission and synchronization; it is not GPU kernel time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_cuda_kernels_20261005 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ10 WORK_AUDIT_STUDY=multisession_overlap WORK_AUDIT_TRACE_PROFILE=kv_decode_overlap WORK_AUDIT_FORWARD_TRACE=1 WORK_AUDIT_NSYS_ENABLE=1 WORK_AUDIT_CASE_ORDER=early-post_short WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=10000 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_cuda_kernels_20261005/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_cuda_kernels_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_cuda_kernels_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_cuda_kernels_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_cuda_kernels_20261005/backend_trace.jsonl.gz) · [Captured GPU kernels](docs/reports/work_audit/work_audit_cuda_kernels_20261005/nsys/kernel_attribution_pair01.json) · [CUDA launch gaps](docs/reports/work_audit/work_audit_cuda_kernels_20261005/nsys/launch_gap_attribution_pair01.json) · [Nsight SQLite trace](docs/reports/work_audit/work_audit_cuda_kernels_20261005/nsys/backend.sqlite.gz) · [Nsight SQLite trace](docs/reports/work_audit/work_audit_cuda_kernels_20261005/nsys/backend.sqlite.gz)

</details>

<a id="run-work_audit_decode_forward_repeated_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 4:05:08 p.m. CDT · Decode overlap attribution</strong> · work_audit_decode_forward_repeated_20261005</summary>

**Question (RQ10).** When an early worker KV load overlaps a different session's decode, is that session delayed before its first token, between backend batches, or inside model forward?

**Finding.** Early worker loading delayed the short response in every pair; the added batch time was in model forward, not queue gaps. HBM contention is not established.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; trace kv_decode_overlap; model-forward trace on. Each case began three equal-importance sessions: a short tool wait, a long tool wait, and one session that ended. The long prefix was explicitly evicted to host. Its worker load began either during short decode or after short completion; both loads had to finish before long tool return. Tool waits: 900 / 10000 ms; early load at 1200 ms. 1 warmup pairs excluded; order reversed by pair. Prompt target 4090 words, output cap 16 tokens, host cache 8 GB, GPU memory fraction 0.7.

**Key measurements**

| Trial / mode | Short first token (ms) | Short finish (ms) | Batch time (ms) | Model forward (ms) | Other batch time (ms) | Between batches (ms) | Load overlap (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trial 1 · post_short | 303.0 | 813.6 | 241.9 | 237.8 | 4.1 | 233.7 | 0.0 |
| Trial 1 · early | 306.4 | 967.2 | 428.3 | 422.7 | 5.6 | 185.9 | 278.7 |
| Trial 2 · early | 311.6 | 976.4 | 429.1 | 423.4 | 5.8 | 190.4 | 285.1 |
| Trial 2 · post_short | 313.8 | 823.6 | 237.2 | 233.2 | 4.0 | 238.7 | 0.0 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Limits**

- Worker start-to-commit is an upper bound on GPU-copy activity, not exact HBM overlap.
- Client stream content chunks need not equal model tokens; backend decode steps are separate evidence.
- Batch wall time includes CPU submission and synchronization; it is not GPU kernel time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_decode_forward_repeated_20261005 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ10 WORK_AUDIT_STUDY=multisession_overlap WORK_AUDIT_TRACE_PROFILE=kv_decode_overlap WORK_AUDIT_FORWARD_TRACE=1 WORK_AUDIT_CASE_ORDER=early-post_short WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=10000 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_decode_forward_repeated_20261005/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_decode_forward_repeated_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_decode_forward_repeated_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_decode_forward_repeated_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_decode_forward_repeated_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_decode_overlap_repeated_20261005"></a>
<details>
<summary><strong>Oct 5, 2026, 3:52:46 p.m. CDT · Decode overlap attribution</strong> · work_audit_decode_overlap_repeated_20261005</summary>

**Question (RQ10).** When an early worker KV load overlaps a different session's decode, is that session delayed before its first token, between backend batches, or inside model forward?

**Finding.** Early worker loading delayed the short response in every pair. The trace places the added time inside backend batches, not queue gaps.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; trace kv_decode_overlap; model-forward trace off. Each case began three equal-importance sessions: a short tool wait, a long tool wait, and one session that ended. The long prefix was explicitly evicted to host. Its worker load began either during short decode or after short completion; both loads had to finish before long tool return. Tool waits: 900 / 10000 ms; early load at 1200 ms. 1 warmup pairs excluded; order reversed by pair. Prompt target 4090 words, output cap 16 tokens, host cache 8 GB, GPU memory fraction 0.7.

**Key measurements**

| Trial / mode | Short first token (ms) | Short finish (ms) | Batch time (ms) | Model forward (ms) | Other batch time (ms) | Between batches (ms) | Load overlap (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trial 1 · post_short | 298.9 | 808.0 | 230.0 | not recorded | not recorded | 244.2 | 0.0 |
| Trial 1 · early | 306.3 | 955.4 | 412.9 | not recorded | not recorded | 189.2 | 270.1 |
| Trial 2 · early | 311.0 | 954.0 | 406.9 | not recorded | not recorded | 189.7 | 298.9 |
| Trial 2 · post_short | 317.2 | 825.9 | 233.9 | not recorded | not recorded | 240.3 | 0.0 |
| Trial 3 · post_short | 319.8 | 829.2 | 236.5 | not recorded | not recorded | 238.9 | 0.0 |
| Trial 3 · early | 317.5 | 1011.3 | 461.8 | not recorded | not recorded | 184.0 | 351.0 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Limits**

- Worker start-to-commit is an upper bound on GPU-copy activity, not exact HBM overlap.
- Client stream content chunks need not equal model tokens; backend decode steps are separate evidence.
- Batch wall time includes CPU submission and synchronization; it is not GPU kernel time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_decode_overlap_repeated_20261005 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ10 WORK_AUDIT_STUDY=multisession_overlap WORK_AUDIT_TRACE_PROFILE=kv_decode_overlap WORK_AUDIT_FORWARD_TRACE=0 WORK_AUDIT_CASE_ORDER=early-post_short WORK_AUDIT_PAIRS=3 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_SHORT_WAIT_MS=900 WORK_AUDIT_LONG_WAIT_MS=10000 WORK_AUDIT_EARLY_AT_MS=1200 WORK_AUDIT_ESTIMATED_LOAD_MS=250 WORK_AUDIT_LOAD_MARGIN_MS=150 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_decode_overlap_repeated_20261005/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_decode_overlap_repeated_20261005/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_decode_overlap_repeated_20261005/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_decode_overlap_repeated_20261005/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_decode_overlap_repeated_20261005/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_async_fixed_worker_20261005_01"></a>
<details>
<summary><strong>Oct 5, 2026, 1:28:34 p.m. CDT · Concurrent early vs late · worker load</strong> · work_audit_async_fixed_worker_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** Early loading sped the long replay and whole workflow, but delayed the short session.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution worker. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. 1 warmup pair were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 89.9 | 1006.4 | 7900.2 | -959.3 |
| Trial 1 · late_nonblocking | 1131.2 | 808.5 | 8953.5 | 523.8 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 5, 2026, 1:26:11 p.m. CDT · Concurrent early vs late · scheduler load</strong> · work_audit_async_fixed_scheduler_20261005_02</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** Early loading sped the long replay and whole workflow, but delayed the short session.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1; trace kv_lifecycle_lean; load execution scheduler. Each case started three equal-importance sessions; two waited for tools and one ended. The long prefix was explicitly evicted to host. The ended session's prefix was released in every mode before load-back, enforcing a logical two-prefix budget. Only load timing changed: late control submitted at tool return without blocking replay, or early control submitted at 1200 ms during the long tool wait. 1 warmup pair were excluded. Condition order reversed on alternate trials. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. The logical cap is not a measurement of physical GPU occupancy.

**Key measurements**

| Trial / mode | Long first token after tool (ms) | Short finish after tool (ms) | Workflow (ms) | Load complete relative to tool return (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 89.5 | 991.1 | 7896.4 | -1078.5 |
| Trial 1 · late_nonblocking | 276.0 | 810.7 | 8089.1 | 201.7 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 5, 2026, 1:12:20 p.m. CDT · Busy workload · KV-load attribution · scheduler load</strong> · work_audit_async_scheduler_busy_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 421934.4 | 90257.4 | 0 |
| Seed 1 · check_only | 36 | 423615.9 | 90480.8 | 0 |
| Seed 1 · controller | 36 | 436774.1 | 91272.0 | 3 |

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 5, 2026, 1:03:46 p.m. CDT · Busy workload · KV-load attribution · worker load</strong> · work_audit_async_worker_busy_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution worker, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 423918.7 | 92364.6 | 0 |
| Seed 1 · check_only | 36 | 420453.7 | 92069.0 | 0 |
| Seed 1 · controller | 36 | 442849.6 | 91877.7 | 2 |

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 5, 2026, 1:00:42 p.m. CDT · Lifecycle validation · worker load</strong> · work_audit_async_worker_verify_20261005_01</summary>

**Question (RQ9).** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: host → warm; tool wait: 500 ms; replays/case: 2; measured pairs: 1; warmup pairs: 0; trace: kv_lifecycle; exact-index limit: 256; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load. Prompt target: 4090 words; output cap: 16 tokens; minimum host prefix: 512 tokens; eviction attempts: 4; host cache: 8 GB; GPU memory fraction: 0.7.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 222.7 | 0 | 0 | 4163 |
| host_backed | 899.5 | 2048 | 4096 | 4162 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_async_worker_verify_20261005_01 WORK_AUDIT_RESEARCH_QUESTION_ID=RQ9 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle WORK_AUDIT_CASE_ORDER=host-warm WORK_AUDIT_SECOND_REPLAY=1 WORK_AUDIT_WAIT_MS=500 WORK_AUDIT_PROMPT_WORDS=4090 WORK_AUDIT_MAX_OUTPUT_TOKENS=16 WORK_AUDIT_MINIMUM_HOST_TOKENS=512 WORK_AUDIT_EVICTION_ROUNDS=4 AGENTIC_KV_PREPARE_LOAD_WORKER=1 HICACHE_SIZE_GB=8 MEM_FRACTION_STATIC=0.7 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_async_worker_verify_20261005_01/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_load_kernel_20261005_01"></a>
<details>
<summary><strong>Oct 5, 2026, 11:35:35 a.m. CDT · Busy workload · KV-load attribution</strong> · work_audit_load_kernel_20261005_01</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend kernel, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 437550.8 | 94228.2 | 0 |
| Seed 1 · check_only | 36 | 435266.4 | 94086.8 | 0 |
| Seed 1 · controller | 36 | 452880.2 | 95508.3 | 3 |

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 5, 2026, 9:31:03 a.m. CDT · Busy workload · KV-load attribution</strong> · work_audit_load_phase_20261005_01</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** The actual-load arm had higher replay TTFT in both seeds, while substantive post-first-token generation was not slower. Check-only effects varied by seed; the comparison does not prove copy-engine or HBM contention.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; focused ingress + KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Recorded load phases |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 420768.6 | 94256.0 | 0 |
| Seed 1 · check_only | 36 | 423830.5 | 92192.5 | 0 |
| Seed 1 · controller | 36 | 435562.2 | 92621.3 | 3 |

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 8:19:35 p.m. CDT · Busy workload · KV-load attribution</strong> · work_audit_kv_attribution_20261002_04</summary>

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

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 5:14:42 p.m. CDT · Busy workload · controller KV timing</strong> · work_audit_busy_pair_20261002_01</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 5:08:50 p.m. CDT · Busy workload · controller KV timing</strong> · work_audit_busy_pilot_20261002</summary>

**Question (RQ8).** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**Finding.** Controller-timed KV preparation worsened workflow and total replay TTFT in every paired seed.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1. Fresh backend per arm, order reversed by seed. Prefix target 8192 tokens, replay cap 64 tokens, tool waits [800, 3500] ms, host cache 8.0 GB, KV I/O backend direct, load execution scheduler, GPU memory fraction 0.8. Controller requires a host-resident prefix and at least 250.0 + 150.0 ms before expected tool return. No frontend importance ranks or forced eviction; lean KV trace.

**Key measurements**

| Arm | Replays | Total replay TTFT (ms) | Workflow (ms) | Native loads |
| --- | --- | --- | --- | --- |
| Seed 1 · baseline | 36 | 335038.7 | 79697.6 | 35 |
| Seed 1 · controller | 36 | 355726.5 | 79846.9 | 43 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 4:16:50 p.m. CDT · Controller-chosen load window</strong> · work_audit_controller_window_20261002_01</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 3:34:20 p.m. CDT · Three concurrent load windows</strong> · work_audit_post_short_20261002_01</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 2:48:50 p.m. CDT · Concurrent early vs late</strong> · work_audit_concurrent_compare_20261002_02</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 1:42:21 p.m. CDT · Concurrent timeline</strong> · work_audit_multisession_20261002_01</summary>

**Question (RQ4).** Can a small, equal-importance, concurrent workload link tool waits, one controlled host eviction/load, replays, and an ending session?

**Finding.** The trace linked overlapping tool waits, host-KV movement, and replays across separate sessions.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; trace kv_lifecycle_lean. Three equal-importance sessions began together. Two tool waits overlapped; the third session ended without replay. An explicit control command evicted the long-wait session's GPU prefix under a synthetic two-prefix budget, then the client proved host residency. Its load was requested when the tool returned without gating replay on the control response. Prompt target: 4090 words; output cap: 16 tokens; host cache: 8 GB; GPU memory fraction: 0.7. No frontend task had higher semantic priority.

**Key measurements**

| Session | Tool wait (ms) | First token after tool (ms) | Replay TTFT (ms) | Cached prefix tokens |
| --- | --- | --- | --- | --- |
| short | 901.5 | 81.6 | 81.5 | 4177 |
| long | 2501.3 | 282.4 | 282.0 | 4177 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 1:33:37 p.m. CDT · Early vs late</strong> · work_audit_nonblocking_20261002_01</summary>

**Question (RQ3).** Does not waiting for the late-load control response remove the observed tool-return-to-first-token penalty?

**Finding.** Nonblocking submission shortened the client gap, but the strict nonblocking comparison was withheld.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: early → late → late_nonblocking; tool wait: 2000 ms; replays/case: 2; measured pairs: 1; warmup pairs: 1; trace: kv_lifecycle_lean; exact-index limit: 256; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load. Prompt target: 4090 words; output cap: 16 tokens; minimum host prefix: 512 tokens; eviction attempts: 4; host cache: 8 GB; GPU memory fraction: 0.7.

**Key measurements**

| Trial / mode | Submit after due (ms) | First token after due (ms) | Replay TTFT (ms) | Task duration (ms) |
| --- | --- | --- | --- | --- |
| Trial 1 · early | 0.1 | 88.2 | 88.0 | 7455.2 |
| Trial 1 · late | 170.4 | 253.3 | 82.9 | 7014.9 |
| Trial 1 · late_nonblocking | 0.5 | 243.7 | 243.2 | 7265.2 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 11:41:32 a.m. CDT · Early vs late</strong> · work_audit_timing_sampled_late_early_20261002</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 11:38:54 a.m. CDT · Early vs late</strong> · work_audit_timing_sampled_early_late_20261002</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

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
<summary><strong>Oct 2, 2026, 11:25:50 a.m. CDT · Early vs late</strong> · work_audit_timing_exact_late_early_20261002</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_timing_exact_late_early_20261002 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=late-early WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=4096 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_timing_exact_late_early_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_timing_exact_early_late_20261002"></a>
<details>
<summary><strong>Oct 2, 2026, 11:22:26 a.m. CDT · Early vs late</strong> · work_audit_timing_exact_early_late_20261002</summary>

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

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_timing_exact_early_late_20261002 WORK_AUDIT_STUDY=timing WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=early-late WORK_AUDIT_PAIRS=2 WORK_AUDIT_WARMUP_PAIRS=1 WORK_AUDIT_WAIT_MS=2000 WORK_AUDIT_EXACT_INDICES=4096 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_timing_exact_early_late_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_lean_reverse_20261002"></a>
<details>
<summary><strong>Oct 2, 2026, 10:34:04 a.m. CDT · Lifecycle validation</strong> · work_audit_two_replays_lean_reverse_20261002</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: host → warm; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle_lean; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 82.3 | 0 | 0 | 4178 |
| host_backed | 436.3 | 2048 | 4096 | 4177 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_lean_reverse_20261002 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=host-warm WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_lean_reverse_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_lean_pump_20261002"></a>
<details>
<summary><strong>Oct 2, 2026, 10:26:59 a.m. CDT · Lifecycle validation</strong> · work_audit_two_replays_lean_pump_20261002</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: warm → host; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle_lean; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 81.5 | 0 | 0 | 4179 |
| host_backed | 82.6 | 2048 | 4096 | 4178 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_lean_pump_20261002 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle_lean WORK_AUDIT_CASE_ORDER=warm-host WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_lean_pump_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_20261002_reverse"></a>
<details>
<summary><strong>Oct 2, 2026, 9:54:21 a.m. CDT · Lifecycle validation</strong> · work_audit_two_replays_20261002_reverse</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: host → warm; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 217.1 | 0 | 0 | 4161 |
| host_backed | 573.9 | 2048 | 4096 | 4160 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_20261002_reverse WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle WORK_AUDIT_CASE_ORDER=host-warm WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_20261002_reverse/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_two_replays_20261002"></a>
<details>
<summary><strong>Oct 2, 2026, 9:50:49 a.m. CDT · Lifecycle validation</strong> · work_audit_two_replays_20261002</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: warm → host; tool wait: not recorded ms; replays/case: 2; measured pairs: not recorded; warmup pairs: not recorded; trace: kv_lifecycle; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 220.9 | 0 | 0 | 4160 |
| host_backed | 229.1 | 2048 | 4096 | 4159 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_two_replays_20261002 WORK_AUDIT_STUDY=validation WORK_AUDIT_TRACE_PROFILE=kv_lifecycle WORK_AUDIT_CASE_ORDER=warm-host WORK_AUDIT_SECOND_REPLAY=1 bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_two_replays_20261002/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_two_replays_20261002/run_manifest.json) · [Hook gate](docs/reports/work_audit/work_audit_two_replays_20261002/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/work_audit_two_replays_20261002/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/work_audit_two_replays_20261002/backend_trace.jsonl.gz)

</details>

<a id="run-work_audit_a10g_20261001_final"></a>
<details>
<summary><strong>Oct 1, 2026, 5:42:31 p.m. CDT · Lifecycle validation</strong> · work_audit_a10g_20261001_final</summary>

**Question (RQ1).** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**Finding.** Host-backed KV movement and replay were linked; this was not a policy-speed comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; SGLang 0.5.10.post1 with v0510 adapter. Case order: warm → host; tool wait: not recorded ms; replays/case: not recorded; measured pairs: not recorded; warmup pairs: not recorded; trace: not recorded; exact-index limit: not recorded; frontend priority: none. Cases ran sequentially with no intentionally competing filler requests. The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load.

**Key measurements**

| Case | Replay TTFT (ms) | Host tokens | Loaded tokens | Largest matched prefix (tokens) |
| --- | --- | --- | --- | --- |
| warm_control | 227.1 | 0 | 0 | 4162 |
| host_backed | 230.3 | 2048 | 4096 | 4161 |

**Evidence gate.** validated. Timestamp: First request; displayed in Central Time.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID=work_audit_a10g_20261001_final WORK_AUDIT_STUDY=validation WORK_AUDIT_CASE_ORDER=warm-host bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/work_audit_a10g_20261001_final/summary.json) · [Run manifest](docs/reports/work_audit/work_audit_a10g_20261001_final/run_manifest.json)

</details>
