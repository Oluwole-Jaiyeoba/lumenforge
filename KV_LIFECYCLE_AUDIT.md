# KV Lifecycle Audit

**Research question:** Given what the harness knew, was KV moved, kept, or rebuilt at the wrong time?

This audit compares observed cache work with replay timing and whole-workload outcomes. A faster control call is not automatically a faster agent task. All times below are from saved runs; experimental and hypothetical claims are kept separate.

## Key Research Pivots

Manager-facing experiments that materially changed the direction or interpretation of the research.

| Research pivot | Main finding | Why it matters |
| --- | --- | --- |
| [Capacity-safe memory tiering](#rq-RQ24) | When every admitted session fit in GPU KV, CPU tiering added 10-16% to workload time, while storage still added 93-110%. | It separated raw tier-recovery cost from GPU-cache overcommit and showed that admission pressure caused much of the earlier CPU slowdown. |
| [Unrestricted memory-tier pressure](#rq-RQ23) | With all six sessions free to compete, CPU tiering added 94-141% to workload time and storage added 202-264%. | It revealed that lower-tier recovery and uncontrolled concurrent demand can compound into a severe full-system penalty. |

## Research progress

<a id="rq-RQ24"></a>
### RQ24: What is the tiering cost when active KV always fits?

**Question.** When every currently active session fits within the restricted GPU KV budget, how much performance is lost solely because waiting-session KV is kept on CPU or storage and restored only after its tool call returns? Does a near-simultaneous return burst make that unprepared tiering penalty worse?

**What the evidence says.** Eighteen fresh-backend A10G arms completed: three seeds, six equal-priority sessions, ten one-second tool waits per session, two return patterns, and three memory placements. At most two sessions ran at once. Their maximum estimated active KV was 9342 tokens, safely below the 12288-token GPU limit, with zero capacity violations across 1080 measured replays. For spread returns, median whole-workload time rose from 33.384 seconds with resident GPU KV to 36.622 seconds with CPU KV and 64.543 seconds with storage KV: increases of 9.7% and 93.3%. Median mean due-to-first-token delay rose from 516 to 752 and 2449 ms. For 75 ms burst returns, median whole-workload time rose from 30.954 to 35.945 and 64.853 seconds: increases of 16.0% and 109.5%. Median mean due-to-first-token delay rose from 875 to 1129 and 2883 ms. Backend TTFT after submission stayed close across tiers; the main penalty happened before submission while a due session waited for an active slot and complete KV restoration. Native evidence proved all 360 resident-mode replays came from GPU KV, all 360 CPU-mode replays came from host KV, and all 360 storage-mode replays came from storage before becoming GPU-ready.

**Working hypothesis.** Lower-tier placement can materially slow an agentic workload even when the KV of every admitted session fits in GPU memory. CPU recovery is a modest cost in this setup, while unprepared storage recovery is large enough to nearly double the workload. Bursty tool returns make the relative penalty worse because several due sessions serialize behind admission and restoration. Exact tool-return knowledge may close part of this gap by staging waiting-session KV before it becomes due.

**Not yet proved.** This intentionally measures an unprepared worst-credible case: no KV is prefetched before tool return, and a session is admitted only after its complete prefix is restored. The prompts, one-second waits and return patterns are synthetic. File-backed storage used normal operating-system caching and the page cache was not flushed, so this is not a physical SSD-latency claim. Priming was excluded, generated text was not compared byte-for-byte, and the active-KV limit is an estimate based on prompt tokens rather than physical HBM occupancy. The run does not yet show how much advance staging can recover or whether an optimized implementation can overlap recovery without delaying active sessions.

<a id="rq-RQ23"></a>
### RQ23: How large is the GPU, CPU and storage KV gap?

**Question.** With a 7B coding model, equal-priority multi-turn sessions and no prefetch before tool return, how far do whole-workload time and replay response time separate when KV stays on GPU, spills to CPU, or can fall through CPU to file-backed storage? Does a near-simultaneous tool-return burst make the gap worse than normally spread returns?

**What the evidence says.** Eighteen fresh-backend A10G arms completed: three rotated seeds, six sessions, ten one-second tool waits per session, and both spread and 75 ms burst returns. With spread returns, median whole-workload time was 34.796 seconds all-GPU, 67.632 seconds with CPU-tier KV, and 104.891 seconds with GPU+CPU+storage tiering. Median mean due-to-first-token delay was 130, 1799 and 3936 ms. With burst returns, whole-workload time was 28.070, 67.695 and 102.487 seconds, while mean replay delay was 260, 2267 and 3956 ms. Relative to all-GPU, CPU tiering added about 94% workload time for spread returns and 141% for burst returns; storage tiering added about 202% and 264%. Client submission waiting stayed small. Native SGLang evidence showed all 360 all-GPU replays using GPU KV and all 360 CPU-tier replays using host KV. Across the 360 storage-mode replays, 148 had positive native storage hits, 41 had positive CPU hits, 33 had positive GPU hits, and the remaining replays showed zero reused cache tokens.

**Working hypothesis.** Unprepared tool returns expose a large placement gap even though earlier experiments showed that an isolated host-to-GPU copy can be cheap. The dominant cost here is the full lower-tier replay recovery and serialized service behavior under repeated multi-session demand, not late client submission. Simultaneous returns increase the relative penalty because the all-GPU reference batches them efficiently while lower-tier recovery remains expensive.

**Not yet proved.** This is a synthetic worst-credible setup, not a production distribution: every session used the same 4096-token starting context, ten fixed one-second waits and 16 output tokens. GPU and CPU cache caps were deliberately different to force tier exposure. File-backed storage used normal operating-system caching and the page cache was not flushed, so a native storage hit is logical L3 evidence rather than proof of a physical SSD read. The experiment measures the unprepared gap; it does not yet show how much exact tool-return knowledge and advance staging can close it, or isolate storage I/O from cache reconstruction and scheduler service time.

<a id="rq-RQ22"></a>
### RQ22: Can session-level advance loading make CPU spill invisible?

**Question.** With twenty independent sessions, identical tool clocks in all modes, reserved GPU preparation space and exact tool-return times, can individual early KV restores overlap active model work and approach a fully GPU-resident reference without a group barrier?

**What the evidence says.** The session pipeline worked correctly but did not make swapping invisible in the short gate. After removing a wasteful control-poll loop and moving preparation 750 ms ahead, whole-workload time was 7.200 seconds for normal restricted-cache operation, 9.804 seconds for coordinated CPU/GPU operation and 6.170 seconds for the GPU-resident reference. Coordinated mode was 36.2% slower than normal and 58.9% slower than resident; all twenty sessions finished later. Its mean TTFT after submission was 448 ms, but mean delay from tool due time to first token was 1634 ms because 71.180 seconds of submission waiting accumulated across sixty replays. Only 24 of 60 replays had KV ready by their tool deadline. All 44 actual loads were requested while other model requests were active, proving that session-level transfer/computation overlap occurred. Native CUDA-stream intervals totaled 0.971 seconds; release, residency, prepare and status controls totaled about 4.669 seconds of overlapping control wall time.

**Working hypothesis.** The remaining gap is primarily a preparation-throughput and software-control problem, not absence of overlap. Closely spaced session returns required 44 separate release/load handovers. Even with a 750 ms lead and two-session headroom, individual controls could not keep up, so late preparation became submission delay. Dynamic micro-batching of several soon-due independent sessions may retain session-level scheduling while reducing control transactions and native load launches.

**Not yet proved.** This is one three-turn calibration trial with synthetic prompts, fixed one-second waits, eight output tokens, at most eight in-flight requests and an imposed GPU KV-token cap. It does not establish production performance or show that micro-batching will solve the delay. The controller used exact synthetic tool-return times, and initial priming/preparation was excluded. Native CUDA intervals are not pure copy-engine busy time, and control wall times overlap each other and model work. The first pilot used a 350 ms lead and repeated residency polling; it is retained as calibration evidence and must not be pooled with the corrected pilot.

<a id="rq-RQ21"></a>
### RQ21: Can ideal paired tool waits hide CPU/GPU KV swaps?

**Question.** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**What the evidence says.** The native swaps and replay-prefix checks worked, but the waits did not hide the handovers. All nine arms completed 800 replays each. Across three rotated trials, independent sessions finished in 80.272-81.557 seconds, coordinated sessions in 99.703-101.347 seconds, and the same paired schedule with resident KV in 80.728-80.749 seconds. Coordination was 24.2-24.3% slower than the independent baseline, and all twenty sessions finished later in every paired comparison. Its mean TTFT after submission was only 316-321 ms versus 719-761 ms independently, but mean delay from tool due time to first token rose to 803-849 ms versus 724-766 ms. All 790 measured session restores per coordinated arm were observed ready after their original deadlines; the first ten sessions were prepared during excluded setup. Coordinated and resident replays reused at least 98.9% of their input prefix on GPU. CUDA graphs and overlap scheduling were on. This is a functional swapping proof, not GPU-resident-like performance.

**Working hypothesis.** The current release, restore and readiness-check sequence exceeds the available handover window. Coordinated controls took about 40-42 seconds of combined wall time per run; 79 native CUDA-stream load intervals totaled about 16.9 seconds. Those intervals overlap control time and must not be added together. The roughly 19-21 second whole-workload penalty versus the resident schedule appears before replay submission; after-submission TTFT was similar. This does not isolate copy-engine occupancy, scheduler bookkeeping or each control round trip as the single cause.

**Not yet proved.** This uses twenty separate synthetic contexts, eight output tokens per replay, forty one-second tool waits, and an imposed GPU KV-pool cap, not exhausted physical GPU memory. The independent baseline has different ready times and batching. Initial priming is excluded; measured control, alignment and submission waits count. Count-only tracing avoids reading tensor index values but does not prove exact slot identity or zero tracing overhead. One setup reply was compared before and after a restore in each arm, not every replay or tensor. Earlier short pilots used different tracing, transfer or control settings and must not be pooled with the full run. These results do not establish production behavior or a hardware-offload benefit.

<a id="rq-RQ20"></a>
### RQ20: Does selective early storage staging improve a full busy workload?

**Question.** With repeated tool returns and natural file-backed KV displacement, can one-at-a-time storage-to-host staging during sufficiently long waits improve whole-workload completion and replay delay without shifting too much delay to other sessions?

**What the evidence says.** A six-session baseline calibration produced only one storage candidate, so the paired test used eight equal-priority sessions, six tool returns each, 1 GiB host cache, 10240 GPU KV tokens, and two reversed-order seeds. Baselines had 19 and 24 native storage-tier replay hits. Selective staging completed two early loads in seed 1 and one in seed 2; none finished after the tool-return deadline. Whole-workload duration changed from 28.546 to 27.842 seconds in seed 1, but from 29.978 to 30.575 seconds in seed 2. Median due-to-first-token delay worsened from 1096 to 1158 ms and from 847 to 936 ms. Individual session finish times moved in both directions, with some delayed by over one second. Of the three matched early-staged replays, one improved by 211 ms and two worsened by 582 and 505 ms. This policy reliably exercised the storage path but did not show a consistent workload or replay-latency benefit.

**Working hypothesis.** Under busy multi-session pressure, a small number of feasible storage stages and interference with shared scheduling can outweigh the benefit of moving data before a tool returns. An early completion flag alone is not evidence that a replay will be faster or that the whole workload will finish sooner.

**Not yet proved.** Only two paired seeds and three successful stages were observed. Storage-candidate and native-hit counts varied across arms despite matched logical sessions. Requests used synthetic prompts and file-backed storage that may be served by the OS page cache. Generated text was not compared byte-for-byte. The host ran synced working-tree files, so its manifest has an empty Git commit; the archived source change and trace identify the implementation. The experiment does not isolate the precise source of delay to peer sessions, establish production prevalence, or prove a hardware-offload benefit.

<a id="rq-RQ19"></a>
### RQ19: Where do storage-stage delays come from?

**Question.** When early storage-to-host KV staging delays peer requests, how much is associated with the scheduler-control path versus the native staging action, and what explains the apparent delay from storage data readiness to cache availability?

**What the evidence says.** Two order-reversed, four-session controlled runs compared no early action, three scheduler-control status checks without transfer, and real storage-to-host staging. Median peer completion moved from 2585 to 2618 to 2647 ms in seed 1 and from 2588 to 2607 to 2667 ms in seed 2. Thus control-only checks added about 33 and 20 ms, while real staging added about 62 and 79 ms versus baseline. The returning session's due-to-first-token delay fell from 380 to 162 ms and 353 to 163 ms with staging; whole-workload duration fell by 224 and 212 ms. Native data readiness took 76 and 84 ms. The later status-observed timestamp followed after 189/183 ms until the next poll, 702/722 ms waiting for scheduler-control dequeue, and only about 6 ms inside the final status action. Earlier four- and eight-session traces show the same poll/queue pattern. The long data-ready-to-status interval is not a measurement of storage copying or cache-commit execution. Peer first-token differences appeared both before the first backend cache lookup and after it, depending on the peer.

**Working hypothesis.** The busy scheduler delays execution of status checks and may postpone publication of prefetched KV; issuing control checks also has some peer cost. Native staging adds further cost, but these runs do not separate storage I/O, host-cache bookkeeping, CPU scheduling, and GPU effects within that remainder.

**Not yet proved.** Only two seeds, three fresh peer requests, a synthetic five-second tool wait, and a deliberately forced storage-only target prefix were tested. The three control-only status checks approximate the staging arm's control-call pattern but do not execute the same native prefetch action. File-backed data may have come from the OS page cache. The exact moment prefetched KV became usable is not independently timestamped; the reported commit timestamp is taken when a status check executes. These results do not establish physical SSD latency, HBM contention, production frequency, or a hardware-offload benefit.

<a id="rq-RQ18"></a>
### RQ18: When does early storage staging stop helping everyone?

**Question.** If a returning session's storage-resident KV is staged during its tool wait, how do its replay delay, peer latency, and whole-workload time change as equal-priority session count rises?

**What the evidence says.** In paired controlled runs with one returning session and 0, 1, 3, or 7 concurrent peers, host staging reduced the returning session's deadline-to-first-token delay by 150, 186-207, 206, and 157 ms respectively. Whole-workload duration fell by 116-190 ms across these pairs. With one peer, that peer completed 60-64 ms later in both seeds; with three peers, all completed 54-56 ms later; with seven peers, all completed 66-106 ms later. Thus the returning-session gain is not a win for every session. Replay output hashes matched across arms in the four- and eight-session runs. These controlled runs deliberately made a 2048-token prefix storage-only before the tool wait; natural-pressure loading is tested separately in RQ17.

**Working hypothesis.** Tool-return timing can hide a storage load for the returning session, but loading during concurrent work shifts some delay to peers.

**Not yet proved.** The controlled ladder used one seed at one, four, and eight sessions, two seeds at two sessions, synthetic prompts, a five-second tool wait, and deliberate GPU/host eviction. Peer sessions issued concurrent requests rather than full independent multi-turn workflows. File-backed L3 hits may be served by the OS page cache and do not prove physical SSD I/O. These data do not establish hardware-offload value or a production threshold.

<a id="rq-RQ17"></a>
### RQ17: Does storage staging help across repeated tool returns?

**Question.** With eight equal-priority agent sessions, six tool returns each, growing prompt histories, and natural GPU/host cache displacement, can storage-to-host staging during known tool waits improve replay delay and whole-workload time without harming other sessions?

**What the evidence says.** The original three-seed study was blocked: one completed pair used 48 replays per arm, with median deadline-to-first-token delay 1705.9 to 1702.0 ms, p95 4358.7 to 4843.8 ms, and whole-workload duration 43.153 to 44.153 s. A later host-stage arm hit SGLang's evict_host cache-tree assertion; a full-GPU-prepare pilot hit a separate scheduler assertion. The unguarded partial-prefix path was replaced by a capacity- and rate-limit-guarded suffix path. In a subsequent one-seed eight-session, six-tool-return gate, that guarded path completed without a cache-tree failure. Only one of nine observed storage candidates staged before due time, and eight were skipped for insufficient host capacity. Median replay delay changed from 1046 to 1038 ms; whole-workload time rose from 27.336 to 27.468 s. This is a successful safety gate under one pressure setup, not a demonstrated whole-workload gain or a completed three-seed study.

**Working hypothesis.** When many sessions contend for the host cache, staging a missing storage suffix may need reservation and cache-tree synchronization; simply issuing the native prefetch early can move work into tool waits but cannot be assumed to improve end-to-end performance or remain safe under pressure.

**Not yet proved.** There is no validated cross-seed performance conclusion. The exact internal cause of the historical assertions remains unknown. The first guarded attempt recorded a native prefetch refusal and was blocked by the audit gate; the adapter now explicitly checks SGLang's prefetch rate limit. The completed retry had only one successful early stage and was not a full repeat of the three-seed study. The guard prevents prefetch when host capacity is insufficient; it does not fix SGLang's underlying eviction assertion or establish safety across versions and loads. The workload is synthetic, and file-backed L3 hits do not prove physical SSD reads.

<a id="rq-RQ16"></a>
### RQ16: Does storage preparation still help under peer pressure?

**Question.** With a storage-resident target prefix and equal-priority peer requests, does preparing KV during a five-second tool wait still improve target replay as concurrent peer pressure rises, and what do peers pay?

**What the evidence says.** On the A10G with pinned SGLang 0.5.10.post1, a two-seed, three-arm synthetic sweep used the same 2048-token target prefix, 1024-token peer prompts, 64-token file-cache pages, fresh backend per arm, and 0, 2, or 6 peers starting at tool-wait onset. Median target deadline-to-first-token time (on demand / host stage / full prepare) was 370 / 171 / 60 ms at 0 peers, 364 / 170 / 61 ms at 2 peers, and 468 / 168 / 60 ms at 6 peers. All staged loads completed before tool return; both peers at level 2 and all six at level 6 overlapped preparation. Peer median TTFT was 157 / 393 / 456 ms at 2 peers and 1026 / 1075 / 1184 ms at 6 peers. Peer median completion was 1923 / 2158 / 2306 ms at 2 peers and 2963 / 3049 / 3150 ms at 6 peers. Whole-workflow duration was 16195 / 16141 / 15982 ms at 2 peers and 16445 / 16108 / 16045 ms at 6 peers. Native prefetch-completion evidence separated data readiness from later host-cache commit/status polling: the latter interval grew to roughly 581-586 ms for staged arms at six peers. Target replay benefited at every tested pressure, but peers sometimes waited longer; this is not an unconditional system-wide win.

**Working hypothesis.** Known tool-return timing can hide a returning session's storage preparation even amid concurrent work, but the backend's status-poll/commit and other requests' startup can compete for software scheduling time. Pressure-aware admission or pacing may be needed to avoid shifting delay onto peers.

**Not yet proved.** These were two seeds per pressure level, with synthetic prompts, deliberate device/host eviction, one returning target, and at most six peer requests. At two peers, peer TTFT varied substantially between seeds. File-backed L3 hits do not establish physical SSD reads; OS page-cache warming may explain why native data-readiness times were shorter in later pressure runs. The ready-to-commit interval includes status-poll delay and must not be read as pure commit cost. The whole-workflow metric is the last completion among these requests, not a production throughput measure. The remote manifests have an empty Git commit because the host used a synced source copy; this change archives the evidence with the experiment code. This study does not isolate the exact cause of peer delay, prove net benefit across independent workflows, or establish hardware-offload value.

<a id="rq-RQ15"></a>
### RQ15: Can storage KV be prepared during a tool wait?

**Question.** When a session prefix is present in file-backed storage but absent from host and GPU cache, can native storage-to-host and host-to-GPU preparation during a known tool wait reduce replay delay, and what happens to peer requests?

**What the evidence says.** Three paired A10G trials on pinned SGLang 0.5.10.post1 each used a fresh backend, the same 2048-token prefix, a 5-second synthetic tool wait, explicit device and host eviction, and 64-token file-cache pages. Native trace gates proved an L3 hit on on-demand replay, 2048 tokens staged during the wait in the early arms, a matched prefix on replay, and preparation complete before tool return. Time from tool-return deadline to first token was 359 to 170 to 62 ms in the quiet trial, 330 to 165 to 59 ms with two peers started after preparation, and 337 to 166 to 65 ms with two peers whose requests both overlapped preparation (on demand, host staging, full preparation). In the overlapping trial, peer median TTFT rose from 154 ms on demand to 196 ms with host staging and 203 ms with full preparation; peer median completion was 1934, 1939, and 2071 ms respectively. Early preparation hid much of the returning session's startup delay in these runs, but was not a demonstrated win-win for peer sessions.

**Working hypothesis.** A known tool-return time can make storage-to-host and host-to-GPU movement useful before replay is submitted, but contention with other work may shift some delay onto peer requests. A capacity-aware policy would need to account for that tradeoff.

**Not yet proved.** The file-backed L3 hit does not prove physical SSD reads because the OS page cache may serve data. The workload used deliberate evictions, one returning session, at most two synthetic peers, one seed per concurrency setup, and a fixed five-second wait. Peer latency differences are measured associations, not a fully isolated transfer penalty. The experiment invokes explicit preparation controls rather than an autonomous controller; it does not establish production frequency, net system benefit at scale, or hardware-offload value. The remote run manifests have an empty git commit because the host used a synced source copy; the archived raw traces, settings, and local source provide reproducibility evidence instead.

<a id="rq-RQ14"></a>
### RQ14: Do startup delays recur across many tool returns?

**Question.** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**What the evidence says.** On the pinned A10G backend, two equal-priority active sessions each made 12 tool returns with prompt history growing from 935 to 2291 tokens. Four backend-flag combinations ran at low pressure (no donors) and high pressure (eight competing sessions), each with two seeds and a fresh backend. At low pressure, active median first-token delay was 92-101 ms and no host-to-device KV load-backs were recorded. At high pressure, the median was 132-177 ms; 3-4 of 24 active replays per run had recorded KV load-back, and first-token p95 ranged from 471 to 958 ms. In one crowded baseline, those four loaded turns spent 15-21 ms inside the native load-back call but 196-563 ms from cache lookup completion to first batch start. Two trace-off controls per pressure level still had high-pressure first-token p95 of 465 and 504 ms, so the tail does not require tracing, although traced baseline high-pressure workflow time was 27.80 and 28.45 s versus 25.64 and 25.99 s trace-off. With both features enabled, high-pressure median first-token delay was higher than both-off in both seeds, while active workflow completed sooner. Thus the earlier 385 ms is not a fixed tax after every tool return; pressure creates occasional large startup tails, and feature settings trade first-token latency against whole-workload time.

**Working hypothesis.** Under cache pressure, replay startup can wait for scheduler admission or competing backend work beyond the duration of the native KV load-back call. Repeated tool returns make these tail events relevant to a full agent workflow.

**Not yet proved.** The trace places time before the replay's first backend batch but does not identify the exact scheduler, admission, CPU, or GPU mechanism. The observed load-back call duration is not physical device-copy time. Trace-on and trace-off runs alter timing and may change scheduling, so their difference is a perturbation check, not a calibrated instrumentation-overhead subtraction. These deterministic synthetic runs use only two seeds, two active sessions, at most 2291 prompt tokens, and changing natural eviction counts across modes; they do not establish production prevalence, an optimal flag setting, or a hardware-offload benefit. The remote run manifests record the host's older Git HEAD: the new experiment code was synced from a working tree before it was committed. A later move of raw-event translation into the backend adapter reproduced the saved metrics on a representative trace, but that does not make the recorded HEAD an exact source snapshot.

<a id="rq-RQ13"></a>
### RQ13: Can existing backend features absorb KV-load overlap?

**Question.** On the pinned A10G backend, do CUDA graphs and overlap scheduling reduce the extra target-decode time from four native host-KV loads, without moving KV management into hardware?

**What the evidence says.** In matched six-session, 2048-word, 96-output-token A10G runs, four worker KV loads added 475 and 446 ms to target tool-return-to-completion time with both features off (13.5% and 12.6%). With CUDA graphs and overlap scheduling both enabled, the added time was 6.6 and 1.2 ms (0.18% and 0.03%) across the same two seeds. Single-seed checks found +142 ms (4.1%) with graphs alone and +458 ms (11.4%) with overlap scheduling alone. First-token timing was essentially unchanged within each mode. One separate profiler run with both features on verified four physical host-to-GPU loads during target decode: 37.145 ms of copy activity in the decode interval, but only 0.044 ms concurrent with recorded decode kernels. Thus existing software features largely removed the measured incremental overlap penalty in this small workload, without proving a hardware bandwidth effect or hardware-offload benefit.

**Working hypothesis.** The original slowdown depends strongly on backend work-submission behavior; CUDA graphs together with overlap scheduling may keep decode work moving despite concurrent KV-load orchestration.

**Not yet proved.** Two seeds cover only the both-off and both-on comparison; single-feature arms have one seed. The combined mode raised no-load target first-token time from about 65-66 ms to 385-387 ms and no-load completion from about 3.53 to 3.65 seconds, so minimal added overlap cost is not an unconditional win. Worker-window overlap is a proxy in the unprofiled comparisons; the separate Nsight run verifies physical copy overlap but barely any copy/kernel concurrency and must not be used for latency estimates. The 2048-word prompts were chosen because a 4090-word graph-enabled four-load attempt hit KV-capacity admission limits. The exact reason the combined features remove the incremental penalty, production prevalence, and any hardware benefit remain unproven. The first-seed summaries retain their reused RQ11 driver label; their run manifests and this milestone identify them as RQ13, and later summaries emit RQ13 directly.

<a id="rq-RQ12"></a>
### RQ12: Where does the overlap slowdown occur?

**Question.** When native host-to-GPU KV copies physically overlap another session's decode, does added time appear inside decode kernels or in the host's cadence of submitting them?

**What the evidence says.** Two order-balanced six-session A10G profile pairs each captured zero versus four physical KV-load overlaps while the target had the same 94 model forwards and 32,712 linked kernels. Four loads contributed about 73 ms of physical copy overlap in each overlap arm. Summed decode-kernel execution changed by only +1.1 and +1.4 ms, while time inside forwards before the CPU began the next CUDA launch grew by 481 and 546 ms; gaps between forwards grew by another 135 and 138 ms. Earlier unprofiled six-session runs found 21-23% longer target completion with four worker-window overlaps. The profile supports delayed host-side launch cadence, not slower decode kernels, as the main measured locus of the delay.

**Working hypothesis.** KV-load orchestration or competition for host-side scheduler/launch resources delays submission of subsequent decode work. Physical copies do overlap kernels, but their direct bandwidth effect is not shown to explain the large completion penalty.

**Not yet proved.** The profiler locates time before CUDA launches but does not show why the host waited: Python/CPU contention, SGLang scheduling, batching, copy-launch overhead, or a mixture remain possible. Nsight perturbs timing, so profiled completion differences are not the clean slowdown estimate. This is synthetic traffic on one A10G and does not establish a hardware-offload speedup or a production-wide frequency. The first reverse-order pair had a post-run manifest recovery and a corrected research-question label; raw traces were unchanged.

<a id="rq-RQ11"></a>
### RQ11: Does more KV-load overlap delay other decoders?

**Question.** With equal-importance sessions and the same four host-resident donor prefixes, does shifting more native worker KV loads into an active replay's decode window increase other sessions' latency as session count grows?

**What the evidence says.** On the pinned A10G setup, both six- and twelve-session pilots realized 0, 1, 2, and 4 worker-window overlaps as scheduled. Target tool-return-to-finish time rose with dose: 3.631 to 4.399 seconds in the six-session first seed (+21.2%) and 3.895 to 4.640 seconds in the twelve-session first seed (+19.1%). Zero-versus-four endpoints repeated in a second seed: 3.630 to 4.475 seconds (+23.3%) with six sessions and 3.896 to 4.683 seconds (+20.2%) with twelve. Peer decoders finished later in the high-dose arms too. First-token timing did not rise consistently, so the extra time was mainly after first token. All doses executed the same four native donor loads; only timing changed within each series.

**Working hypothesis.** The added decode time may come primarily from delayed host-side submission of GPU work while KV loads are active, rather than slower decode kernels or direct HBM-bandwidth contention. The partial RQ10 captures suggested this; the RQ11 sweep itself did not capture enough physical CUDA activity to test it. RQ12 later tests this hypothesis in a separate six-session profile.

**Not yet proved.** Worker start-to-commit windows are proxies, not verified physical host-to-GPU copy overlap. A profiled repeat recorded all four NVTX load ranges but no CUDA kernel or memcpy activity, so the physical-overlap gate failed. The cause could be host launch cadence, scheduler/batch effects, GPU copies, or a mixture; this does not isolate HBM bandwidth or prove a hardware fix. The six-session series used 4090-word active prompts and 20- or 40-second donor waits, while the twelve-session series used 512-word active prompts and 40-second donor waits to fit capacity; compare doses within each series, not absolute times across series. These synthetic runs are small, and only zero/four endpoints have a second seed. Excluded capacity and timeout attempts appear in the experiment details.

<a id="rq-RQ10"></a>
### RQ10: Which stage slows another decoder?

**Question.** When an early worker KV load overlaps a different session's decode, is that session delayed before its first token, between backend batches, or inside model forward?

**What the evidence says.** In three warmed, order-balanced A10G pairs, early loading made the short session finish 128-182 ms later than loading after that session finished; first-token timing was nearly unchanged. Backend batch time increased 173-225 ms while gaps between batches decreased. A follow-up two-pair run placed 185-190 ms of added time inside model forward. In two later Nsight captures, the first pair in each had complete CUDA linkage: both modes launched 4872 short-replay kernels, total kernel execution changed by only 0.3-0.6 ms, but between-kernel gaps grew by 126 and 174 ms. Of those added gaps, 122 and 167 ms occurred before the CPU began the next CUDA launch; launch-call and after-launch time changed much less. Recorded stream-wait activity was about 1 ms per arm, and no blocking CUDA synchronization API was observed in those forward calls. No host-to-device copy was recorded during the short-forward kernel spans. The captured evidence points to delayed host-side launch cadence, not slower kernels or direct concurrent copy-bandwidth contention.

**Not yet proved.** Nsight lost CUDA activity for later cases in both two-pair profiling runs; the launch-gap result is a validated subset, not a complete order-balanced profiler experiment. A reverse-order run had no CUDA kernel capture and is excluded. The trace does not show why the host delayed its next launch: CPU scheduling, Python work, other software waits, or competition from the worker remain candidates. Gaps between this request's kernels are not global GPU-idle measurements. The workload used explicit host eviction, a synthetic logical two-prefix budget, and a 10-second long tool wait; hardware offload benefits and production frequency remain unproven.

<a id="rq-RQ9"></a>
### RQ9: Does off-scheduler KV loading help?

**Question.** If a host-resident prefix reserves device slots and copies on a worker stream while the scheduler continues serving other requests, does that reduce replay or whole-workload time without exposing incomplete KV?

**What the evidence says.** An opt-in prototype for pinned SGLang 0.5.10.post1 passed a live exact host-versus-device K/V equality check across all model layers before publishing a 4096-token prefix; the trace recorded 56 K/V copy events. In one warmed, comparable three-session pair per execution mode, early-load replay delay was 89.5 ms with scheduler copies and 89.9 ms with worker copies. For a load requested at tool return, the worker control response fell from about 200 ms to 16 ms, but the replay arrived before commit, recomputed its prefix, and its first-token delay rose from 276 to 1131 ms. In separate one-seed 12-session busy runs, summed replay TTFT was 13.16 seconds above check-only for scheduler copies and 22.40 seconds above check-only for worker copies; the runs admitted three and two loads respectively, so this is not a matched causal difference. Moving copy submission off the scheduler is feasible, but these runs do not show an end-to-end performance win.

**Not yet proved.** The prototype only handles explicit prepare-control loads, direct host I/O, and one tensor-parallel rank. A replay reaching the backend before commit may recompute rather than wait for the prefix. One warmed pair and one busy seed per mode cannot establish general performance. The busy admissions differed. Dedicated-stream copy time also changed under overlap, so scheduler relief is not an isolated hardware-speed measurement. Production replay gating, multi-rank safety, and native automatic load paths remain unimplemented.

<a id="rq-RQ8"></a>
### RQ8: Does timed KV preparation help a busy system?

**Question.** With 12 equal-importance sessions, three tool waits each, and natural SGLang cache pressure, does using each session's expected tool return to prepare host KV improve total replay timing and whole-workload completion?

**What the evidence says.** The original two-seed A10G comparison found higher summed replay TTFT and slower workflows with timed preparation. A three-arm comparison separated no checks, plan-only checks, and real early loads: the load arm added +11.04 and +19.42 seconds of summed replay TTFT versus checks-only across 36 replays per seed, while sustained generation did not slow. In a one-seed phase-timing repeat, three completed early loads spent 121, 308, and 1362 ms inside SGLang's ready-to-load call on the scheduler path; control-queue waits were only 2-12 ms. A separate diagnostic profile of one 2048-token direct load found 117.44 MB moved in 3920 small copies: 10.884 ms of active host-to-device copies spread across 156.859 ms. A one-seed control using SGLang's supported kernel I/O backend also held the scheduler for 106, 314, and 1314 ms on its three accepted loads and added 17.61 seconds summed replay TTFT versus checks-only. Neither I/O backend moved the load off the scheduler path; post-first-token generation did not slow in these runs.

**Not yet proved.** The profiler diagnostic perturbs timing and is not a performance comparison; gaps between copy events cannot be assumed recoverable. The phase-timing and kernel controls each have only one seed and three accepted early loads. CUDA-event duration overlaps wall-clock ready-call time and must not be added to it. Matched-arm differences locate affected stages but do not prove per-copy causality, HBM contention, or how much a hardware offload would save; concurrent batch trajectories can diverge. An opt-in off-scheduler prototype is evaluated separately in RQ9. A separate sequential tracing-off/on microcheck measured +3.05% median latency but does not bound overhead in the busy workload. Prompts and tool waits remain synthetic.

<a id="rq-RQ7"></a>
### RQ7: Can a controller choose the quiet load window?

**Question.** Using observed short-replay completion, host residency, slot release, and the long tool-return estimate, can a controller policy decide when to load KV without frontend importance ranks?

**What the evidence says.** Yes in this controlled A10G run. All four warmed, order-balanced four-mode trials passed the evidence gates. The controller chose load four times, each native load completed before long tool return, and no overshoot was recorded. Median long return-to-first-token was 85.6 ms with the controller versus 351.7 ms with late loading; median short return-to-finish was 859.6 ms versus 1105.3 ms with early loading. The controller's outcome was close to scripted post-short loading.

**Not yet proved.** This policy was invoked by the experiment harness, not yet by the production gateway. The load-time estimate was a fixed 250 ms plus a 150 ms margin; this run did not test the defer path under real load or naturally arising cache pressure. Four synthetic trials do not establish a universal win or identify a hardware bottleneck.

<a id="rq-RQ6"></a>
### RQ6: Can we load after the short replay finishes?

**Question.** Under the same equal-importance, logical two-prefix workload, can loading just after the short replay completes preserve the long replay benefit while avoiding the short-session penalty?

**What the evidence says.** In two warmed, order-balanced A10G triplets, post-short loading saved 188.2 and 525.5 ms in long replay tool-return-to-first-token time versus nonblocking late loading. The short session finished 203.7 and 208.6 ms sooner than with early loading; its completion was close to the late control. Whole workflows finished 208.1 and 510.1 ms sooner than late. All three-mode evidence gates passed.

**Not yet proved.** The post-short trigger used the observed client completion event, not a prediction available to a production controller. The two-prefix budget was explicit, not natural capacity pressure. Two synthetic triplets do not establish a general policy win or identify GPU bandwidth versus backend scheduling as the cause.

<a id="rq-RQ5"></a>
### RQ5: Does early loading help the whole shared system?

**Question.** With equal-importance sessions and the same logical two-prefix budget, does loading during the wait help the returning session without delaying another session or the whole workflow?

**What the evidence says.** In two warmed, order-balanced A10G pairs, early loading moved the long replay first token forward by 201 and 513 ms and shortened workflow completion by 212 and 503 ms. The short session finished 178 and 195 ms later. This is a measured tradeoff, not a no-cost win.

**Not yet proved.** The short-session delay is not attributed to GPU bandwidth versus backend scheduling. The two-prefix cap was an explicit logical policy, not measured physical occupancy or natural capacity pressure. Two synthetic pairs do not establish a production-wide benefit.

<a id="rq-RQ4"></a>
### RQ4: What happens when equal-importance sessions overlap?

**Question.** Can a small, equal-importance, concurrent workload link tool waits, one controlled host eviction/load, replays, and an ending session?

**What the evidence says.** Yes. The three-session A10G trace passed the identity and ordering gates. The 900 ms and 2500 ms tool waits overlapped; the long session was explicitly evicted to host, loaded, and replayed; the third session ended without replay.

**Not yet proved.** The eviction enforced a synthetic two-prefix policy, not natural capacity pressure. Different-session TTFTs do not prove a causal eviction penalty; RQ5 supplies the separate matched timing comparison.

<a id="rq-RQ3"></a>
### RQ3: Does nonblocking late loading remove the delay?

**Question.** Does not waiting for the late-load control response remove the observed tool-return-to-first-token penalty?

**What the evidence says.** In one warmed triple, nonblocking submission removed the roughly 170 ms client submission gap, but first token still arrived 243.659 ms after tool return versus 88.159 ms with early loading. The delay appeared in replay TTFT instead.

**Not yet proved.** The native load was accepted after nonblocking replay submission, so the strict comparable delta was withheld. More order-balanced runs are needed; the full-task comparison was also withheld because second-replay TTFT drifted.

<a id="rq-RQ2"></a>
### RQ2: Does loading during the tool wait help?

**Question.** In a controlled replay, does requesting host-KV load during a tool wait shorten time from tool return to first token compared with requesting it after the wait?

**What the evidence says.** In these sampled runs, yes: late preparation added 166-178 ms across four comparable pairs. Most of that difference occurred before replay submission.

**Not yet proved.** The sequential runs did not test competing-session cost; RQ5 now measures that separately under explicit logical capacity. Neither result establishes a production or hardware benefit.

<a id="rq-RQ1"></a>
### RQ1: Can we trace one session's KV lifecycle reliably?

**Question.** Can host residency, GPU eviction and load-back, and replay be linked to the same session across one or two tool waits?

**What the evidence says.** Yes for these controlled A10G validations: the host-backed runs linked eviction, host residency, native load, layer copies, and replay to a session. Two-replay probes also checked a later replay's cache match.

**Not yet proved.** A matched prefix does not prove exact model-kernel consumption. These sequential validations do not show a policy speedup, natural capacity pressure, or avoidable work.

## Experiment index

Newest first. Each arrow goes from the named control to the changed case in the **Compared** column; lower times are better. Three-session rows show the long replay's first token and the short session's finish after tool return. Busy rows show summed replay TTFT across all replays. Workflow is total elapsed time. Rows without a validated comparison have no arrow. For multi-trial runs, arrows compare each mode's median time, which can differ from the median trial-by-trial improvement. Select an experiment for exact trial values and limits.

| Central date / time | Experiment | Question | Setup | Compared | Replay / long session | Other session | Whole workflow | Plain-English finding | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oct&nbsp;8,&nbsp;2026,&nbsp;9:04:52&nbsp;p.m.&nbsp;CDT | <kbd>Research pivot</kbd><br>[GPU / CPU / storage KV gap](#run-capacity_safe_tiers_7b_full_20261008) | RQ24 | All-GPU&nbsp;vs&nbsp;CPU-tiered&nbsp;vs&nbsp;storage-tiered&nbsp;KV | All GPU → CPU tier → storage tier (burst returns) | 875&nbsp;ms&nbsp;→&nbsp;1129&nbsp;ms&nbsp;→&nbsp;2883&nbsp;ms | All&nbsp;equal-priority&nbsp;sessions&nbsp;included | 30.95&nbsp;s&nbsp;→&nbsp;35.95&nbsp;s&nbsp;→&nbsp;64.85&nbsp;s | spread&nbsp;CPU&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+9.7%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+239.4&nbsp;ms&nbsp;vs&nbsp;all-GPU.&nbsp;spread&nbsp;storage&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+93.3%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+1933.0&nbsp;ms&nbsp;vs&nbsp;all-GPU.&nbsp;burst&nbsp;CPU&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+16.0%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+249.7&nbsp;ms&nbsp;vs&nbsp;all-GPU.&nbsp;burst&nbsp;storage&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+109.5%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+2000.9&nbsp;ms&nbsp;vs&nbsp;all-GPU. | complete; 720 proven lower-tier replay hits |
| Oct&nbsp;8,&nbsp;2026,&nbsp;5:39:28&nbsp;p.m.&nbsp;CDT | <kbd>Research pivot</kbd><br>[GPU / CPU / storage KV gap](#run-memory_tiers_7b_full_20261008) | RQ23 | All-GPU&nbsp;vs&nbsp;CPU-tiered&nbsp;vs&nbsp;storage-tiered&nbsp;KV | All GPU → CPU tier → storage tier (burst returns) | 260&nbsp;ms&nbsp;→&nbsp;2267&nbsp;ms&nbsp;→&nbsp;3956&nbsp;ms | All&nbsp;equal-priority&nbsp;sessions&nbsp;included | 28.07&nbsp;s&nbsp;→&nbsp;67.70&nbsp;s&nbsp;→&nbsp;102.49&nbsp;s | spread&nbsp;CPU&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+94.3%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+1669.0&nbsp;ms&nbsp;vs&nbsp;all-GPU.&nbsp;spread&nbsp;storage&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+201.6%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+3805.9&nbsp;ms&nbsp;vs&nbsp;all-GPU.&nbsp;burst&nbsp;CPU&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+141.1%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+2006.5&nbsp;ms&nbsp;vs&nbsp;all-GPU.&nbsp;burst&nbsp;storage&nbsp;tier:&nbsp;whole&nbsp;workload&nbsp;+264.5%&nbsp;and&nbsp;mean&nbsp;replay&nbsp;delay&nbsp;+3696.8&nbsp;ms&nbsp;vs&nbsp;all-GPU. | complete; 549 proven lower-tier replay hits |
| Oct&nbsp;8,&nbsp;2026,&nbsp;4:51:13&nbsp;p.m.&nbsp;CDT | [Session-level coordinated KV (calibration)](#run-coordinated_session_pipeline_pilot_v2_20261008) | RQ22 | 20&nbsp;sessions;&nbsp;session-level&nbsp;advance&nbsp;KV&nbsp;loading | Independent / coordinated / resident (per-mode trial medians) | independent:&nbsp;1073.3ms;&nbsp;coordinated:&nbsp;1634.0ms;&nbsp;resident:&nbsp;792.1ms | All&nbsp;20&nbsp;sessions&nbsp;included | independent:&nbsp;7.2s;&nbsp;coordinated:&nbsp;9.8s;&nbsp;resident:&nbsp;6.2s | Coordinated&nbsp;workload&nbsp;was&nbsp;36.2%&nbsp;longer&nbsp;than&nbsp;independent&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;Coordinated&nbsp;workload&nbsp;was&nbsp;58.9%&nbsp;longer&nbsp;than&nbsp;resident&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;36&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;4:42:28&nbsp;p.m.&nbsp;CDT | [Session-level coordinated KV (calibration)](#run-coordinated_session_pipeline_pilot_20261008) | RQ22 | 20&nbsp;sessions;&nbsp;session-level&nbsp;advance&nbsp;KV&nbsp;loading | Independent / coordinated / resident (per-mode trial medians) | independent:&nbsp;992.6ms;&nbsp;coordinated:&nbsp;2352.7ms;&nbsp;resident:&nbsp;791.0ms | All&nbsp;20&nbsp;sessions&nbsp;included | independent:&nbsp;7.1s;&nbsp;coordinated:&nbsp;12.2s;&nbsp;resident:&nbsp;6.2s | Coordinated&nbsp;workload&nbsp;was&nbsp;71.5%&nbsp;longer&nbsp;than&nbsp;independent&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;Coordinated&nbsp;workload&nbsp;was&nbsp;97.2%&nbsp;longer&nbsp;than&nbsp;resident&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;46&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;2:28:45&nbsp;p.m.&nbsp;CDT | [Coordinated CPU/GPU swaps](#run-coordinated_swap_count_full_20261008) | RQ21 | 20&nbsp;sessions;&nbsp;ideal&nbsp;paired&nbsp;CPU/GPU&nbsp;swaps | Independent / coordinated / resident (per-mode trial medians) | independent:&nbsp;748.3ms;&nbsp;coordinated:&nbsp;826.9ms;&nbsp;resident:&nbsp;339.6ms | All&nbsp;20&nbsp;sessions&nbsp;included | independent:&nbsp;81.0s;&nbsp;coordinated:&nbsp;100.6s;&nbsp;resident:&nbsp;80.7s | Coordinated&nbsp;workload&nbsp;was&nbsp;24.3%&nbsp;longer&nbsp;than&nbsp;independent&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;Coordinated&nbsp;workload&nbsp;was&nbsp;24.6%&nbsp;longer&nbsp;than&nbsp;resident&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;2370&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;2:20:38&nbsp;p.m.&nbsp;CDT | [Coordinated CPU/GPU swaps (calibration)](#run-coordinated_swap_count_pilot_20261008) | RQ21 | 20&nbsp;sessions;&nbsp;ideal&nbsp;paired&nbsp;CPU/GPU&nbsp;swaps | Independent / coordinated / resident (per-mode trial medians) | independent:&nbsp;1143.2ms;&nbsp;coordinated:&nbsp;1664.6ms;&nbsp;resident:&nbsp;732.5ms | All&nbsp;20&nbsp;sessions&nbsp;included | independent:&nbsp;7.5s;&nbsp;coordinated:&nbsp;9.6s;&nbsp;resident:&nbsp;6.6s | Coordinated&nbsp;workload&nbsp;was&nbsp;29.0%&nbsp;longer&nbsp;than&nbsp;independent&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;Coordinated&nbsp;workload&nbsp;was&nbsp;45.6%&nbsp;longer&nbsp;than&nbsp;resident&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;50&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;2:13:43&nbsp;p.m.&nbsp;CDT | [Coordinated CPU/GPU swaps (calibration)](#run-coordinated_swap_batched_pilot_20261008) | RQ21 | 20&nbsp;sessions;&nbsp;ideal&nbsp;paired&nbsp;CPU/GPU&nbsp;swaps | Independent / coordinated / resident (per-mode trial medians) | coordinated:&nbsp;6828.2ms | All&nbsp;20&nbsp;sessions&nbsp;included | coordinated:&nbsp;27.1s | 50&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;2:04:28&nbsp;p.m.&nbsp;CDT | [Coordinated CPU/GPU swaps (calibration)](#run-coordinated_swap_kernel_pilot_20261008) | RQ21 | 20&nbsp;sessions;&nbsp;ideal&nbsp;paired&nbsp;CPU/GPU&nbsp;swaps | Independent / coordinated / resident (per-mode trial medians) | independent:&nbsp;5643.7ms;&nbsp;coordinated:&nbsp;7012.3ms;&nbsp;resident:&nbsp;916.4ms | All&nbsp;20&nbsp;sessions&nbsp;included | independent:&nbsp;24.1s;&nbsp;coordinated:&nbsp;27.6s;&nbsp;resident:&nbsp;7.1s | Coordinated&nbsp;workload&nbsp;was&nbsp;14.3%&nbsp;longer&nbsp;than&nbsp;independent&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;Coordinated&nbsp;workload&nbsp;was&nbsp;287.1%&nbsp;longer&nbsp;than&nbsp;resident&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;50&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;1:55:10&nbsp;p.m.&nbsp;CDT | [Coordinated CPU/GPU swaps (calibration)](#run-coordinated_swap_pilot_v2_20261008) | RQ21 | 20&nbsp;sessions;&nbsp;ideal&nbsp;paired&nbsp;CPU/GPU&nbsp;swaps | Independent / coordinated / resident (per-mode trial medians) | independent:&nbsp;4781.6ms;&nbsp;coordinated:&nbsp;7393.6ms;&nbsp;resident:&nbsp;1101.2ms | All&nbsp;20&nbsp;sessions&nbsp;included | independent:&nbsp;22.5s;&nbsp;coordinated:&nbsp;29.4s;&nbsp;resident:&nbsp;8.6s | Coordinated&nbsp;workload&nbsp;was&nbsp;30.4%&nbsp;longer&nbsp;than&nbsp;independent&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;Coordinated&nbsp;workload&nbsp;was&nbsp;243.3%&nbsp;longer&nbsp;than&nbsp;resident&nbsp;(median&nbsp;paired&nbsp;change).&nbsp;50&nbsp;coordinated&nbsp;replays&nbsp;had&nbsp;KV&nbsp;become&nbsp;ready&nbsp;after&nbsp;their&nbsp;tool&nbsp;deadline. | complete |
| Oct&nbsp;8,&nbsp;2026,&nbsp;1:51:36&nbsp;p.m.&nbsp;CDT | [Coordinated CPU/GPU swaps (calibration)](#run-coordinated_swap_pilot_20261008) | RQ21 | 20&nbsp;sessions;&nbsp;ideal&nbsp;paired&nbsp;CPU/GPU&nbsp;swaps | Independent / coordinated / resident (per-mode trial medians) | unavailable | All&nbsp;20&nbsp;sessions&nbsp;included | unavailable | Evidence&nbsp;incomplete;&nbsp;no&nbsp;performance&nbsp;conclusion.&nbsp;trial1_coordinated:&nbsp;RuntimeError:&nbsp;Restored-prefix&nbsp;diagnostic&nbsp;output&nbsp;mismatch | blocked |
| Oct&nbsp;7,&nbsp;2026,&nbsp;11:00:30&nbsp;a.m.&nbsp;CDT | [Selective storage staging across repeated sessions](#run-rq20_selective_storage_pair_20261007) | RQ20 | 8&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;6&nbsp;tool&nbsp;returns&nbsp;each&nbsp;·&nbsp;natural&nbsp;file-cache&nbsp;pressure | On demand → selective staging | Replay&nbsp;delay:&nbsp;seed&nbsp;1&nbsp;1096&nbsp;→&nbsp;1158&nbsp;ms;&nbsp;seed&nbsp;2&nbsp;847&nbsp;→&nbsp;936&nbsp;ms | Session&nbsp;finish&nbsp;changes&nbsp;range&nbsp;from&nbsp;-818&nbsp;to&nbsp;+1750&nbsp;ms | Whole&nbsp;workload:&nbsp;seed&nbsp;1&nbsp;28.55&nbsp;→&nbsp;27.84&nbsp;s;&nbsp;seed&nbsp;2&nbsp;29.98&nbsp;→&nbsp;30.57&nbsp;s | Mixed&nbsp;result&nbsp;across&nbsp;2&nbsp;paired&nbsp;seeds:&nbsp;seed&nbsp;1&nbsp;whole&nbsp;workload&nbsp;-703&nbsp;ms,&nbsp;median&nbsp;replay&nbsp;delay&nbsp;+62&nbsp;ms;&nbsp;seed&nbsp;2&nbsp;whole&nbsp;workload&nbsp;+596&nbsp;ms,&nbsp;median&nbsp;replay&nbsp;delay&nbsp;+89&nbsp;ms.&nbsp;3&nbsp;stages&nbsp;finished&nbsp;before&nbsp;due&nbsp;time;&nbsp;there&nbsp;is&nbsp;no&nbsp;consistent&nbsp;whole-workload&nbsp;gain.&nbsp;Synthetic&nbsp;file-backed&nbsp;storage&nbsp;does&nbsp;not&nbsp;establish&nbsp;a&nbsp;physical-SSD&nbsp;or&nbsp;hardware&nbsp;benefit. | native L3 hits verified |
| Oct&nbsp;7,&nbsp;2026,&nbsp;10:57:02&nbsp;a.m.&nbsp;CDT | [Selective storage staging across repeated sessions](#run-rq20_exposure_h1_g10240_w900_s6_20261007) | RQ20 | 6&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;6&nbsp;tool&nbsp;returns&nbsp;each&nbsp;·&nbsp;natural&nbsp;file-cache&nbsp;pressure | Calibration only | No&nbsp;paired&nbsp;comparison | No&nbsp;peer&nbsp;comparison | Not&nbsp;established | Calibration&nbsp;run&nbsp;only;&nbsp;no&nbsp;paired&nbsp;mode&nbsp;comparison. | pilot |
| Oct&nbsp;7,&nbsp;2026,&nbsp;9:45:46&nbsp;a.m.&nbsp;CDT | [Storage staging and peer-delay control](#run-rq19_control_placebo_n4_s2_20261007) | RQ19 | 1&nbsp;returning&nbsp;+&nbsp;3&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | No early action → control checks → KV staging | First&nbsp;token:&nbsp;353&nbsp;→&nbsp;352&nbsp;→&nbsp;163&nbsp;ms | Peer&nbsp;finish:&nbsp;2588&nbsp;→&nbsp;2607&nbsp;→&nbsp;2667&nbsp;ms | Whole&nbsp;workload:&nbsp;6.62&nbsp;→&nbsp;6.56&nbsp;→&nbsp;6.41&nbsp;s | Early&nbsp;staging&nbsp;helped&nbsp;the&nbsp;returning&nbsp;session&nbsp;but&nbsp;delayed&nbsp;peers;&nbsp;control&nbsp;checks&nbsp;alone&nbsp;explain&nbsp;only&nbsp;part&nbsp;of&nbsp;the&nbsp;peer&nbsp;delay. | Native L3 hits and replay reuse verified |
| Oct&nbsp;7,&nbsp;2026,&nbsp;9:38:46&nbsp;a.m.&nbsp;CDT | [Storage staging and peer-delay control](#run-rq19_control_placebo_n4_s1_20261007) | RQ19 | 1&nbsp;returning&nbsp;+&nbsp;3&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | No early action → control checks → KV staging | First&nbsp;token:&nbsp;380&nbsp;→&nbsp;319&nbsp;→&nbsp;162&nbsp;ms | Peer&nbsp;finish:&nbsp;2585&nbsp;→&nbsp;2618&nbsp;→&nbsp;2647&nbsp;ms | Whole&nbsp;workload:&nbsp;6.59&nbsp;→&nbsp;6.53&nbsp;→&nbsp;6.37&nbsp;s | Early&nbsp;staging&nbsp;helped&nbsp;the&nbsp;returning&nbsp;session&nbsp;but&nbsp;delayed&nbsp;peers;&nbsp;control&nbsp;checks&nbsp;alone&nbsp;explain&nbsp;only&nbsp;part&nbsp;of&nbsp;the&nbsp;peer&nbsp;delay. | Native L3 hits and replay reuse verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:50:41&nbsp;p.m.&nbsp;CDT | [Repeated storage-resume timing](#run-rq18_guarded_cycles_retry_20261007) | RQ17 | 8&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;6&nbsp;tool&nbsp;returns&nbsp;each&nbsp;·&nbsp;natural&nbsp;file-cache&nbsp;pressure | On demand → host staging | Replay&nbsp;delay:&nbsp;1046&nbsp;→&nbsp;1038&nbsp;ms | Synthetic&nbsp;file-backed&nbsp;storage;&nbsp;full-GPU&nbsp;preparation&nbsp;not&nbsp;tested | Whole&nbsp;workload:&nbsp;27.34&nbsp;→&nbsp;27.47&nbsp;s | Across&nbsp;1&nbsp;paired&nbsp;seeds,&nbsp;host&nbsp;staging&nbsp;changed&nbsp;median&nbsp;replay&nbsp;delay&nbsp;by&nbsp;-8&nbsp;ms&nbsp;and&nbsp;whole-workload&nbsp;duration&nbsp;by&nbsp;+131&nbsp;ms&nbsp;(negative&nbsp;is&nbsp;faster).&nbsp;On-demand&nbsp;replay&nbsp;had&nbsp;15&nbsp;native&nbsp;storage&nbsp;hits;&nbsp;staging&nbsp;completed&nbsp;before&nbsp;due&nbsp;time&nbsp;on&nbsp;1&nbsp;waits.&nbsp;These&nbsp;are&nbsp;associations&nbsp;in&nbsp;a&nbsp;small&nbsp;synthetic,&nbsp;file-backed&nbsp;workload,&nbsp;not&nbsp;an&nbsp;isolated&nbsp;physical-SSD&nbsp;or&nbsp;hardware&nbsp;speedup. | native L3 hits verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:43:07&nbsp;p.m.&nbsp;CDT | [Repeated storage-resume timing](#run-rq18_guarded_cycles_20261007) | RQ17 | 8&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;6&nbsp;tool&nbsp;returns&nbsp;each&nbsp;·&nbsp;natural&nbsp;file-cache&nbsp;pressure | Blocked: on demand → host staging | Replay&nbsp;delay:&nbsp;1084&nbsp;→&nbsp;1073&nbsp;ms | Synthetic&nbsp;file-backed&nbsp;storage;&nbsp;full-GPU&nbsp;preparation&nbsp;not&nbsp;tested | Blocked&nbsp;by&nbsp;preparation&nbsp;error&nbsp;or&nbsp;incomplete&nbsp;arm | Blocked&nbsp;after&nbsp;one&nbsp;completed&nbsp;pair:&nbsp;host&nbsp;staging&nbsp;changed&nbsp;median&nbsp;replay&nbsp;delay&nbsp;by&nbsp;-11.2&nbsp;ms&nbsp;and&nbsp;whole-workload&nbsp;time&nbsp;by&nbsp;+394.2&nbsp;ms.&nbsp;Native&nbsp;prefetch&nbsp;was&nbsp;declined&nbsp;once;&nbsp;an&nbsp;explicit&nbsp;rate-limit&nbsp;check&nbsp;was&nbsp;added&nbsp;afterward.&nbsp;This&nbsp;one-pair&nbsp;observation&nbsp;is&nbsp;not&nbsp;a&nbsp;validated&nbsp;performance&nbsp;conclusion. | blocked |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:40:39&nbsp;p.m.&nbsp;CDT | [Safe storage-stage session ladder](#run-rq18_safe_ladder_n8_20261007) | RQ18 | 1&nbsp;returning&nbsp;+&nbsp;7&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage | First&nbsp;token:&nbsp;325&nbsp;→&nbsp;168&nbsp;ms | Peer&nbsp;TTFT:&nbsp;1326&nbsp;→&nbsp;1438&nbsp;ms | Whole&nbsp;workload:&nbsp;6.55&nbsp;→&nbsp;6.43&nbsp;s | Native&nbsp;storage-only&nbsp;KV&nbsp;and&nbsp;replay&nbsp;reuse&nbsp;were&nbsp;verified.&nbsp;First-token&nbsp;delay&nbsp;changed&nbsp;from&nbsp;325&nbsp;to&nbsp;168&nbsp;ms&nbsp;with&nbsp;host&nbsp;staging&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s).&nbsp;Peer&nbsp;timing&nbsp;and&nbsp;whole-workload&nbsp;changes&nbsp;are&nbsp;recorded&nbsp;separately;&nbsp;a&nbsp;single&nbsp;pair&nbsp;is&nbsp;not&nbsp;enough&nbsp;to&nbsp;claim&nbsp;a&nbsp;win-win. | L3 hit and replay reuse verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:35:33&nbsp;p.m.&nbsp;CDT | [Safe storage-stage session ladder](#run-rq18_safe_ladder_n4_20261007) | RQ18 | 1&nbsp;returning&nbsp;+&nbsp;3&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage | First&nbsp;token:&nbsp;376&nbsp;→&nbsp;170&nbsp;ms | Peer&nbsp;TTFT:&nbsp;1168&nbsp;→&nbsp;1223&nbsp;ms | Whole&nbsp;workload:&nbsp;6.59&nbsp;→&nbsp;6.41&nbsp;s | Native&nbsp;storage-only&nbsp;KV&nbsp;and&nbsp;replay&nbsp;reuse&nbsp;were&nbsp;verified.&nbsp;First-token&nbsp;delay&nbsp;changed&nbsp;from&nbsp;376&nbsp;to&nbsp;170&nbsp;ms&nbsp;with&nbsp;host&nbsp;staging&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s).&nbsp;Peer&nbsp;timing&nbsp;and&nbsp;whole-workload&nbsp;changes&nbsp;are&nbsp;recorded&nbsp;separately;&nbsp;a&nbsp;single&nbsp;pair&nbsp;is&nbsp;not&nbsp;enough&nbsp;to&nbsp;claim&nbsp;a&nbsp;win-win. | L3 hit and replay reuse verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:29:59&nbsp;p.m.&nbsp;CDT | [Safe storage-stage session ladder](#run-rq18_safe_ladder_n2_s2_20261007) | RQ18 | 1&nbsp;returning&nbsp;+&nbsp;1&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage | First&nbsp;token:&nbsp;348&nbsp;→&nbsp;162&nbsp;ms | Peer&nbsp;TTFT:&nbsp;98&nbsp;→&nbsp;137&nbsp;ms | Whole&nbsp;workload:&nbsp;6.52&nbsp;→&nbsp;6.37&nbsp;s | Native&nbsp;storage-only&nbsp;KV&nbsp;and&nbsp;replay&nbsp;reuse&nbsp;were&nbsp;verified.&nbsp;First-token&nbsp;delay&nbsp;changed&nbsp;from&nbsp;348&nbsp;to&nbsp;162&nbsp;ms&nbsp;with&nbsp;host&nbsp;staging&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s).&nbsp;Peer&nbsp;timing&nbsp;and&nbsp;whole-workload&nbsp;changes&nbsp;are&nbsp;recorded&nbsp;separately;&nbsp;a&nbsp;single&nbsp;pair&nbsp;is&nbsp;not&nbsp;enough&nbsp;to&nbsp;claim&nbsp;a&nbsp;win-win. | L3 hit and replay reuse verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:24:59&nbsp;p.m.&nbsp;CDT | [Safe storage-stage session ladder](#run-rq18_safe_ladder_n2_20261007) | RQ18 | 1&nbsp;returning&nbsp;+&nbsp;1&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage | First&nbsp;token:&nbsp;374&nbsp;→&nbsp;167&nbsp;ms | Peer&nbsp;TTFT:&nbsp;98&nbsp;→&nbsp;145&nbsp;ms | Whole&nbsp;workload:&nbsp;6.57&nbsp;→&nbsp;6.40&nbsp;s | Native&nbsp;storage-only&nbsp;KV&nbsp;and&nbsp;replay&nbsp;reuse&nbsp;were&nbsp;verified.&nbsp;First-token&nbsp;delay&nbsp;changed&nbsp;from&nbsp;374&nbsp;to&nbsp;167&nbsp;ms&nbsp;with&nbsp;host&nbsp;staging&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s).&nbsp;Peer&nbsp;timing&nbsp;and&nbsp;whole-workload&nbsp;changes&nbsp;are&nbsp;recorded&nbsp;separately;&nbsp;a&nbsp;single&nbsp;pair&nbsp;is&nbsp;not&nbsp;enough&nbsp;to&nbsp;claim&nbsp;a&nbsp;win-win. | L3 hit and replay reuse verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;11:19:42&nbsp;p.m.&nbsp;CDT | [Safe storage-stage session ladder](#run-rq18_safe_ladder_n1b_20261007) | RQ18 | 1&nbsp;returning&nbsp;+&nbsp;0&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage | First&nbsp;token:&nbsp;317&nbsp;→&nbsp;167&nbsp;ms | No&nbsp;peer&nbsp;session | Whole&nbsp;workload:&nbsp;6.76&nbsp;→&nbsp;6.63&nbsp;s | Native&nbsp;storage-only&nbsp;KV&nbsp;and&nbsp;replay&nbsp;reuse&nbsp;were&nbsp;verified.&nbsp;First-token&nbsp;delay&nbsp;changed&nbsp;from&nbsp;317&nbsp;to&nbsp;167&nbsp;ms&nbsp;with&nbsp;host&nbsp;staging&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s).&nbsp;No&nbsp;competing&nbsp;session&nbsp;was&nbsp;present;&nbsp;this&nbsp;only&nbsp;validates&nbsp;the&nbsp;load&nbsp;path. | L3 hit and replay reuse verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;9:59:24&nbsp;p.m.&nbsp;CDT | [Repeated storage-resume timing](#run-storage_cycles_rq17_20261006) | RQ17 | 8&nbsp;equal-priority&nbsp;sessions&nbsp;·&nbsp;6&nbsp;tool&nbsp;returns&nbsp;each&nbsp;·&nbsp;natural&nbsp;file-cache&nbsp;pressure | Blocked: on demand → host staging | Replay&nbsp;delay:&nbsp;1706&nbsp;→&nbsp;1702&nbsp;ms | Synthetic&nbsp;file-backed&nbsp;storage;&nbsp;full-GPU&nbsp;preparation&nbsp;not&nbsp;tested | Blocked&nbsp;by&nbsp;backend&nbsp;assertion;&nbsp;incomplete&nbsp;paired&nbsp;study | Blocked&nbsp;after&nbsp;one&nbsp;completed&nbsp;pair:&nbsp;host&nbsp;staging&nbsp;changed&nbsp;median&nbsp;replay&nbsp;delay&nbsp;by&nbsp;-3.9&nbsp;ms&nbsp;and&nbsp;whole-workload&nbsp;time&nbsp;by&nbsp;+1000.1&nbsp;ms.&nbsp;Native&nbsp;prefetch&nbsp;was&nbsp;declined&nbsp;once;&nbsp;an&nbsp;explicit&nbsp;rate-limit&nbsp;check&nbsp;was&nbsp;added&nbsp;afterward.&nbsp;This&nbsp;one-pair&nbsp;observation&nbsp;is&nbsp;not&nbsp;a&nbsp;validated&nbsp;performance&nbsp;conclusion. | blocked |
| Oct&nbsp;6,&nbsp;2026,&nbsp;7:40:44&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_pressure_20261006_peers6) | RQ16 | 1&nbsp;returning&nbsp;+&nbsp;6&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;468&nbsp;→&nbsp;168&nbsp;→&nbsp;59&nbsp;ms | Peer&nbsp;TTFT:&nbsp;1026&nbsp;→&nbsp;1075&nbsp;→&nbsp;1184&nbsp;ms | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;468&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;168&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;59&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;2&nbsp;paired&nbsp;seed(s)).&nbsp;Peers&nbsp;overlapped&nbsp;preparation;&nbsp;their&nbsp;median&nbsp;TTFT&nbsp;changed&nbsp;from&nbsp;1026&nbsp;to&nbsp;1184&nbsp;ms.&nbsp;This&nbsp;is&nbsp;not&nbsp;a&nbsp;proven&nbsp;win-win.&nbsp;Native&nbsp;data&nbsp;readiness&nbsp;took&nbsp;a&nbsp;median&nbsp;80&nbsp;ms;&nbsp;the&nbsp;later&nbsp;status-poll/host-commit&nbsp;interval&nbsp;took&nbsp;585&nbsp;ms. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;7:29:17&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_pressure_20261006_peers2) | RQ16 | 1&nbsp;returning&nbsp;+&nbsp;2&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;364&nbsp;→&nbsp;169&nbsp;→&nbsp;61&nbsp;ms | Peer&nbsp;TTFT:&nbsp;157&nbsp;→&nbsp;393&nbsp;→&nbsp;456&nbsp;ms | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;364&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;169&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;61&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;2&nbsp;paired&nbsp;seed(s)).&nbsp;Peers&nbsp;overlapped&nbsp;preparation;&nbsp;their&nbsp;median&nbsp;TTFT&nbsp;changed&nbsp;from&nbsp;157&nbsp;to&nbsp;456&nbsp;ms.&nbsp;This&nbsp;is&nbsp;not&nbsp;a&nbsp;proven&nbsp;win-win.&nbsp;Native&nbsp;data&nbsp;readiness&nbsp;took&nbsp;a&nbsp;median&nbsp;85&nbsp;ms;&nbsp;the&nbsp;later&nbsp;status-poll/host-commit&nbsp;interval&nbsp;took&nbsp;326&nbsp;ms. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;7:17:50&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_pressure_20261006_peers0) | RQ16 | 1&nbsp;returning&nbsp;+&nbsp;0&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;370&nbsp;→&nbsp;171&nbsp;→&nbsp;60&nbsp;ms | No&nbsp;peer&nbsp;session | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;370&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;171&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;60&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;2&nbsp;paired&nbsp;seed(s)).&nbsp;No&nbsp;peer-session&nbsp;or&nbsp;whole-system&nbsp;benefit&nbsp;is&nbsp;established&nbsp;by&nbsp;this&nbsp;single-session&nbsp;run.&nbsp;Native&nbsp;data&nbsp;readiness&nbsp;took&nbsp;a&nbsp;median&nbsp;187&nbsp;ms;&nbsp;the&nbsp;later&nbsp;status-poll/host-commit&nbsp;interval&nbsp;took&nbsp;105&nbsp;ms. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;7:06:07&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_native_gate_20261006) | RQ16 | 1&nbsp;returning&nbsp;+&nbsp;0&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;369&nbsp;→&nbsp;170&nbsp;→&nbsp;64&nbsp;ms | No&nbsp;peer&nbsp;session | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;369&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;170&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;64&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s)).&nbsp;No&nbsp;peer-session&nbsp;or&nbsp;whole-system&nbsp;benefit&nbsp;is&nbsp;established&nbsp;by&nbsp;this&nbsp;single-session&nbsp;run.&nbsp;Native&nbsp;data&nbsp;readiness&nbsp;took&nbsp;a&nbsp;median&nbsp;262&nbsp;ms;&nbsp;the&nbsp;later&nbsp;status-poll/host-commit&nbsp;interval&nbsp;took&nbsp;62&nbsp;ms. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;4:42:18&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_audit_20261006_213611) | RQ15 | 1&nbsp;returning&nbsp;+&nbsp;2&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;337&nbsp;→&nbsp;166&nbsp;→&nbsp;65&nbsp;ms | Peer&nbsp;TTFT:&nbsp;154&nbsp;→&nbsp;196&nbsp;→&nbsp;203&nbsp;ms | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;337&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;166&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;65&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s)).&nbsp;Peers&nbsp;overlapped&nbsp;preparation;&nbsp;their&nbsp;median&nbsp;TTFT&nbsp;changed&nbsp;from&nbsp;154&nbsp;to&nbsp;203&nbsp;ms.&nbsp;This&nbsp;is&nbsp;not&nbsp;a&nbsp;proven&nbsp;win-win. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;4:35:31&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_audit_20261006_212924) | RQ15 | 1&nbsp;returning&nbsp;+&nbsp;2&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;330&nbsp;→&nbsp;165&nbsp;→&nbsp;59&nbsp;ms | Peer&nbsp;TTFT:&nbsp;160&nbsp;→&nbsp;159&nbsp;→&nbsp;155&nbsp;ms | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;330&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;165&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;59&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s)).&nbsp;Peers&nbsp;began&nbsp;after&nbsp;preparation&nbsp;finished,&nbsp;so&nbsp;this&nbsp;run&nbsp;does&nbsp;not&nbsp;measure&nbsp;contention&nbsp;during&nbsp;the&nbsp;transfer. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;4:28:44&nbsp;p.m.&nbsp;CDT | [Storage-tier KV timing](#run-storage_audit_20261006_212239) | RQ15 | 1&nbsp;returning&nbsp;+&nbsp;0&nbsp;peer&nbsp;session(s)&nbsp;·&nbsp;2048&nbsp;prompt&nbsp;tokens&nbsp;·&nbsp;5000&nbsp;ms&nbsp;tool&nbsp;wait&nbsp;·&nbsp;file-backed&nbsp;L3 | On demand → host stage → full prepare | First&nbsp;token:&nbsp;359&nbsp;→&nbsp;170&nbsp;→&nbsp;62&nbsp;ms | No&nbsp;peer&nbsp;session | Small&nbsp;synthetic&nbsp;workload | With&nbsp;storage-only&nbsp;KV&nbsp;proven,&nbsp;first&nbsp;token&nbsp;after&nbsp;tool&nbsp;due&nbsp;was&nbsp;359&nbsp;ms&nbsp;on&nbsp;demand,&nbsp;170&nbsp;ms&nbsp;after&nbsp;host&nbsp;staging,&nbsp;and&nbsp;62&nbsp;ms&nbsp;after&nbsp;full&nbsp;preparation&nbsp;(median&nbsp;across&nbsp;1&nbsp;paired&nbsp;seed(s)).&nbsp;No&nbsp;peer-session&nbsp;or&nbsp;whole-system&nbsp;benefit&nbsp;is&nbsp;established&nbsp;by&nbsp;this&nbsp;single-session&nbsp;run. | L3 hit verified |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:11:20&nbsp;p.m.&nbsp;CDT | [Repeated tool returns · trace-off control](#run-rq14_traceoff_d0_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | Trace-off control; 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;79.7&nbsp;ms | Backend&nbsp;stages:&nbsp;not&nbsp;captured | Active&nbsp;workflow:&nbsp;22.0&nbsp;s | Trace-off&nbsp;control:&nbsp;first-token&nbsp;delay&nbsp;79.7&nbsp;ms&nbsp;median,&nbsp;136.6&nbsp;ms&nbsp;at&nbsp;p95&nbsp;across&nbsp;24&nbsp;active&nbsp;replays.&nbsp;Backend&nbsp;stage&nbsp;and&nbsp;KV-load&nbsp;evidence&nbsp;was&nbsp;deliberately&nbsp;not&nbsp;captured. | trace_disabled |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:09:27&nbsp;p.m.&nbsp;CDT | [Repeated tool returns · trace-off control](#run-rq14_traceoff_d8_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | Trace-off control; 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;111.9&nbsp;ms | Backend&nbsp;stages:&nbsp;not&nbsp;captured | Active&nbsp;workflow:&nbsp;25.6&nbsp;s | Trace-off&nbsp;control:&nbsp;first-token&nbsp;delay&nbsp;111.9&nbsp;ms&nbsp;median,&nbsp;464.7&nbsp;ms&nbsp;at&nbsp;p95&nbsp;across&nbsp;24&nbsp;active&nbsp;replays.&nbsp;Backend&nbsp;stage&nbsp;and&nbsp;KV-load&nbsp;evidence&nbsp;was&nbsp;deliberately&nbsp;not&nbsp;captured. | trace_disabled |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:07:27&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g0_o0_s1_recheck_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;92.8&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;22.6&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;92.833&nbsp;ms&nbsp;median&nbsp;and&nbsp;146.335&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:04:59&nbsp;p.m.&nbsp;CDT | [Repeated tool returns · trace-off control](#run-rq14_traceoff_d8_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | Trace-off control; 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;103.7&nbsp;ms | Backend&nbsp;stages:&nbsp;not&nbsp;captured | Active&nbsp;workflow:&nbsp;26.0&nbsp;s | Trace-off&nbsp;control:&nbsp;first-token&nbsp;delay&nbsp;103.7&nbsp;ms&nbsp;median,&nbsp;503.7&nbsp;ms&nbsp;at&nbsp;p95&nbsp;across&nbsp;24&nbsp;active&nbsp;replays.&nbsp;Backend&nbsp;stage&nbsp;and&nbsp;KV-load&nbsp;evidence&nbsp;was&nbsp;deliberately&nbsp;not&nbsp;captured. | trace_disabled |
| Oct&nbsp;6,&nbsp;2026,&nbsp;2:03:08&nbsp;p.m.&nbsp;CDT | [Repeated tool returns · trace-off control](#run-rq14_traceoff_d0_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | Trace-off control; 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;80.9&nbsp;ms | Backend&nbsp;stages:&nbsp;not&nbsp;captured | Active&nbsp;workflow:&nbsp;22.2&nbsp;s | Trace-off&nbsp;control:&nbsp;first-token&nbsp;delay&nbsp;80.9&nbsp;ms&nbsp;median,&nbsp;119.3&nbsp;ms&nbsp;at&nbsp;p95&nbsp;across&nbsp;24&nbsp;active&nbsp;replays.&nbsp;Backend&nbsp;stage&nbsp;and&nbsp;KV-load&nbsp;evidence&nbsp;was&nbsp;deliberately&nbsp;not&nbsp;captured. | trace_disabled |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:54:42&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g1_o1_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;163.3&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;2.8&nbsp;ms | Active&nbsp;workflow:&nbsp;26.7&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;163.299&nbsp;ms&nbsp;median&nbsp;and&nbsp;717.435&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:52:25&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g1_o0_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;132.0&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;28.6&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;131.976&nbsp;ms&nbsp;median&nbsp;and&nbsp;788.576&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;3&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:50:14&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g0_o1_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;150.0&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;2.0&nbsp;ms | Active&nbsp;workflow:&nbsp;26.5&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;150.021&nbsp;ms&nbsp;median&nbsp;and&nbsp;470.966&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:47:55&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g0_o0_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;134.5&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;28.4&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;134.517&nbsp;ms&nbsp;median&nbsp;and&nbsp;638.199&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:45:36&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g1_o1_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;100.5&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;22.0&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;100.516&nbsp;ms&nbsp;median&nbsp;and&nbsp;141.645&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:43:29&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g1_o0_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;94.3&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;22.9&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;94.265&nbsp;ms&nbsp;median&nbsp;and&nbsp;401.106&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:41:28&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g0_o1_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;101.0&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;22.2&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;100.951&nbsp;ms&nbsp;median&nbsp;and&nbsp;147.002&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:39:20&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g0_o0_s2_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;92.2&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;22.7&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;92.218&nbsp;ms&nbsp;median&nbsp;and&nbsp;142.942&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:36:27&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g1_o0_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;150.4&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;2.8&nbsp;ms | Active&nbsp;workflow:&nbsp;28.3&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;150.379&nbsp;ms&nbsp;median&nbsp;and&nbsp;608.649&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:34:14&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g0_o1_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;166.0&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;2.8&nbsp;ms | Active&nbsp;workflow:&nbsp;26.5&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;165.976&nbsp;ms&nbsp;median&nbsp;and&nbsp;957.5&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:31:44&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g1_o1_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;176.6&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;2.7&nbsp;ms | Active&nbsp;workflow:&nbsp;26.2&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;176.596&nbsp;ms&nbsp;median&nbsp;and&nbsp;770.334&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:28:04&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_capacity_d8_g0_o0_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;8&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 8 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;139.3&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;27.8&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;139.292&nbsp;ms&nbsp;median&nbsp;and&nbsp;825.248&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;4&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:25:33&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_busy_g1_o1_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;4&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 4 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;126.6&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;23.6&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;126.646&nbsp;ms&nbsp;median&nbsp;and&nbsp;367.543&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:23:10&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_busy_g1_o0_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;4&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 4 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;118.0&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;24.6&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;117.993&nbsp;ms&nbsp;median&nbsp;and&nbsp;341.125&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:20:52&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_busy_g0_o1_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;4&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 4 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;124.8&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;23.8&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;124.843&nbsp;ms&nbsp;median&nbsp;and&nbsp;421.335&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:18:02&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_busy_g0_o0_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;4&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 4 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;109.9&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;24.9&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;109.874&nbsp;ms&nbsp;median&nbsp;and&nbsp;479.657&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:15:27&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g1_o1_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;98.7&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;22.0&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;98.698&nbsp;ms&nbsp;median&nbsp;and&nbsp;146.319&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:13:11&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g1_o0_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;92.4&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.7&nbsp;ms | Active&nbsp;workflow:&nbsp;22.6&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;92.39&nbsp;ms&nbsp;median&nbsp;and&nbsp;402.143&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
| Oct&nbsp;6,&nbsp;2026,&nbsp;1:11:00&nbsp;p.m.&nbsp;CDT | [Repeated tool-return startup](#run-rq14_pilot_g0_o1_s1_20261006) | RQ14 | 2&nbsp;active&nbsp;+&nbsp;0&nbsp;donor&nbsp;sessions&nbsp;·&nbsp;12&nbsp;tool&nbsp;returns&nbsp;each | 12 tool returns per session; 0 donor sessions | Median&nbsp;first&nbsp;token:&nbsp;98.1&nbsp;ms | Lookup&nbsp;→&nbsp;batch:&nbsp;1.8&nbsp;ms | Active&nbsp;workflow:&nbsp;21.8&nbsp;s | Across&nbsp;24&nbsp;active&nbsp;replays,&nbsp;first-token&nbsp;delay&nbsp;was&nbsp;98.126&nbsp;ms&nbsp;median&nbsp;and&nbsp;141.573&nbsp;ms&nbsp;at&nbsp;p95.&nbsp;0&nbsp;active&nbsp;replays&nbsp;had&nbsp;a&nbsp;recorded&nbsp;KV&nbsp;load-back.&nbsp;Stage&nbsp;timing&nbsp;identifies&nbsp;where&nbsp;time&nbsp;was&nbsp;spent,&nbsp;not&nbsp;why&nbsp;the&nbsp;backend&nbsp;waited. | complete_stage_join |
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

<a id="run-capacity_safe_tiers_7b_full_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 9:04:52 p.m. CDT · GPU / CPU / storage KV gap</strong> · capacity_safe_tiers_7b_full_20261008</summary>

**Question (RQ24).** When every currently active session fits within the restricted GPU KV budget, how much performance is lost solely because waiting-session KV is kept on CPU or storage and restored only after its tool call returns? Does a near-simultaneous return burst make that unprepared tiering penalty worse?

**Finding.** spread CPU tier: whole workload +9.7% and mean replay delay +239.4 ms vs all-GPU. spread storage tier: whole workload +93.3% and mean replay delay +1933.0 ms vs all-GPU. burst CPU tier: whole workload +16.0% and mean replay delay +249.7 ms vs all-GPU. burst storage tier: whole workload +109.5% and mean replay delay +2000.9 ms vs all-GPU.

**Setup.** 6 equal-priority sessions; 10 tool returns per session; 4096 initial prompt tokens; 16 output tokens per replay; 1000 ms tool waits. Return patterns: spread, burst; the burst spans 75 ms and the spread control spans 1000 ms. Fresh backend per arm; no KV prefetch; all-GPU cap 40960 tokens; lower-tier GPU cap 12288 tokens; CPU caches 3.0 GiB allocated but unused for measured replays in all-GPU mode, 2.0 GiB for CPU mode and 1.0 GiB for storage mode. CUDA graphs on; overlap scheduling on; frontend priority equal; kv_lifecycle_lean tracing. Model: Qwen/Qwen2.5-Coder-7B-Instruct; hardware: nvidia_a10g_24gb; backend: 0.5.10.post1. At most 2 sessions could run at once, and their estimated combined active KV had to remain below 12288 tokens. Waiting sessions were restored completely before admission.

**Key measurements**

| Seed / return pattern / mode | Whole workload (s) | Due to first token mean / p95 (ms) | TTFT mean / p95 (ms) | Total due delay / TTFT (s) | Submission waiting (ms) | Proven GPU / CPU / storage source requests | Proven GPU / CPU / storage source tokens | Active sessions max / limit | Active KV estimate max / limit | Slot wait / KV preparation mean (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / burst / host | 35.724 | 1123.9 / 2133.2 | 155.6 / 195.1 | 67.433 / 9.338 | 58095.7 | 0 / 60 / 0 | 0 / 258,432 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 854.2 / 82.9 |
| 1 / burst / resident | 30.954 | 874.2 / 1769.9 | 161.3 / 160.1 | 52.450 / 9.681 | 42769.3 | 60 / 0 / 0 | 258,432 / 0 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 692.0 / 0.0 |
| 1 / burst / storage | 64.853 | 2902.4 / 5070.4 | 152.0 / 153.4 | 174.144 / 9.121 | 165023.3 | 0 / 0 / 60 | 0 / 0 / 260,004 | 2 / 2 | 9,342 / 12,288 tokens | 1921.0 / 601.6 |
| 1 / spread / host | 36.536 | 764.9 / 1480.7 | 158.1 / 195.0 | 45.892 / 9.489 | 36403.6 | 0 / 60 / 0 | 0 / 258,432 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 494.6 / 103.7 |
| 1 / spread / resident | 33.384 | 516.1 / 1280.4 | 145.2 / 150.4 | 30.966 / 8.710 | 22256.1 | 60 / 0 / 0 | 258,432 / 0 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 344.0 / 0.0 |
| 1 / spread / storage | 64.259 | 2449.1 / 4040.7 | 151.4 / 153.6 | 146.945 / 9.084 | 137861.2 | 0 / 0 / 60 | 0 / 0 / 260,004 | 2 / 2 | 9,342 / 12,288 tokens | 1500.8 / 595.2 |
| 2 / burst / host | 35.966 | 1155.1 / 2019.4 | 168.2 / 192.1 | 69.304 / 10.094 | 59210.1 | 0 / 60 / 0 | 0 / 258,432 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 867.5 / 88.3 |
| 2 / burst / resident | 30.947 | 874.9 / 1764.1 | 162.1 / 159.1 | 52.493 / 9.728 | 42764.6 | 60 / 0 / 0 | 258,432 / 0 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 691.8 / 0.0 |
| 2 / burst / storage | 64.234 | 2857.4 / 4850.9 | 151.4 / 153.3 | 171.442 / 9.086 | 162355.7 | 0 / 0 / 60 | 0 / 0 / 260,004 | 2 / 2 | 9,342 / 12,288 tokens | 1890.0 / 593.2 |
| 2 / spread / host | 36.650 | 752.2 / 1520.4 | 156.9 / 180.5 | 45.132 / 9.411 | 35721.1 | 0 / 60 / 0 | 0 / 258,432 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 487.4 / 105.2 |
| 2 / spread / resident | 33.347 | 512.9 / 1257.5 | 145.2 / 151.1 | 30.771 / 8.712 | 22059.2 | 60 / 0 / 0 | 258,432 / 0 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 341.5 / 0.0 |
| 2 / spread / storage | 65.565 | 2494.6 / 4037.5 | 151.5 / 153.7 | 149.673 / 9.091 | 140582.6 | 0 / 0 / 60 | 0 / 0 / 260,004 | 2 / 2 | 9,342 / 12,288 tokens | 1522.1 / 613.0 |
| 3 / burst / host | 35.945 | 1128.5 / 2040.4 | 159.9 / 194.0 | 67.711 / 9.594 | 58116.8 | 0 / 60 / 0 | 0 / 258,432 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 853.3 / 82.7 |
| 3 / burst / resident | 30.982 | 881.9 / 1780.2 | 166.4 / 159.9 | 52.917 / 9.985 | 42931.8 | 60 / 0 / 0 | 258,432 / 0 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 693.9 / 0.0 |
| 3 / burst / storage | 64.924 | 2882.9 / 5176.1 | 153.7 / 153.6 | 172.973 / 9.221 | 163751.9 | 0 / 0 / 60 | 0 / 0 / 260,004 | 2 / 2 | 9,342 / 12,288 tokens | 1894.6 / 600.1 |
| 3 / spread / host | 36.622 | 750.0 / 1467.0 | 158.8 / 194.5 | 45.003 / 9.528 | 35474.4 | 0 / 60 / 0 | 0 / 258,432 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 483.3 / 102.8 |
| 3 / spread / resident | 33.386 | 518.5 / 1280.5 | 147.5 / 150.6 | 31.111 / 8.853 | 22258.4 | 60 / 0 / 0 | 258,432 / 0 / 0 | 2 / 2 | 9,342 / 12,288 tokens | 344.2 / 0.0 |
| 3 / spread / storage | 64.543 | 2390.6 / 4074.5 | 153.2 / 153.4 | 143.435 / 9.194 | 134240.5 | 0 / 0 / 60 | 0 / 0 / 260,004 | 2 / 2 | 9,342 / 12,288 tokens | 1454.3 / 597.1 |

| Seed / return pattern / tier | Whole-workload change vs GPU | Mean / p95 replay-delay increase (ms) | Mean TTFT increase (ms) |
| --- | --- | --- | --- |
| 1 / spread / host | +3.153s (+9.4%) | +248.8 / +200.3 | +13.0 |
| 1 / spread / storage | +30.876s (+92.5%) | +1933.0 / +2760.3 | +6.2 |
| 1 / burst / host | +4.770s (+15.4%) | +249.7 / +363.3 | -5.7 |
| 1 / burst / storage | +33.899s (+109.5%) | +2028.2 / +3300.5 | -9.3 |
| 2 / spread / host | +3.304s (+9.9%) | +239.4 / +262.9 | +11.7 |
| 2 / spread / storage | +32.218s (+96.6%) | +1981.7 / +2780.0 | +6.3 |
| 2 / burst / host | +5.020s (+16.2%) | +280.2 / +255.2 | +6.1 |
| 2 / burst / storage | +33.288s (+107.6%) | +1982.5 / +3086.8 | -10.7 |
| 3 / spread / host | +3.235s (+9.7%) | +231.5 / +186.5 | +11.3 |
| 3 / spread / storage | +31.157s (+93.3%) | +1872.1 / +2794.0 | +5.7 |
| 3 / burst / host | +4.963s (+16.0%) | +246.6 / +260.2 | -6.5 |
| 3 / burst / storage | +33.942s (+109.6%) | +2000.9 / +3395.9 | -12.7 |

Positive changes mean the lower tier was slower than the all-GPU reference. Due-to-first-token includes any delay before submission plus backend TTFT. Tier-source counts come from native SGLang evidence, not from the requested mode name. Capacity-safe arms use pre-admission residency; original pressure arms use replay cache matches. File-backed storage uses normal operating-system caching; the page cache was not flushed.

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Synthetic equal-priority coding sessions with at most two active at once.
- Lower-tier restoration starts only after tool return; this is an unprepared worst case.
- File-backed storage can be served by the operating-system page cache.

**Reproduce** (set the container image and model cache for the target host):

```bash
CAPACITY_TIER_RUN_ID=capacity_safe_tiers_7b_full_20261008_repeat \
CAPACITY_TIER_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
CAPACITY_TIER_SEEDS='1 2 3' \
CAPACITY_TIER_PATTERNS='spread burst' \
CAPACITY_TIER_MODES='resident host storage' \
CAPACITY_TIER_SESSIONS=6 \
CAPACITY_TIER_TURNS=10 \
CAPACITY_TIER_INITIAL_TOKENS=4096 \
CAPACITY_TIER_TOOL_WORDS=16 \
CAPACITY_TIER_DECODE_TOKENS=16 \
CAPACITY_TIER_WAIT_MS=1000 \
CAPACITY_TIER_BURST_WINDOW_MS=75 \
CAPACITY_TIER_SPREAD_WINDOW_MS=1000 \
CAPACITY_TIER_RESIDENT_GPU_TOKENS=40960 \
CAPACITY_TIER_RESIDENT_HOST_GB=3.0 \
CAPACITY_TIER_RESTRICTED_GPU_TOKENS=12288 \
CAPACITY_TIER_HOST_GB=2.0 \
CAPACITY_TIER_STORAGE_HOST_GB=1.0 \
CAPACITY_TIER_MAX_ACTIVE=2 \
CAPACITY_TIER_ACTIVE_TOKEN_LIMIT=12288 \
bash infra/container/run_work_audit_capacity_safe_tiers.sh
```

**Evidence:** [Summary](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/summary.json) · [Run manifest](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/run_manifest.json) · [Source hashes](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/source_sha256.txt) · [seed1/burst_host timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_host/case_results.json) · [seed1/burst_host trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_host/backend_trace.jsonl.gz) · [seed1/burst_host hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_host/instrumentation_audit.json) · [seed1/burst_resident timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_resident/case_results.json) · [seed1/burst_resident trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_resident/backend_trace.jsonl.gz) · [seed1/burst_resident hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_resident/instrumentation_audit.json) · [seed1/burst_storage timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_storage/case_results.json) · [seed1/burst_storage trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_storage/backend_trace.jsonl.gz) · [seed1/burst_storage hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/burst_storage/instrumentation_audit.json) · [seed1/spread_host timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_host/case_results.json) · [seed1/spread_host trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_host/backend_trace.jsonl.gz) · [seed1/spread_host hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_host/instrumentation_audit.json) · [seed1/spread_resident timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_resident/case_results.json) · [seed1/spread_resident trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_resident/backend_trace.jsonl.gz) · [seed1/spread_resident hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_resident/instrumentation_audit.json) · [seed1/spread_storage timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_storage/case_results.json) · [seed1/spread_storage trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_storage/backend_trace.jsonl.gz) · [seed1/spread_storage hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed1/spread_storage/instrumentation_audit.json) · [seed2/burst_host timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_host/case_results.json) · [seed2/burst_host trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_host/backend_trace.jsonl.gz) · [seed2/burst_host hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_host/instrumentation_audit.json) · [seed2/burst_resident timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_resident/case_results.json) · [seed2/burst_resident trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_resident/backend_trace.jsonl.gz) · [seed2/burst_resident hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_resident/instrumentation_audit.json) · [seed2/burst_storage timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_storage/case_results.json) · [seed2/burst_storage trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_storage/backend_trace.jsonl.gz) · [seed2/burst_storage hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/burst_storage/instrumentation_audit.json) · [seed2/spread_host timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_host/case_results.json) · [seed2/spread_host trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_host/backend_trace.jsonl.gz) · [seed2/spread_host hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_host/instrumentation_audit.json) · [seed2/spread_resident timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_resident/case_results.json) · [seed2/spread_resident trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_resident/backend_trace.jsonl.gz) · [seed2/spread_resident hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_resident/instrumentation_audit.json) · [seed2/spread_storage timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_storage/case_results.json) · [seed2/spread_storage trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_storage/backend_trace.jsonl.gz) · [seed2/spread_storage hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed2/spread_storage/instrumentation_audit.json) · [seed3/burst_host timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_host/case_results.json) · [seed3/burst_host trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_host/backend_trace.jsonl.gz) · [seed3/burst_host hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_host/instrumentation_audit.json) · [seed3/burst_resident timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_resident/case_results.json) · [seed3/burst_resident trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_resident/backend_trace.jsonl.gz) · [seed3/burst_resident hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_resident/instrumentation_audit.json) · [seed3/burst_storage timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_storage/case_results.json) · [seed3/burst_storage trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_storage/backend_trace.jsonl.gz) · [seed3/burst_storage hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/burst_storage/instrumentation_audit.json) · [seed3/spread_host timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_host/case_results.json) · [seed3/spread_host trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_host/backend_trace.jsonl.gz) · [seed3/spread_host hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_host/instrumentation_audit.json) · [seed3/spread_resident timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_resident/case_results.json) · [seed3/spread_resident trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_resident/backend_trace.jsonl.gz) · [seed3/spread_resident hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_resident/instrumentation_audit.json) · [seed3/spread_storage timings](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_storage/case_results.json) · [seed3/spread_storage trace](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_storage/backend_trace.jsonl.gz) · [seed3/spread_storage hook gate](docs/reports/work_audit/capacity_safe_tiers_7b_full_20261008/arms/seed3/spread_storage/instrumentation_audit.json)

</details>

<a id="run-memory_tiers_7b_full_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 5:39:28 p.m. CDT · GPU / CPU / storage KV gap</strong> · memory_tiers_7b_full_20261008</summary>

**Question (RQ23).** With a 7B coding model, equal-priority multi-turn sessions and no prefetch before tool return, how far do whole-workload time and replay response time separate when KV stays on GPU, spills to CPU, or can fall through CPU to file-backed storage? Does a near-simultaneous tool-return burst make the gap worse than normally spread returns?

**Finding.** spread CPU tier: whole workload +94.3% and mean replay delay +1669.0 ms vs all-GPU. spread storage tier: whole workload +201.6% and mean replay delay +3805.9 ms vs all-GPU. burst CPU tier: whole workload +141.1% and mean replay delay +2006.5 ms vs all-GPU. burst storage tier: whole workload +264.5% and mean replay delay +3696.8 ms vs all-GPU.

**Setup.** 6 equal-priority sessions; 10 tool returns per session; 4096 initial prompt tokens; 16 output tokens per replay; 1000 ms tool waits. Return patterns: spread, burst; the burst spans 75 ms and the spread control spans 1000 ms. Fresh backend per arm; no KV prefetch; all-GPU cap 40960 tokens; lower-tier GPU cap 8192 tokens; CPU caches 3.0 GiB allocated but unused for measured replays in all-GPU mode, 2.0 GiB for CPU mode and 1.0 GiB for storage mode. CUDA graphs on; overlap scheduling on; frontend priority equal; kv_lifecycle_lean tracing. Model: Qwen/Qwen2.5-Coder-7B-Instruct; hardware: nvidia_a10g_24gb; backend: 0.5.10.post1.

**Key measurements**

| Seed / return pattern / mode | Whole workload (s) | Due to first token mean / p95 (ms) | TTFT mean / p95 (ms) | Total due delay / TTFT (s) | Submission waiting (ms) | Proven GPU / CPU / storage source requests | Proven GPU / CPU / storage source tokens | Active sessions max / limit | Active KV estimate max / limit | Slot wait / KV preparation mean (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / burst / host | 67.698 | 2267.0 / 4235.9 | 2265.8 / 4234.2 | 136.022 / 135.949 | 69.0 | 0 / 60 / 0 | 0 / 263,040 / 0 | not recorded | not recorded | not recorded |
| 1 / burst / resident | 28.207 | 272.6 / 520.3 | 271.4 / 518.8 | 16.354 / 16.282 | 68.1 | 60 / 0 / 0 | 263,040 / 0 / 0 | not recorded | not recorded | not recorded |
| 1 / burst / storage | 102.826 | 3970.0 / 7997.4 | 3968.8 / 7996.0 | 238.201 / 238.128 | 70.1 | 8 / 8 / 20 | 28,160 / 23,872 / 87,680 | not recorded | not recorded | not recorded |
| 1 / spread / host | 67.554 | 1795.7 / 3284.9 | 1794.8 / 3284.5 | 107.741 / 107.689 | 48.7 | 0 / 60 / 0 | 0 / 263,040 / 0 | not recorded | not recorded | not recorded |
| 1 / spread / resident | 34.796 | 130.7 / 167.2 | 129.9 / 166.1 | 7.841 / 7.794 | 44.2 | 60 / 0 / 0 | 263,040 / 0 / 0 | not recorded | not recorded | not recorded |
| 1 / spread / storage | 103.264 | 3933.6 / 7451.1 | 3932.7 / 7449.8 | 236.015 / 235.962 | 49.1 | 2 / 3 / 32 | 4,096 / 12,608 / 137,856 | not recorded | not recorded | not recorded |
| 2 / burst / host | 67.691 | 2266.9 / 4206.4 | 2265.5 / 4205.7 | 136.012 / 135.933 | 76.1 | 0 / 60 / 0 | 0 / 263,040 / 0 | not recorded | not recorded | not recorded |
| 2 / burst / resident | 28.070 | 260.3 / 496.5 | 259.1 / 494.7 | 15.620 / 15.548 | 69.2 | 60 / 0 / 0 | 263,040 / 0 / 0 | not recorded | not recorded | not recorded |
| 2 / burst / storage | 102.311 | 3955.8 / 7994.6 | 3954.6 / 7993.3 | 237.346 / 237.275 | 67.8 | 8 / 6 / 20 | 35,328 / 18,752 / 87,680 | not recorded | not recorded | not recorded |
| 2 / spread / host | 67.646 | 1800.1 / 3288.2 | 1799.3 / 3287.4 | 108.007 / 107.955 | 49.1 | 0 / 60 / 0 | 0 / 263,040 / 0 | not recorded | not recorded | not recorded |
| 2 / spread / resident | 34.781 | 129.8 / 164.4 | 129.0 / 163.6 | 7.788 / 7.740 | 44.6 | 60 / 0 / 0 | 263,040 / 0 / 0 | not recorded | not recorded | not recorded |
| 2 / spread / storage | 104.891 | 3935.7 / 7410.1 | 3934.9 / 7409.6 | 236.145 / 236.091 | 50.5 | 4 / 9 / 28 | 8,192 / 29,888 / 123,072 | not recorded | not recorded | not recorded |
| 3 / burst / host | 67.695 | 2268.1 / 4212.8 | 2266.9 / 4211.1 | 136.085 / 136.015 | 66.6 | 0 / 60 / 0 | 0 / 263,040 / 0 | not recorded | not recorded | not recorded |
| 3 / burst / resident | 28.045 | 257.2 / 496.7 | 256.0 / 494.9 | 15.434 / 15.361 | 70.1 | 60 / 0 / 0 | 263,040 / 0 / 0 | not recorded | not recorded | not recorded |
| 3 / burst / storage | 102.487 | 3954.0 / 8002.6 | 3952.8 / 8000.5 | 237.242 / 237.165 | 73.2 | 8 / 8 / 20 | 30,400 / 25,728 / 87,680 | not recorded | not recorded | not recorded |
| 3 / spread / host | 67.632 | 1799.1 / 3285.1 | 1798.3 / 3284.1 | 107.948 / 107.898 | 47.5 | 0 / 60 / 0 | 0 / 263,040 / 0 | not recorded | not recorded | not recorded |
| 3 / spread / resident | 34.806 | 130.1 / 166.0 | 129.3 / 164.9 | 7.806 / 7.758 | 44.4 | 60 / 0 / 0 | 263,040 / 0 / 0 | not recorded | not recorded | not recorded |
| 3 / spread / storage | 105.814 | 4045.4 / 7432.5 | 4044.5 / 7431.3 | 242.725 / 242.669 | 51.9 | 3 / 7 / 28 | 6,144 / 23,424 / 122,816 | not recorded | not recorded | not recorded |

| Seed / return pattern / tier | Whole-workload change vs GPU | Mean / p95 replay-delay increase (ms) | Mean TTFT increase (ms) |
| --- | --- | --- | --- |
| 1 / spread / host | +32.758s (+94.1%) | +1665.0 / +3117.7 | +1664.9 |
| 1 / spread / storage | +68.468s (+196.8%) | +3802.9 / +7283.9 | +3802.8 |
| 1 / burst / host | +39.491s (+140.0%) | +1994.5 / +3715.5 | +1994.5 |
| 1 / burst / storage | +74.619s (+264.5%) | +3697.5 / +7477.0 | +3697.4 |
| 2 / spread / host | +32.865s (+94.5%) | +1670.3 / +3123.8 | +1670.2 |
| 2 / spread / storage | +70.110s (+201.6%) | +3805.9 / +7245.6 | +3805.8 |
| 2 / burst / host | +39.621s (+141.1%) | +2006.5 / +3709.9 | +2006.4 |
| 2 / burst / storage | +74.241s (+264.5%) | +3695.4 / +7498.0 | +3695.4 |
| 3 / spread / host | +32.825s (+94.3%) | +1669.0 / +3119.1 | +1669.0 |
| 3 / spread / storage | +71.008s (+204.0%) | +3915.3 / +7266.4 | +3915.2 |
| 3 / burst / host | +39.650s (+141.4%) | +2010.8 / +3716.1 | +2010.9 |
| 3 / burst / storage | +74.441s (+265.4%) | +3696.8 / +7505.9 | +3696.7 |

Positive changes mean the lower tier was slower than the all-GPU reference. Due-to-first-token includes any delay before submission plus backend TTFT. Tier-source counts come from native SGLang evidence, not from the requested mode name. Capacity-safe arms use pre-admission residency; original pressure arms use replay cache matches. File-backed storage uses normal operating-system caching; the page cache was not flushed.

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Synthetic equal-priority coding sessions; no semantic request priority.
- File-backed L3 is normal system storage; the OS page cache is not flushed.
- This exposes an unprepared return burst, not the later tool-aware optimized policy.

**Reproduce** (set the container image and model cache for the target host):

```bash
TIER_RUN_ID=memory_tiers_7b_full_20261008_repeat \
TIER_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
TIER_SEEDS='1 2 3' \
TIER_PATTERNS='spread burst' \
TIER_MODES='resident host storage' \
TIER_SESSIONS=6 \
TIER_TURNS=10 \
TIER_INITIAL_TOKENS=4096 \
TIER_TOOL_WORDS=16 \
TIER_DECODE_TOKENS=16 \
TIER_WAIT_MS=1000 \
TIER_BURST_WINDOW_MS=75 \
TIER_SPREAD_WINDOW_MS=1000 \
TIER_RESIDENT_GPU_TOKENS=40960 \
TIER_RESIDENT_HOST_GB=3.0 \
TIER_RESTRICTED_GPU_TOKENS=8192 \
TIER_HOST_GB=2.0 \
TIER_STORAGE_HOST_GB=1.0 \
TIER_MAX_INFLIGHT=6 \
bash infra/container/run_work_audit_memory_tiers.sh
```

**Evidence:** [Summary](docs/reports/work_audit/memory_tiers_7b_full_20261008/summary.json) · [Run manifest](docs/reports/work_audit/memory_tiers_7b_full_20261008/run_manifest.json) · [Source hashes](docs/reports/work_audit/memory_tiers_7b_full_20261008/source_sha256.txt) · [seed1/burst_host timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_host/case_results.json) · [seed1/burst_host trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_host/backend_trace.jsonl.gz) · [seed1/burst_host hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_host/instrumentation_audit.json) · [seed1/burst_resident timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_resident/case_results.json) · [seed1/burst_resident trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_resident/backend_trace.jsonl.gz) · [seed1/burst_resident hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_resident/instrumentation_audit.json) · [seed1/burst_storage timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_storage/case_results.json) · [seed1/burst_storage trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_storage/backend_trace.jsonl.gz) · [seed1/burst_storage hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/burst_storage/instrumentation_audit.json) · [seed1/spread_host timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_host/case_results.json) · [seed1/spread_host trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_host/backend_trace.jsonl.gz) · [seed1/spread_host hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_host/instrumentation_audit.json) · [seed1/spread_resident timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_resident/case_results.json) · [seed1/spread_resident trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_resident/backend_trace.jsonl.gz) · [seed1/spread_resident hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_resident/instrumentation_audit.json) · [seed1/spread_storage timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_storage/case_results.json) · [seed1/spread_storage trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_storage/backend_trace.jsonl.gz) · [seed1/spread_storage hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed1/spread_storage/instrumentation_audit.json) · [seed2/burst_host timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_host/case_results.json) · [seed2/burst_host trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_host/backend_trace.jsonl.gz) · [seed2/burst_host hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_host/instrumentation_audit.json) · [seed2/burst_resident timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_resident/case_results.json) · [seed2/burst_resident trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_resident/backend_trace.jsonl.gz) · [seed2/burst_resident hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_resident/instrumentation_audit.json) · [seed2/burst_storage timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_storage/case_results.json) · [seed2/burst_storage trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_storage/backend_trace.jsonl.gz) · [seed2/burst_storage hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/burst_storage/instrumentation_audit.json) · [seed2/spread_host timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_host/case_results.json) · [seed2/spread_host trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_host/backend_trace.jsonl.gz) · [seed2/spread_host hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_host/instrumentation_audit.json) · [seed2/spread_resident timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_resident/case_results.json) · [seed2/spread_resident trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_resident/backend_trace.jsonl.gz) · [seed2/spread_resident hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_resident/instrumentation_audit.json) · [seed2/spread_storage timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_storage/case_results.json) · [seed2/spread_storage trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_storage/backend_trace.jsonl.gz) · [seed2/spread_storage hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed2/spread_storage/instrumentation_audit.json) · [seed3/burst_host timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_host/case_results.json) · [seed3/burst_host trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_host/backend_trace.jsonl.gz) · [seed3/burst_host hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_host/instrumentation_audit.json) · [seed3/burst_resident timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_resident/case_results.json) · [seed3/burst_resident trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_resident/backend_trace.jsonl.gz) · [seed3/burst_resident hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_resident/instrumentation_audit.json) · [seed3/burst_storage timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_storage/case_results.json) · [seed3/burst_storage trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_storage/backend_trace.jsonl.gz) · [seed3/burst_storage hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/burst_storage/instrumentation_audit.json) · [seed3/spread_host timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_host/case_results.json) · [seed3/spread_host trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_host/backend_trace.jsonl.gz) · [seed3/spread_host hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_host/instrumentation_audit.json) · [seed3/spread_resident timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_resident/case_results.json) · [seed3/spread_resident trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_resident/backend_trace.jsonl.gz) · [seed3/spread_resident hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_resident/instrumentation_audit.json) · [seed3/spread_storage timings](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_storage/case_results.json) · [seed3/spread_storage trace](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_storage/backend_trace.jsonl.gz) · [seed3/spread_storage hook gate](docs/reports/work_audit/memory_tiers_7b_full_20261008/arms/seed3/spread_storage/instrumentation_audit.json)

</details>

<a id="run-coordinated_session_pipeline_pilot_v2_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 4:51:13 p.m. CDT · Session-level coordinated KV (calibration)</strong> · coordinated_session_pipeline_pilot_v2_20261008</summary>

**Question (RQ22).** With twenty independent sessions, identical tool clocks in all modes, reserved GPU preparation space and exact tool-return times, can individual early KV restores overlap active model work and approach a fully GPU-resident reference without a group barrier?

**Finding.** Coordinated workload was 36.2% longer than independent (median paired change). Coordinated workload was 58.9% longer than resident (median paired change). 36 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; individual session clocks; no group barrier; 750 ms prefetch lead; 2 session headroom; at most 8 requests in flight; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 8 output tokens per replay; native kernel KV transfer; group restore submission; group control calls; kv_lifecycle_counts tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: nvidia_a10g_24gb; backend version: 0.5.10.post1. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / independent | 7.200 | 1073.3 | 2226.5 | 34.505 | 64.398 | 29.892 | 0.000 | 0 | 0 / 0 | 0.0% |
| 1 / coordinated | 9.804 | 1634.0 | 3078.3 | 26.860 | 98.040 | 71.180 | 0.000 | 44 | 24 / 36 | 99.0% |
| 1 / resident | 6.170 | 792.1 | 2201.1 | 24.586 | 47.527 | 22.940 | 0.000 | 0 | 0 / 0 | 99.0% |

| Trial / reference | Whole-workload change | Mean replay-delay change (ms) | Sessions finishing sooner / later |
| --- | --- | --- | --- |
| 1 / independent | +36.2% | +560.7 | 0 / 20 |
| 1 / resident | +58.9% | +841.9 | 0 / 20 |

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / independent | 16.830 | 0 | 0.000 | not measured | 0 / 0 |
| 1 / coordinated | 17.875 | 253 | 4.669 | 44 / 0.971 | 362,688 / 364,416 |
| 1 / resident | 17.349 | 0 | 0.000 | not measured | 0 / 0 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | 5.996 | 8.486 | 4.885 |
| 1 / swap-s01 | 5.998 | 7.233 | 4.886 |
| 1 / swap-s02 | 5.998 | 7.882 | 4.885 |
| 1 / swap-s03 | 5.997 | 7.233 | 5.583 |
| 1 / swap-s04 | 5.997 | 7.232 | 5.581 |
| 1 / swap-s05 | 5.998 | 7.233 | 4.886 |
| 1 / swap-s06 | 5.998 | 7.882 | 5.583 |
| 1 / swap-s07 | 6.001 | 9.804 | 5.583 |
| 1 / swap-s08 | 6.814 | 8.654 | 5.583 |
| 1 / swap-s09 | 6.814 | 7.882 | 5.866 |
| 1 / swap-s10 | 6.814 | 7.883 | 5.866 |
| 1 / swap-s11 | 6.815 | 8.536 | 5.585 |
| 1 / swap-s12 | 6.817 | 8.876 | 5.867 |
| 1 / swap-s13 | 6.814 | 8.654 | 5.584 |
| 1 / swap-s14 | 6.813 | 8.652 | 5.583 |
| 1 / swap-s15 | 6.815 | 8.876 | 5.867 |
| 1 / swap-s16 | 7.200 | 8.653 | 6.169 |
| 1 / swap-s17 | 7.200 | 8.877 | 6.169 |
| 1 / swap-s18 | 7.200 | 8.877 | 6.169 |
| 1 / swap-s19 | 7.200 | 9.638 | 6.170 |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_session_pipeline_pilot_v2_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=8 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=kernel \
SWAP_RESTORE_STYLE=group \
SWAP_CONTROL_STYLE=group \
SWAP_SCHEDULE_STYLE=session_pipeline \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_counts \
SWAP_TRIALS=1 \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/summary.json) · [Run manifest](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/run_manifest.json) · [source_sha256.txt](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/source_sha256.txt) · [source_bundle.tar.gz](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/source_bundle.tar.gz) · [trial1_independent per-turn timings](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/arms/trial1_independent/case_results.json) · [trial1_independent trace](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/arms/trial1_independent/backend_trace.jsonl.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz) · [trial1_resident per-turn timings](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/arms/trial1_resident/case_results.json) · [trial1_resident trace](docs/reports/work_audit/coordinated_session_pipeline_pilot_v2_20261008/arms/trial1_resident/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_session_pipeline_pilot_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 4:42:28 p.m. CDT · Session-level coordinated KV (calibration)</strong> · coordinated_session_pipeline_pilot_20261008</summary>

**Question (RQ22).** With twenty independent sessions, identical tool clocks in all modes, reserved GPU preparation space and exact tool-return times, can individual early KV restores overlap active model work and approach a fully GPU-resident reference without a group barrier?

**Finding.** Coordinated workload was 71.5% longer than independent (median paired change). Coordinated workload was 97.2% longer than resident (median paired change). 46 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; individual session clocks; no group barrier; 350 ms prefetch lead; 2 session headroom; at most 8 requests in flight; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 8 output tokens per replay; native kernel KV transfer; group restore submission; group control calls; kv_lifecycle_counts tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: nvidia_a10g_24gb; backend version: 0.5.10.post1. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / independent | 7.088 | 992.6 | 2222.8 | 30.688 | 59.554 | 28.866 | 0.000 | 0 | 0 / 0 | 0.0% |
| 1 / coordinated | 12.154 | 2352.7 | 4177.8 | 23.078 | 141.161 | 118.083 | 0.000 | 48 | 14 / 46 | 99.0% |
| 1 / resident | 6.162 | 791.0 | 2199.6 | 24.518 | 47.459 | 22.940 | 0.000 | 0 | 0 / 0 | 99.0% |

| Trial / reference | Whole-workload change | Mean replay-delay change (ms) | Sessions finishing sooner / later |
| --- | --- | --- | --- |
| 1 / independent | +71.5% | +1360.1 | 0 / 20 |
| 1 / resident | +97.2% | +1561.7 | 0 / 20 |

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / independent | 16.985 | 0 | 0.000 | not measured | 0 / 0 |
| 1 / coordinated | 18.309 | 525 | 10.609 | 48 / 1.060 | 395,712 / 397,504 |
| 1 / resident | 17.288 | 0 | 0.000 | not measured | 0 / 0 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | 6.311 | 9.063 | 4.881 |
| 1 / swap-s01 | 5.701 | 9.328 | 4.880 |
| 1 / swap-s02 | 5.701 | 9.328 | 4.880 |
| 1 / swap-s03 | 6.313 | 9.679 | 5.575 |
| 1 / swap-s04 | 5.699 | 9.328 | 5.574 |
| 1 / swap-s05 | 5.701 | 10.452 | 4.879 |
| 1 / swap-s06 | 6.312 | 10.253 | 5.573 |
| 1 / swap-s07 | 6.313 | 10.453 | 5.574 |
| 1 / swap-s08 | 6.790 | 10.453 | 5.575 |
| 1 / swap-s09 | 6.313 | 11.115 | 5.857 |
| 1 / swap-s10 | 6.789 | 11.271 | 5.857 |
| 1 / swap-s11 | 6.790 | 11.116 | 5.857 |
| 1 / swap-s12 | 6.310 | 11.272 | 5.858 |
| 1 / swap-s13 | 6.312 | 11.116 | 5.575 |
| 1 / swap-s14 | 6.312 | 11.001 | 5.574 |
| 1 / swap-s15 | 6.790 | 11.052 | 5.575 |
| 1 / swap-s16 | 7.088 | 11.116 | 6.162 |
| 1 / swap-s17 | 7.087 | 11.398 | 6.162 |
| 1 / swap-s18 | 7.088 | 11.051 | 6.162 |
| 1 / swap-s19 | 7.087 | 12.154 | 6.162 |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_session_pipeline_pilot_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=8 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=kernel \
SWAP_RESTORE_STYLE=group \
SWAP_CONTROL_STYLE=group \
SWAP_SCHEDULE_STYLE=session_pipeline \
SWAP_PREFETCH_LEAD_MS=350 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_counts \
SWAP_TRIALS=1 \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/summary.json) · [Run manifest](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/run_manifest.json) · [source_sha256.txt](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/source_sha256.txt) · [source_bundle.tar.gz](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/source_bundle.tar.gz) · [trial1_independent per-turn timings](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/arms/trial1_independent/case_results.json) · [trial1_independent trace](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/arms/trial1_independent/backend_trace.jsonl.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz) · [trial1_resident per-turn timings](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/arms/trial1_resident/case_results.json) · [trial1_resident trace](docs/reports/work_audit/coordinated_session_pipeline_pilot_20261008/arms/trial1_resident/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_swap_count_full_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 2:28:45 p.m. CDT · Coordinated CPU/GPU swaps</strong> · coordinated_swap_count_full_20261008</summary>

**Question (RQ21).** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**Finding.** Coordinated workload was 24.3% longer than independent (median paired change). Coordinated workload was 24.6% longer than resident (median paired change). 2370 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 40 tool rounds; 1,000 ms waits; paired AB/CD group rotation; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 8 output tokens per replay; native kernel KV transfer; group restore submission; group control calls; kv_lifecycle_counts tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: nvidia_a10g_24gb; backend version: 0.5.10.post1. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / independent | 81.557 | 766.0 | 1051.3 | 609.193 | 612.767 | 3.574 | 0.000 | 0 | 0 / 0 | 0.0% |
| 1 / coordinated | 101.347 | 848.8 | 1193.1 | 257.192 | 679.044 | 421.851 | 318.076 | 79 | 0 / 790 | 98.9% |
| 1 / resident | 80.749 | 339.6 | 499.8 | 257.578 | 271.683 | 14.105 | 318.226 | 0 | 0 / 0 | 98.9% |
| 2 / independent | 80.272 | 723.7 | 954.7 | 575.565 | 578.994 | 3.429 | 0.000 | 0 | 0 / 0 | 0.0% |
| 2 / coordinated | 99.703 | 803.4 | 1118.5 | 252.518 | 642.682 | 390.164 | 322.600 | 79 | 0 / 790 | 98.9% |
| 2 / resident | 80.731 | 340.8 | 556.7 | 258.556 | 272.666 | 14.110 | 319.108 | 0 | 0 / 0 | 98.9% |
| 3 / independent | 80.959 | 748.3 | 1021.8 | 595.783 | 598.653 | 2.870 | 0.000 | 0 | 0 / 0 | 0.0% |
| 3 / coordinated | 100.605 | 826.9 | 1054.1 | 253.809 | 661.508 | 407.699 | 321.564 | 79 | 0 / 790 | 98.9% |
| 3 / resident | 80.728 | 334.6 | 492.8 | 253.753 | 267.672 | 13.919 | 321.897 | 0 | 0 / 0 | 98.9% |

| Trial / reference | Whole-workload change | Mean replay-delay change (ms) | Sessions finishing sooner / later |
| --- | --- | --- | --- |
| 1 / independent | +24.3% | +82.8 | 0 / 20 |
| 1 / resident | +25.5% | +509.2 | 0 / 20 |
| 2 / independent | +24.2% | +79.6 | 0 / 20 |
| 2 / resident | +23.5% | +462.5 | 0 / 20 |
| 3 / independent | +24.3% | +78.6 | 0 / 20 |
| 3 / resident | +24.6% | +492.3 | 0 / 20 |

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / independent | 16.871 | 0 | 0.000 | not measured | 0 / 0 |
| 1 / coordinated | 18.087 | 1377 | 41.820 | 79 / 16.933 | 6,833,920 / 6,853,760 |
| 1 / resident | 17.112 | 0 | 0.000 | not measured | 0 / 0 |
| 2 / independent | 16.962 | 0 | 0.000 | not measured | 0 / 0 |
| 2 / coordinated | 17.875 | 1444 | 39.986 | 79 / 16.926 | 6,833,920 / 6,853,760 |
| 2 / resident | 17.163 | 0 | 0.000 | not measured | 0 / 0 |
| 3 / independent | 16.881 | 0 | 0.000 | not measured | 0 / 0 |
| 3 / coordinated | 17.991 | 1351 | 41.710 | 79 / 16.931 | 6,833,920 / 6,853,760 |
| 3 / resident | 17.104 | 0 | 0.000 | not measured | 0 / 0 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | 80.803 | 99.932 | 79.741 |
| 1 / swap-s01 | 80.803 | 99.931 | 79.740 |
| 1 / swap-s02 | 81.556 | 99.930 | 79.741 |
| 1 / swap-s03 | 80.801 | 99.932 | 79.742 |
| 1 / swap-s04 | 80.803 | 99.930 | 79.741 |
| 1 / swap-s05 | 80.804 | 99.932 | 79.742 |
| 1 / swap-s06 | 80.803 | 99.931 | 79.742 |
| 1 / swap-s07 | 80.803 | 99.931 | 79.742 |
| 1 / swap-s08 | 80.804 | 99.932 | 79.742 |
| 1 / swap-s09 | 80.802 | 99.931 | 79.741 |
| 1 / swap-s10 | 80.803 | 101.346 | 80.748 |
| 1 / swap-s11 | 80.803 | 101.346 | 80.749 |
| 1 / swap-s12 | 80.802 | 101.346 | 80.748 |
| 1 / swap-s13 | 81.557 | 101.346 | 80.748 |
| 1 / swap-s14 | 81.557 | 101.346 | 80.747 |
| 1 / swap-s15 | 81.556 | 101.347 | 80.748 |
| 1 / swap-s16 | 81.557 | 101.347 | 80.748 |
| 1 / swap-s17 | 81.556 | 101.345 | 80.748 |
| 1 / swap-s18 | 81.556 | 101.345 | 80.747 |
| 1 / swap-s19 | 81.556 | 101.346 | 80.748 |
| 2 / swap-s00 | 79.516 | 98.368 | 79.733 |
| 2 / swap-s01 | 80.271 | 98.367 | 79.732 |
| 2 / swap-s02 | 79.519 | 98.369 | 79.732 |
| 2 / swap-s03 | 79.518 | 98.369 | 79.732 |
| 2 / swap-s04 | 79.518 | 98.369 | 79.733 |
| 2 / swap-s05 | 79.519 | 98.369 | 79.732 |
| 2 / swap-s06 | 79.519 | 98.368 | 79.732 |
| 2 / swap-s07 | 79.518 | 98.368 | 79.731 |
| 2 / swap-s08 | 79.519 | 98.368 | 79.732 |
| 2 / swap-s09 | 79.519 | 98.368 | 79.732 |
| 2 / swap-s10 | 79.518 | 99.702 | 80.731 |
| 2 / swap-s11 | 79.518 | 99.702 | 80.729 |
| 2 / swap-s12 | 79.518 | 99.702 | 80.731 |
| 2 / swap-s13 | 80.271 | 99.702 | 80.730 |
| 2 / swap-s14 | 80.272 | 99.702 | 80.730 |
| 2 / swap-s15 | 80.272 | 99.702 | 80.730 |
| 2 / swap-s16 | 80.271 | 99.702 | 80.730 |
| 2 / swap-s17 | 80.270 | 99.703 | 80.730 |
| 2 / swap-s18 | 80.272 | 99.701 | 80.731 |
| 2 / swap-s19 | 80.271 | 99.703 | 80.730 |
| 3 / swap-s00 | 80.387 | 99.243 | 79.726 |
| 3 / swap-s01 | 80.389 | 99.243 | 79.726 |
| 3 / swap-s02 | 79.062 | 99.243 | 79.726 |
| 3 / swap-s03 | 80.389 | 99.244 | 79.727 |
| 3 / swap-s04 | 80.386 | 99.242 | 79.726 |
| 3 / swap-s05 | 80.388 | 99.243 | 79.727 |
| 3 / swap-s06 | 80.388 | 99.243 | 79.726 |
| 3 / swap-s07 | 80.388 | 99.244 | 79.725 |
| 3 / swap-s08 | 80.388 | 99.244 | 79.726 |
| 3 / swap-s09 | 80.387 | 99.243 | 79.726 |
| 3 / swap-s10 | 79.062 | 100.604 | 80.728 |
| 3 / swap-s11 | 80.388 | 100.604 | 80.728 |
| 3 / swap-s12 | 80.388 | 100.604 | 80.728 |
| 3 / swap-s13 | 80.958 | 100.604 | 80.726 |
| 3 / swap-s14 | 80.958 | 100.603 | 80.727 |
| 3 / swap-s15 | 80.959 | 100.605 | 80.727 |
| 3 / swap-s16 | 80.959 | 100.603 | 80.728 |
| 3 / swap-s17 | 80.958 | 100.604 | 80.727 |
| 3 / swap-s18 | 80.959 | 100.604 | 80.727 |
| 3 / swap-s19 | 80.388 | 100.604 | 80.727 |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_swap_count_full_20261008_repeat \
SWAP_TURNS=40 \
SWAP_DECODE_TOKENS=8 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=kernel \
SWAP_RESTORE_STYLE=group \
SWAP_CONTROL_STYLE=group \
SWAP_SCHEDULE_STYLE=barrier \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_counts \
SWAP_TRIALS='1 2 3' \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_swap_count_full_20261008/summary.json) · [Run manifest](docs/reports/work_audit/coordinated_swap_count_full_20261008/run_manifest.json) · [source_sha256.txt](docs/reports/work_audit/coordinated_swap_count_full_20261008/source_sha256.txt) · [source_bundle.tar.gz](docs/reports/work_audit/coordinated_swap_count_full_20261008/source_bundle.tar.gz) · [package_sources.tar.gz](docs/reports/work_audit/coordinated_swap_count_full_20261008/package_sources.tar.gz) · [source_validation.json](docs/reports/work_audit/coordinated_swap_count_full_20261008/source_validation.json) · [postrun_analysis_sources.tar.gz](docs/reports/work_audit/coordinated_swap_count_full_20261008/postrun_analysis_sources.tar.gz) · [trial1_independent per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial1_independent/case_results.json) · [trial1_independent trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial1_independent/backend_trace.jsonl.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz) · [trial1_resident per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial1_resident/case_results.json) · [trial1_resident trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial1_resident/backend_trace.jsonl.gz) · [trial2_independent per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial2_independent/case_results.json) · [trial2_independent trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial2_independent/backend_trace.jsonl.gz) · [trial2_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial2_coordinated/case_results.json) · [trial2_coordinated trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial2_coordinated/backend_trace.jsonl.gz) · [trial2_resident per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial2_resident/case_results.json) · [trial2_resident trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial2_resident/backend_trace.jsonl.gz) · [trial3_independent per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial3_independent/case_results.json) · [trial3_independent trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial3_independent/backend_trace.jsonl.gz) · [trial3_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial3_coordinated/case_results.json) · [trial3_coordinated trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial3_coordinated/backend_trace.jsonl.gz) · [trial3_resident per-turn timings](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial3_resident/case_results.json) · [trial3_resident trace](docs/reports/work_audit/coordinated_swap_count_full_20261008/arms/trial3_resident/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_swap_count_pilot_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 2:20:38 p.m. CDT · Coordinated CPU/GPU swaps (calibration)</strong> · coordinated_swap_count_pilot_20261008</summary>

**Question (RQ21).** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**Finding.** Coordinated workload was 29.0% longer than independent (median paired change). Coordinated workload was 45.6% longer than resident (median paired change). 50 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; paired AB/CD group rotation; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 8 output tokens per replay; native kernel KV transfer; group restore submission; individual control calls; kv_lifecycle_counts tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: nvidia_a10g_24gb; backend version: 0.5.10.post1. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / independent | 7.455 | 1143.2 | 2278.3 | 68.469 | 68.593 | 0.124 | 0.000 | 0 | 0 / 0 | 0.0% |
| 1 / coordinated | 9.613 | 1664.6 | 2434.5 | 33.818 | 99.874 | 66.056 | 18.269 | 5 | 0 / 50 | 99.0% |
| 1 / resident | 6.602 | 732.5 | 1834.7 | 33.339 | 43.953 | 10.613 | 18.356 | 0 | 0 / 0 | 99.0% |

| Trial / reference | Whole-workload change | Mean replay-delay change (ms) | Sessions finishing sooner / later |
| --- | --- | --- | --- |
| 1 / independent | +29.0% | +521.4 | 0 / 20 |
| 1 / resident | +45.6% | +932.0 | 0 / 20 |

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / independent | 16.858 | 0 | 0.000 | not measured | 0 / 0 |
| 1 / coordinated | 18.169 | 221 | 3.735 | 5 / 1.021 | 412,160 / 414,080 |
| 1 / resident | 17.129 | 0 | 0.000 | not measured | 0 / 0 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | 6.791 | 8.164 | 5.601 |
| 1 / swap-s01 | 6.792 | 8.163 | 5.601 |
| 1 / swap-s02 | 6.792 | 8.163 | 5.602 |
| 1 / swap-s03 | 6.791 | 8.164 | 5.602 |
| 1 / swap-s04 | 6.791 | 8.165 | 5.602 |
| 1 / swap-s05 | 6.791 | 8.164 | 5.601 |
| 1 / swap-s06 | 6.792 | 8.164 | 5.602 |
| 1 / swap-s07 | 6.792 | 8.163 | 5.600 |
| 1 / swap-s08 | 6.789 | 8.165 | 5.602 |
| 1 / swap-s09 | 6.792 | 8.164 | 5.602 |
| 1 / swap-s10 | 6.791 | 9.613 | 6.601 |
| 1 / swap-s11 | 6.791 | 9.613 | 6.600 |
| 1 / swap-s12 | 6.791 | 9.613 | 6.602 |
| 1 / swap-s13 | 7.455 | 9.612 | 6.601 |
| 1 / swap-s14 | 7.455 | 9.611 | 6.601 |
| 1 / swap-s15 | 7.454 | 9.612 | 6.601 |
| 1 / swap-s16 | 7.455 | 9.612 | 6.601 |
| 1 / swap-s17 | 7.454 | 9.613 | 6.601 |
| 1 / swap-s18 | 7.455 | 9.613 | 6.602 |
| 1 / swap-s19 | 7.455 | 9.613 | 6.601 |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_swap_count_pilot_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=8 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=kernel \
SWAP_RESTORE_STYLE=group \
SWAP_CONTROL_STYLE=individual \
SWAP_SCHEDULE_STYLE=barrier \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_counts \
SWAP_TRIALS=1 \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/summary.json) · [Run manifest](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/run_manifest.json) · [source_sha256.txt](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/source_sha256.txt) · [source_bundle.tar.gz](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/source_bundle.tar.gz) · [trial1_independent per-turn timings](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/arms/trial1_independent/case_results.json) · [trial1_independent trace](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/arms/trial1_independent/backend_trace.jsonl.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz) · [trial1_resident per-turn timings](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/arms/trial1_resident/case_results.json) · [trial1_resident trace](docs/reports/work_audit/coordinated_swap_count_pilot_20261008/arms/trial1_resident/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_swap_batched_pilot_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 2:13:43 p.m. CDT · Coordinated CPU/GPU swaps (calibration)</strong> · coordinated_swap_batched_pilot_20261008</summary>

**Question (RQ21).** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**Finding.** 50 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; paired AB/CD group rotation; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 8 output tokens per replay; native kernel KV transfer; group restore submission; individual control calls; kv_lifecycle_lean tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: nvidia_a10g_24gb; backend version: 0.5.10.post1. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / coordinated | 27.075 | 6828.2 | 8876.4 | 41.468 | 409.692 | 368.224 | 7.109 | 5 | 0 / 50 | 99.0% |

No comparable measurements recorded.

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / coordinated | 22.730 | 160 | 20.423 | 5 / 15.306 | 412,160 / 414,080 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | unavailable | 21.824 | unavailable |
| 1 / swap-s01 | unavailable | 21.826 | unavailable |
| 1 / swap-s02 | unavailable | 21.824 | unavailable |
| 1 / swap-s03 | unavailable | 21.826 | unavailable |
| 1 / swap-s04 | unavailable | 21.825 | unavailable |
| 1 / swap-s05 | unavailable | 21.826 | unavailable |
| 1 / swap-s06 | unavailable | 21.826 | unavailable |
| 1 / swap-s07 | unavailable | 21.826 | unavailable |
| 1 / swap-s08 | unavailable | 21.825 | unavailable |
| 1 / swap-s09 | unavailable | 21.826 | unavailable |
| 1 / swap-s10 | unavailable | 27.074 | unavailable |
| 1 / swap-s11 | unavailable | 27.074 | unavailable |
| 1 / swap-s12 | unavailable | 27.075 | unavailable |
| 1 / swap-s13 | unavailable | 27.073 | unavailable |
| 1 / swap-s14 | unavailable | 27.075 | unavailable |
| 1 / swap-s15 | unavailable | 27.074 | unavailable |
| 1 / swap-s16 | unavailable | 27.075 | unavailable |
| 1 / swap-s17 | unavailable | 27.075 | unavailable |
| 1 / swap-s18 | unavailable | 27.075 | unavailable |
| 1 / swap-s19 | unavailable | 27.075 | unavailable |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_swap_batched_pilot_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=8 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=kernel \
SWAP_RESTORE_STYLE=group \
SWAP_CONTROL_STYLE=individual \
SWAP_SCHEDULE_STYLE=barrier \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_lean \
SWAP_TRIALS=1 \
SWAP_MODES=coordinated \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_swap_batched_pilot_20261008/summary.json) · [Run manifest](docs/reports/work_audit/coordinated_swap_batched_pilot_20261008/run_manifest.json) · [source_sha256.txt](docs/reports/work_audit/coordinated_swap_batched_pilot_20261008/source_sha256.txt) · [source_bundle.tar.gz](docs/reports/work_audit/coordinated_swap_batched_pilot_20261008/source_bundle.tar.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_batched_pilot_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_swap_batched_pilot_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_swap_kernel_pilot_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 2:04:28 p.m. CDT · Coordinated CPU/GPU swaps (calibration)</strong> · coordinated_swap_kernel_pilot_20261008</summary>

**Question (RQ21).** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**Finding.** Coordinated workload was 14.3% longer than independent (median paired change). Coordinated workload was 287.1% longer than resident (median paired change). 50 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; paired AB/CD group rotation; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 8 output tokens per replay; native kernel KV transfer; serial restore submission; individual control calls; kv_lifecycle_lean tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: nvidia_a10g_24gb; backend version: 0.5.10.post1. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / independent | 24.125 | 5643.7 | 9810.9 | 338.483 | 338.620 | 0.137 | 0.000 | 0 | 0 / 0 | 0.0% |
| 1 / coordinated | 27.575 | 7012.3 | 8885.5 | 41.471 | 420.737 | 379.266 | 7.128 | 50 | 0 / 50 | 99.0% |
| 1 / resident | 7.124 | 916.4 | 2057.4 | 40.737 | 54.983 | 14.246 | 7.654 | 0 | 0 / 0 | 99.0% |

| Trial / reference | Whole-workload change | Mean replay-delay change (ms) | Sessions finishing sooner / later |
| --- | --- | --- | --- |
| 1 / independent | +14.3% | +1368.6 | 2 / 18 |
| 1 / resident | +287.1% | +6095.9 | 0 / 20 |

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / independent | 18.533 | 0 | 0.000 | not measured | 0 / 0 |
| 1 / coordinated | 22.928 | 300 | 20.920 | 50 / 14.891 | 412,160 / 414,080 |
| 1 / resident | 18.793 | 0 | 0.000 | not measured | 0 / 0 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | 15.398 | 22.421 | 6.407 |
| 1 / swap-s01 | 20.572 | 22.422 | 6.407 |
| 1 / swap-s02 | 24.125 | 22.421 | 6.407 |
| 1 / swap-s03 | 15.396 | 22.421 | 6.407 |
| 1 / swap-s04 | 20.572 | 22.421 | 6.407 |
| 1 / swap-s05 | 15.396 | 22.421 | 6.408 |
| 1 / swap-s06 | 20.571 | 22.420 | 6.407 |
| 1 / swap-s07 | 15.397 | 22.421 | 6.408 |
| 1 / swap-s08 | 20.571 | 22.421 | 6.406 |
| 1 / swap-s09 | 24.125 | 22.422 | 6.408 |
| 1 / swap-s10 | 20.571 | 27.575 | 7.124 |
| 1 / swap-s11 | 20.570 | 27.574 | 7.122 |
| 1 / swap-s12 | 20.571 | 27.574 | 7.124 |
| 1 / swap-s13 | 24.124 | 27.574 | 7.122 |
| 1 / swap-s14 | 24.124 | 27.575 | 7.124 |
| 1 / swap-s15 | 24.125 | 27.574 | 7.123 |
| 1 / swap-s16 | 24.125 | 27.574 | 7.123 |
| 1 / swap-s17 | 24.125 | 27.575 | 7.123 |
| 1 / swap-s18 | 20.571 | 27.575 | 7.123 |
| 1 / swap-s19 | 24.124 | 27.575 | 7.124 |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_swap_kernel_pilot_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=8 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=kernel \
SWAP_RESTORE_STYLE=serial \
SWAP_CONTROL_STYLE=individual \
SWAP_SCHEDULE_STYLE=barrier \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_lean \
SWAP_TRIALS=1 \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/summary.json) · [Run manifest](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/run_manifest.json) · [source_sha256.txt](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/source_sha256.txt) · [source_bundle.tar.gz](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/source_bundle.tar.gz) · [trial1_independent per-turn timings](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/arms/trial1_independent/case_results.json) · [trial1_independent trace](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/arms/trial1_independent/backend_trace.jsonl.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz) · [trial1_resident per-turn timings](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/arms/trial1_resident/case_results.json) · [trial1_resident trace](docs/reports/work_audit/coordinated_swap_kernel_pilot_20261008/arms/trial1_resident/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_swap_pilot_v2_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 1:55:10 p.m. CDT · Coordinated CPU/GPU swaps (calibration)</strong> · coordinated_swap_pilot_v2_20261008</summary>

**Question (RQ21).** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**Finding.** Coordinated workload was 30.4% longer than independent (median paired change). Coordinated workload was 243.3% longer than resident (median paired change). 50 coordinated replays had KV become ready after their tool deadline.

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; paired AB/CD group rotation; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 32 output tokens per replay; native direct KV transfer; serial restore submission; individual control calls; kv_lifecycle_lean tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: see manifest; backend version: see runtime record. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

| Trial / mode | Whole workload (s) | Due to first token mean (ms) | Due to first token p95 (ms) | Total TTFT (s) | Total due delay (s) | Submission waiting total (s) | Scheduled slot padding sum (s) | Explicit native load batches | KV ready on time / late | Minimum GPU prefix reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / independent | 22.547 | 4781.6 | 9180.9 | 286.735 | 286.894 | 0.159 | 0.000 | 0 | 0 / 0 | 0.0% |
| 1 / coordinated | 29.396 | 7393.6 | 9560.6 | 40.828 | 443.616 | 402.788 | 0.042 | 50 | 0 / 50 | 99.0% |
| 1 / resident | 8.563 | 1101.2 | 2355.5 | 40.846 | 66.070 | 25.223 | 0.040 | 0 | 0 / 0 | 99.0% |

| Trial / reference | Whole-workload change | Mean replay-delay change (ms) | Sessions finishing sooner / later |
| --- | --- | --- | --- |
| 1 / independent | +30.4% | +2612.0 | 0 / 20 |
| 1 / resident | +243.3% | +6292.4 | 0 / 20 |

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

| Trial / mode | Excluded setup (s) | Measured control calls | Total control wall time (s) | Native CUDA-stream intervals / total (s) | Explicit restored / released KV tokens |
| --- | --- | --- | --- | --- | --- |
| 1 / independent | 15.999 | 0 | 0.000 | not measured | 0 / 0 |
| 1 / coordinated | 21.266 | 300 | 20.814 | 50 / 13.604 | 412,160 / 414,720 |
| 1 / resident | 16.348 | 0 | 0.000 | not measured | 0 / 0 |

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

| Trial / session | Independent finish (s) | Coordinated finish (s) | Resident finish (s) |
| --- | --- | --- | --- |
| 1 / swap-s00 | 14.450 | 24.407 | 7.506 |
| 1 / swap-s01 | 19.597 | 24.408 | 7.505 |
| 1 / swap-s02 | 14.451 | 24.408 | 7.506 |
| 1 / swap-s03 | 22.359 | 24.408 | 7.504 |
| 1 / swap-s04 | 14.451 | 24.406 | 7.505 |
| 1 / swap-s05 | 19.596 | 24.408 | 7.506 |
| 1 / swap-s06 | 22.360 | 24.408 | 7.506 |
| 1 / swap-s07 | 19.596 | 24.407 | 7.506 |
| 1 / swap-s08 | 19.598 | 24.408 | 7.505 |
| 1 / swap-s09 | 19.598 | 24.408 | 7.505 |
| 1 / swap-s10 | 14.451 | 29.396 | 8.563 |
| 1 / swap-s11 | 14.451 | 29.395 | 8.563 |
| 1 / swap-s12 | 19.595 | 29.396 | 8.563 |
| 1 / swap-s13 | 22.547 | 29.395 | 8.562 |
| 1 / swap-s14 | 22.547 | 29.395 | 8.563 |
| 1 / swap-s15 | 22.547 | 29.395 | 8.563 |
| 1 / swap-s16 | 22.547 | 29.395 | 8.563 |
| 1 / swap-s17 | 22.547 | 29.395 | 8.562 |
| 1 / swap-s18 | 22.546 | 29.395 | 8.562 |
| 1 / swap-s19 | 19.598 | 29.394 | 8.563 |

</details>

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Limits**

- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.
- Setup excluded and reported separately; measured control, swap and alignment delays included.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_swap_pilot_v2_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=32 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=direct \
SWAP_RESTORE_STYLE=serial \
SWAP_CONTROL_STYLE=individual \
SWAP_SCHEDULE_STYLE=barrier \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_lean \
SWAP_TRIALS=1 \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/summary.json) · [trial1_independent per-turn timings](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/arms/trial1_independent/case_results.json) · [trial1_independent trace](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/arms/trial1_independent/backend_trace.jsonl.gz) · [trial1_coordinated per-turn timings](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/arms/trial1_coordinated/case_results.json) · [trial1_coordinated trace](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/arms/trial1_coordinated/backend_trace.jsonl.gz) · [trial1_resident per-turn timings](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/arms/trial1_resident/case_results.json) · [trial1_resident trace](docs/reports/work_audit/coordinated_swap_pilot_v2_20261008/arms/trial1_resident/backend_trace.jsonl.gz)

</details>

<a id="run-coordinated_swap_pilot_20261008"></a>
<details>
<summary><strong>Oct 8, 2026, 1:51:36 p.m. CDT · Coordinated CPU/GPU swaps (calibration)</strong> · coordinated_swap_pilot_20261008</summary>

**Question (RQ21).** With twenty independent contexts that exceed the GPU cache budget, can perfectly staggered A/B and C/D pairs swap their KV during one-second tool waits and approach a GPU-resident reference?

**Finding.** Evidence incomplete; no performance conclusion. trial1_coordinated: RuntimeError: Restored-prefix diagnostic output mismatch

**Setup.** 20 separate sessions; 4 labels of 5; 3 tool rounds; 1,000 ms waits; paired AB/CD group rotation; restricted GPU/8 GiB host KV; 8192 initial prompt words and 16 new tool words per round; GPU token caps 110592 restricted / 262144 resident; 32 output tokens per replay; native direct KV transfer; serial restore submission; individual control calls; kv_lifecycle_lean tracing; no storage; CUDA graphs and overlap scheduling on. Model: Qwen/Qwen2.5-1.5B-Instruct; hardware: see manifest; backend version: see runtime record. Initial setup is excluded. The restricted KV budget is imposed on the same GPU, not a claim that all its physical memory was exhausted. Setup is excluded; real swap, control and late-submission time count. The resident reference has a larger GPU cache. Contexts are not merged.

**Key measurements**

No comparable measurements recorded.

No comparable measurements recorded.

Negative changes mean coordinated finished sooner or had less delay. Session finish times start at the measured workload start, not initial setup. TTFT, due delay, submission waiting and scheduled slot padding totals add time across requests; they are not elapsed workload time. Scheduled padding is the gap from each reply to its planned slot boundary, including the last slot's unused padding; whole-workload time stops at the actual last reply. Explicit load batches count prepare controls, not automatic loads in the independent mode. KV ready on time / late counts every coordinated replay in the session-pipeline setup; older barrier runs count only explicit restores. The modes deliberately use different residency policies; this is an optimistic comparison, not a production fairness test.

No comparable measurements recorded.

Control wall time includes waiting for the backend and checking its reply. The CUDA-stream interval can include gaps between launching copies; it is not a measurement of copy-engine busy time alone. These intervals can overlap control wall time, so do not add them together. Initial priming and diagnostics are reported as excluded setup, not hidden inside the workload duration.

<details>
<summary>Every session's finish time</summary>

No comparable measurements recorded.

</details>

**Evidence gate.** blocked. Timestamp: First request; displayed in Central Time.

**Limits**

- trial1_coordinated: RuntimeError: Restored-prefix diagnostic output mismatch
- Optimistic paired timeline; not a fair causal comparison of grouping alone.
- Twenty separate synthetic contexts; fixed tool-result text and forced output length.

**Reproduce** (set the container image and model cache for the target host):

```bash
SWAP_RUN_ID=coordinated_swap_pilot_20261008_repeat \
SWAP_TURNS=3 \
SWAP_DECODE_TOKENS=32 \
SWAP_INITIAL_TOKENS=8192 \
SWAP_GPU_TOKENS=110592 \
SWAP_RESIDENT_TOKENS=262144 \
SWAP_TOOL_WORDS=16 \
SWAP_IO_BACKEND=direct \
SWAP_RESTORE_STYLE=serial \
SWAP_CONTROL_STYLE=individual \
SWAP_SCHEDULE_STYLE=barrier \
SWAP_PREFETCH_LEAD_MS=750 \
SWAP_HEADROOM_SESSIONS=2 \
SWAP_MAX_INFLIGHT=8 \
SWAP_TRACE_PROFILE=kv_lifecycle_lean \
SWAP_TRIALS=1 \
SWAP_MODES='independent coordinated resident' \
bash infra/container/run_work_audit_coordinated_swap.sh
```

**Evidence:** [Summary](docs/reports/work_audit/coordinated_swap_pilot_20261008/summary.json)

</details>

<a id="run-rq20_selective_storage_pair_20261007"></a>
<details>
<summary><strong>Oct 7, 2026, 11:00:30 a.m. CDT · Selective storage staging across repeated sessions</strong> · rq20_selective_storage_pair_20261007</summary>

**Question (RQ20).** With repeated tool returns and natural file-backed KV displacement, can one-at-a-time storage-to-host staging during sufficiently long waits improve whole-workload completion and replay delay without shifting too much delay to other sessions?

**Finding.** Mixed result across 2 paired seeds: seed 1 whole workload -703 ms, median replay delay +62 ms; seed 2 whole workload +596 ms, median replay delay +89 ms. 3 stages finished before due time; there is no consistent whole-workload gain. Synthetic file-backed storage does not establish a physical-SSD or hardware benefit.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned SGLang 0.5.10.post1. Each session began near 2048 prompt tokens and gained 900 tool-result words per turn. Tool waits varied from 1000 to 4000 ms by a fixed seed. Fresh backend and file-cache path per arm; GPU KV cap 10240 tokens, host cache 1.0 GiB, 64-token pages. No explicit eviction or frontend priority. The staging arm observes natural residency during each wait; replay is submitted at its due time even if staging misses it. Selective staging starts inspection at 0.2 of the tool wait, requires at least 1200 ms until return, and permits one stage at a time. The adapter validates the suffix anchor, host capacity, and native rate limit before native prefetch. CUDA graphs on; overlap scheduling on. Full GPU preparation was not tested in this run.

**Key measurements**

| Seed | Mode | Sessions | Replays | Whole workload (ms) | Due → first token median (ms) | Due → first token p95 (ms) | Replay TTFT median (ms) | Storage candidates | Stages admitted | Stage ready by due | Stage after due | Stage skips | Tokens staged from L3 | Native L3 replay hits | L3 replay tokens | Stage errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | on_demand | 8 | 48 | 28545.6 | 1096.0 | 2050.6 | 1094.9 | 5 | 0 | 0 | 0 | 0 | 0 | 19 | 98112 | 0 |
| 1 | selective_stage | 8 | 48 | 27842.2 | 1158.2 | 2083.9 | 1157.0 | 5 | 2 | 2 | 0 | 3 | 11520 | 22 | 100800 | 0 |
| 2 | on_demand | 8 | 48 | 29978.5 | 847.2 | 2406.8 | 846.7 | 2 | 0 | 0 | 0 | 0 | 0 | 24 | 126016 | 0 |
| 2 | selective_stage | 8 | 48 | 30574.7 | 936.1 | 2406.2 | 935.4 | 3 | 1 | 1 | 0 | 2 | 6720 | 24 | 126016 | 0 |

| Seed | Session | Finish-time change (ms) |
| --- | --- | --- |
| 1 | storagecycles-seed1-session0 | -10.8 |
| 1 | storagecycles-seed1-session1 | -703.5 |
| 1 | storagecycles-seed1-session2 | 1132.7 |
| 1 | storagecycles-seed1-session3 | -818.5 |
| 1 | storagecycles-seed1-session4 | -252.2 |
| 1 | storagecycles-seed1-session5 | -611.0 |
| 1 | storagecycles-seed1-session6 | -248.0 |
| 1 | storagecycles-seed1-session7 | 540.1 |
| 2 | storagecycles-seed2-session0 | 6.9 |
| 2 | storagecycles-seed2-session1 | 596.2 |
| 2 | storagecycles-seed2-session2 | -241.2 |
| 2 | storagecycles-seed2-session3 | 587.2 |
| 2 | storagecycles-seed2-session4 | 50.3 |
| 2 | storagecycles-seed2-session5 | -47.2 |
| 2 | storagecycles-seed2-session6 | -65.1 |
| 2 | storagecycles-seed2-session7 | 1749.7 |

| Seed | Early-staged replay | Baseline delay (ms) | Staged delay (ms) | Delay change (ms) | Tokens staged | Baseline replay L3 tokens | Staged replay L3 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | storagecycles-seed1-session0-turn05 | 2360.1 | 2148.7 | -211.4 | 5760 | 5760 | 5760 |
| 1 | storagecycles-seed1-session1-turn05 | 1913.5 | 2495.7 | 582.2 | 5760 | 5760 | 5760 |
| 2 | storagecycles-seed2-session3-turn06 | 1180.3 | 1685.1 | 504.7 | 6720 | 6720 | 6720 |

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Stage skip reasons.** host_capacity_insufficient (3), stage_already_in_flight (1), storage_prefetch_no_hit (1)

**Limits**

- file-backed storage is not an independently benchmarked physical SSD
- a missing-suffix residency observation is not proof of a physical SSD read
- requests are synthetic equal-priority coding sessions

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_CYCLES_RESEARCH_QUESTION_ID='RQ20' WORK_AUDIT_CYCLES_SEEDS='1 2' WORK_AUDIT_CYCLES_ARMS='on_demand selective_stage' WORK_AUDIT_CYCLES_SESSIONS='8' WORK_AUDIT_CYCLES_TURNS='6' WORK_AUDIT_CYCLES_INITIAL_TOKENS='2048' WORK_AUDIT_CYCLES_TOOL_WORDS='900' WORK_AUDIT_CYCLES_DECODE_TOKENS='16' WORK_AUDIT_CYCLES_WAIT_MS='1000' WORK_AUDIT_CYCLES_WAIT_SPREAD_MS='3000' WORK_AUDIT_CYCLES_STAGGER_MS='250' WORK_AUDIT_CYCLES_GPU_TOKENS='10240' WORK_AUDIT_CYCLES_HOST_GB='1' WORK_AUDIT_CYCLES_INSPECT_FRACTION='0.2' WORK_AUDIT_CYCLES_MIN_STAGE_SLACK_MS='1200' WORK_AUDIT_CYCLES_CUDA_GRAPH='1' WORK_AUDIT_CYCLES_OVERLAP_SCHEDULE='1' WORK_AUDIT_CYCLES_RUN_ID='new_unique_id' bash infra/container/run_work_audit_storage_cycles.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq20_selective_storage_pair_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq20_selective_storage_pair_20261007/run_manifest.json) · [seed1_on_demand per-turn timings](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed1_on_demand hook gate](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed1_on_demand/instrumentation_audit.json) · [seed1_selective_stage per-turn timings](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed1_selective_stage/case_results.json) · [seed1_selective_stage trace](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed1_selective_stage/backend_trace.jsonl.gz) · [seed1_selective_stage hook gate](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed1_selective_stage/instrumentation_audit.json) · [seed2_on_demand per-turn timings](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed2_on_demand/backend_trace.jsonl.gz) · [seed2_on_demand hook gate](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed2_on_demand/instrumentation_audit.json) · [seed2_selective_stage per-turn timings](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed2_selective_stage/case_results.json) · [seed2_selective_stage trace](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed2_selective_stage/backend_trace.jsonl.gz) · [seed2_selective_stage hook gate](docs/reports/work_audit/rq20_selective_storage_pair_20261007/arms/seed2_selective_stage/instrumentation_audit.json)

</details>

<a id="run-rq20_exposure_h1_g10240_w900_s6_20261007"></a>
<details>
<summary><strong>Oct 7, 2026, 10:57:02 a.m. CDT · Selective storage staging across repeated sessions</strong> · rq20_exposure_h1_g10240_w900_s6_20261007</summary>

**Question (RQ20).** With repeated tool returns and natural file-backed KV displacement, can one-at-a-time storage-to-host staging during sufficiently long waits improve whole-workload completion and replay delay without shifting too much delay to other sessions?

**Finding.** Calibration run only; no paired mode comparison.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned SGLang 0.5.10.post1. Each session began near 2048 prompt tokens and gained 900 tool-result words per turn. Tool waits varied from 1000 to 4000 ms by a fixed seed. Fresh backend and file-cache path per arm; GPU KV cap 10240 tokens, host cache 1.0 GiB, 64-token pages. No explicit eviction or frontend priority. The staging arm observes natural residency during each wait; replay is submitted at its due time even if staging misses it. The adapter validates the suffix anchor, host capacity, and native rate limit before native prefetch. CUDA graphs on; overlap scheduling on. Full GPU preparation was not tested in this run.

**Key measurements**

| Seed | Mode | Sessions | Replays | Whole workload (ms) | Due → first token median (ms) | Due → first token p95 (ms) | Replay TTFT median (ms) | Storage candidates | Stages admitted | Stage ready by due | Stage after due | Stage skips | Tokens staged from L3 | Native L3 replay hits | L3 replay tokens | Stage errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | on_demand | 6 | 36 | 24430.9 | 543.4 | 1355.6 | 542.3 | 1 | 0 | 0 | 0 | 0 | 0 | 5 | 14080 | 0 |

**Evidence gate.** calibration. Timestamp: First request; displayed in Central Time.

**Limits**

- file-backed storage is not an independently benchmarked physical SSD
- a missing-suffix residency observation is not proof of a physical SSD read
- requests are synthetic equal-priority coding sessions

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_CYCLES_RESEARCH_QUESTION_ID='RQ20' WORK_AUDIT_CYCLES_SEEDS='1' WORK_AUDIT_CYCLES_ARMS='on_demand' WORK_AUDIT_CYCLES_SESSIONS='6' WORK_AUDIT_CYCLES_TURNS='6' WORK_AUDIT_CYCLES_INITIAL_TOKENS='2048' WORK_AUDIT_CYCLES_TOOL_WORDS='900' WORK_AUDIT_CYCLES_DECODE_TOKENS='16' WORK_AUDIT_CYCLES_WAIT_MS='1000' WORK_AUDIT_CYCLES_WAIT_SPREAD_MS='3000' WORK_AUDIT_CYCLES_STAGGER_MS='250' WORK_AUDIT_CYCLES_GPU_TOKENS='10240' WORK_AUDIT_CYCLES_HOST_GB='1' WORK_AUDIT_CYCLES_INSPECT_FRACTION='0.4' WORK_AUDIT_CYCLES_MIN_STAGE_SLACK_MS='1200' WORK_AUDIT_CYCLES_CUDA_GRAPH='1' WORK_AUDIT_CYCLES_OVERLAP_SCHEDULE='1' WORK_AUDIT_CYCLES_RUN_ID='new_unique_id' bash infra/container/run_work_audit_storage_cycles.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq20_exposure_h1_g10240_w900_s6_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq20_exposure_h1_g10240_w900_s6_20261007/run_manifest.json) · [seed1_on_demand per-turn timings](docs/reports/work_audit/rq20_exposure_h1_g10240_w900_s6_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq20_exposure_h1_g10240_w900_s6_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed1_on_demand hook gate](docs/reports/work_audit/rq20_exposure_h1_g10240_w900_s6_20261007/arms/seed1_on_demand/instrumentation_audit.json)

</details>

<a id="run-rq19_control_placebo_n4_s2_20261007"></a>
<details>
<summary><strong>Oct 7, 2026, 9:45:46 a.m. CDT · Storage staging and peer-delay control</strong> · rq19_control_placebo_n4_s2_20261007</summary>

**Question (RQ19).** When early storage-to-host KV staging delays peer requests, how much is associated with the scheduler-control path versus the native staging action, and what explains the apparent delay from storage data readiness to cache availability?

**Finding.** Three-way control: peer median first-token time was 1159 ms without early activity, 1174 ms with control probes only, and 1237 ms with actual KV staging. This small synthetic comparison locates an effect; it does not prove a hardware cause.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The control-only arm made three residency checks without transferring KV; the host-stage arm fetched KV from L3 to L2 before tool return. There were 3 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on. Matching replay output hashes were required across arms.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | control_only | 351.9 | 348.9 | 1174.4 | 2607.3 | 0 | not recorded | not recorded | 6559.1 | 2048 | 0 | 2112 | not applicable |
| 2 | host_stage | 163.0 | 160.0 | 1237.1 | 2666.9 | 3 | 83.5 | 910.8 | 6409.3 | 0 | 2048 | 2112 | yes |
| 2 | on_demand | 352.8 | 349.4 | 1158.7 | 2587.7 | 0 | not recorded | not recorded | 6621.3 | 2048 | 0 | 2112 | not applicable |

| Seed | Arm | Peer request → cache lookup median (ms) | Peer cache lookup → first token median (ms) |
| --- | --- | --- | --- |
| 2 | control_only | 49.2 | 1113.3 |
| 2 | host_stage | 96.0 | 1126.5 |
| 2 | on_demand | 49.9 | 1100.3 |

| Seed | Data ready → poll queued (ms) | Poll queued → dequeued (ms) | Poll execution (ms) |
| --- | --- | --- | --- |
| 2 | 183.2 | 721.8 | 5.8 |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. The three status checks approximate the staged arm's control-call pattern but do not execute native prefetch. The status-commit timestamp is recorded when that check runs; it is not an independent timestamp of first host-cache usability. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='2' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ19 WORK_AUDIT_STORAGE_PROMPT_ID=rq19_control_placebo_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=3 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand control_only host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 WORK_AUDIT_STORAGE_VERIFY_OUTPUT=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/run_manifest.json) · [seed2_control_only timings](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/arms/seed2_control_only/case_results.json) · [seed2_control_only trace](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/arms/seed2_control_only/backend_trace.jsonl.gz) · [seed2_host_stage timings](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/arms/seed2_host_stage/case_results.json) · [seed2_host_stage trace](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/arms/seed2_host_stage/backend_trace.jsonl.gz) · [seed2_on_demand timings](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/rq19_control_placebo_n4_s2_20261007/arms/seed2_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq19_control_placebo_n4_s1_20261007"></a>
<details>
<summary><strong>Oct 7, 2026, 9:38:46 a.m. CDT · Storage staging and peer-delay control</strong> · rq19_control_placebo_n4_s1_20261007</summary>

**Question (RQ19).** When early storage-to-host KV staging delays peer requests, how much is associated with the scheduler-control path versus the native staging action, and what explains the apparent delay from storage data readiness to cache availability?

**Finding.** Three-way control: peer median first-token time was 1157 ms without early activity, 1176 ms with control probes only, and 1216 ms with actual KV staging. This small synthetic comparison locates an effect; it does not prove a hardware cause.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The control-only arm made three residency checks without transferring KV; the host-stage arm fetched KV from L3 to L2 before tool return. There were 3 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on. Matching replay output hashes were required across arms.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | control_only | 319.0 | 316.2 | 1176.4 | 2618.4 | 0 | not recorded | not recorded | 6529.8 | 2048 | 0 | 2112 | not applicable |
| 1 | host_stage | 161.7 | 158.6 | 1216.3 | 2647.1 | 3 | 76.0 | 896.3 | 6366.8 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 379.5 | 376.8 | 1157.0 | 2585.1 | 0 | not recorded | not recorded | 6591.0 | 2048 | 0 | 2112 | not applicable |

| Seed | Arm | Peer request → cache lookup median (ms) | Peer cache lookup → first token median (ms) |
| --- | --- | --- | --- |
| 1 | control_only | 47.9 | 1116.5 |
| 1 | host_stage | 85.1 | 1116.9 |
| 1 | on_demand | 44.8 | 1101.4 |

| Seed | Data ready → poll queued (ms) | Poll queued → dequeued (ms) | Poll execution (ms) |
| --- | --- | --- | --- |
| 1 | 188.9 | 701.7 | 5.7 |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. The three status checks approximate the staged arm's control-call pattern but do not execute native prefetch. The status-commit timestamp is recorded when that check runs; it is not an independent timestamp of first host-cache usability. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ19 WORK_AUDIT_STORAGE_PROMPT_ID=rq19_control_placebo_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=3 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand control_only host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 WORK_AUDIT_STORAGE_VERIFY_OUTPUT=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/run_manifest.json) · [seed1_control_only timings](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/arms/seed1_control_only/case_results.json) · [seed1_control_only trace](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/arms/seed1_control_only/backend_trace.jsonl.gz) · [seed1_host_stage timings](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq19_control_placebo_n4_s1_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq18_guarded_cycles_retry_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:50:41 p.m. CDT · Repeated storage-resume timing</strong> · rq18_guarded_cycles_retry_20261007</summary>

**Question (RQ17).** With eight equal-priority agent sessions, six tool returns each, growing prompt histories, and natural GPU/host cache displacement, can storage-to-host staging during known tool waits improve replay delay and whole-workload time without harming other sessions?

**Finding.** Across 1 paired seeds, host staging changed median replay delay by -8 ms and whole-workload duration by +131 ms (negative is faster). On-demand replay had 15 native storage hits; staging completed before due time on 1 waits. These are associations in a small synthetic, file-backed workload, not an isolated physical-SSD or hardware speedup.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned SGLang 0.5.10.post1. Each session began near 2048 prompt tokens and gained 900 tool-result words per turn. Tool waits varied from 1000 to 4000 ms by a fixed seed. Fresh backend and file-cache path per arm; GPU KV cap 10240 tokens, host cache 1 GiB, 64-token pages. No explicit eviction or frontend priority. The staging arm observes natural residency during each wait; replay is submitted at its due time even if staging misses it. The adapter validates the suffix anchor, host capacity, and native rate limit before native prefetch. CUDA graphs on; overlap scheduling on. Full GPU preparation was not tested in this run.

**Key measurements**

| Seed | Mode | Sessions | Replays | Whole workload (ms) | Due → first token median (ms) | Due → first token p95 (ms) | Replay TTFT median (ms) | Storage candidates | Stages admitted | Stage ready by due | Stage after due | Stage skips | Tokens staged from L3 | Native L3 replay hits | L3 replay tokens | Stage errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | on_demand | 8 | 48 | 27336.2 | 1046.1 | 1873.2 | 1045.2 | 9 | 0 | 0 | 0 | 0 | 0 | 15 | 73088 | 0 |
| 1 | host_stage | 8 | 48 | 27467.6 | 1038.1 | 1911.5 | 1037.0 | 9 | 0 | 1 | 0 | 8 | 6720 | 13 | 66368 | 0 |

**Evidence gate.** complete. Timestamp: First request; displayed in Central Time.

**Stage skip reasons.** host_capacity_insufficient (8)

**Limits**

- file-backed storage is not an independently benchmarked physical SSD
- a missing-suffix residency observation is not proof of a physical SSD read
- requests are synthetic equal-priority coding sessions

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_CYCLES_RESEARCH_QUESTION_ID='RQ17' WORK_AUDIT_CYCLES_SEEDS='1' WORK_AUDIT_CYCLES_ARMS='on_demand host_stage' WORK_AUDIT_CYCLES_SESSIONS='8' WORK_AUDIT_CYCLES_TURNS='6' WORK_AUDIT_CYCLES_INITIAL_TOKENS='2048' WORK_AUDIT_CYCLES_TOOL_WORDS='900' WORK_AUDIT_CYCLES_DECODE_TOKENS='16' WORK_AUDIT_CYCLES_WAIT_MS='1000' WORK_AUDIT_CYCLES_WAIT_SPREAD_MS='3000' WORK_AUDIT_CYCLES_STAGGER_MS='250' WORK_AUDIT_CYCLES_GPU_TOKENS='10240' WORK_AUDIT_CYCLES_HOST_GB='1' WORK_AUDIT_CYCLES_CUDA_GRAPH='1' WORK_AUDIT_CYCLES_OVERLAP_SCHEDULE='1' WORK_AUDIT_CYCLES_RUN_ID='new_unique_id' bash infra/container/run_work_audit_storage_cycles.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/run_manifest.json) · [seed1_on_demand per-turn timings](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed1_on_demand hook gate](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/arms/seed1_on_demand/instrumentation_audit.json) · [seed1_host_stage per-turn timings](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_host_stage hook gate](docs/reports/work_audit/rq18_guarded_cycles_retry_20261007/arms/seed1_host_stage/instrumentation_audit.json)

</details>

<a id="run-rq18_guarded_cycles_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:43:07 p.m. CDT · Repeated storage-resume timing</strong> · rq18_guarded_cycles_20261007</summary>

**Question (RQ17).** With eight equal-priority agent sessions, six tool returns each, growing prompt histories, and natural GPU/host cache displacement, can storage-to-host staging during known tool waits improve replay delay and whole-workload time without harming other sessions?

**Finding.** Blocked after one completed pair: host staging changed median replay delay by -11.2 ms and whole-workload time by +394.2 ms. Native prefetch was declined once; an explicit rate-limit check was added afterward. This one-pair observation is not a validated performance conclusion.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned SGLang 0.5.10.post1. Each session began near 2048 prompt tokens and gained 900 tool-result words per turn. Tool waits varied from 1000 to 4000 ms by a fixed seed. Fresh backend and file-cache path per arm; GPU KV cap 10240 tokens, host cache 1 GiB, 64-token pages. No explicit eviction or frontend priority. The staging arm observes natural residency during each wait; replay is submitted at its due time even if staging misses it. The run checked suffix anchor and host capacity, but did not yet classify a native rate-limit refusal; the current adapter checks that limit too. CUDA graphs on; overlap scheduling on. Full GPU preparation was not tested in this run.

**Key measurements**

| Seed | Mode | Sessions | Replays | Whole workload (ms) | Due → first token median (ms) | Due → first token p95 (ms) | Replay TTFT median (ms) | Storage candidates | Stages admitted | Stage ready by due | Stage after due | Stage skips | Tokens staged from L3 | Native L3 replay hits | L3 replay tokens | Stage errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | on_demand | 8 | 48 | 28831.5 | 1083.9 | 2192.8 | 1082.6 | 11 | 0 | 0 | 0 | 0 | 0 | 18 | 98048 | 0 |
| 1 | host_stage | 8 | 48 | 29225.6 | 1072.8 | 2448.4 | 1072.3 | 11 | 0 | 2 | 0 | 8 | 12480 | 17 | 89280 | 1 |

**Evidence gate.** blocked. Timestamp: First request; displayed in Central Time.

**Stage skip reasons.** host_capacity_insufficient (8)

**Limits**

- file-backed storage is not an independently benchmarked physical SSD
- a missing-suffix residency observation is not proof of a physical SSD read
- requests are synthetic equal-priority coding sessions

**Reproduction note.** The current adapter checks native rate limiting before prefetch; this command will not reproduce the earlier unclassified refusal exactly.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_CYCLES_RESEARCH_QUESTION_ID='RQ17' WORK_AUDIT_CYCLES_SEEDS='1' WORK_AUDIT_CYCLES_ARMS='on_demand host_stage' WORK_AUDIT_CYCLES_SESSIONS='8' WORK_AUDIT_CYCLES_TURNS='6' WORK_AUDIT_CYCLES_INITIAL_TOKENS='2048' WORK_AUDIT_CYCLES_TOOL_WORDS='900' WORK_AUDIT_CYCLES_DECODE_TOKENS='16' WORK_AUDIT_CYCLES_WAIT_MS='1000' WORK_AUDIT_CYCLES_WAIT_SPREAD_MS='3000' WORK_AUDIT_CYCLES_STAGGER_MS='250' WORK_AUDIT_CYCLES_GPU_TOKENS='10240' WORK_AUDIT_CYCLES_HOST_GB='1' WORK_AUDIT_CYCLES_CUDA_GRAPH='1' WORK_AUDIT_CYCLES_OVERLAP_SCHEDULE='1' WORK_AUDIT_CYCLES_RUN_ID='new_unique_id' bash infra/container/run_work_audit_storage_cycles.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_guarded_cycles_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_guarded_cycles_20261007/run_manifest.json) · [seed1_on_demand per-turn timings](docs/reports/work_audit/rq18_guarded_cycles_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq18_guarded_cycles_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed1_on_demand hook gate](docs/reports/work_audit/rq18_guarded_cycles_20261007/arms/seed1_on_demand/instrumentation_audit.json) · [seed1_host_stage per-turn timings](docs/reports/work_audit/rq18_guarded_cycles_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq18_guarded_cycles_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_host_stage hook gate](docs/reports/work_audit/rq18_guarded_cycles_20261007/arms/seed1_host_stage/instrumentation_audit.json)

</details>

<a id="run-rq18_safe_ladder_n8_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:40:39 p.m. CDT · Safe storage-stage session ladder</strong> · rq18_safe_ladder_n8_20261007</summary>

**Question (RQ18).** If a returning session's storage-resident KV is staged during its tool wait, how do its replay delay, peer latency, and whole-workload time change as equal-priority session count rises?

**Finding.** Native storage-only KV and replay reuse were verified. First-token delay changed from 325 to 168 ms with host staging across 1 paired seed(s). Peer timing and whole-workload changes are recorded separately; a single pair is not enough to claim a win-win.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 7 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on. Matching replay output hashes were required across arms.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | host_stage | 168.1 | 165.8 | 1437.9 | 3154.5 | 7 | 73.8 | 872.4 | 6432.5 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 324.8 | 322.6 | 1326.3 | 3053.3 | 0 | not recorded | not recorded | 6548.2 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ18 WORK_AUDIT_STORAGE_PROMPT_ID=rq18_safe_ladder_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=7 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 WORK_AUDIT_STORAGE_VERIFY_OUTPUT=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_safe_ladder_n8_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_safe_ladder_n8_20261007/run_manifest.json) · [seed1_host_stage timings](docs/reports/work_audit/rq18_safe_ladder_n8_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq18_safe_ladder_n8_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/rq18_safe_ladder_n8_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq18_safe_ladder_n8_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq18_safe_ladder_n4_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:35:33 p.m. CDT · Safe storage-stage session ladder</strong> · rq18_safe_ladder_n4_20261007</summary>

**Question (RQ18).** If a returning session's storage-resident KV is staged during its tool wait, how do its replay delay, peer latency, and whole-workload time change as equal-priority session count rises?

**Finding.** Native storage-only KV and replay reuse were verified. First-token delay changed from 376 to 170 ms with host staging across 1 paired seed(s). Peer timing and whole-workload changes are recorded separately; a single pair is not enough to claim a win-win.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 3 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on. Matching replay output hashes were required across arms.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | host_stage | 170.1 | 167.1 | 1223.1 | 2653.9 | 3 | 75.0 | 896.0 | 6405.0 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 376.3 | 373.7 | 1167.8 | 2598.1 | 0 | not recorded | not recorded | 6594.9 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ18 WORK_AUDIT_STORAGE_PROMPT_ID=rq18_safe_ladder_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=3 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 WORK_AUDIT_STORAGE_VERIFY_OUTPUT=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_safe_ladder_n4_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_safe_ladder_n4_20261007/run_manifest.json) · [seed1_host_stage timings](docs/reports/work_audit/rq18_safe_ladder_n4_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq18_safe_ladder_n4_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/rq18_safe_ladder_n4_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq18_safe_ladder_n4_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq18_safe_ladder_n2_s2_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:29:59 p.m. CDT · Safe storage-stage session ladder</strong> · rq18_safe_ladder_n2_s2_20261007</summary>

**Question (RQ18).** If a returning session's storage-resident KV is staged during its tool wait, how do its replay delay, peer latency, and whole-workload time change as equal-priority session count rises?

**Finding.** Native storage-only KV and replay reuse were verified. First-token delay changed from 348 to 162 ms with host staging across 1 paired seed(s). Peer timing and whole-workload changes are recorded separately; a single pair is not enough to claim a win-win.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 1 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | host_stage | 162.1 | 160.4 | 136.9 | 1144.8 | 1 | 76.9 | 200.0 | 6368.0 | 0 | 2048 | 2112 | yes |
| 2 | on_demand | 347.7 | 342.6 | 97.7 | 1084.4 | 0 | not recorded | not recorded | 6517.5 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='2' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ18 WORK_AUDIT_STORAGE_PROMPT_ID=rq18_safe_ladder_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=1 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_safe_ladder_n2_s2_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_safe_ladder_n2_s2_20261007/run_manifest.json) · [seed2_host_stage timings](docs/reports/work_audit/rq18_safe_ladder_n2_s2_20261007/arms/seed2_host_stage/case_results.json) · [seed2_host_stage trace](docs/reports/work_audit/rq18_safe_ladder_n2_s2_20261007/arms/seed2_host_stage/backend_trace.jsonl.gz) · [seed2_on_demand timings](docs/reports/work_audit/rq18_safe_ladder_n2_s2_20261007/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/rq18_safe_ladder_n2_s2_20261007/arms/seed2_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq18_safe_ladder_n2_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:24:59 p.m. CDT · Safe storage-stage session ladder</strong> · rq18_safe_ladder_n2_20261007</summary>

**Question (RQ18).** If a returning session's storage-resident KV is staged during its tool wait, how do its replay delay, peer latency, and whole-workload time change as equal-priority session count rises?

**Finding.** Native storage-only KV and replay reuse were verified. First-token delay changed from 374 to 167 ms with host staging across 1 paired seed(s). Peer timing and whole-workload changes are recorded separately; a single pair is not enough to claim a win-win.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 1 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | host_stage | 166.8 | 162.1 | 145.3 | 1149.0 | 1 | 86.4 | 193.4 | 6401.6 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 373.6 | 369.0 | 97.9 | 1084.9 | 0 | not recorded | not recorded | 6574.2 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ18 WORK_AUDIT_STORAGE_PROMPT_ID=rq18_safe_ladder_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=1 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_safe_ladder_n2_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_safe_ladder_n2_20261007/run_manifest.json) · [seed1_host_stage timings](docs/reports/work_audit/rq18_safe_ladder_n2_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq18_safe_ladder_n2_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/rq18_safe_ladder_n2_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq18_safe_ladder_n2_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq18_safe_ladder_n1b_20261007"></a>
<details>
<summary><strong>Oct 6, 2026, 11:19:42 p.m. CDT · Safe storage-stage session ladder</strong> · rq18_safe_ladder_n1b_20261007</summary>

**Question (RQ18).** If a returning session's storage-resident KV is staged during its tool wait, how do its replay delay, peer latency, and whole-workload time change as equal-priority session count rises?

**Finding.** Native storage-only KV and replay reuse were verified. First-token delay changed from 317 to 167 ms with host staging across 1 paired seed(s). No competing session was present; this only validates the load path.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 0 peer session(s), started 0 ms into the tool wait. CUDA graphs on; overlap scheduling on.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | host_stage | 167.2 | 162.0 | not recorded | not recorded | 0 | 269.6 | 58.1 | 6625.8 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 317.4 | 312.2 | not recorded | not recorded | 0 | not recorded | not recorded | 6762.4 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Single-session storage timing only. Peer-session cost and a busy-workload benefit are not measured. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ18 WORK_AUDIT_STORAGE_PROMPT_ID=rq18_safe_ladder_n1b_20261007 WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=0 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 WORK_AUDIT_STORAGE_ARMS='on_demand host_stage' WORK_AUDIT_STORAGE_CUDA_GRAPH=1 WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE=1 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/rq18_safe_ladder_n1b_20261007/summary.json) · [Run manifest](docs/reports/work_audit/rq18_safe_ladder_n1b_20261007/run_manifest.json) · [seed1_host_stage timings](docs/reports/work_audit/rq18_safe_ladder_n1b_20261007/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/rq18_safe_ladder_n1b_20261007/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/rq18_safe_ladder_n1b_20261007/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/rq18_safe_ladder_n1b_20261007/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_cycles_rq17_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 9:59:24 p.m. CDT · Repeated storage-resume timing</strong> · storage_cycles_rq17_20261006</summary>

**Question (RQ17).** With eight equal-priority agent sessions, six tool returns each, growing prompt histories, and natural GPU/host cache displacement, can storage-to-host staging during known tool waits improve replay delay and whole-workload time without harming other sessions?

**Finding.** Blocked after one completed pair: host staging changed median replay delay by -3.9 ms and whole-workload time by +1000.1 ms. Native prefetch was declined once; an explicit rate-limit check was added afterward. This one-pair observation is not a validated performance conclusion.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned SGLang 0.5.10.post1. Each session began near 2048 prompt tokens and gained 900 tool-result words per turn. Tool waits varied from 1000 to 4000 ms by a fixed seed. Fresh backend and file-cache path per arm; GPU KV cap 10240 tokens, host cache 1 GiB, 64-token pages. No explicit eviction or frontend priority. The staging arm observes natural residency during each wait; replay is submitted at its due time even if staging misses it. The historical blocked run used an unguarded partial-suffix prefetch; the current adapter refuses prefetch when it would require host eviction. A separate full-GPU-prepare pilot hit a scheduler assertion and is excluded.

**Key measurements**

| Seed | Mode | Sessions | Replays | Whole workload (ms) | Due → first token median (ms) | Due → first token p95 (ms) | Replay TTFT median (ms) | Storage candidates | Stages admitted | Stage ready by due | Stage after due | Stage skips | Tokens staged from L3 | Native L3 replay hits | L3 replay tokens | Stage errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | on_demand | 8 | 48 | 43152.9 | 1705.9 | 4358.7 | 1705.1 | 10 | 0 | 0 | 0 | 0 | 0 | 19 | 94720 | 0 |
| 1 | host_stage | 8 | 48 | 44153.0 | 1702.0 | 4843.8 | 1701.4 | 11 | 0 | 9 | 0 | 0 | 38400 | 16 | 76672 | 2 |

**Evidence gate.** blocked. Timestamp: First request; displayed in Central Time.

**Limits**

- file-backed storage is not an independently benchmarked physical SSD
- a missing-suffix residency observation is not proof of a physical SSD read
- requests are synthetic equal-priority coding sessions

**Safety note.** The failed partial-prefetch behavior is intentionally unavailable in current code. This command runs the guarded safe variant, not an exact replay of the failed prototype.

**Reproduction note.** The current adapter checks native rate limiting before prefetch; this command will not reproduce the earlier unclassified refusal exactly.

**Run guarded variant** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_CYCLES_RESEARCH_QUESTION_ID='RQ17' WORK_AUDIT_CYCLES_SEEDS='1 2 3' WORK_AUDIT_CYCLES_ARMS='on_demand host_stage' WORK_AUDIT_CYCLES_SESSIONS='8' WORK_AUDIT_CYCLES_TURNS='6' WORK_AUDIT_CYCLES_INITIAL_TOKENS='2048' WORK_AUDIT_CYCLES_TOOL_WORDS='900' WORK_AUDIT_CYCLES_DECODE_TOKENS='16' WORK_AUDIT_CYCLES_WAIT_MS='1000' WORK_AUDIT_CYCLES_WAIT_SPREAD_MS='3000' WORK_AUDIT_CYCLES_STAGGER_MS='250' WORK_AUDIT_CYCLES_GPU_TOKENS='10240' WORK_AUDIT_CYCLES_HOST_GB='1' WORK_AUDIT_CYCLES_RUN_ID='new_unique_id' bash infra/container/run_work_audit_storage_cycles.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_cycles_rq17_20261006/summary.json) · [Run manifest](docs/reports/work_audit/storage_cycles_rq17_20261006/run_manifest.json) · [seed1_on_demand per-turn timings](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed1_on_demand hook gate](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed1_on_demand/instrumentation_audit.json) · [seed1_host_stage per-turn timings](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_host_stage hook gate](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed1_host_stage/instrumentation_audit.json) · [seed2_host_stage server failure](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed2_host_stage/server.log) · [seed2_host_stage partial trace](docs/reports/work_audit/storage_cycles_rq17_20261006/arms/seed2_host_stage/backend_trace.jsonl.gz) · [Full-prepare pilot failure](docs/reports/work_audit/storage_cycles_full_pilot_20261006/arms/seed1_full_prepare/server.log)

</details>

<a id="run-storage_pressure_20261006_peers6"></a>
<details>
<summary><strong>Oct 6, 2026, 7:40:44 p.m. CDT · Storage-tier KV timing</strong> · storage_pressure_20261006_peers6</summary>

**Question (RQ16).** With a storage-resident target prefix and equal-priority peer requests, does preparing KV during a five-second tool wait still improve target replay as concurrent peer pressure rises, and what do peers pay?

**Finding.** With storage-only KV proven, first token after tool due was 468 ms on demand, 168 ms after host staging, and 59 ms after full preparation (median across 2 paired seed(s)). Peers overlapped preparation; their median TTFT changed from 1026 to 1184 ms. This is not a proven win-win. Native data readiness took a median 80 ms; the later status-poll/host-commit interval took 585 ms.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 6 peer session(s), started 0 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full_prepare | 59.0 | 57.5 | 1185.5 | 3148.7 | 6 | 78.9 | 584.6 | 15950.2 | 0 | 2048 | 2112 | yes |
| 1 | host_stage | 166.6 | 164.2 | 1075.2 | 3013.6 | 6 | 82.0 | 585.2 | 16119.2 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 432.9 | 430.6 | 1023.4 | 2978.1 | 0 | not recorded | not recorded | 16405.4 | 2048 | 0 | 2112 | not applicable |
| 2 | full_prepare | 59.9 | 56.8 | 1182.0 | 3151.6 | 6 | 88.4 | 576.8 | 16138.9 | 0 | 2048 | 2112 | yes |
| 2 | host_stage | 169.9 | 168.8 | 1075.0 | 3083.9 | 6 | 76.1 | 586.0 | 16097.2 | 0 | 2048 | 2112 | yes |
| 2 | on_demand | 502.7 | 499.9 | 1028.7 | 2947.8 | 0 | not recorded | not recorded | 16484.3 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1 2' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ16 WORK_AUDIT_STORAGE_PROMPT_ID=storage_pressure_20261006_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=6 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_pressure_20261006_peers6/summary.json) · [Run manifest](docs/reports/work_audit/storage_pressure_20261006_peers6/run_manifest.json) · [seed1_full_prepare timings](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed1_full_prepare/case_results.json) · [seed1_full_prepare trace](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed1_full_prepare/backend_trace.jsonl.gz) · [seed1_host_stage timings](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed2_full_prepare timings](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed2_full_prepare/case_results.json) · [seed2_full_prepare trace](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed2_full_prepare/backend_trace.jsonl.gz) · [seed2_host_stage timings](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed2_host_stage/case_results.json) · [seed2_host_stage trace](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed2_host_stage/backend_trace.jsonl.gz) · [seed2_on_demand timings](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/storage_pressure_20261006_peers6/arms/seed2_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_pressure_20261006_peers2"></a>
<details>
<summary><strong>Oct 6, 2026, 7:29:17 p.m. CDT · Storage-tier KV timing</strong> · storage_pressure_20261006_peers2</summary>

**Question (RQ16).** With a storage-resident target prefix and equal-priority peer requests, does preparing KV during a five-second tool wait still improve target replay as concurrent peer pressure rises, and what do peers pay?

**Finding.** With storage-only KV proven, first token after tool due was 364 ms on demand, 169 ms after host staging, and 61 ms after full preparation (median across 2 paired seed(s)). Peers overlapped preparation; their median TTFT changed from 157 to 456 ms. This is not a proven win-win. Native data readiness took a median 85 ms; the later status-poll/host-commit interval took 326 ms.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 2 peer session(s), started 0 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full_prepare | 61.0 | 57.3 | 711.6 | 2451.5 | 2 | 96.0 | 448.3 | 15927.8 | 0 | 2048 | 2112 | yes |
| 1 | host_stage | 172.3 | 168.1 | 199.9 | 1979.0 | 2 | 82.9 | 191.2 | 16170.0 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 316.0 | 312.2 | 155.3 | 1943.4 | 0 | not recorded | not recorded | 16097.3 | 2048 | 0 | 2112 | not applicable |
| 2 | full_prepare | 60.3 | 56.3 | 200.5 | 2160.0 | 2 | 78.1 | 208.5 | 16036.0 | 0 | 2048 | 2112 | yes |
| 2 | host_stage | 166.7 | 163.3 | 585.6 | 2337.3 | 2 | 86.5 | 444.1 | 16111.0 | 0 | 2048 | 2112 | yes |
| 2 | on_demand | 411.1 | 407.8 | 157.9 | 1903.4 | 0 | not recorded | not recorded | 16292.9 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1 2' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ16 WORK_AUDIT_STORAGE_PROMPT_ID=storage_pressure_20261006_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=2 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_pressure_20261006_peers2/summary.json) · [Run manifest](docs/reports/work_audit/storage_pressure_20261006_peers2/run_manifest.json) · [seed1_full_prepare timings](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed1_full_prepare/case_results.json) · [seed1_full_prepare trace](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed1_full_prepare/backend_trace.jsonl.gz) · [seed1_host_stage timings](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed2_full_prepare timings](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed2_full_prepare/case_results.json) · [seed2_full_prepare trace](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed2_full_prepare/backend_trace.jsonl.gz) · [seed2_host_stage timings](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed2_host_stage/case_results.json) · [seed2_host_stage trace](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed2_host_stage/backend_trace.jsonl.gz) · [seed2_on_demand timings](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/storage_pressure_20261006_peers2/arms/seed2_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_pressure_20261006_peers0"></a>
<details>
<summary><strong>Oct 6, 2026, 7:17:50 p.m. CDT · Storage-tier KV timing</strong> · storage_pressure_20261006_peers0</summary>

**Question (RQ16).** With a storage-resident target prefix and equal-priority peer requests, does preparing KV during a five-second tool wait still improve target replay as concurrent peer pressure rises, and what do peers pay?

**Finding.** With storage-only KV proven, first token after tool due was 370 ms on demand, 171 ms after host staging, and 60 ms after full preparation (median across 2 paired seed(s)). No peer-session or whole-system benefit is established by this single-session run. Native data readiness took a median 187 ms; the later status-poll/host-commit interval took 105 ms.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 0 peer session(s), started 0 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full_prepare | 61.0 | 56.7 | not recorded | not recorded | 0 | 199.4 | 121.8 | 16103.4 | 0 | 2048 | 2112 | yes |
| 1 | host_stage | 170.3 | 166.5 | not recorded | not recorded | 0 | 169.5 | 111.7 | 16257.0 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 355.0 | 349.7 | not recorded | not recorded | 0 | not recorded | not recorded | 16255.6 | 2048 | 0 | 2112 | not applicable |
| 2 | full_prepare | 58.3 | 56.6 | not recorded | not recorded | 0 | 174.3 | 99.1 | 15922.4 | 0 | 2048 | 2112 | yes |
| 2 | host_stage | 170.8 | 165.0 | not recorded | not recorded | 0 | 223.2 | 63.8 | 16263.7 | 0 | 2048 | 2112 | yes |
| 2 | on_demand | 385.7 | 380.6 | not recorded | not recorded | 0 | not recorded | not recorded | 16376.7 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Single-session storage timing only. Peer-session cost and a busy-workload benefit are not measured. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1 2' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ16 WORK_AUDIT_STORAGE_PROMPT_ID=storage_pressure_20261006_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=0 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_pressure_20261006_peers0/summary.json) · [Run manifest](docs/reports/work_audit/storage_pressure_20261006_peers0/run_manifest.json) · [seed1_full_prepare timings](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed1_full_prepare/case_results.json) · [seed1_full_prepare trace](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed1_full_prepare/backend_trace.jsonl.gz) · [seed1_host_stage timings](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed1_on_demand/backend_trace.jsonl.gz) · [seed2_full_prepare timings](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed2_full_prepare/case_results.json) · [seed2_full_prepare trace](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed2_full_prepare/backend_trace.jsonl.gz) · [seed2_host_stage timings](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed2_host_stage/case_results.json) · [seed2_host_stage trace](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed2_host_stage/backend_trace.jsonl.gz) · [seed2_on_demand timings](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/storage_pressure_20261006_peers0/arms/seed2_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_native_gate_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 7:06:07 p.m. CDT · Storage-tier KV timing</strong> · storage_native_gate_20261006</summary>

**Question (RQ16).** With a storage-resident target prefix and equal-priority peer requests, does preparing KV during a five-second tool wait still improve target replay as concurrent peer pressure rises, and what do peers pay?

**Finding.** With storage-only KV proven, first token after tool due was 369 ms on demand, 170 ms after host staging, and 64 ms after full preparation (median across 1 paired seed(s)). No peer-session or whole-system benefit is established by this single-session run. Native data readiness took a median 262 ms; the later status-poll/host-commit interval took 62 ms.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 0 peer session(s), started 0 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full_prepare | 64.2 | 59.1 | not recorded | not recorded | 0 | 264.9 | 91.3 | 15940.6 | 0 | 2048 | 2112 | yes |
| 1 | host_stage | 170.2 | 164.6 | not recorded | not recorded | 0 | 258.5 | 32.6 | 16102.0 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 369.3 | 365.9 | not recorded | not recorded | 0 | not recorded | not recorded | 16260.5 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Single-session storage timing only. Peer-session cost and a busy-workload benefit are not measured. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ16 WORK_AUDIT_STORAGE_PROMPT_ID=storage_pressure_20261006_shared WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=0 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_native_gate_20261006/summary.json) · [Run manifest](docs/reports/work_audit/storage_native_gate_20261006/run_manifest.json) · [seed1_full_prepare timings](docs/reports/work_audit/storage_native_gate_20261006/arms/seed1_full_prepare/case_results.json) · [seed1_full_prepare trace](docs/reports/work_audit/storage_native_gate_20261006/arms/seed1_full_prepare/backend_trace.jsonl.gz) · [seed1_host_stage timings](docs/reports/work_audit/storage_native_gate_20261006/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/storage_native_gate_20261006/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/storage_native_gate_20261006/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/storage_native_gate_20261006/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_audit_20261006_213611"></a>
<details>
<summary><strong>Oct 6, 2026, 4:42:18 p.m. CDT · Storage-tier KV timing</strong> · storage_audit_20261006_213611</summary>

**Question (RQ15).** When a session prefix is present in file-backed storage but absent from host and GPU cache, can native storage-to-host and host-to-GPU preparation during a known tool wait reduce replay delay, and what happens to peer requests?

**Finding.** With storage-only KV proven, first token after tool due was 337 ms on demand, 166 ms after host staging, and 65 ms after full preparation (median across 1 paired seed(s)). Peers overlapped preparation; their median TTFT changed from 154 to 203 ms. This is not a proven win-win.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 2 peer session(s), started 0 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | full_prepare | 64.7 | 60.0 | 202.8 | 2070.6 | 2 | not recorded | not recorded | 16109.2 | 0 | 2048 | 2112 | yes |
| 3 | host_stage | 165.6 | 161.6 | 196.3 | 1938.8 | 2 | not recorded | not recorded | 16267.5 | 0 | 2048 | 2112 | yes |
| 3 | on_demand | 336.6 | 333.3 | 153.7 | 1934.1 | 0 | not recorded | not recorded | 16281.5 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='3' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ15 WORK_AUDIT_STORAGE_PROMPT_ID= WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=2 WORK_AUDIT_STORAGE_PEER_START_MS=0 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_audit_20261006_213611/summary.json) · [Run manifest](docs/reports/work_audit/storage_audit_20261006_213611/run_manifest.json) · [seed3_full_prepare timings](docs/reports/work_audit/storage_audit_20261006_213611/arms/seed3_full_prepare/case_results.json) · [seed3_full_prepare trace](docs/reports/work_audit/storage_audit_20261006_213611/arms/seed3_full_prepare/backend_trace.jsonl.gz) · [seed3_host_stage timings](docs/reports/work_audit/storage_audit_20261006_213611/arms/seed3_host_stage/case_results.json) · [seed3_host_stage trace](docs/reports/work_audit/storage_audit_20261006_213611/arms/seed3_host_stage/backend_trace.jsonl.gz) · [seed3_on_demand timings](docs/reports/work_audit/storage_audit_20261006_213611/arms/seed3_on_demand/case_results.json) · [seed3_on_demand trace](docs/reports/work_audit/storage_audit_20261006_213611/arms/seed3_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_audit_20261006_212924"></a>
<details>
<summary><strong>Oct 6, 2026, 4:35:31 p.m. CDT · Storage-tier KV timing</strong> · storage_audit_20261006_212924</summary>

**Question (RQ15).** When a session prefix is present in file-backed storage but absent from host and GPU cache, can native storage-to-host and host-to-GPU preparation during a known tool wait reduce replay delay, and what happens to peer requests?

**Finding.** With storage-only KV proven, first token after tool due was 330 ms on demand, 165 ms after host staging, and 59 ms after full preparation (median across 1 paired seed(s)). Peers began after preparation finished, so this run does not measure contention during the transfer.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 2 peer session(s), started 1000 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | full_prepare | 59.4 | 56.7 | 155.5 | 1887.7 | 0 | not recorded | not recorded | 16568.0 | 0 | 2048 | 2112 | yes |
| 2 | host_stage | 165.3 | 162.5 | 158.8 | 1972.5 | 0 | not recorded | not recorded | 15896.9 | 0 | 2048 | 2112 | yes |
| 2 | on_demand | 330.3 | 328.3 | 160.1 | 1898.9 | 0 | not recorded | not recorded | 16154.3 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Small concurrent-peer timing only; this does not establish a production-workload benefit. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='2' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ15 WORK_AUDIT_STORAGE_PROMPT_ID= WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=2 WORK_AUDIT_STORAGE_PEER_START_MS=1000 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_audit_20261006_212924/summary.json) · [Run manifest](docs/reports/work_audit/storage_audit_20261006_212924/run_manifest.json) · [seed2_full_prepare timings](docs/reports/work_audit/storage_audit_20261006_212924/arms/seed2_full_prepare/case_results.json) · [seed2_full_prepare trace](docs/reports/work_audit/storage_audit_20261006_212924/arms/seed2_full_prepare/backend_trace.jsonl.gz) · [seed2_host_stage timings](docs/reports/work_audit/storage_audit_20261006_212924/arms/seed2_host_stage/case_results.json) · [seed2_host_stage trace](docs/reports/work_audit/storage_audit_20261006_212924/arms/seed2_host_stage/backend_trace.jsonl.gz) · [seed2_on_demand timings](docs/reports/work_audit/storage_audit_20261006_212924/arms/seed2_on_demand/case_results.json) · [seed2_on_demand trace](docs/reports/work_audit/storage_audit_20261006_212924/arms/seed2_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-storage_audit_20261006_212239"></a>
<details>
<summary><strong>Oct 6, 2026, 4:28:44 p.m. CDT · Storage-tier KV timing</strong> · storage_audit_20261006_212239</summary>

**Question (RQ15).** When a session prefix is present in file-backed storage but absent from host and GPU cache, can native storage-to-host and host-to-GPU preparation during a known tool wait reduce replay delay, and what happens to peer requests?

**Finding.** With storage-only KV proven, first token after tool due was 359 ms on demand, 170 ms after host staging, and 62 ms after full preparation (median across 1 paired seed(s)). No peer-session or whole-system benefit is established by this single-session run.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-1.5B-Instruct; pinned backend 0.5.10.post1. Fresh backend and storage path per arm; write-through storage, 64-token pages, equal frontend priority, lean KV trace. Each arm populated a prefix, evicted it from GPU and host, then replayed it after the same synthetic tool wait. A positive native L3 hit was required. The early arms staged L3→L2 or L3→L2→L1 before tool return. There were 0 peer session(s), started 1000 ms into the tool wait.

**Key measurements**

| Seed | Arm | Due → first token (ms) | Replay TTFT (ms) | Peer TTFT median (ms) | Peer completion median (ms) | Peers overlapping preparation | Native L3 → host ready (ms) | Ready → status observed (ms) | Whole workflow (ms) | L3 tokens at replay | L3 tokens in wait | Matched prefix tokens | Stage ready by due |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full_prepare | 61.9 | 56.8 | not recorded | not recorded | 0 | not recorded | not recorded | 15780.3 | 0 | 2048 | 2112 | yes |
| 1 | host_stage | 169.7 | 163.2 | not recorded | not recorded | 0 | not recorded | not recorded | 16102.9 | 0 | 2048 | 2112 | yes |
| 1 | on_demand | 358.5 | 355.8 | not recorded | not recorded | 0 | not recorded | not recorded | 16159.7 | 2048 | 0 | 2112 | not applicable |

**Evidence gate.** complete. Timestamp: Manifest completion time; start unavailable; displayed in Central Time.

**Limits**

- Single-session storage timing only. Peer-session cost and a busy-workload benefit are not measured. A file-backend L3 hit does not prove physical SSD I/O; the OS page cache may serve reads.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_STORAGE_SEEDS='1' WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID=RQ15 WORK_AUDIT_STORAGE_PROMPT_ID= WORK_AUDIT_STORAGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct WORK_AUDIT_STORAGE_WAIT_MS=5000 WORK_AUDIT_STORAGE_PROMPT_TOKENS=2048 WORK_AUDIT_STORAGE_PAGE_SIZE=64 WORK_AUDIT_STORAGE_HOST_GB=14.0 WORK_AUDIT_STORAGE_MEM_FRACTION=0.7 WORK_AUDIT_STORAGE_PEERS=0 WORK_AUDIT_STORAGE_PEER_START_MS=1000 WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS=1024 WORK_AUDIT_STORAGE_PEER_MAX_TOKENS=96 bash infra/container/run_work_audit_storage.sh
```

**Evidence:** [Summary](docs/reports/work_audit/storage_audit_20261006_212239/summary.json) · [Run manifest](docs/reports/work_audit/storage_audit_20261006_212239/run_manifest.json) · [seed1_full_prepare timings](docs/reports/work_audit/storage_audit_20261006_212239/arms/seed1_full_prepare/case_results.json) · [seed1_full_prepare trace](docs/reports/work_audit/storage_audit_20261006_212239/arms/seed1_full_prepare/backend_trace.jsonl.gz) · [seed1_host_stage timings](docs/reports/work_audit/storage_audit_20261006_212239/arms/seed1_host_stage/case_results.json) · [seed1_host_stage trace](docs/reports/work_audit/storage_audit_20261006_212239/arms/seed1_host_stage/backend_trace.jsonl.gz) · [seed1_on_demand timings](docs/reports/work_audit/storage_audit_20261006_212239/arms/seed1_on_demand/case_results.json) · [seed1_on_demand trace](docs/reports/work_audit/storage_audit_20261006_212239/arms/seed1_on_demand/backend_trace.jsonl.gz)

</details>

<a id="run-rq14_traceoff_d0_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:11:20 p.m. CDT · Repeated tool returns · trace-off control</strong> · rq14_traceoff_d0_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Trace-off control: first-token delay 79.7 ms median, 136.6 ms at p95 across 24 active replays. Backend stage and KV-load evidence was deliberately not captured.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement. Backend tracing was disabled; stage and load-back evidence is unavailable.

**Key measurements**

| Session · turn | Prompt tokens | Tool return → first token (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 70.0 | 939.2 |
| toolcycles-seed1-active0 · 2 | 1058 | 70.4 | 941.6 |
| toolcycles-seed1-active0 · 3 | 1181 | 70.9 | 932.2 |
| toolcycles-seed1-active0 · 4 | 1304 | 71.7 | 935.5 |
| toolcycles-seed1-active0 · 5 | 1427 | 73.0 | 947.6 |
| toolcycles-seed1-active0 · 6 | 1550 | 73.8 | 953.3 |
| toolcycles-seed1-active0 · 7 | 1673 | 97.3 | 909.3 |
| toolcycles-seed1-active0 · 8 | 1796 | 75.5 | 955.7 |
| toolcycles-seed1-active0 · 9 | 1919 | 75.8 | 946.1 |
| toolcycles-seed1-active0 · 10 | 2043 | 78.0 | 960.3 |
| toolcycles-seed1-active0 · 11 | 2167 | 92.8 | 906.7 |
| toolcycles-seed1-active0 · 12 | 2291 | 79.2 | 963.7 |
| toolcycles-seed1-active1 · 1 | 935 | 103.4 | 910.0 |
| toolcycles-seed1-active1 · 2 | 1058 | 125.6 | 933.5 |
| toolcycles-seed1-active1 · 3 | 1181 | 81.6 | 888.5 |
| toolcycles-seed1-active1 · 4 | 1304 | 105.3 | 913.6 |
| toolcycles-seed1-active1 · 5 | 1427 | 114.6 | 923.9 |
| toolcycles-seed1-active1 · 6 | 1550 | 119.0 | 932.6 |
| toolcycles-seed1-active1 · 7 | 1673 | 74.6 | 943.7 |
| toolcycles-seed1-active1 · 8 | 1796 | 104.1 | 917.5 |
| toolcycles-seed1-active1 · 9 | 1919 | 80.5 | 892.8 |
| toolcycles-seed1-active1 · 10 | 2043 | 139.3 | 953.6 |
| toolcycles-seed1-active1 · 11 | 2167 | 78.1 | 951.2 |
| toolcycles-seed1-active1 · 12 | 2291 | 136.8 | 952.5 |

Trace disabled: no backend stages or KV-load evidence was captured.

**Evidence gate.** trace_off_control. Timestamp: First request; displayed in Central Time.

**Limits**

- Trace disabled: no backend stage or KV-load attribution is available.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_traceoff_d0_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='0' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_traceoff_d0_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_traceoff_d0_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_traceoff_d0_s1_20261006/instrumentation_audit.json)

</details>

<a id="run-rq14_traceoff_d8_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:09:27 p.m. CDT · Repeated tool returns · trace-off control</strong> · rq14_traceoff_d8_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Trace-off control: first-token delay 111.9 ms median, 464.7 ms at p95 across 24 active replays. Backend stage and KV-load evidence was deliberately not captured.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement. Backend tracing was disabled; stage and load-back evidence is unavailable.

**Key measurements**

| Session · turn | Prompt tokens | Tool return → first token (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 344.3 | 1181.5 |
| toolcycles-seed1-active0 · 2 | 1058 | 70.4 | 1354.7 |
| toolcycles-seed1-active0 · 3 | 1181 | 71.2 | 1316.4 |
| toolcycles-seed1-active0 · 4 | 1304 | 76.1 | 1328.4 |
| toolcycles-seed1-active0 · 5 | 1427 | 130.2 | 1381.3 |
| toolcycles-seed1-active0 · 6 | 1550 | 169.3 | 1011.6 |
| toolcycles-seed1-active0 · 7 | 1673 | 133.0 | 1483.2 |
| toolcycles-seed1-active0 · 8 | 1796 | 78.6 | 1659.9 |
| toolcycles-seed1-active0 · 9 | 1919 | 564.2 | 1377.6 |
| toolcycles-seed1-active0 · 10 | 2043 | 111.1 | 926.0 |
| toolcycles-seed1-active0 · 11 | 2167 | 95.3 | 909.6 |
| toolcycles-seed1-active0 · 12 | 2291 | 79.9 | 965.8 |
| toolcycles-seed1-active1 · 1 | 935 | 316.8 | 1153.4 |
| toolcycles-seed1-active1 · 2 | 1058 | 124.2 | 1345.6 |
| toolcycles-seed1-active1 · 3 | 1181 | 81.9 | 1272.4 |
| toolcycles-seed1-active1 · 4 | 1304 | 76.3 | 1274.9 |
| toolcycles-seed1-active1 · 5 | 1427 | 72.7 | 1388.9 |
| toolcycles-seed1-active1 · 6 | 1550 | 147.7 | 989.8 |
| toolcycles-seed1-active1 · 7 | 1673 | 113.2 | 1561.2 |
| toolcycles-seed1-active1 · 8 | 1796 | 342.0 | 1578.4 |
| toolcycles-seed1-active1 · 9 | 1919 | 464.9 | 1278.2 |
| toolcycles-seed1-active1 · 10 | 2043 | 78.0 | 961.8 |
| toolcycles-seed1-active1 · 11 | 2167 | 79.2 | 953.2 |
| toolcycles-seed1-active1 · 12 | 2291 | 137.9 | 954.5 |

Trace disabled: no backend stages or KV-load evidence was captured.

**Evidence gate.** trace_off_control. Timestamp: First request; displayed in Central Time.

**Limits**

- Trace disabled: no backend stage or KV-load attribution is available.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_traceoff_d8_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='0' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_traceoff_d8_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_traceoff_d8_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_traceoff_d8_s1_20261006/instrumentation_audit.json)

</details>

<a id="run-rq14_pilot_g0_o0_s1_recheck_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:07:27 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g0_o0_s1_recheck_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 92.833 ms median and 146.335 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 81.6 | 1.7 | 0 | not recorded | 981.9 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 81.5 | 1.8 | 0 | not recorded | 983.8 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 82.3 | 1.7 | 0 | not recorded | 969.0 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.1 | 1.7 | 0 | not recorded | 973.3 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 161.2 | 1.7 | 0 | not recorded | 994.3 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 85.7 | 1.8 | 0 | not recorded | 999.1 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 89.2 | 1.8 | 0 | not recorded | 926.2 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 88.7 | 1.7 | 0 | not recorded | 989.1 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 89.2 | 1.7 | 0 | not recorded | 990.4 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 93.8 | 1.7 | 0 | not recorded | 995.1 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 140.0 | 1.7 | 0 | not recorded | 981.3 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 93.6 | 1.7 | 0 | not recorded | 1019.7 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 127.8 | 1.7 | 0 | not recorded | 954.5 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 146.6 | 1.7 | 0 | not recorded | 974.9 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 103.6 | 1.7 | 0 | not recorded | 933.0 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 87.2 | 1.7 | 0 | not recorded | 918.6 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 85.4 | 1.7 | 0 | not recorded | 995.3 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 143.6 | 1.8 | 0 | not recorded | 978.5 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.4 | 1.7 | 0 | not recorded | 1003.6 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 122.6 | 1.7 | 0 | not recorded | 961.6 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 123.1 | 1.7 | 0 | not recorded | 960.8 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 122.6 | 1.7 | 0 | not recorded | 963.7 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 92.4 | 1.7 | 0 | not recorded | 1015.4 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 122.8 | 1.8 | 0 | not recorded | 965.5 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g0_o0_s1_recheck_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g0_o0_s1_recheck_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g0_o0_s1_recheck_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g0_o0_s1_recheck_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g0_o0_s1_recheck_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g0_o0_s1_recheck_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g0_o0_s1_recheck_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_traceoff_d8_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:04:59 p.m. CDT · Repeated tool returns · trace-off control</strong> · rq14_traceoff_d8_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Trace-off control: first-token delay 103.7 ms median, 503.7 ms at p95 across 24 active replays. Backend stage and KV-load evidence was deliberately not captured.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement. Backend tracing was disabled; stage and load-back evidence is unavailable.

**Key measurements**

| Session · turn | Prompt tokens | Tool return → first token (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 329.2 | 1167.5 |
| toolcycles-seed2-active0 · 2 | 1058 | 73.6 | 1293.5 |
| toolcycles-seed2-active0 · 3 | 1181 | 70.9 | 1388.6 |
| toolcycles-seed2-active0 · 4 | 1304 | 131.9 | 1361.0 |
| toolcycles-seed2-active0 · 5 | 1427 | 119.0 | 1368.6 |
| toolcycles-seed2-active0 · 6 | 1550 | 84.4 | 993.4 |
| toolcycles-seed2-active0 · 7 | 1673 | 171.0 | 1561.0 |
| toolcycles-seed2-active0 · 8 | 1796 | 78.8 | 1654.8 |
| toolcycles-seed2-active0 · 9 | 1919 | 653.8 | 1466.9 |
| toolcycles-seed2-active0 · 10 | 2043 | 78.4 | 960.8 |
| toolcycles-seed2-active0 · 11 | 2167 | 79.2 | 951.6 |
| toolcycles-seed2-active0 · 12 | 2291 | 79.6 | 956.0 |
| toolcycles-seed2-active1 · 1 | 935 | 326.4 | 1164.0 |
| toolcycles-seed2-active1 · 2 | 1058 | 71.1 | 1228.1 |
| toolcycles-seed2-active1 · 3 | 1181 | 112.5 | 1366.6 |
| toolcycles-seed2-active1 · 4 | 1304 | 90.9 | 1384.1 |
| toolcycles-seed2-active1 · 5 | 1427 | 72.6 | 1387.2 |
| toolcycles-seed2-active1 · 6 | 1550 | 102.2 | 945.0 |
| toolcycles-seed2-active1 · 7 | 1673 | 113.2 | 1600.0 |
| toolcycles-seed2-active1 · 8 | 1796 | 183.3 | 1621.4 |
| toolcycles-seed2-active1 · 9 | 1919 | 503.9 | 1317.1 |
| toolcycles-seed2-active1 · 10 | 2043 | 126.9 | 940.9 |
| toolcycles-seed2-active1 · 11 | 2167 | 104.1 | 916.2 |
| toolcycles-seed2-active1 · 12 | 2291 | 103.7 | 915.5 |

Trace disabled: no backend stages or KV-load evidence was captured.

**Evidence gate.** trace_off_control. Timestamp: First request; displayed in Central Time.

**Limits**

- Trace disabled: no backend stage or KV-load attribution is available.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_traceoff_d8_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='0' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_traceoff_d8_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_traceoff_d8_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_traceoff_d8_s2_20261006/instrumentation_audit.json)

</details>

<a id="run-rq14_traceoff_d0_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 2:03:08 p.m. CDT · Repeated tool returns · trace-off control</strong> · rq14_traceoff_d0_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Trace-off control: first-token delay 80.9 ms median, 119.3 ms at p95 across 24 active replays. Backend stage and KV-load evidence was deliberately not captured.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement. Backend tracing was disabled; stage and load-back evidence is unavailable.

**Key measurements**

| Session · turn | Prompt tokens | Tool return → first token (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 71.6 | 939.3 |
| toolcycles-seed2-active0 · 2 | 1058 | 71.3 | 933.5 |
| toolcycles-seed2-active0 · 3 | 1181 | 72.4 | 930.9 |
| toolcycles-seed2-active0 · 4 | 1304 | 71.5 | 944.3 |
| toolcycles-seed2-active0 · 5 | 1427 | 119.5 | 929.1 |
| toolcycles-seed2-active0 · 6 | 1550 | 73.6 | 950.1 |
| toolcycles-seed2-active0 · 7 | 1673 | 101.7 | 912.0 |
| toolcycles-seed2-active0 · 8 | 1796 | 75.3 | 942.9 |
| toolcycles-seed2-active0 · 9 | 1919 | 76.0 | 943.8 |
| toolcycles-seed2-active0 · 10 | 2043 | 77.4 | 950.0 |
| toolcycles-seed2-active0 · 11 | 2167 | 78.0 | 948.0 |
| toolcycles-seed2-active0 · 12 | 2291 | 78.8 | 949.6 |
| toolcycles-seed2-active1 · 1 | 935 | 132.3 | 937.6 |
| toolcycles-seed2-active1 · 2 | 1058 | 102.7 | 912.1 |
| toolcycles-seed2-active1 · 3 | 1181 | 103.0 | 908.7 |
| toolcycles-seed2-active1 · 4 | 1304 | 115.8 | 924.2 |
| toolcycles-seed2-active1 · 5 | 1427 | 73.2 | 947.8 |
| toolcycles-seed2-active1 · 6 | 1550 | 92.9 | 903.5 |
| toolcycles-seed2-active1 · 7 | 1673 | 74.6 | 951.7 |
| toolcycles-seed2-active1 · 8 | 1796 | 98.9 | 908.1 |
| toolcycles-seed2-active1 · 9 | 1919 | 95.4 | 905.0 |
| toolcycles-seed2-active1 · 10 | 2043 | 83.2 | 896.5 |
| toolcycles-seed2-active1 · 11 | 2167 | 93.9 | 902.8 |
| toolcycles-seed2-active1 · 12 | 2291 | 101.4 | 910.4 |

Trace disabled: no backend stages or KV-load evidence was captured.

**Evidence gate.** trace_off_control. Timestamp: First request; displayed in Central Time.

**Limits**

- Trace disabled: no backend stage or KV-load attribution is available.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_traceoff_d0_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='0' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_traceoff_d0_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_traceoff_d0_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_traceoff_d0_s2_20261006/instrumentation_audit.json)

</details>

<a id="run-rq14_capacity_d8_g1_o1_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:54:42 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g1_o1_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 163.299 ms median and 717.435 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 493.5 | 5.3 | 0 | not recorded | 1312.1 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 140.4 | 1.8 | 0 | not recorded | 964.6 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.7 | 1.7 | 0 | not recorded | 1398.4 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 358.7 | 10.0 | 0 | not recorded | 1300.7 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 160.9 | 1.8 | 0 | not recorded | 1006.9 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 213.1 | 5.5 | 0 | not recorded | 988.4 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 143.3 | 2.4 | 0 | not recorded | 1658.8 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 402.8 | 57.7 | 0 | not recorded | 1602.4 |
| toolcycles-seed2-active0 · 9 | 1919 | 34 | 1 | 378.4 | 156.7 | 15.3 | 139.6 | 1414.2 |
| toolcycles-seed2-active0 · 10 | 2043 | 34 | 1 | 717.6 | 405.8 | 19.0 | 146.5 | 1826.5 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 93.4 | 1.7 | 0 | not recorded | 915.9 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 95.9 | 3.2 | 0 | not recorded | 919.3 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 490.9 | 1.8 | 0 | not recorded | 1308.9 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 118.9 | 1.7 | 0 | not recorded | 898.8 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 152.8 | 1.8 | 0 | not recorded | 1392.9 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 365.4 | 13.4 | 0 | not recorded | 1307.2 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 117.9 | 2.0 | 0 | not recorded | 1025.6 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 166.0 | 1.7 | 0 | not recorded | 941.2 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 87.8 | 1.8 | 0 | not recorded | 1566.6 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 387.4 | 53.6 | 0 | not recorded | 1587.1 |
| toolcycles-seed2-active1 · 9 | 1919 | 34 | 1 | 743.1 | 538.0 | 25.2 | 133.1 | 2533.2 |
| toolcycles-seed2-active1 · 10 | 2043 | 34 | 1 | 376.9 | 258.2 | 19.9 | 236.4 | 1137.1 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 145.5 | 4.4 | 0 | not recorded | 903.6 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 119.3 | 1.9 | 0 | not recorded | 878.1 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g1_o1_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g1_o0_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:52:25 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g1_o0_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 131.976 ms median and 788.576 ms at p95. 3 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 531.2 | 5.4 | 0 | not recorded | 1423.6 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 88.6 | 1.7 | 0 | not recorded | 1051.2 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.7 | 1.7 | 0 | not recorded | 1450.3 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 237.5 | 8.2 | 0 | not recorded | 1434.3 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 143.1 | 1.7 | 0 | not recorded | 1047.2 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 227.2 | 5.6 | 0 | not recorded | 1576.5 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 129.3 | 1.8 | 0 | not recorded | 1725.3 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 354.7 | 13.7 | 0 | not recorded | 1464.2 |
| toolcycles-seed2-active0 · 9 | 1919 | 34 | 1 | 828.1 | 565.3 | 19.6 | 171.1 | 2642.3 |
| toolcycles-seed2-active0 · 10 | 2043 | 1913 | 0 | 140.4 | 1.7 | 0 | not recorded | 927.1 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 101.3 | 3.8 | 0 | not recorded | 904.4 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 99.6 | 1.8 | 0 | not recorded | 910.1 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 528.0 | 1.8 | 0 | not recorded | 1420.3 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 96.6 | 1.7 | 0 | not recorded | 985.4 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 135.1 | 1.8 | 0 | not recorded | 1427.5 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 83.9 | 1.7 | 0 | not recorded | 1456.4 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 86.8 | 1.7 | 0 | not recorded | 1067.8 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 180.1 | 1.8 | 0 | not recorded | 1528.7 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 88.7 | 1.7 | 0 | not recorded | 1764.6 |
| toolcycles-seed2-active1 · 8 | 1796 | 1298 | 1 | 788.8 | 609.4 | 8.2 | 112.2 | 2571.3 |
| toolcycles-seed2-active1 · 9 | 1919 | 34 | 1 | 266.0 | 146.4 | 18.0 | 126.5 | 2376.4 |
| toolcycles-seed2-active1 · 10 | 2043 | 1913 | 0 | 91.5 | 1.7 | 0 | not recorded | 968.0 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 95.3 | 4.1 | 0 | not recorded | 972.9 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 93.9 | 1.7 | 0 | not recorded | 881.7 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g1_o0_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g0_o1_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:50:14 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g0_o1_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 150.021 ms median and 470.966 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 373.1 | 5.4 | 0 | not recorded | 1199.0 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 136.5 | 1.7 | 0 | not recorded | 967.2 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.9 | 1.7 | 0 | not recorded | 1424.0 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 345.9 | 9.1 | 0 | not recorded | 1304.1 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 246.4 | 1.7 | 0 | not recorded | 1099.5 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 138.1 | 2.0 | 0 | not recorded | 1059.0 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 143.4 | 2.0 | 0 | not recorded | 1487.4 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 134.7 | 1.8 | 0 | not recorded | 1676.3 |
| toolcycles-seed2-active0 · 9 | 1919 | 34 | 1 | 471.1 | 266.9 | 15.1 | 250.0 | 1709.1 |
| toolcycles-seed2-active0 · 10 | 2043 | 34 | 1 | 290.9 | 187.8 | 19.3 | 166.7 | 1384.7 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 93.7 | 1.7 | 0 | not recorded | 924.3 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 96.1 | 3.2 | 0 | not recorded | 927.7 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 369.6 | 1.9 | 0 | not recorded | 1196.0 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 112.4 | 1.8 | 0 | not recorded | 900.2 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 121.3 | 2.3 | 0 | not recorded | 1402.5 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 234.8 | 1.8 | 0 | not recorded | 1326.1 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 264.7 | 6.1 | 0 | not recorded | 1117.9 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 153.0 | 1.7 | 0 | not recorded | 1011.4 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 88.6 | 1.8 | 0 | not recorded | 1488.8 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 153.8 | 13.8 | 0 | not recorded | 1604.8 |
| toolcycles-seed2-active1 · 9 | 1919 | 34 | 1 | 775.1 | 426.0 | 26.9 | 150.4 | 2335.2 |
| toolcycles-seed2-active1 · 10 | 2043 | 34 | 1 | 354.7 | 232.3 | 20.1 | 210.1 | 1122.7 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 147.4 | 4.5 | 0 | not recorded | 913.2 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 119.7 | 2.0 | 0 | not recorded | 886.4 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g0_o1_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g0_o0_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:47:55 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g0_o0_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 134.517 ms median and 638.199 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 419.8 | 5.3 | 0 | not recorded | 1328.6 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 257.2 | 4.8 | 0 | not recorded | 1509.0 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.4 | 1.7 | 0 | not recorded | 1547.4 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 160.0 | 1.7 | 0 | not recorded | 1078.6 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 155.3 | 1.7 | 0 | not recorded | 1085.7 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 95.5 | 1.8 | 0 | not recorded | 1101.7 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 130.2 | 1.8 | 0 | not recorded | 1666.1 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 121.5 | 1.7 | 0 | not recorded | 2022.6 |
| toolcycles-seed2-active0 · 9 | 1919 | 34 | 1 | 563.4 | 252.1 | 15.1 | 235.2 | 1987.6 |
| toolcycles-seed2-active0 · 10 | 2043 | 34 | 1 | 638.4 | 349.9 | 18.0 | 159.2 | 1686.3 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 92.7 | 1.7 | 0 | not recorded | 986.5 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 95.5 | 3.2 | 0 | not recorded | 988.1 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 417.4 | 1.8 | 0 | not recorded | 1326.2 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 191.9 | 1.7 | 0 | not recorded | 1443.6 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 134.1 | 1.7 | 0 | not recorded | 1524.1 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 105.2 | 1.8 | 0 | not recorded | 1101.9 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 176.2 | 5.5 | 0 | not recorded | 1106.5 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 127.7 | 1.8 | 0 | not recorded | 1055.4 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 91.3 | 1.8 | 0 | not recorded | 1706.3 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 135.4 | 15.8 | 0 | not recorded | 1928.9 |
| toolcycles-seed2-active1 · 9 | 1919 | 34 | 1 | 943.8 | 770.1 | 21.0 | 138.2 | 3134.4 |
| toolcycles-seed2-active1 · 10 | 2043 | 929 | 1 | 237.6 | 141.0 | 12.8 | 126.3 | 1060.6 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 124.3 | 3.9 | 0 | not recorded | 942.6 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 94.8 | 1.8 | 0 | not recorded | 914.2 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g0_o0_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g1_o1_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:45:36 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g1_o1_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 100.516 ms median and 141.645 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 82.3 | 1.8 | 0 | not recorded | 919.6 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 81.7 | 1.7 | 0 | not recorded | 903.5 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.8 | 1.7 | 0 | not recorded | 905.4 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 83.6 | 1.8 | 0 | not recorded | 927.3 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 127.6 | 1.6 | 0 | not recorded | 911.4 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 86.5 | 1.7 | 0 | not recorded | 914.6 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 142.8 | 1.9 | 0 | not recorded | 928.8 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 89.0 | 1.7 | 0 | not recorded | 918.6 |
| toolcycles-seed2-active0 · 9 | 1919 | 1790 | 0 | 90.0 | 1.7 | 0 | not recorded | 919.9 |
| toolcycles-seed2-active0 · 10 | 2043 | 1913 | 0 | 91.1 | 1.7 | 0 | not recorded | 923.2 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 92.2 | 1.7 | 0 | not recorded | 922.8 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 93.7 | 1.7 | 0 | not recorded | 925.2 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 139.7 | 1.6 | 0 | not recorded | 918.9 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 107.7 | 1.8 | 0 | not recorded | 869.7 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 120.6 | 2.1 | 0 | not recorded | 883.0 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 135.9 | 1.8 | 0 | not recorded | 919.2 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 84.7 | 1.7 | 0 | not recorded | 929.5 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 134.5 | 2.3 | 0 | not recorded | 900.4 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 87.9 | 1.8 | 0 | not recorded | 936.6 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 137.5 | 1.7 | 0 | not recorded | 903.8 |
| toolcycles-seed2-active1 · 9 | 1919 | 1790 | 0 | 116.6 | 2.2 | 0 | not recorded | 882.9 |
| toolcycles-seed2-active1 · 10 | 2043 | 1913 | 0 | 135.0 | 2.4 | 0 | not recorded | 903.4 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 141.8 | 2.3 | 0 | not recorded | 907.8 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 117.7 | 2.0 | 0 | not recorded | 884.3 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g1_o1_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g1_o1_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g1_o1_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g1_o1_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g1_o1_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g1_o1_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g1_o1_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g1_o0_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:43:29 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g1_o0_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 94.265 ms median and 401.106 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 401.4 | 320.7 | 0 | not recorded | 1284.3 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 81.5 | 1.7 | 0 | not recorded | 967.8 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.8 | 1.7 | 0 | not recorded | 970.1 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 137.7 | 1.8 | 0 | not recorded | 952.0 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 143.1 | 1.7 | 0 | not recorded | 957.6 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 86.6 | 1.7 | 0 | not recorded | 981.0 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 127.4 | 1.7 | 0 | not recorded | 946.8 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 91.6 | 1.7 | 0 | not recorded | 976.4 |
| toolcycles-seed2-active0 · 9 | 1919 | 1790 | 0 | 93.2 | 1.8 | 0 | not recorded | 979.3 |
| toolcycles-seed2-active0 · 10 | 2043 | 1913 | 0 | 94.4 | 1.7 | 0 | not recorded | 983.4 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 92.3 | 1.7 | 0 | not recorded | 976.3 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 93.5 | 1.7 | 0 | not recorded | 979.5 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 474.2 | 1.7 | 0 | not recorded | 1283.5 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 90.0 | 1.8 | 0 | not recorded | 902.4 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 135.1 | 1.8 | 0 | not recorded | 947.1 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 84.1 | 1.7 | 0 | not recorded | 974.8 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 84.9 | 1.7 | 0 | not recorded | 975.8 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 118.0 | 1.9 | 0 | not recorded | 934.7 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 87.6 | 1.7 | 0 | not recorded | 986.5 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 94.4 | 1.7 | 0 | not recorded | 916.8 |
| toolcycles-seed2-active1 · 9 | 1919 | 1790 | 0 | 118.0 | 1.7 | 0 | not recorded | 941.3 |
| toolcycles-seed2-active1 · 10 | 2043 | 1913 | 0 | 103.9 | 1.8 | 0 | not recorded | 929.4 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 111.6 | 1.8 | 0 | not recorded | 930.8 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 118.5 | 1.8 | 0 | not recorded | 938.6 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g1_o0_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g1_o0_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g1_o0_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g1_o0_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g1_o0_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g1_o0_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g1_o0_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g0_o1_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:41:28 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g0_o1_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 100.951 ms median and 147.002 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 82.4 | 1.7 | 0 | not recorded | 928.5 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 82.0 | 1.7 | 0 | not recorded | 911.6 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.8 | 1.7 | 0 | not recorded | 913.7 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 83.7 | 1.7 | 0 | not recorded | 935.9 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 128.4 | 2.2 | 0 | not recorded | 921.2 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 86.6 | 1.8 | 0 | not recorded | 923.8 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 147.1 | 1.9 | 0 | not recorded | 941.6 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 88.7 | 1.7 | 0 | not recorded | 926.7 |
| toolcycles-seed2-active0 · 9 | 1919 | 1790 | 0 | 89.7 | 1.7 | 0 | not recorded | 928.0 |
| toolcycles-seed2-active0 · 10 | 2043 | 1913 | 0 | 91.4 | 1.7 | 0 | not recorded | 931.4 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 92.6 | 1.7 | 0 | not recorded | 931.2 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 94.0 | 1.7 | 0 | not recorded | 932.5 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 139.9 | 2.1 | 0 | not recorded | 927.2 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 108.3 | 2.3 | 0 | not recorded | 880.7 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 119.4 | 1.7 | 0 | not recorded | 892.9 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 133.2 | 2.2 | 0 | not recorded | 924.3 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 85.2 | 1.8 | 0 | not recorded | 940.0 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 135.4 | 1.8 | 0 | not recorded | 913.3 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 88.1 | 1.8 | 0 | not recorded | 944.9 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 138.6 | 1.8 | 0 | not recorded | 913.0 |
| toolcycles-seed2-active1 · 9 | 1919 | 1790 | 0 | 116.7 | 2.4 | 0 | not recorded | 891.4 |
| toolcycles-seed2-active1 · 10 | 2043 | 1913 | 0 | 134.6 | 2.3 | 0 | not recorded | 910.8 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 144.0 | 2.2 | 0 | not recorded | 917.4 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 151.2 | 2.3 | 0 | not recorded | 924.9 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g0_o1_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g0_o1_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g0_o1_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g0_o1_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g0_o1_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g0_o1_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g0_o1_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g0_o0_s2_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:39:20 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g0_o0_s2_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 92.218 ms median and 142.942 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 2. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed2-active0 · 1 | 935 | 806 | 0 | 81.9 | 1.8 | 0 | not recorded | 982.4 |
| toolcycles-seed2-active0 · 2 | 1058 | 929 | 0 | 81.3 | 1.7 | 0 | not recorded | 983.4 |
| toolcycles-seed2-active0 · 3 | 1181 | 1052 | 0 | 82.2 | 1.7 | 0 | not recorded | 986.3 |
| toolcycles-seed2-active0 · 4 | 1304 | 1175 | 0 | 136.5 | 1.7 | 0 | not recorded | 968.3 |
| toolcycles-seed2-active0 · 5 | 1427 | 1298 | 0 | 143.1 | 1.7 | 0 | not recorded | 975.2 |
| toolcycles-seed2-active0 · 6 | 1550 | 1421 | 0 | 86.1 | 1.8 | 0 | not recorded | 998.4 |
| toolcycles-seed2-active0 · 7 | 1673 | 1544 | 0 | 126.9 | 1.7 | 0 | not recorded | 962.3 |
| toolcycles-seed2-active0 · 8 | 1796 | 1667 | 0 | 88.3 | 1.7 | 0 | not recorded | 986.5 |
| toolcycles-seed2-active0 · 9 | 1919 | 1790 | 0 | 89.8 | 1.7 | 0 | not recorded | 989.1 |
| toolcycles-seed2-active0 · 10 | 2043 | 1913 | 0 | 90.7 | 1.7 | 0 | not recorded | 993.0 |
| toolcycles-seed2-active0 · 11 | 2167 | 2037 | 0 | 92.2 | 1.7 | 0 | not recorded | 993.1 |
| toolcycles-seed2-active0 · 12 | 2291 | 2161 | 0 | 93.0 | 1.7 | 0 | not recorded | 996.7 |
| toolcycles-seed2-active1 · 1 | 935 | 806 | 0 | 154.0 | 1.8 | 0 | not recorded | 980.9 |
| toolcycles-seed2-active1 · 2 | 1058 | 929 | 0 | 89.6 | 1.7 | 0 | not recorded | 918.2 |
| toolcycles-seed2-active1 · 3 | 1181 | 1052 | 0 | 134.2 | 1.7 | 0 | not recorded | 963.9 |
| toolcycles-seed2-active1 · 4 | 1304 | 1175 | 0 | 83.7 | 1.7 | 0 | not recorded | 991.0 |
| toolcycles-seed2-active1 · 5 | 1427 | 1298 | 0 | 84.9 | 1.7 | 0 | not recorded | 994.0 |
| toolcycles-seed2-active1 · 6 | 1550 | 1421 | 0 | 118.0 | 1.7 | 0 | not recorded | 952.2 |
| toolcycles-seed2-active1 · 7 | 1673 | 1544 | 0 | 87.4 | 1.7 | 0 | not recorded | 1001.3 |
| toolcycles-seed2-active1 · 8 | 1796 | 1667 | 0 | 92.5 | 1.7 | 0 | not recorded | 928.2 |
| toolcycles-seed2-active1 · 9 | 1919 | 1790 | 0 | 114.1 | 1.7 | 0 | not recorded | 950.6 |
| toolcycles-seed2-active1 · 10 | 2043 | 1913 | 0 | 98.9 | 1.8 | 0 | not recorded | 938.2 |
| toolcycles-seed2-active1 · 11 | 2167 | 2037 | 0 | 112.5 | 1.7 | 0 | not recorded | 948.9 |
| toolcycles-seed2-active1 · 12 | 2291 | 2161 | 0 | 120.4 | 1.7 | 0 | not recorded | 958.1 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g0_o0_s2_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='2' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g0_o0_s2_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g0_o0_s2_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g0_o0_s2_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g0_o0_s2_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g0_o0_s2_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g0_o0_s2_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g1_o0_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:36:27 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g1_o0_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 150.379 ms median and 608.649 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 608.8 | 14.2 | 0 | not recorded | 1500.2 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 206.4 | 4.8 | 0 | not recorded | 1440.6 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 81.8 | 1.7 | 0 | not recorded | 1467.1 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 103.6 | 1.7 | 0 | not recorded | 1063.6 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 157.7 | 1.7 | 0 | not recorded | 1429.2 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 85.8 | 1.7 | 0 | not recorded | 1564.1 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 270.5 | 29.5 | 0 | not recorded | 1193.7 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 354.2 | 13.6 | 0 | not recorded | 1437.9 |
| toolcycles-seed1-active0 · 9 | 1919 | 34 | 1 | 558.0 | 406.6 | 20.2 | 146.2 | 2307.8 |
| toolcycles-seed1-active0 · 10 | 2043 | 34 | 1 | 248.1 | 150.1 | 18.0 | 130.2 | 1057.5 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 126.3 | 3.8 | 0 | not recorded | 930.2 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 108.1 | 1.8 | 0 | not recorded | 916.1 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 580.4 | 10.6 | 0 | not recorded | 1471.5 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 198.3 | 1.7 | 0 | not recorded | 1431.8 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 103.1 | 1.7 | 0 | not recorded | 1430.4 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 109.7 | 1.7 | 0 | not recorded | 1008.4 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 84.4 | 1.8 | 0 | not recorded | 1432.6 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 143.3 | 1.8 | 0 | not recorded | 1544.1 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 347.7 | 33.3 | 0 | not recorded | 1271.0 |
| toolcycles-seed1-active1 · 8 | 1796 | 1298 | 1 | 827.9 | 582.5 | 8.2 | 86.1 | 2465.8 |
| toolcycles-seed1-active1 · 9 | 1919 | 34 | 1 | 280.2 | 177.9 | 17.9 | 158.1 | 2492.5 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 91.3 | 1.7 | 0 | not recorded | 967.8 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 95.4 | 4.0 | 0 | not recorded | 972.4 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 93.6 | 1.8 | 0 | not recorded | 881.5 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g1_o0_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g1_o0_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g0_o1_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:34:14 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g0_o1_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 165.976 ms median and 957.5 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 359.2 | 12.0 | 0 | not recorded | 1235.7 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 161.9 | 4.9 | 0 | not recorded | 942.0 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 82.6 | 1.7 | 0 | not recorded | 1308.7 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.9 | 1.7 | 0 | not recorded | 1297.8 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 263.9 | 5.9 | 0 | not recorded | 1119.4 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 211.2 | 6.0 | 0 | not recorded | 1053.6 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 150.1 | 2.2 | 0 | not recorded | 1508.2 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 170.6 | 39.8 | 0 | not recorded | 1676.1 |
| toolcycles-seed1-active0 · 9 | 1919 | 34 | 1 | 1029.6 | 663.0 | 24.5 | 184.2 | 2670.9 |
| toolcycles-seed1-active0 · 10 | 2043 | 34 | 1 | 346.5 | 253.0 | 20.6 | 230.5 | 1118.3 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 139.1 | 4.5 | 0 | not recorded | 910.2 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 118.8 | 2.0 | 0 | not recorded | 894.1 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 332.1 | 7.9 | 0 | not recorded | 1208.5 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 153.7 | 1.8 | 0 | not recorded | 933.8 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 120.3 | 2.3 | 0 | not recorded | 1375.2 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 354.2 | 1.8 | 0 | not recorded | 1153.8 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 251.9 | 2.2 | 0 | not recorded | 1107.5 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 188.8 | 2.2 | 0 | not recorded | 1031.3 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 132.6 | 1.9 | 0 | not recorded | 1085.6 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 133.2 | 1.7 | 0 | not recorded | 1727.7 |
| toolcycles-seed1-active1 · 9 | 1919 | 34 | 1 | 660.4 | 284.1 | 18.7 | 263.1 | 1772.3 |
| toolcycles-seed1-active1 · 10 | 2043 | 34 | 1 | 957.7 | 585.4 | 19.2 | 201.7 | 2023.1 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 93.7 | 1.8 | 0 | not recorded | 929.8 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 96.5 | 3.2 | 0 | not recorded | 936.6 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g0_o1_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g0_o1_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g1_o1_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:31:44 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g1_o1_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 176.596 ms median and 770.334 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 552.3 | 14.1 | 0 | not recorded | 1360.8 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 205.6 | 4.8 | 0 | not recorded | 977.8 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 83.0 | 1.7 | 0 | not recorded | 1343.0 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 305.1 | 5.6 | 0 | not recorded | 1348.4 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 167.5 | 1.7 | 0 | not recorded | 990.5 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 186.1 | 2.0 | 0 | not recorded | 972.0 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 105.4 | 2.3 | 0 | not recorded | 1460.5 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 148.2 | 15.6 | 0 | not recorded | 1626.6 |
| toolcycles-seed1-active0 · 9 | 1919 | 34 | 1 | 356.9 | 155.6 | 16.5 | 137.2 | 1408.8 |
| toolcycles-seed1-active0 · 10 | 2043 | 34 | 1 | 967.2 | 602.4 | 18.0 | 166.8 | 1922.1 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 93.6 | 1.8 | 0 | not recorded | 925.5 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 96.6 | 3.2 | 0 | not recorded | 929.5 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 526.5 | 10.6 | 0 | not recorded | 1334.0 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 197.5 | 1.7 | 0 | not recorded | 969.7 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 119.6 | 2.3 | 0 | not recorded | 1374.2 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 350.9 | 1.7 | 0 | not recorded | 1173.0 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 123.9 | 2.1 | 0 | not recorded | 1010.3 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 150.4 | 2.2 | 0 | not recorded | 983.7 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.9 | 1.8 | 0 | not recorded | 1503.5 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 419.7 | 42.0 | 0 | not recorded | 1580.1 |
| toolcycles-seed1-active1 · 9 | 1919 | 34 | 1 | 770.5 | 644.8 | 21.9 | 134.6 | 2506.1 |
| toolcycles-seed1-active1 · 10 | 2043 | 34 | 1 | 257.3 | 150.9 | 20.3 | 128.7 | 1021.7 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 147.8 | 4.4 | 0 | not recorded | 915.1 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 139.9 | 1.8 | 0 | not recorded | 910.0 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g1_o1_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g1_o1_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_capacity_d8_g0_o0_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:28:04 p.m. CDT · Repeated tool-return startup</strong> · rq14_capacity_d8_g0_o0_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 139.292 ms median and 825.248 ms at p95. 4 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 435.2 | 13.7 | 0 | not recorded | 1342.8 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 199.3 | 4.9 | 0 | not recorded | 1494.9 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 81.6 | 1.7 | 0 | not recorded | 1484.9 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 96.9 | 1.7 | 0 | not recorded | 1076.8 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 167.8 | 1.7 | 0 | not recorded | 1094.9 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 99.3 | 1.8 | 0 | not recorded | 1106.8 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 89.1 | 1.8 | 0 | not recorded | 1628.4 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 121.2 | 1.7 | 0 | not recorded | 1996.2 |
| toolcycles-seed1-active0 · 9 | 1919 | 34 | 1 | 520.3 | 250.2 | 15.0 | 233.4 | 1701.4 |
| toolcycles-seed1-active0 · 10 | 2043 | 34 | 1 | 825.5 | 323.8 | 16.2 | 305.7 | 1936.4 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 93.3 | 1.8 | 0 | not recorded | 997.6 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 95.7 | 3.2 | 0 | not recorded | 999.9 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 407.2 | 7.4 | 0 | not recorded | 1314.7 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 190.2 | 1.8 | 0 | not recorded | 1485.1 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 102.4 | 1.7 | 0 | not recorded | 1448.4 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 104.8 | 1.7 | 0 | not recorded | 1021.6 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 168.3 | 5.4 | 0 | not recorded | 1095.5 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 157.7 | 1.7 | 0 | not recorded | 1086.3 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.5 | 1.7 | 0 | not recorded | 1707.0 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 163.5 | 15.5 | 0 | not recorded | 1913.6 |
| toolcycles-seed1-active1 · 9 | 1919 | 34 | 1 | 1062.0 | 562.7 | 20.9 | 146.4 | 2749.5 |
| toolcycles-seed1-active1 · 10 | 2043 | 34 | 1 | 318.6 | 196.4 | 17.9 | 176.7 | 1152.4 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 113.4 | 3.8 | 0 | not recorded | 950.3 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 105.4 | 1.7 | 0 | not recorded | 943.7 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_capacity_d8_g0_o0_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='8' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_capacity_d8_g0_o0_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_busy_g1_o1_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:25:33 p.m. CDT · Repeated tool-return startup</strong> · rq14_busy_g1_o1_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 126.646 ms median and 367.543 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 367.8 | 2.0 | 0 | not recorded | 1248.0 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 81.8 | 1.8 | 0 | not recorded | 1158.1 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 83.0 | 1.8 | 0 | not recorded | 1118.2 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.9 | 1.7 | 0 | not recorded | 1153.2 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 85.3 | 1.7 | 0 | not recorded | 1141.2 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 114.6 | 1.7 | 0 | not recorded | 968.6 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 105.2 | 2.2 | 0 | not recorded | 1171.2 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 91.9 | 1.8 | 0 | not recorded | 1120.5 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 139.9 | 2.2 | 0 | not recorded | 972.7 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 91.6 | 1.9 | 0 | not recorded | 926.0 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 134.2 | 1.8 | 0 | not recorded | 905.0 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 93.6 | 1.7 | 0 | not recorded | 950.8 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 437.3 | 4.9 | 0 | not recorded | 1220.6 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 132.0 | 2.0 | 0 | not recorded | 1149.1 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 223.6 | 2.5 | 0 | not recorded | 1063.0 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 121.6 | 2.2 | 0 | not recorded | 1132.6 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 132.6 | 1.7 | 0 | not recorded | 1127.7 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 155.3 | 1.7 | 0 | not recorded | 947.4 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.4 | 1.7 | 0 | not recorded | 1229.8 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 264.4 | 2.1 | 0 | not recorded | 1091.5 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 137.7 | 2.0 | 0 | not recorded | 905.8 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 159.8 | 2.3 | 0 | not recorded | 929.9 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 92.9 | 1.7 | 0 | not recorded | 928.2 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 136.6 | 1.7 | 0 | not recorded | 928.7 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_busy_g1_o1_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_busy_g1_o1_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_busy_g1_o1_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_busy_g1_o1_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_busy_g1_o1_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_busy_g1_o1_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_busy_g1_o1_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_busy_g1_o0_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:23:10 p.m. CDT · Repeated tool-return startup</strong> · rq14_busy_g1_o0_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 117.993 ms median and 341.125 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 341.3 | 1.9 | 0 | not recorded | 1331.7 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 81.6 | 1.8 | 0 | not recorded | 1298.2 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 230.3 | 4.8 | 0 | not recorded | 1229.4 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.7 | 1.7 | 0 | not recorded | 1268.5 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 109.4 | 1.8 | 0 | not recorded | 1231.9 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 185.4 | 5.5 | 0 | not recorded | 1116.6 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 88.7 | 1.7 | 0 | not recorded | 1275.7 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 209.7 | 1.8 | 0 | not recorded | 1166.2 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 114.9 | 1.7 | 0 | not recorded | 1067.3 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 137.4 | 1.7 | 0 | not recorded | 959.0 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 121.5 | 1.7 | 0 | not recorded | 943.7 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 93.6 | 1.8 | 0 | not recorded | 1001.8 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 451.0 | 4.9 | 0 | not recorded | 1302.5 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 146.9 | 1.7 | 0 | not recorded | 1290.2 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 284.3 | 1.8 | 0 | not recorded | 1141.5 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 104.2 | 1.8 | 0 | not recorded | 1213.5 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 84.7 | 1.7 | 0 | not recorded | 1284.0 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 164.4 | 1.7 | 0 | not recorded | 1095.8 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.3 | 1.7 | 0 | not recorded | 1353.2 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 207.6 | 1.7 | 0 | not recorded | 1084.6 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 96.9 | 1.7 | 0 | not recorded | 968.1 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 91.1 | 1.7 | 0 | not recorded | 994.6 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 97.0 | 1.7 | 0 | not recorded | 978.7 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 176.4 | 1.7 | 0 | not recorded | 1000.8 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_busy_g1_o0_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_busy_g1_o0_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_busy_g1_o0_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_busy_g1_o0_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_busy_g1_o0_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_busy_g1_o0_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_busy_g1_o0_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_busy_g0_o1_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:20:52 p.m. CDT · Repeated tool-return startup</strong> · rq14_busy_g0_o1_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 124.843 ms median and 421.335 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 188.8 | 1.8 | 0 | not recorded | 1092.8 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 82.4 | 2.3 | 0 | not recorded | 1167.4 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 83.2 | 1.7 | 0 | not recorded | 1162.2 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.7 | 1.7 | 0 | not recorded | 1157.2 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 84.8 | 1.7 | 0 | not recorded | 1148.6 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 212.1 | 5.5 | 0 | not recorded | 1003.0 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 104.5 | 1.7 | 0 | not recorded | 1173.2 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 433.4 | 1.7 | 0 | not recorded | 1348.4 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 117.5 | 2.2 | 0 | not recorded | 958.2 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 92.4 | 1.8 | 0 | not recorded | 955.7 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 136.6 | 2.1 | 0 | not recorded | 914.3 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 153.0 | 1.7 | 0 | not recorded | 954.1 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 273.2 | 5.3 | 0 | not recorded | 1064.9 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 133.4 | 1.6 | 0 | not recorded | 1158.2 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 269.2 | 1.7 | 0 | not recorded | 1109.9 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 119.4 | 1.7 | 0 | not recorded | 1134.3 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 130.7 | 1.6 | 0 | not recorded | 1132.8 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 189.5 | 1.7 | 0 | not recorded | 980.6 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.3 | 1.7 | 0 | not recorded | 1214.9 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 421.5 | 2.3 | 0 | not recorded | 1303.4 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 115.5 | 2.3 | 0 | not recorded | 897.5 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 155.9 | 1.7 | 0 | not recorded | 954.5 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 96.7 | 1.8 | 0 | not recorded | 936.0 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 94.7 | 1.8 | 0 | not recorded | 961.3 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_busy_g0_o1_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_busy_g0_o1_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_busy_g0_o1_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_busy_g0_o1_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_busy_g0_o1_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_busy_g0_o1_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_busy_g0_o1_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_busy_g0_o0_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:18:02 p.m. CDT · Repeated tool-return startup</strong> · rq14_busy_g0_o0_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 109.874 ms median and 479.657 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 176.3 | 1.8 | 0 | not recorded | 1183.1 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 81.0 | 1.7 | 0 | not recorded | 1321.5 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 303.5 | 8.3 | 0 | not recorded | 1179.0 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.1 | 1.8 | 0 | not recorded | 1284.5 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 109.2 | 1.7 | 0 | not recorded | 1248.3 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 105.9 | 1.8 | 0 | not recorded | 1133.9 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 88.8 | 1.9 | 0 | not recorded | 1251.3 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 482.0 | 1.7 | 0 | not recorded | 1458.8 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 111.0 | 1.7 | 0 | not recorded | 1087.2 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 138.9 | 1.9 | 0 | not recorded | 982.2 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 123.9 | 1.8 | 0 | not recorded | 966.5 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 102.4 | 1.8 | 0 | not recorded | 1032.5 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 285.8 | 5.0 | 0 | not recorded | 1154.8 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 147.2 | 1.8 | 0 | not recorded | 1314.2 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 215.3 | 1.7 | 0 | not recorded | 1090.4 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 103.4 | 1.7 | 0 | not recorded | 1229.0 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 84.8 | 1.8 | 0 | not recorded | 1300.6 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 162.9 | 1.7 | 0 | not recorded | 1113.5 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.4 | 1.7 | 0 | not recorded | 1329.4 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 479.9 | 1.7 | 0 | not recorded | 1376.3 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 92.6 | 1.8 | 0 | not recorded | 987.1 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 92.2 | 1.8 | 0 | not recorded | 1017.4 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 96.6 | 1.7 | 0 | not recorded | 999.7 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 186.5 | 1.8 | 0 | not recorded | 1031.6 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_busy_g0_o0_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='4' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_busy_g0_o0_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_busy_g0_o0_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_busy_g0_o0_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_busy_g0_o0_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_busy_g0_o0_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_busy_g0_o0_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g1_o1_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:15:27 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g1_o1_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 98.698 ms median and 146.319 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 82.8 | 1.8 | 0 | not recorded | 901.8 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 82.0 | 1.7 | 0 | not recorded | 903.6 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 83.2 | 1.7 | 0 | not recorded | 904.6 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 84.7 | 1.7 | 0 | not recorded | 1202.0 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 85.8 | 1.7 | 0 | not recorded | 911.9 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 87.4 | 1.7 | 0 | not recorded | 915.4 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 137.7 | 2.2 | 0 | not recorded | 904.4 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 89.8 | 1.8 | 0 | not recorded | 920.8 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 90.1 | 1.7 | 0 | not recorded | 920.8 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 91.9 | 1.7 | 0 | not recorded | 925.8 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 135.1 | 1.9 | 0 | not recorded | 905.6 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 93.6 | 1.7 | 0 | not recorded | 951.1 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 146.5 | 2.3 | 0 | not recorded | 906.8 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 133.2 | 2.1 | 0 | not recorded | 895.0 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 120.7 | 2.3 | 0 | not recorded | 881.1 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 418.3 | 320.8 | 0 | not recorded | 1180.2 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 104.1 | 2.3 | 0 | not recorded | 868.4 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 129.0 | 2.0 | 0 | not recorded | 894.9 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 88.7 | 1.7 | 0 | not recorded | 917.4 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 136.7 | 2.2 | 0 | not recorded | 904.8 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 120.7 | 2.2 | 0 | not recorded | 887.8 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 125.4 | 2.0 | 0 | not recorded | 895.9 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 92.9 | 1.7 | 0 | not recorded | 927.7 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 136.7 | 1.8 | 0 | not recorded | 928.8 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g1_o1_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g1_o1_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g1_o1_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g1_o1_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g1_o1_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g1_o1_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g1_o1_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g1_o0_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:13:11 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g1_o0_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 92.39 ms median and 402.143 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs on; overlap scheduling off. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 402.5 | 321.9 | 0 | not recorded | 1284.4 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 81.6 | 1.7 | 0 | not recorded | 965.0 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 82.2 | 1.7 | 0 | not recorded | 950.0 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 83.6 | 1.7 | 0 | not recorded | 955.0 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 161.2 | 1.7 | 0 | not recorded | 975.3 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 86.0 | 1.7 | 0 | not recorded | 981.4 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 89.3 | 1.7 | 0 | not recorded | 905.6 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 88.4 | 1.7 | 0 | not recorded | 968.7 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 89.4 | 1.7 | 0 | not recorded | 969.8 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 90.9 | 1.7 | 0 | not recorded | 994.1 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 121.4 | 1.8 | 0 | not recorded | 942.8 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 93.5 | 1.8 | 0 | not recorded | 1001.0 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 449.4 | 1.8 | 0 | not recorded | 1257.6 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 146.8 | 1.7 | 0 | not recorded | 955.8 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 103.4 | 1.7 | 0 | not recorded | 913.0 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 87.6 | 1.7 | 0 | not recorded | 899.9 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 85.6 | 1.7 | 0 | not recorded | 976.2 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 143.4 | 1.7 | 0 | not recorded | 960.7 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 87.2 | 1.7 | 0 | not recorded | 983.3 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 122.2 | 1.7 | 0 | not recorded | 940.4 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 121.4 | 1.7 | 0 | not recorded | 938.1 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 91.6 | 1.7 | 0 | not recorded | 912.3 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 96.8 | 1.7 | 0 | not recorded | 977.5 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 175.9 | 1.7 | 0 | not recorded | 1000.0 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g1_o0_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='1' WORK_AUDIT_OVERLAP_SCHEDULE='0' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g1_o0_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g1_o0_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g1_o0_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g1_o0_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g1_o0_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g1_o0_s1_20261006/runtime/backend_features.json)

</details>

<a id="run-rq14_pilot_g0_o1_s1_20261006"></a>
<details>
<summary><strong>Oct 6, 2026, 1:11:00 p.m. CDT · Repeated tool-return startup</strong> · rq14_pilot_g0_o1_s1_20261006</summary>

**Question (RQ14).** Across repeated tool returns and growing agent context, does the earlier one-off first-token delay recur, and how do competing sessions, CUDA graphs, and overlap scheduling change the result?

**Finding.** Across 24 active replays, first-token delay was 98.126 ms median and 141.573 ms at p95. 0 active replays had a recorded KV load-back. Stage timing identifies where time was spent, not why the backend waited.

**Setup.** nvidia_a10g_24gb; Qwen/Qwen2.5-Coder-7B-Instruct; backend 0.5.10.post1; seed 1. Repeated, deterministic synthetic tool outputs grew each session's history from 768 initial words by 96 words per turn. Tool wait 800 ms plus seeded jitter; output cap 24 tokens. CUDA graphs off; overlap scheduling on. No frontend importance ranks. Donor traffic does not by itself prove KV movement.

**Key measurements**

| Session · turn | Prompt tokens | Cached prefix tokens | KV load-backs | Tool return → first token (ms) | Lookup → batch (ms) | Load call (ms) | Load end → batch (ms) | Tool return → finish (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toolcycles-seed1-active0 · 1 | 935 | 806 | 0 | 82.0 | 1.8 | 0 | not recorded | 909.5 |
| toolcycles-seed1-active0 · 2 | 1058 | 929 | 0 | 81.7 | 1.7 | 0 | not recorded | 911.5 |
| toolcycles-seed1-active0 · 3 | 1181 | 1052 | 0 | 83.1 | 1.7 | 0 | not recorded | 911.9 |
| toolcycles-seed1-active0 · 4 | 1304 | 1175 | 0 | 84.2 | 1.7 | 0 | not recorded | 915.2 |
| toolcycles-seed1-active0 · 5 | 1427 | 1298 | 0 | 85.1 | 1.7 | 0 | not recorded | 919.6 |
| toolcycles-seed1-active0 · 6 | 1550 | 1421 | 0 | 86.5 | 1.7 | 0 | not recorded | 922.8 |
| toolcycles-seed1-active0 · 7 | 1673 | 1544 | 0 | 141.8 | 2.1 | 0 | not recorded | 919.9 |
| toolcycles-seed1-active0 · 8 | 1796 | 1667 | 0 | 89.1 | 1.7 | 0 | not recorded | 928.8 |
| toolcycles-seed1-active0 · 9 | 1919 | 1790 | 0 | 89.9 | 1.7 | 0 | not recorded | 928.9 |
| toolcycles-seed1-active0 · 10 | 2043 | 1913 | 0 | 91.7 | 1.7 | 0 | not recorded | 933.9 |
| toolcycles-seed1-active0 · 11 | 2167 | 2037 | 0 | 139.6 | 2.2 | 0 | not recorded | 922.4 |
| toolcycles-seed1-active0 · 12 | 2291 | 2161 | 0 | 93.9 | 1.8 | 0 | not recorded | 960.0 |
| toolcycles-seed1-active1 · 1 | 935 | 806 | 0 | 145.9 | 2.4 | 0 | not recorded | 916.0 |
| toolcycles-seed1-active1 · 2 | 1058 | 929 | 0 | 132.4 | 1.8 | 0 | not recorded | 903.6 |
| toolcycles-seed1-active1 · 3 | 1181 | 1052 | 0 | 121.1 | 2.2 | 0 | not recorded | 889.8 |
| toolcycles-seed1-active1 · 4 | 1304 | 1175 | 0 | 122.1 | 2.4 | 0 | not recorded | 892.2 |
| toolcycles-seed1-active1 · 5 | 1427 | 1298 | 0 | 102.8 | 1.9 | 0 | not recorded | 878.7 |
| toolcycles-seed1-active1 · 6 | 1550 | 1421 | 0 | 125.9 | 1.7 | 0 | not recorded | 903.3 |
| toolcycles-seed1-active1 · 7 | 1673 | 1544 | 0 | 88.3 | 1.7 | 0 | not recorded | 925.7 |
| toolcycles-seed1-active1 · 8 | 1796 | 1667 | 0 | 140.6 | 2.0 | 0 | not recorded | 920.2 |
| toolcycles-seed1-active1 · 9 | 1919 | 1790 | 0 | 117.7 | 2.2 | 0 | not recorded | 893.0 |
| toolcycles-seed1-active1 · 10 | 2043 | 1913 | 0 | 125.8 | 2.4 | 0 | not recorded | 908.2 |
| toolcycles-seed1-active1 · 11 | 2167 | 2037 | 0 | 92.6 | 1.7 | 0 | not recorded | 936.2 |
| toolcycles-seed1-active1 · 12 | 2291 | 2161 | 0 | 141.4 | 1.8 | 0 | not recorded | 942.3 |

The batch boundary is a scheduler-method timestamp, not measured GPU completion. Donor traffic is not proof of KV movement.

**Evidence gate.** observed. Timestamp: First request; displayed in Central Time.

**Limits**

- The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.
- These are deterministic synthetic tool results, not autonomous tool or model decisions.
- Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.

**Reproduce** (set the container image and model cache for the target host):

```bash
WORK_AUDIT_RUN_ID='rq14_pilot_g0_o1_s1_20261006' WORK_AUDIT_STUDY='tool_cycles' WORK_AUDIT_SEED='1' WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT='2' WORK_AUDIT_DONOR_COUNT='0' WORK_AUDIT_TOOL_CYCLE_TURNS='12' WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS='768' WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS='512' WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS='96' WORK_AUDIT_TOOL_CYCLE_WAIT_MS='800' WORK_AUDIT_DECODE_TOKENS='24' WORK_AUDIT_CUDA_GRAPH='0' WORK_AUDIT_OVERLAP_SCHEDULE='1' WORK_AUDIT_TRACE_ENABLE='1' HICACHE_SIZE_GB='8' MEM_FRACTION_STATIC='0.7' bash infra/container/run_work_audit_validation.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

**Evidence:** [Summary](docs/reports/work_audit/rq14_pilot_g0_o1_s1_20261006/summary.json) · [Run manifest](docs/reports/work_audit/rq14_pilot_g0_o1_s1_20261006/run_manifest.json) · [Hook gate](docs/reports/work_audit/rq14_pilot_g0_o1_s1_20261006/instrumentation_audit.json) · [Harness timeline](docs/reports/work_audit/rq14_pilot_g0_o1_s1_20261006/harness_events.jsonl) · [Raw trace](docs/reports/work_audit/rq14_pilot_g0_o1_s1_20261006/backend_trace.jsonl.gz) · [Backend features](docs/reports/work_audit/rq14_pilot_g0_o1_s1_20261006/runtime/backend_features.json)

</details>

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
