# Hint Benchmark Runbook

Compact command table for the Agentic Hint Benchmark Suite.

Run these commands on the EC2 machine from the repo root:

```bash
cd ~/agentic_hardware
```

From the Mac, SSH with the configured EC2 host:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware
bash -lc 'source aws/config.sh && ssh $(ssh_opts hintbench) "$EC2_USER@${SERVERS[0]}"'
cd ~/agentic_hardware
```

## Where Attached Legend

| Value | Meaning |
| --- | --- |
| Request | Applies to one outgoing model request. |
| Task | Applies to one task or workflow run. |
| Session | Applies to a session, so many requests in that session may carry it. |
| Configuration | Comes from a client/profile setting, so any run using that setting may carry it. |

## Source Lane Legend

| Value | Meaning |
| --- | --- |
| Native Claude Code | The real Claude Code CLI emitted the signal. |
| Claude Code + Provider | Claude Code was used, but provider routing/config may have produced the signal. |
| Native Qwen Code | The real Qwen Code CLI emitted or carried the signal at the request boundary. |
| Qwen Code + Provider Config | Qwen Code carried provider/model configuration such as `extra_body`, custom headers, or cache-control settings. |
| Native Hermes Agent | The real Hermes Agent CLI emitted or carried the signal at the request boundary. |
| Hermes Agent + Provider Config | Hermes carried provider/model configuration such as `service_tier`, `extra_body`, headers, or prompt-cache settings. |
| Native NAT Workflow | The real NeMo/NAT transport emitted the signal from workflow or transport settings. |
| NAT Pass-through | The signal was supplied before or around NAT, and NAT preserved it at the boundary. |
| Mixed NAT | The run intentionally combines NAT workflow signals and NAT pass-through signals. |

## Claude Code

<table>
<colgroup>
<col width="12%" style="width: 12%;">
<col width="18%" style="width: 18%;">
<col width="12%" style="width: 12%;">
<col width="25%" style="width: 25%;">
<col width="15%" style="width: 15%;">
<col width="16%" style="width: 16%;">
<col width="9%" style="width: 9%;">
<col width="8%" style="width: 8%;">
</colgroup>
<thead>
<tr>
<th>Run</th>
<th>Plain Purpose</th>
<th>Source Lane</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>When It Appears</th>
<th>Where Attached</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>Claude native baseline</td>
<td>Confirm Claude emits no benchmark hints when knobs are off.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_baseline_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile baseline \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li>none</li>
</ul>
</td>
<td>No benchmark hint knobs are enabled: <code>--knob-profile baseline</code>.</td>
<td>Request</td>
<td>Native Claude Code request-boundary control case.</td>
</tr>
<tr>
<td>All Claude native request-boundary probes</td>
<td>Produce all Claude Code signals observed today in one run.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile all_request_boundary \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
<li><code>messages.*.content.*.cache_control.type="ephemeral"</code></li>
<li><code>cache_control.ttl="1h"</code></li>
<li><code>anthropic-beta: fast-mode-2026-02-01</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile all_request_boundary</code>; observed signals come from stable-context cache probes, <code>claude_fast_mode_setting</code> with <code>fastMode=true</code>, <code>ENABLE_PROMPT_CACHING_1H=1</code>, and <code>FORCE_PROMPT_CACHING_5M=1</code>.</td>
<td>Configuration</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native QoS probes</td>
<td>Produce the Claude fast-mode header.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_qos_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile qos_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>anthropic-beta: fast-mode-2026-02-01</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile qos_only</code>, including scenario <code>claude_fast_mode_setting</code>.</td>
<td>Configuration</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native cache probes</td>
<td>Produce Claude prompt-cache markers and TTL.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_cache_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile cache_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
<li><code>messages.*.content.*.cache_control.type="ephemeral"</code></li>
<li><code>cache_control.ttl="1h"</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile cache_only</code>, including stable-prefix cache scenarios plus <code>claude_provider_retention_1h</code> with <code>ENABLE_PROMPT_CACHING_1H=1</code> and <code>claude_provider_retention_5m</code> with <code>FORCE_PROMPT_CACHING_5M=1</code>.</td>
<td>Configuration</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native prewarm probe</td>
<td>Send a request that warms a cacheable prompt block.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_prewarm_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile prewarm_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile prewarm_only</code> with scenario <code>claude_prewarm_like_probe</code>.</td>
<td>Request</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native fast mode only</td>
<td>Produce only the Claude fast-mode header.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_fast_mode_setting_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_fast_mode_setting \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>anthropic-beta: fast-mode-2026-02-01</code></li>
</ul>
</td>
<td>Runs scenario <code>claude_fast_mode_setting</code>, which passes Claude settings with <code>fastMode=true</code>.</td>
<td>Configuration</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native 1h cache TTL</td>
<td>Produce Claude cache control with a 1-hour TTL.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_provider_retention_1h_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_provider_retention_1h \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>cache_control.type="ephemeral"</code></li>
<li><code>cache_control.ttl="1h"</code></li>
</ul>
</td>
<td>Runs scenario <code>claude_provider_retention_1h</code> with <code>ENABLE_PROMPT_CACHING_1H=1</code>.</td>
<td>Configuration</td>
<td>Native Claude Code request-boundary capture using `ENABLE_PROMPT_CACHING_1H=1`.</td>
</tr>
<tr>
<td>Claude native 5m cache control</td>
<td>Produce Claude cache control with default 5-minute retention.</td>
<td>Native Claude Code</td>
<td>

```bash
RUN_ID="claude_provider_retention_5m_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_provider_retention_5m \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>cache_control.type="ephemeral"</code></li>
</ul>
</td>
<td>Runs scenario <code>claude_provider_retention_5m</code> with <code>FORCE_PROMPT_CACHING_5M=1</code>.</td>
<td>Configuration</td>
<td>Native Claude Code request-boundary capture using `FORCE_PROMPT_CACHING_5M=1`.</td>
</tr>
</tbody>
</table>

## Qwen Code

The current Qwen suite is request-boundary only. It proves Qwen Code can carry
configured provider/body/header/cache fields. It does not prove a real provider
or SGLang consumed them.

<table>
<colgroup>
<col width="12%" style="width: 12%;">
<col width="18%" style="width: 18%;">
<col width="12%" style="width: 12%;">
<col width="25%" style="width: 25%;">
<col width="15%" style="width: 15%;">
<col width="16%" style="width: 16%;">
<col width="9%" style="width: 9%;">
<col width="8%" style="width: 8%;">
</colgroup>
<thead>
<tr>
<th>Run</th>
<th>Plain Purpose</th>
<th>Source Lane</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>When It Appears</th>
<th>Where Attached</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>All Qwen request-boundary probes</td>
<td>Produce every Qwen signal we can observe today in one run.</td>
<td>Qwen Code + Provider Config</td>
<td>

```bash
RUN_ID="qwen_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness qwen_code \
  --knob-profile all_request_boundary \
  --qwen-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>service_tier="priority"</code></li>
<li><code>agentic_hints.priority_class="urgent"</code></li>
<li><code>x-hintbench-cache-namespace</code></li>
<li><code>cacheRetention="1h"</code></li>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
<li><code>system.*.cache_control.ttl="1h"</code></li>
<li><code>agentic_hints.eviction_priority="high"</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile all_request_boundary</code>, combining OpenAI-compatible extra-body/header probes and the Anthropic-compatible cache-control probe.</td>
<td>Configuration</td>
<td>Native Qwen Code request-boundary capture. Current artifact: <code>qwen_native_full_20260921_local</code>.</td>
</tr>
<tr>
<td>Qwen native baseline</td>
<td>Confirm Qwen emits no benchmark hints when knobs are off.</td>
<td>Native Qwen Code</td>
<td>

```bash
RUN_ID="qwen_baseline_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness qwen_code \
  --knob-profile baseline \
  --qwen-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li>none</li>
</ul>
</td>
<td>No Qwen benchmark hint knobs are enabled: <code>--knob-profile baseline</code>.</td>
<td>Request</td>
<td>Native Qwen Code request-boundary control case.</td>
</tr>
<tr>
<td>Qwen provider QoS extra body</td>
<td>Carry provider QoS metadata through Qwen's OpenAI-compatible request body.</td>
<td>Qwen Code + Provider Config</td>
<td>

```bash
RUN_ID="qwen_qos_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness qwen_code \
  --knob-profile qos_only \
  --qwen-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>service_tier="priority"</code></li>
<li><code>agentic_hints.priority_class="urgent"</code></li>
</ul>
</td>
<td>Runs scenario <code>qwen_provider_qos_extra_body</code>, whose Qwen <code>generationConfig.extra_body</code> supplies provider metadata.</td>
<td>Configuration</td>
<td>Provider/config carried by Qwen Code, not organic priority inference.</td>
</tr>
<tr>
<td>Qwen cache signals</td>
<td>Produce Qwen cache-retention and cache-control markers.</td>
<td>Qwen Code + Provider Config</td>
<td>

```bash
RUN_ID="qwen_cache_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness qwen_code \
  --knob-profile cache_only \
  --qwen-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>cacheRetention="1h"</code></li>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
<li><code>system.*.cache_control.ttl="1h"</code></li>
<li><code>anthropic-beta</code> cache TTL marker</li>
</ul>
</td>
<td>Runs OpenAI-compatible retention and Anthropic-compatible cache-control scenarios. The repeated-prefix cache-key probe is optional and did not expose a literal cache-key field today.</td>
<td>Configuration</td>
<td>Request-boundary cache metadata only; real provider cache-hit feedback is separate.</td>
</tr>
<tr>
<td>Qwen pass-through probes</td>
<td>Carry namespace and eviction metadata through Qwen configuration.</td>
<td>Qwen Code + Provider Config</td>
<td>

```bash
RUN_ID="qwen_passthrough_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness qwen_code \
  --knob-profile passthrough_only \
  --qwen-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>x-hintbench-cache-namespace</code></li>
<li><code>agentic_hints.eviction_priority="high"</code></li>
</ul>
</td>
<td>Runs namespace-header and eviction-metadata scenarios. These fields are supplied by benchmark config and preserved by Qwen.</td>
<td>Configuration</td>
<td>Pass-through/config evidence, not native Qwen scheduling or eviction policy.</td>
</tr>
</tbody>
</table>

## NeMo Agent Toolkit / NAT

<table>
<colgroup>
<col width="12%" style="width: 12%;">
<col width="18%" style="width: 18%;">
<col width="12%" style="width: 12%;">
<col width="25%" style="width: 25%;">
<col width="15%" style="width: 15%;">
<col width="16%" style="width: 16%;">
<col width="9%" style="width: 9%;">
<col width="8%" style="width: 8%;">
</colgroup>
<thead>
<tr>
<th>Run</th>
<th>Plain Purpose</th>
<th>Source Lane</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>When It Appears</th>
<th>Where Attached</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>NAT baseline</td>
<td>Confirm NAT emits no hints when knobs are off.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_baseline_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile baseline \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li>none</li>
</ul>
</td>
<td>No NAT benchmark hint knobs are enabled: <code>--knob-profile baseline</code>.</td>
<td>Request</td>
<td>Native NAT transport capture control case.</td>
</tr>
<tr>
<td>All NAT request-boundary signals</td>
<td>Produce every NAT signal we can observe today in one run.</td>
<td>Mixed NAT</td>
<td>

```bash
RUN_ID="nat_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>priority</code></li>
<li><code>latency_sensitivity</code></li>
<li><code>osl</code></li>
<li><code>iat</code></li>
<li><code>total_requests</code></li>
<li><code>prefix_id</code></li>
<li><code>nvext.cache_control.ttl</code></li>
<li><code>nvext.cache_control.type</code></li>
<li><code>nvext.cache_salt</code></li>
<li><code>provider.qos_tier</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile all_request_boundary</code>, which combines scheduling, cache-control, namespace, and provider QoS scenarios.</td>
<td>Configuration</td>
<td>Native NAT `_DynamoTransport` request-boundary capture.</td>
</tr>
<tr>
<td>Scheduling signals only</td>
<td>Produce NAT priority and request-planning signals.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_scheduling_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile scheduling_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>priority</code></li>
<li><code>latency_sensitivity</code></li>
<li><code>osl</code></li>
<li><code>iat</code></li>
<li><code>total_requests</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile scheduling_only</code>, including high/low priority, latency sensitivity, output length, cadence, and planned-count scenarios.</td>
<td>Task</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache signals only</td>
<td>Produce NAT cache reuse and cache-control signals.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_cache_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile cache_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>prefix_id</code></li>
<li><code>nvext.cache_control.ttl</code></li>
<li><code>nvext.cache_control.type</code></li>
<li><code>priority</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile cache_only</code>, including prefix reuse, TTL, ephemeral cache control, first-only cache control, and eviction-priority intent.</td>
<td>Configuration</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Pass-through signals only</td>
<td>Preserve provider and session metadata through NAT.</td>
<td>NAT Pass-through</td>
<td>

```bash
RUN_ID="nat_passthrough_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile passthrough_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>nvext.cache_salt</code></li>
<li><code>provider.qos_tier</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile passthrough_only</code>, including <code>nat_cache_namespace</code> and <code>nat_provider_qos</code>.</td>
<td>Session</td>
<td>Preserved through NAT transport; provider QoS is pass-through, not NAT-invented.</td>
</tr>
<tr>
<td>Priority high</td>
<td>Produce a high-priority NAT request.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_priority_high_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_high \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>priority=100</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_priority_high</code>, whose workflow metadata sets <code>priority="high"</code>.</td>
<td>Task</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Priority low</td>
<td>Produce a low-priority NAT request.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_priority_low_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_low \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>priority=2</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_priority_low</code>, whose workflow metadata sets <code>priority="low"</code>.</td>
<td>Task</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Latency sensitive</td>
<td>Produce a latency-sensitive NAT request.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_latency_sensitive_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_latency_sensitive \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>latency_sensitivity</code></li>
<li><code>priority</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_latency_sensitive</code>, whose workflow metadata sets <code>latency_sensitivity=100</code>.</td>
<td>Task</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Expected output length</td>
<td>Produce the expected output length hint.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_expected_output_length_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_output_length \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>osl</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_expected_output_length</code>, whose workflow metadata sets <code>osl=128</code>.</td>
<td>Task</td>
<td>Native NAT transport capture from configured workload metadata.</td>
</tr>
<tr>
<td>Expected interarrival time</td>
<td>Produce the expected request spacing hint.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_expected_interarrival_time_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_interarrival_time \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>iat</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_expected_interarrival_time</code>, whose workflow metadata sets <code>iat=750</code>.</td>
<td>Task</td>
<td>Native NAT transport capture from configured workload metadata.</td>
</tr>
<tr>
<td>Planned request count</td>
<td>Produce the planned request count hint.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_remaining_calls_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_remaining_calls \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>total_requests</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_remaining_calls</code>, whose workflow metadata sets <code>total_requests=4</code>.</td>
<td>Task</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Prefix reuse ID</td>
<td>Produce the reusable prefix/session ID hint.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_prefix_reuse_id_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_prefix_reuse_id \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>prefix_id</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_prefix_reuse_id</code>, whose workflow metadata sets a shared <code>prefix_id</code>.</td>
<td>Session</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache TTL</td>
<td>Produce a cache lifetime hint.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_cache_control_ttl_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_ttl \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>nvext.cache_control.ttl</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_cache_control_ttl</code>, whose workflow metadata sets <code>cache_control.ttl="1s"</code>.</td>
<td>Request</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Ephemeral cache entry</td>
<td>Produce an ephemeral cache-control hint.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_cache_ephemeral_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_ephemeral \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>nvext.cache_control.type="ephemeral"</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_cache_ephemeral</code>, whose workflow metadata sets <code>cache_control.type="ephemeral"</code>.</td>
<td>Request</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>First-only cache control</td>
<td>Produce cache control only on the first repeated request.</td>
<td>Native NAT Workflow</td>
<td>

```bash
RUN_ID="nat_cache_control_first_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_first_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>nvext.cache_control</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_cache_control_first_only</code>, whose workflow metadata sets <code>cache_control.mode="first_only"</code>.</td>
<td>Task</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache namespace</td>
<td>Produce a cache namespace hint.</td>
<td>NAT Pass-through</td>
<td>

```bash
RUN_ID="nat_cache_namespace_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_namespace \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>nvext.cache_salt</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_cache_namespace</code>, whose client metadata supplies <code>nvext.cache_salt</code>.</td>
<td>Session</td>
<td>Pass-through preserved by NAT transport.</td>
</tr>
<tr>
<td>Provider QoS pass-through</td>
<td>Preserve provider QoS metadata through NAT.</td>
<td>NAT Pass-through</td>
<td>

```bash
RUN_ID="nat_provider_qos_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_provider_qos \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>provider.qos_tier</code></li>
</ul>
</td>
<td>Runs scenario <code>nat_provider_qos</code>, whose provider metadata supplies <code>qos_tier</code>.</td>
<td>Configuration</td>
<td>Pass-through preserved by NAT transport; not NAT-invented.</td>
</tr>
</tbody>
</table>

## Hermes Agent

The current Hermes suite has request-boundary setup and fixture validation. A
native Hermes run requires a Hermes CLI on the host, for example
`HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes`.
Do not count fixture output as native Hermes evidence.

<table>
<colgroup>
<col width="12%" style="width: 12%;">
<col width="18%" style="width: 18%;">
<col width="12%" style="width: 12%;">
<col width="25%" style="width: 25%;">
<col width="15%" style="width: 15%;">
<col width="16%" style="width: 16%;">
<col width="9%" style="width: 9%;">
<col width="8%" style="width: 8%;">
</colgroup>
<thead>
<tr>
<th>Run</th>
<th>Plain Purpose</th>
<th>Source Lane</th>
<th>Command</th>
<th>Signals Targeted</th>
<th>When It Appears</th>
<th>Where Attached</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>All Hermes request-boundary probes</td>
<td>Produce every Hermes request-boundary signal recipe in one run.</td>
<td>Hermes Agent + Provider Config</td>
<td>

```bash
RUN_ID="hermes_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
HARNESS_HERMES_BIN="$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes" \
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness hermes_agent \
  --knob-profile all_request_boundary \
  --hermes-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>service_tier="priority"</code></li>
<li><code>agentic_hints.priority_class="urgent"</code></li>
<li><code>x-hintbench-cache-namespace</code></li>
<li><code>system.*.cache_control.type</code></li>
<li><code>system.*.cache_control.ttl="1h"</code></li>
<li>absence of literal <code>cache_pinning</code></li>
</ul>
</td>
<td>Runs <code>--knob-profile all_request_boundary</code>, combining provider QoS, namespace/isolation, prompt-cache TTL, and cache-pinning negative probes.</td>
<td>Configuration</td>
<td>Native Hermes Agent request-boundary capture once the Hermes CLI is installed. Local fixture check: <code>hermes_fixture_check</code>.</td>
</tr>
<tr>
<td>Hermes setup smoke</td>
<td>Verify the Hermes manifest, scenarios, and validator plumbing without claiming native evidence.</td>
<td>Hermes Agent + Provider Config</td>
<td>

```bash
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness hermes_agent \
  --knob-profile all_request_boundary \
  --fixture-observations \
  --run-id hermes_fixture_check
```

</td>
<td>
<ul>
<li>fixture-only validator rows</li>
<li>no unknown hints</li>
</ul>
</td>
<td>Use before native capture, or on machines without the Hermes CLI.</td>
<td>Configuration</td>
<td><code>fixture_plumbing_only</code>; not native harness evidence.</td>
</tr>
<tr>
<td>Hermes provider QoS</td>
<td>Carry service-tier metadata through Hermes provider config.</td>
<td>Hermes Agent + Provider Config</td>
<td>

```bash
RUN_ID="hermes_qos_only_$(date +%Y%m%d_%H%M%S)"
HARNESS_HERMES_BIN="$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes" \
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness hermes_agent \
  --knob-profile qos_only \
  --hermes-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>service_tier="priority"</code></li>
<li><code>agentic_hints.priority_class="urgent"</code></li>
</ul>
</td>
<td>Runs scenario <code>hermes_provider_qos_request_overrides</code>, whose provider config supplies service-tier metadata.</td>
<td>Configuration</td>
<td>Provider/config evidence, not organic urgency inference.</td>
</tr>
<tr>
<td>Hermes cache probes</td>
<td>Probe prompt-cache TTL/type and provider-derived cache-key behavior.</td>
<td>Hermes Agent + Provider Config</td>
<td>

```bash
RUN_ID="hermes_cache_only_$(date +%Y%m%d_%H%M%S)"
HARNESS_HERMES_BIN="$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes" \
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness hermes_agent \
  --knob-profile cache_only \
  --hermes-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>
<ul>
<li><code>system.*.cache_control.type</code></li>
<li><code>system.*.cache_control.ttl="1h"</code></li>
<li><code>model</code> as part of provider cache identity</li>
</ul>
</td>
<td>Runs prompt-cache TTL, repeated-prefix cache-key, and cache-pinning negative scenarios.</td>
<td>Configuration</td>
<td>Request-boundary cache metadata only; real provider cache-hit feedback is separate.</td>
</tr>
</tbody>
</table>

## Missing Or Blocked Today

| Harness | Signal | Current State | Needed To See It |
| --- | --- | --- | --- |
| NAT | cache-hit runtime feedback | Not request-boundary metadata; needs backend/runtime cache metrics. | Run a real backend cache-hit experiment and collect post-execution profiler/metrics output, not only request-boundary capture. |
| NAT | literal `cache_pinning=true` | Not observed in NAT 1.8.0; NAT exposes ephemeral/first-only cache control instead. | A NAT version/path would need to expose an actual pinning field; today use first-only/ephemeral cache control as the observable pin-like behavior. |
| Claude Code | native `service_tier=auto` / `standard_only` body field | Not observed through tested Claude Code env-var path. | A Claude Code/provider path would need to forward <code>ANTHROPIC_SERVICE_TIER</code> or <code>CLAUDE_CODE_SERVICE_TIER</code> into the request body. |
| Claude Code | literal cache key | Not observed; Claude appears to use provider-derived exact-prefix matching. | Claude Code or the provider would need to expose a literal request cache-key field; current native capture did not. |
| Claude Code | tool-level `cache_control` | Not observed in the tested tool-heavy request. | Run a Claude Code setup with stable tool definitions that the CLI/provider marks directly as <code>tools.*.cache_control</code>; our observed path cached system/message blocks instead. |
| Claude Code | native `max_tokens=0` prewarm | Not observed from Claude Code CLI in our capture. | Claude Code would need a native zero-token request path; direct API can represent this, but the native CLI path has not emitted it. |
| Claude Code | real cache-hit usage counters | Implemented runner path, but EC2 Claude CLI is not logged in today. | Use an authenticated Claude/provider run with repeated identical cacheable prefix requests inside the TTL, then capture response usage counters. |
| Qwen Code | literal cache key | Not observed in request-boundary capture; the deck frames Qwen cache keying as automatic prefix matching. | A provider or future Qwen path would need to expose a literal request cache-key field. |
| Qwen Code | real cache-hit usage counters | Request-boundary capture cannot prove provider cache hits. | Use a real provider run with repeated cacheable prefix requests and collect provider usage counters such as cached tokens. |
| Hermes Agent | native request-boundary evidence | Runner/configs are implemented, but no Hermes CLI is installed on this local machine. | Run `--hermes-native-capture` on EC2/GH200 or any host with `HARNESS_HERMES_BIN` set to a working Hermes CLI. |
| Hermes Agent | literal cache key | The deck frames Hermes cache keying as prompt-tier/model/provider behavior; a literal cache-key field is optional. | A provider or future Hermes path would need to expose a literal request cache-key field. |
| Hermes Agent | real cache-hit usage counters | Request-boundary capture cannot prove provider cache hits. | Use a real provider run with repeated cacheable prefix requests and collect provider prompt-cache metrics. |

## Inspect Results

```bash
export RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
column -s, -t "$RUN_DIR/scenario_validation_summary.csv" | less -S
column -s, -t "$RUN_DIR/hint_support_matrix.csv" | less -S
column -s, -t "$RUN_DIR/unknown_hints.csv" | less -S
```

Key files:

```text
run.json
knob_profile.json
scenario_records.jsonl
observed_hint_evidence.jsonl
expected_hint_evidence.csv
hint_validation.csv
scenario_validation_summary.csv
hint_support_matrix.csv
unknown_hints.csv
```
