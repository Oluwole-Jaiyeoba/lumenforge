# Hint Signal Findings

This document is the living findings table for harness hint experiments. It
records what each benchmark run taught us about when hints appear, where they
enter the harness path, what JSON shape they use, and what benchmark knobs can
expose or hide them.

The current completed native/request-boundary harnesses are NeMo Agent Toolkit
/ NAT, Claude Code, Qwen Code, Pi Agent Harness, and OpenClaw. Hermes Agent has
benchmark setup and fixture validation, but native Hermes evidence is still
pending a host with the Hermes CLI installed. OpenCode, Deep Agents, DeepSeek
Harness, and Codex now have manifest/scenario/knob setup plus fixture
validation; they still need native/provider capture before we can claim real
harness evidence.

## Evidence

Current NAT evidence:

```text
run_id: nat_full_coverage_missing_paths_v2_ec2
harness: nemo_agent_toolkit
execution_mode: nat_dynamo_transport_capture
scenario_count: 16
validation_rows: 20
unknown_hint_rows: 0
artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2
```

Primary evidence files:

```text
sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2/hint_support_matrix.csv
sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2/scenario_validation_summary.csv
sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2/observed_hint_evidence.jsonl
```

Important boundary: this run captures NAT request bodies after NAT's transport
has injected or preserved hint metadata, but before the request would go to
SGLang. It proves request-boundary emission/preservation. It does not prove
SGLang acted on those hints.

Current Claude status:

```text
native capture adapter: implemented
harness: claude_code
execution_mode: claude_native_capture
native CLI/client requirement: Claude Code must be installed and configured on EC2
current evidence status: updated native request-boundary run completed; real-provider feedback blocked by Claude login
current native client version: Claude Code 2.1.270
current native run_id: claude_signal_coverage_20260914_212511_native_boundary
current direct api run_id: claude_signal_coverage_20260914_212511_direct_api
current bedrock config run_id: claude_signal_coverage_20260914_212511_bedrock_config
current real-provider feedback run_id: claude_signal_coverage_20260914_212511_real_provider_feedback
```

Important boundary: Claude rows below are now native-client probes, not
synthetic payload evidence. A fixture run may test benchmark plumbing, but it
must not be counted as proof that Claude Code organically emitted a signal.

Important boundary: the earlier all-harness SGLang replay experiments are still
valuable adapter/glue evidence. They show that Claude-shaped request metadata
can travel through our gateway/backend path. They do not, by themselves, prove
that the official Claude Code CLI emitted those fields organically.

Current Qwen Code evidence:

```text
run_id: qwen_native_full_20260921_local
harness: qwen_code
execution_mode: qwen_native_capture
scenario_count: 8
validation_rows: 13
unknown_hint_rows: 0
artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/qwen_native_full_20260921_local
```

Important boundary: Qwen rows below are request-boundary captures from the real
Qwen Code CLI pointed at a local capture endpoint. They prove Qwen Code can
carry configured provider/body/header/cache fields. They do not prove a real
provider accepted or acted on those fields, and they do not prove SGLang
consumed them.

Current Hermes Agent status:

```text
native capture adapter: implemented
harness: hermes_agent
native CLI/client requirement: Hermes Agent CLI must be installed or HARNESS_HERMES_BIN must point to it
local dry-run run_id: hermes_dry_run_check
local fixture run_id: hermes_fixture_check
local evidence status: fixture plumbing only; native request-boundary evidence pending
```

Important boundary: Hermes rows below are setup/probe recipes until
`--hermes-native-capture` runs successfully against the real Hermes Agent CLI.
The fixture run validates the scenario mapping and report plumbing only.

Current Pi Agent Harness evidence:

```text
run_id: pi_native_request_boundary_rescored_20260921
harness: pi_agent_harness
execution_mode: pi_native_capture
scenario_count: 5
validation_rows: 13
unknown_hint_rows: 0
artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/pi_native_request_boundary_rescored_20260921
```

Pi signal accounting against `presentation/Harness Signal Tables As-Is.pptx`:

```text
focused serving-control signals for Pi: 16
deck-supported or conditional signals: 6
native request-boundary signals observed: 5
supported/conditional signals not observed yet: 1
deck-unsupported signals: 10
```

Important boundary: Pi rows below are native request-boundary captures from the
real Pi CLI pointed at a local capture endpoint. They prove Pi can carry the
observed provider/config/cache fields. They do not prove a real provider acted
on those fields, and they do not prove cache-hit feedback.

Current OpenClaw evidence:

```text
run_id: openclaw_native_request_boundary_20260921
harness: openclaw
execution_mode: openclaw_native_capture
scenario_count: 6
validation_rows: 15
unknown_hint_rows: 0
artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/openclaw_native_request_boundary_20260921
```

OpenClaw signal accounting against `presentation/Harness Signal Tables As-Is.pptx`:

```text
focused serving-control signals for OpenClaw: 16
deck-supported or conditional signals: 8
native request-boundary signals observed: 1
supported/conditional signals not observed yet: 7
deck-unsupported signals: 8
```

Important boundary: OpenClaw rows below are native request-boundary captures
from the real OpenClaw CLI pointed at a local capture endpoint. The current run
proves OpenClaw can preserve configured namespace headers. It did not expose
literal service-tier, provider cache, cached-WebSocket, cache-retention, or
cache-feedback fields in the local request body.

Current fixture-only setup for remaining deck harnesses:

```text
run_id: opencode_fixture_full_20260922
harness: opencode
execution_mode: fixture_smoke
scenario_count: 4
validation_rows: 5
unknown_hint_rows: 0
fixture-supported deck signals: 3

run_id: deep_agents_fixture_full_20260922
harness: deep_agents
execution_mode: fixture_smoke
scenario_count: 5
validation_rows: 10
unknown_hint_rows: 0
fixture-supported deck signals: 8

run_id: deepseek_harness_fixture_full_20260922
harness: deepseek_harness
execution_mode: fixture_smoke
scenario_count: 3
validation_rows: 6
unknown_hint_rows: 0
fixture-supported deck signals: 4

run_id: codex_fixture_full_20260922
harness: codex
execution_mode: fixture_smoke
scenario_count: 5
validation_rows: 8
unknown_hint_rows: 0
fixture-supported deck signals: 5
```

Important boundary: these four runs validate benchmark plumbing and expected
field shapes only. They do not prove that OpenCode, Deep Agents, DeepSeek
Harness, or Codex organically emitted the signals. Native/provider capture
adapters are still required for real evidence.

Benchmark outputs include an `evidence_tier` column:

| Evidence Tier | How To Interpret It |
| --- | --- |
| `native_client_or_transport_capture` | Claimable native client/transport evidence. |
| `native_client_real_provider_response` | Claimable real-provider response evidence when the client is logged in and the provider executes the request. Not request-boundary evidence. |
| `external_observed_file` | Validate the observed file provenance before citing as native evidence. |
| `fixture_plumbing_only` | Parser/report smoke test only. |
| `recipe_only` | Scenario recipe only; no observed emission. |

## Signal Findings

| Harness | Signal | Observed? | Injection level | Scope / affects | What produces it | Scenario ID | Example JSON shape | Native vs pass-through | Evidence source | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAT | `priority` high | yes | workflow level | request scheduling | Workflow marks request as high priority / latency sensitive | `nat_priority_high` | `{"nvext":{"agent_hints":{"priority":100}}}` | NAT-native emitted by `_DynamoTransport` from workflow context | `nat_dynamo_transport_capture` | Higher number means higher priority in this NAT/Dynamo path. |
| NAT | `priority` low | yes | workflow level | request scheduling | Workflow marks request as low/background priority | `nat_priority_low` | `{"nvext":{"agent_hints":{"priority":2}}}` | NAT-native emitted by `_DynamoTransport` from workflow context | `nat_dynamo_transport_capture` | This is configured workflow behavior, not semantic urgency inference. |
| NAT | `latency_sensitivity` | yes | workflow level | request scheduling | Workflow sets latency sensitivity | `nat_latency_sensitive` | `{"nvext":{"agent_hints":{"latency_sensitivity":100.0}}}` | NAT-native emitted by `_DynamoTransport` from workflow context | `nat_dynamo_transport_capture` | `priority` is derived from the same sensitivity value. |
| NAT | `osl` / expected output length | yes | workflow level | request resource estimate | Workflow config supplies expected output length | `nat_expected_output_length` | `{"nvext":{"agent_hints":{"osl":128}}}` | NAT-native emitted by `_DynamoTransport` from configured value | `nat_dynamo_transport_capture` | Current scenario supplies the value; it does not prove NAT estimates it organically. |
| NAT | `iat` / expected interarrival time | yes | workflow level | workflow/request stream timing | Workflow config supplies request cadence | `nat_expected_interarrival_time` | `{"nvext":{"agent_hints":{"iat":750}}}` | NAT-native emitted by `_DynamoTransport` from configured value | `nat_dynamo_transport_capture` | Also influences NAT's derived cache TTL when cache control is enabled. |
| NAT | `total_requests` | yes | workflow level | workflow/request stream planning | Multi-step workflow config supplies planned request count | `nat_remaining_calls` | `{"nvext":{"agent_hints":{"total_requests":4}}}` | NAT-native emitted by `_DynamoTransport` from configured value | `nat_dynamo_transport_capture` | In this path it is total planned requests, not dynamically decremented remaining calls. |
| NAT | `prefix_id` | yes | workflow level | workflow/session reuse identity | Repeated workflow/session runs under a stable prefix ID | `nat_prefix_reuse_id` | `{"nvext":{"agent_hints":{"prefix_id":"nat_bench_shared_prefix_001"}}}` | NAT-native emitted by `_DynamoTransport` from prefix context | `nat_dynamo_transport_capture` | Useful as a reuse identity, but not identical to a backend cache key. |
| NAT | `nvext.cache_control.ttl` | yes | workflow level | cache entry/request cache lifetime | Cache-control scenario enables NAT cache control | `nat_cache_control_ttl` | `{"nvext":{"cache_control":{"ttl":"1s"}}}` | NAT-native emitted by `_DynamoTransport` | `nat_dynamo_transport_capture` | TTL is computed from `total_requests * iat`, rounded to seconds/minutes. |
| NAT | `nvext.cache_control.type` | yes | cache entry level | cache entry behavior | Cache-control scenario uses NAT `CachePinType.EPHEMERAL` | `nat_cache_ephemeral` | `{"nvext":{"cache_control":{"type":"ephemeral"}}}` | NAT-native emitted by `_DynamoTransport` | `nat_dynamo_transport_capture` | NAT 1.8.0 exposes `ephemeral`, not a larger enum of pinning policies. |
| NAT | first-request cache control | yes | cache entry level | cache entry behavior | `CacheControlMode.FIRST_ONLY` with repeated prefix | `nat_cache_control_first_only` | request 1: `{"nvext":{"cache_control":{"type":"ephemeral","ttl":"1s"}}}`; request 2 omits `cache_control` | NAT-native behavior | `nat_dynamo_transport_capture` | Closest concrete NAT behavior to pin-like first-prefix retention. |
| NAT | `nvext.cache_salt` / cache namespace | yes | session level | session/request cache isolation | Client/session attaches cache namespace before NAT | `nat_cache_namespace` | `{"nvext":{"cache_salt":"nat_bench_tenant_a"}}` | pass-through preserved by NAT | `nat_dynamo_transport_capture` | This benchmark does not show NAT inventing the namespace. |
| NAT | provider QoS | yes | provider level | provider/model behavior | Provider metadata is attached before NAT request capture | `nat_provider_qos` | `{"provider":{"qos_tier":"provider_specific_fast_or_priority"}}` | pass-through preserved by NAT | `nat_dynamo_transport_capture` | Needs a concrete provider integration before treating this as provider-native. |
| NAT | priority-derived eviction intent | yes | workflow level | scheduling/cache-retention intent | Cacheable high-priority workflow exposes priority beside cache control | `nat_eviction_priority` | `{"nvext":{"agent_hints":{"priority":100}}}` | NAT-native priority reused as eviction intent evidence | `nat_dynamo_transport_capture` | No separate eviction-priority field was observed. |
| NAT | cache-hit feedback | no | runtime feedback level | post-execution metrics | Requires real backend execution and cache metrics | `nat_cache_feedback_metrics` | expected future shape: metrics/profiler row | not request-boundary metadata | not observed in direct capture | Needs real SGLang runtime metrics. |
| NAT | separate `cache_pinning=true` | no | cache entry level | cache retention behavior | Would require a NAT version/path with explicit pinning field | `nat_cache_pinning` | expected future shape: `{"cache_pinning":true}` | not observed in NAT 1.8.0 | not observed in direct capture | NAT 1.8.0 exposes FIRST_ONLY/ephemeral cache control instead. |
| Claude Code | provider QoS / service tier | no | session level | provider/model behavior | Client/session config attempted to set Claude `service_tier` | `claude_service_tier_auto`, `claude_service_tier_standard_only` | expected: `{"service_tier":"auto"}` or `{"service_tier":"standard_only"}` | native Claude client probe | `claude_native_full_coverage_20260914_203812` | Claude Code 2.1.270 did not expose `service_tier` through the tested env-var path. |
| Claude Code | fast-mode request behavior | yes | session level | provider/model behavior | Claude Code run used `--settings {"fastMode":true}` | `claude_fast_mode_setting` | `anthropic-beta` contained `fast-mode-2026-02-01` | native Claude client probe | `claude_qos_strict_20260914_212833` | Native Claude Code emitted fast mode as a header token, not as a JSON body like `{"qos_tier":["fast","speed"]}`. |
| Claude direct API | fast speed / QoS | yes | request level | provider/model behavior | Direct Anthropic API request supplies fast speed and beta header | `claude_api_fast_mode` | `{"speed":"fast"}` plus `anthropic-beta: fast-mode-2026-02-01`; response `{"usage":{"speed":"fast"}}` | documented direct API payload | `anthropic_api_payload_capture` | This proves the benchmark can represent the API capability; it is not native Claude Code CLI emission. |
| Claude Bedrock config | service tier priority | yes | provider level | provider/model behavior | Bedrock provider config supplies service-tier header | `claude_bedrock_service_tier_priority` | `{"_capture":{"headers":{"x-amzn-bedrock-service-tier":"priority"}}}` | provider-config payload | `anthropic_api_payload_capture` | This is Bedrock/provider-config evidence, not normal Anthropic API or Claude Code CLI evidence. |
| Claude Code | exact-prefix cache key behavior | optional not observed | cache entry level | prompt cache matching | Repeated identical client context may cause provider-derived exact-prefix reuse | `claude_repeated_session_prefix` | optional expected marker: `{"cache_key_policy":"provider_exact_prefix_hash"}` | provider-derived behavior probe | `claude_native_full_coverage_20260914_203812` | No literal cache-key field was exposed in the captured request body. |
| Claude Code | prompt `cache_control` | yes | content block level | prompt cache control | Claude Code prompt builder marked reusable system/context blocks | `claude_long_running_cache_session` | `system.*.cache_control.type="ephemeral"` and `messages.*.content.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | Original exact-index validator missed this; wildcard re-score found it. |
| Claude Code | cache TTL 1h | yes | cache entry level | cache retention | `ENABLE_PROMPT_CACHING_1H=1` with reusable stable context | `claude_provider_retention_1h` | `system.*.cache_control.ttl="1h"` and `messages.*.content.*.cache_control.ttl="1h"` | native Claude client probe | `claude_signal_coverage_20260914_212511_native_boundary` | This confirms the documented 1h retention knob at the request boundary. |
| Claude Code | cache TTL 5m | partial | cache entry level | cache retention | `FORCE_PROMPT_CACHING_5M=1` with reusable stable context | `claude_provider_retention_5m` | observed `cache_control.type="ephemeral"` | native Claude client probe | `claude_signal_coverage_20260914_212511_native_boundary` | No literal `ttl` field was present for 5m; this likely behaves as default/implicit short retention. |
| Claude direct API | explicit cache TTL | yes | cache entry level | cache retention | Direct Anthropic API request puts TTL on a cache-control block | `claude_api_cache_ttl_1h` | `{"system":[{"cache_control":{"type":"ephemeral","ttl":"1h"}}]}` | documented direct API payload | `anthropic_api_payload_capture` | Represents long-retention/pin-like behavior through TTL, not a separate `cache_pinning=true` field. |
| Claude Code | tool block `cache_control` | optional not observed | content block level | reusable tool definitions | Tool-heavy request checks whether stable tool definitions receive cache markers | `claude_tool_heavy_request` | expected optional `tools.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | The captured request had tools, but no tool-level `cache_control`. |
| Claude Code | system block `cache_control` | yes | content block level | reusable system prompt | Claude Code prompt builder marked stable system blocks | `claude_stable_system_context` | `system.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | Observed at nonzero system indexes. |
| Claude Code | message block `cache_control` | yes | content block level | reusable message prefix | Claude Code prompt builder marked stable environment/context message blocks | `claude_stable_message_context` | `messages.*.content.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | Observed at nonzero message/content indexes. |
| Claude Code | prompt cache prewarm | partial | request level | cache warmup before real request | Prewarm-like client probe checks whether the real client can emit `max_tokens=0` with cache control | `claude_prewarm_like_probe` | observed cache control, but `max_tokens=64000` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | This showed cache-control, but did not become a true `max_tokens=0` prewarm request. |
| Claude direct API | prompt cache prewarm | yes | request level | cache warmup before real request | Direct Anthropic API request uses `max_tokens=0` with cache-controlled prefix | `claude_api_prewarm_max_tokens_zero` | `{"max_tokens":0,"system":[{"cache_control":{"type":"ephemeral"}}]}` | documented direct API payload | `anthropic_api_payload_capture` | Direct API capability only; native Claude Code CLI did not emit this shape in the previous run. |
| Claude Code | cache-hit feedback | blocked | runtime feedback level | post-execution usage metrics | Real provider/backend response must include cache counters | `claude_provider_cache_feedback_probe` | expected: `{"usage":{"cache_creation_input_tokens":...,"cache_read_input_tokens":...}}` | real provider response required | `claude_signal_coverage_20260914_212511_real_provider_feedback` | EC2 Claude CLI returned `Not logged in`, so the usage counters were zero and this cannot prove cache-hit feedback yet. |
| Claude direct API | cache-hit feedback shape | yes | runtime feedback level | post-execution usage metrics | Direct API response fixture includes documented cache usage counters | `claude_api_cache_feedback_fixture` | `{"usage":{"cache_creation_input_tokens":248,"cache_read_input_tokens":1800}}` | documented response payload | `anthropic_api_payload_capture` | Validates reporting/validation path. Real cache proof still requires real provider execution. |
| Claude Code | separate `cache_pinning=true` | no | cache entry level | cache retention behavior | Negative probe checks whether the real client emits literal pinning | `claude_cache_pinning_negative_probe` | no `cache_pinning` field observed | native Claude client negative probe | `claude_native_full_coverage_20260914_203812` | Negative probe passed; Claude documents TTL/cache-control behavior, not a separate pin flag. |
| Qwen Code | provider QoS / service tier | yes | provider level | provider/model behavior | Qwen `generationConfig.extra_body` supplies provider QoS metadata | `qwen_provider_qos_extra_body` | `{"service_tier":"priority","agentic_hints":{"priority_class":"urgent"}}` | provider/config carried by Qwen Code | `qwen_native_full_20260921_local` | This is configured provider metadata, not organic urgency inference. |
| Qwen Code | cache namespace header | yes | session level | cache namespace / gateway routing | Qwen `generationConfig.customHeaders` supplies a namespace header | `qwen_custom_header_namespace` | header `x-hintbench-cache-namespace: qwen_bench_tenant_a` | pass-through preserved by Qwen Code | `qwen_native_full_20260921_local` | This proves header preservation only; Qwen does not invent the namespace. |
| Qwen Code | cache retention body field | yes | cache entry level | provider cache retention | Qwen OpenAI-compatible `extra_body` supplies `cacheRetention` | `qwen_cache_retention_openai_body` | `{"cacheRetention":"1h"}` | provider/config carried by Qwen Code | `qwen_native_full_20260921_local` | Provider must define the meaning of this field. |
| Qwen Code | Anthropic-style cache control | yes | content block level | prompt cache control | Qwen Anthropic-compatible path enables cache control with 1h retention | `qwen_anthropic_cache_control_1h` | `system.*.cache_control.type="ephemeral"` and `system.*.cache_control.ttl="1h"` | Qwen prompt-builder/cache config emitted request markers | `qwen_native_full_20260921_local` | Also emitted Anthropic beta header containing `extended-cache-ttl`. |
| Qwen Code | eviction priority metadata | yes | request level | cache-retention intent / gateway extension | Qwen OpenAI-compatible `extra_body` supplies eviction metadata | `qwen_eviction_priority_passthrough` | `{"agentic_hints":{"eviction_priority":"high"}}` | pass-through preserved by Qwen Code | `qwen_native_full_20260921_local` | No native Qwen eviction-priority policy was observed. |
| Qwen Code | literal cache key | optional not observed | cache entry level | prompt cache matching | Repeated prefix probe checks for a client-visible cache key field | `qwen_repeated_prefix_cache_probe` | optional expected marker: `{"cache_key_policy":"provider_prefix_hash"}` | provider-automatic behavior probe | `qwen_native_full_20260921_local` | The deck labels this as automatic prefix matching; no literal request cache-key field appeared. |
| Qwen Code | cache-hit feedback | optional not observed | runtime feedback level | post-execution usage metrics | Requires real provider response or metrics with cache counters | `qwen_cache_feedback_usage` | optional expected marker: `{"usage":{"cached_tokens":...}}` | real provider/metrics required | `qwen_native_full_20260921_local` | Local request-boundary capture cannot prove provider cache-hit feedback. |
| Hermes Agent | provider QoS / service tier | pending native | provider level | provider/model behavior | Hermes provider config or request overrides supply service-tier metadata | `hermes_provider_qos_request_overrides` | expected: `{"service_tier":"priority","agentic_hints":{"priority_class":"urgent"}}` | provider/config carried by Hermes | `hermes_fixture_check` | Fixture plumbing passes, but no local Hermes CLI was available for native capture. |
| Hermes Agent | provider/session/model namespace | pending native | session level | cache namespace / isolation | Hermes/provider isolation is scoped by provider, session, and model, with optional gateway header metadata | `hermes_provider_session_model_namespace` | expected optional header `x-hintbench-cache-namespace: hermes_provider_session_model` | provider/config or provider-automatic behavior | `hermes_fixture_check` | A literal namespace field may not exist; native capture is pending. |
| Hermes Agent | prompt cache TTL/type | pending native | content block level | prompt cache control | Hermes prompt-caching config requests cache breakpoints and 1h TTL where supported | `hermes_prompt_cache_ttl_1h` | expected optional `system.*.cache_control.type="ephemeral"` and `system.*.cache_control.ttl="1h"` | Hermes prompt-builder/provider config | `hermes_fixture_check` | OpenAI-compatible local capture may only reveal provider config; Anthropic-compatible provider behavior needs native/provider run. |
| Hermes Agent | prompt-tier/model cache key | pending native | cache entry level | prompt cache matching | Repeated-prefix probe checks for a client-visible cache key; model identity is observable | `hermes_prompt_tier_cache_key_probe` | expected optional `{"cache_key_policy":"prompt_tiers_model"}` plus `{"model":"hermes-hint-benchmark-model"}` | provider-automatic behavior probe | `hermes_fixture_check` | The deck frames this as prompt-tier/model/provider behavior, not necessarily a literal cache key. |
| Hermes Agent | cache-hit feedback | pending real provider | runtime feedback level | post-execution usage metrics | Requires real provider response or prompt-cache metrics | `hermes_cache_feedback_metrics` | expected optional `{"usage":{"cache_read_input_tokens":...}}` | real provider/metrics required | `hermes_fixture_check` | Request-boundary capture cannot prove cache-hit feedback. |
| Hermes Agent | separate `cache_pinning=true` | pending native negative probe | cache entry level | cache retention behavior | Negative probe checks whether Hermes emits a literal pinning field separate from TTL retention | `hermes_cache_pinning_negative_probe` | expected absence of `cache_pinning`; optional `system.*.cache_control.ttl="1h"` | provider retention behavior | `hermes_fixture_check` | The deck describes 1h retention where supported, not a separate pin flag. |
| Pi Agent Harness | prompt-cache key | yes | cache entry level | prompt cache matching | Pi provider/model cache configuration emitted a prompt cache key under long-retention cache mode | `pi_cache_retention_long` | `{"prompt_cache_key":"01a0..."}` | provider/config carried by Pi | `pi_native_request_boundary_rescored_20260921` | This is a provider prompt-cache key, not a portable SGLang KV key. |
| Pi Agent Harness | provider/account namespace | yes | session level | cache namespace / isolation | Pi session-affinity and configured namespace headers survived to the request boundary | `pi_provider_account_namespace` | headers `x-session-affinity: pi_hintbench_provider_account`, `x-hintbench-cache-namespace: pi_provider_account` | provider/config carried by Pi | `pi_native_request_boundary_rescored_20260921` | This proves header preservation/session affinity, not provider-side isolation behavior. |
| Pi Agent Harness | cache TTL / long retention | yes | cache entry level | provider cache retention | Pi cache-retention config emitted `prompt_cache_retention` and cache-control TTL markers | `pi_cache_retention_long` | `{"prompt_cache_retention":"24h"}` and `messages.*.content.*.cache_control.ttl="1h"` | provider/config carried by Pi | `pi_native_request_boundary_rescored_20260921` | Pi normalized the requested long retention to `24h` on this path. |
| Pi Agent Harness | cache entry type | yes | content block level | prompt cache control | Pi emitted Anthropic-style ephemeral cache-control markers | `pi_cache_retention_long` | `messages.*.content.*.cache_control.type="ephemeral"` | provider/config carried by Pi | `pi_native_request_boundary_rescored_20260921` | Observed on message content blocks. |
| Pi Agent Harness | cache pinning / long retention | yes | cache entry level | cache retention behavior | Pinning-negative probe observed long retention and absence of literal `cache_pinning` | `pi_cache_pinning_negative_probe` | `{"prompt_cache_retention":"24h"}` and no `cache_pinning` field | provider retention behavior | `pi_native_request_boundary_rescored_20260921` | Matches the deck wording: long retention where provider supports it, not pinning. |
| Pi Agent Harness | cache-hit feedback | no | runtime feedback level | post-execution usage metrics | Requires real provider response or Pi footer/provider cache usage counters | `pi_cache_feedback_footer_usage` | expected future shape: `usage.cacheRead`, `footer.cache_usage`, or provider cache counters | real provider/metrics required | not observed in local request-boundary capture | Local capture returns dummy usage, so it cannot prove a cache hit. |
| OpenClaw | provider QoS / service tier | optional not observed | provider level | provider/model behavior | OpenClaw provider config attempted to supply service-tier metadata | `openclaw_provider_qos_fast_mode` | expected optional `{"service_tier":"priority"}` or `{"extra_body":{"service_tier":"priority"}}` | provider/config candidate carried by OpenClaw | `openclaw_native_request_boundary_20260921` | Slide 8 maps fast mode to `service_tier=priority`, but the tested native path did not forward the field. |
| OpenClaw | provider/session namespace | yes | session level | cache namespace / isolation | Configured OpenClaw provider header survived to the request boundary | `openclaw_provider_session_namespace` | header `x-hintbench-cache-namespace: openclaw_provider_session` | provider/config carried by OpenClaw | `openclaw_native_request_boundary_20260921` | This proves header preservation only, not provider-side cache isolation behavior. |
| OpenClaw | provider cache key | optional not observed | cache entry level | prompt cache matching | Provider cache config attempted to supply a prompt cache key | `openclaw_provider_cache_config` | expected optional `{"prompt_cache_key":...}` | provider-managed cache candidate | `openclaw_native_request_boundary_20260921` | Slide 13 labels cache keying as provider-managed; no literal cache key appeared in the request body. |
| OpenClaw | cache TTL / retention | optional not observed | cache entry level | provider cache retention | Provider cache config attempted to supply long retention | `openclaw_provider_cache_config` | expected optional `{"prompt_cache_retention":"long"}` or `{"prompt_cache_retention":"24h"}` | provider-managed cache candidate | `openclaw_native_request_boundary_20260921` | No literal retention field appeared in the tested native path. |
| OpenClaw | cache entry type | optional not observed | content block level | prompt cache control | Provider cache config attempted to expose cache-control markers | `openclaw_provider_cache_config` | expected optional `messages.*.content.*.cache_control.type="ephemeral"` | provider-managed cache candidate | `openclaw_native_request_boundary_20260921` | The request body did not contain cache-control markers. |
| OpenClaw | cached WebSocket / prewarm | optional not observed | request level | cache warmup / alternate reuse mechanism | Cached-WebSocket probe checked for OpenClaw prewarm metadata | `openclaw_cached_websocket_prewarm_probe` | expected optional `cached_websocket=true` or `websocket_prewarm=true`; no `speculative_prefill` field | different mechanism candidate | `openclaw_native_request_boundary_20260921` | The deck says this is not KV prefill; the tested request did not expose cached-WebSocket metadata either. |
| OpenClaw | cache-hit feedback | no | runtime feedback level | post-execution usage metrics | Requires real provider trace or usage counters | `openclaw_provider_trace_feedback` | expected future shape: `usage.cached_tokens` or `trace.cache_hit` | real provider/metrics required | not observed in local request-boundary capture | Local capture cannot prove provider cache-hit feedback. |
| OpenClaw | cache pinning / retention | optional not observed | cache entry level | cache retention behavior | Pinning-negative probe checked whether OpenClaw emits literal pinning or provider retention | `openclaw_cache_pinning_negative_probe` | no `cache_pinning` field; expected optional `prompt_cache_retention` did not appear | provider-managed retention candidate | `openclaw_native_request_boundary_20260921` | Matches the deck boundary: provider-managed only, with no literal pin flag observed. |

## Benchmark Knobs

The executable knob catalogs are:

```text
sglang_direct_kv/configs/hint_benchmark/nat_knobs.json
sglang_direct_kv/configs/hint_benchmark/claude_knobs.json
sglang_direct_kv/configs/hint_benchmark/qwen_knobs.json
sglang_direct_kv/configs/hint_benchmark/hermes_knobs.json
sglang_direct_kv/configs/hint_benchmark/pi_knobs.json
sglang_direct_kv/configs/hint_benchmark/openclaw_knobs.json
sglang_direct_kv/configs/hint_benchmark/opencode_knobs.json
sglang_direct_kv/configs/hint_benchmark/deep_agents_knobs.json
sglang_direct_kv/configs/hint_benchmark/deepseek_knobs.json
sglang_direct_kv/configs/hint_benchmark/codex_knobs.json
```

The operational runbook with copy-paste commands is:

```text
HINT_BENCHMARK_RUNBOOK.md
```

Use knob profiles when the benchmark user wants to stress one signal family
without manually listing scenario IDs.

| Knob | Values | Signals it can expose | What it changes | Applies to | Backend required? |
| --- | --- | --- | --- | --- | --- |
| `priority_level` | `unset`, `low`, `high` | `priority`, `latency_sensitivity` | Sets NAT workflow latency/priority context. | NAT | no |
| `expected_output_length` | `unset`, `configured` | `osl` | Sets expected output length sent through NAT's Dynamo transport. | NAT | no |
| `request_cadence` | `unset`, `configured` | `iat`, `total_requests`, derived TTL | Sets expected request cadence and planned request count. | NAT | no |
| `prefix_reuse` | `false`, `true` | `prefix_id` | Runs related requests under the same prefix/workflow identity. | NAT | no |
| `cache_control_mode` | `disabled`, `always`, `first_only` | `nvext.cache_control.ttl`, `nvext.cache_control.type` | Controls whether cache control appears never, on every request, or only on the first request. | NAT | no |
| `cache_namespace` | `disabled`, `client_supplied` | `nvext.cache_salt` | Adds client/session cache namespace and verifies preservation. | NAT | no |
| `provider_qos` | `disabled`, `provider_supplied` | `provider.qos_tier` | Adds provider-level QoS metadata and verifies preservation. | NAT | no |
| `runtime_metrics` | `disabled`, `real_backend_required` | `cache_hit_feedback` | Runs a real backend path and inspects post-execution metrics. | NAT future | yes |
| `claude_service_tier` | `auto`, `standard_only`, `fast`, Bedrock `priority` | Claude `service_tier`, `speed`, `usage.speed`, `anthropic-beta`, Bedrock service-tier header | Runs native Claude client probes plus documented API/provider-config capability probes. | Claude | no |
| `claude_cache_control_location` | tools, system, messages, long-running context | Claude `cache_control` | Runs native Claude client scenarios that may cause prompt-builder cache markers. | Claude | no |
| `claude_cache_ttl` | default/provider-managed, explicit `1h` | Claude `cache_control.ttl` | Runs native Claude client long-retention probes and direct API TTL probes. | Claude | no |
| `claude_cache_feedback` | real provider response | `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens` | Requires a real provider/backend response path to observe cache feedback. | Claude | yes |
| `qwen_extra_body` | provider-config fields | `service_tier`, `agentic_hints.priority_class`, `cacheRetention`, `agentic_hints.eviction_priority` | Adds OpenAI-compatible `generationConfig.extra_body` fields and verifies Qwen carries them. | Qwen Code | no |
| `qwen_custom_headers` | static headers | `x-hintbench-cache-namespace` | Adds Qwen `generationConfig.customHeaders` and verifies request-boundary header preservation. | Qwen Code | no |
| `qwen_anthropic_cache_control` | enabled, 1h | `cache_control.type`, `cache_control.ttl`, Anthropic cache beta header | Runs Qwen over the Anthropic-compatible path with cache control enabled. | Qwen Code | no |
| `qwen_cache_feedback` | real provider response | `usage.cached_tokens` or provider-specific cache counters | Requires a real provider/backend response path to observe cache feedback. | Qwen Code | yes |
| `hermes_provider_qos` | service tier / request overrides | `service_tier`, `agentic_hints.priority_class` | Adds Hermes provider config for QoS/service tier and verifies native request-boundary visibility when the CLI is available. | Hermes Agent | no |
| `hermes_prompt_cache` | enabled, 1h | `cache_control.type`, `cache_control.ttl`, `prompt_caching.cache_ttl` | Configures Hermes prompt caching and probes whether cache-control markers or config fields appear. | Hermes Agent | no |
| `hermes_cache_identity` | repeated prefix, model identity | `model`, optional `cache_key_policy` | Repeats a stable prompt to probe provider-derived prompt-tier/model cache identity. | Hermes Agent | no |
| `hermes_cache_feedback` | real provider response | `usage.cache_read_input_tokens` or provider-specific cache counters | Requires a real provider/backend response path to observe cache feedback. | Hermes Agent | yes |
| `pi_prompt_cache` | provider prompt cache, long retention | `prompt_cache_key`, `prompt_cache_retention`, `cache_control.type`, `cache_control.ttl` | Configures Pi provider cache compatibility and verifies native request-boundary cache fields. | Pi Agent Harness | no |
| `pi_namespace` | session affinity / provider account | `x-session-affinity`, namespace header | Enables Pi session affinity and verifies namespace metadata at the request boundary. | Pi Agent Harness | no |
| `pi_cache_feedback` | real provider response | Pi footer/provider usage cache counters | Requires a real provider/backend response path to observe cache feedback. | Pi Agent Harness | yes |
| `openclaw_provider_qos` | service tier / fast mode | optional `service_tier`, `extra_body.service_tier` | Configures OpenClaw provider QoS candidates and checks whether the native request carries them. | OpenClaw | no |
| `openclaw_namespace` | provider/session namespace | `x-hintbench-cache-namespace` | Adds OpenClaw provider header metadata and verifies request-boundary preservation. | OpenClaw | no |
| `openclaw_provider_cache` | provider-managed cache config | optional `prompt_cache_key`, `prompt_cache_retention`, `cache_control.type` | Configures provider cache candidates; current native request-boundary run did not expose literal cache fields. | OpenClaw | no |
| `openclaw_cache_feedback` | real provider trace/usage | `usage.cached_tokens`, `trace.cache_hit` | Requires a real provider/backend response path to observe cache feedback. | OpenClaw | yes |
| `opencode_provider_cache` | provider/plugin cache config | `helicone-cache-key`, namespace header | Adds OpenCode provider/plugin cache recipes; fixture-only until native/provider capture exists. | OpenCode | maybe |
| `opencode_cache_feedback` | provider/plugin usage | `usage.cached_tokens` | Requires provider/plugin usage output for real evidence. | OpenCode | yes |
| `deep_agents_provider_qos` | provider QoS | `provider.qos_tier` | Adds Deep Agents provider QoS recipe; fixture-only until native/provider capture exists. | Deep Agents | maybe |
| `deep_agents_middleware_cache` | middleware cache config | cache key, namespace, TTL, type, eviction, retention | Adds Deep Agents middleware cache recipes; fixture-only until capture exists. | Deep Agents | maybe |
| `deep_agents_cache_feedback` | provider trace | `provider_trace.cache_hit` | Requires provider trace output for real evidence. | Deep Agents | yes |
| `deepseek_provider_cache` | provider-managed cache | prefix hash, provider TTL, provider retention | Adds DeepSeek provider cache recipes; fixture-only until native/provider capture exists. | DeepSeek Harness | maybe |
| `deepseek_cache_feedback` | provider usage | `usage.cacheReadTokens` | Requires provider usage output for real evidence. | DeepSeek Harness | yes |
| `codex_provider_qos` | service tier | `service_tier` | Adds Codex service-tier recipe; fixture-only until native/provider capture exists. | Codex | maybe |
| `codex_prompt_cache` | prompt cache and prewarm | `prompt_cache_key`, cache bucketing, `websocket_prewarm` | Adds Codex prompt-cache and WebSocket prewarm recipes; fixture-only until capture exists. | Codex | maybe |
| `codex_cache_feedback` | provider usage | `usage.cached_input_tokens` | Requires provider/client usage output for real evidence. | Codex | yes |

## Knob Profiles

| Profile | Purpose | Scenario selection | Expected visibility |
| --- | --- | --- | --- |
| `baseline` | Hide intentional hints. | `nat_no_hints_baseline` | none |
| `scheduling_only` | Expose scheduling/workload hints. | priority, sensitivity, `osl`, `iat`, `total_requests` scenarios | `priority`, `latency_sensitivity`, `osl`, `iat`, `total_requests` |
| `cache_only` | Expose prefix and cache-control hints. | prefix, TTL, ephemeral, FIRST_ONLY, eviction-intent scenarios | `prefix_id`, `nvext.cache_control.ttl`, `nvext.cache_control.type`, `priority` |
| `passthrough_only` | Expose client/session and provider pass-through. | namespace and QoS scenarios | `nvext.cache_salt`, `provider.qos_tier` |
| `all_request_boundary` | Show everything visible without SGLang. | `full_nat_coverage` | all request-boundary observed signals |
| `runtime_feedback` | Reserved for cache-hit metrics. | `nat_cache_feedback_metrics` | `cache_hit_feedback` after backend run |
| Claude `baseline` | Hide intentional hints. | `claude_no_hints_baseline` | none |
| Claude `qos_only` | Probe Claude Code native service-tier and fast-mode variants. | native service-tier probes plus `fastMode=true` probe | `service_tier`, `anthropic-beta` |
| Claude `cache_only` | Probe Claude Code native cache-control locations and TTL. | native cache-control probes, `ENABLE_PROMPT_CACHING_1H`, and `FORCE_PROMPT_CACHING_5M` | `cache_control`, optional `cache_control.ttl`, optional cache beta/header behavior |
| Claude `prewarm_only` | Probe whether Claude Code CLI emits max-token-zero prewarm. | native prewarm-like probe | `max_tokens=0`, `cache_control` |
| Claude `feedback_only` | Probe whether Claude Code native path can expose cache feedback. | native real-provider probe | `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens` |
| Claude `real_provider_feedback` | Run real Claude Code provider-response probes. | real provider cache-feedback and TTL probes | cache creation/read usage counters and TTL usage buckets when exposed |
| Claude `native_client_boundary` | Run Claude Code CLI-only probes. | native Claude Code scenarios | native Claude Code probe targets only |
| Claude `direct_api_capabilities` | Run direct Anthropic API capability probes. | `direct_anthropic_api_coverage` | `speed`, explicit TTL, `max_tokens=0`, cache feedback |
| Claude `bedrock_provider_config` | Run Bedrock provider-config probes. | `bedrock_config_coverage` | Bedrock service-tier header |
| Claude `all_request_boundary` | Run every Claude Code native-client boundary probe. | `full_claude_coverage` | native Claude Code probe targets only |
| Claude `all_signal_recipes` | List every Claude signal recipe. | `all_claude_signal_recipes` | native CLI, direct API, and provider-config recipes |
| Qwen `baseline` | Hide intentional hints. | `qwen_no_hints_baseline` | none |
| Qwen `qos_only` | Probe Qwen OpenAI-compatible provider QoS extra body. | `qwen_provider_qos_extra_body` | `service_tier`, `agentic_hints.priority_class` |
| Qwen `cache_only` | Probe Qwen cache retention and cache-control markers. | OpenAI retention, Anthropic cache-control, repeated-prefix probe | `cacheRetention`, `cache_control.type`, `cache_control.ttl` |
| Qwen `passthrough_only` | Probe Qwen configured pass-through fields. | namespace header and eviction metadata scenarios | custom namespace header, `agentic_hints.eviction_priority` |
| Qwen `openai_compatible` | Run Qwen OpenAI-compatible request-boundary probes. | `qwen_openai_compatible_coverage` | OpenAI-compatible provider/config and pass-through fields |
| Qwen `anthropic_cache` | Run Qwen Anthropic-compatible cache-control probes. | `qwen_anthropic_cache_coverage` | Anthropic-style `cache_control` markers |
| Qwen `all_request_boundary` | Run every Qwen request-boundary probe. | `full_qwen_coverage` | native Qwen request-boundary probe targets only |
| Hermes `baseline` | Hide intentional hints. | `hermes_no_hints_baseline` | none |
| Hermes `qos_only` | Probe Hermes provider QoS/service-tier config. | `hermes_provider_qos_request_overrides` | `service_tier`, optional `agentic_hints.priority_class` |
| Hermes `cache_only` | Probe Hermes prompt-cache TTL/type and cache identity. | prompt-cache TTL, repeated-prefix cache-key, and pinning-negative scenarios | optional `cache_control`, `prompt_caching.cache_ttl`, `model` |
| Hermes `namespace_only` | Probe Hermes provider/session/model isolation metadata. | `hermes_provider_session_model_namespace` | optional namespace header or literal provider/session/model marker |
| Hermes `feedback_only` | Probe whether Hermes/provider can expose cache feedback. | `hermes_cache_feedback_metrics` | cache-read/cached-token usage counters |
| Hermes `all_request_boundary` | Run every Hermes request-boundary probe. | `hermes_request_boundary_coverage` | native Hermes request-boundary probe targets only |
| Hermes `full_coverage` | List every Hermes signal recipe. | `full_hermes_coverage` | request-boundary plus provider-automatic recipes |
| Pi `baseline` | Hide intentional hints. | `pi_no_hints_baseline` | none |
| Pi `cache_only` | Probe Pi prompt-cache key, retention, cache-control, and pinning-negative behavior. | prompt-cache config, long-retention, and pinning-negative scenarios | `prompt_cache_key`, `prompt_cache_retention`, `cache_control` |
| Pi `namespace_only` | Probe Pi provider/account namespace and session-affinity metadata. | `pi_provider_account_namespace` | `x-session-affinity`, namespace header |
| Pi `feedback_only` | Probe whether Pi/provider can expose cache feedback. | `pi_cache_feedback_footer_usage` | cache-read/write or footer usage counters |
| Pi `all_request_boundary` | Run every Pi request-boundary probe. | `pi_request_boundary_coverage` | native Pi request-boundary probe targets only |
| Pi `full_coverage` | List every Pi signal recipe. | `full_pi_coverage` | request-boundary plus provider-feedback recipes |
| OpenClaw `baseline` | Hide intentional hints. | `openclaw_no_hints_baseline` | none |
| OpenClaw `qos_only` | Probe OpenClaw provider QoS/service-tier config. | `openclaw_provider_qos_fast_mode` | optional `service_tier`, optional `extra_body.service_tier` |
| OpenClaw `cache_only` | Probe provider-managed cache config, cached-WebSocket, and pinning-negative recipes. | provider cache config, cached-WebSocket, and pinning-negative scenarios | optional cache fields; none observed in current native request-boundary run |
| OpenClaw `namespace_only` | Probe OpenClaw provider/session namespace metadata. | `openclaw_provider_session_namespace` | `x-hintbench-cache-namespace` |
| OpenClaw `feedback_only` | Probe whether OpenClaw/provider can expose cache feedback. | `openclaw_provider_trace_feedback` | cache-read/cached-token usage counters or trace fields |
| OpenClaw `all_request_boundary` | Run every OpenClaw request-boundary probe. | `openclaw_request_boundary_coverage` | native OpenClaw request-boundary probe targets only |
| OpenClaw `full_coverage` | List every OpenClaw signal recipe. | `full_openclaw_coverage` | request-boundary plus provider-feedback recipes |
| OpenCode `full_coverage` | Run every OpenCode fixture recipe. | `full_opencode_coverage` | fixture-only provider/plugin cache recipes |
| Deep Agents `full_coverage` | Run every Deep Agents fixture recipe. | `full_deep_agents_coverage` | fixture-only provider/middleware recipes |
| DeepSeek `full_coverage` | Run every DeepSeek Harness fixture recipe. | `full_deepseek_coverage` | fixture-only provider-managed cache recipes |
| Codex `full_coverage` | Run every Codex fixture recipe. | `full_codex_coverage` | fixture-only provider/cache/prewarm recipes |

Example:

```bash
cd ~/agentic_hardware
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id nat_all_request_boundary \
  --out-dir sglang_direct_kv/artifacts/results/hint_benchmark/nat_all_request_boundary
```

## Naming Rules

Use these labels consistently:

- `NAT-native emitted`: NAT itself added the field during its transport path.
- `pass-through preserved`: the client/provider supplied the field before NAT,
  and NAT preserved it.
- `runtime feedback`: the value appears after real backend execution, not in
  the outgoing request body.
- `optional-not-observed`: the manifest tracks the hint, but this benchmark
  mode did not observe a concrete emitted field.

These distinctions prevent us from overclaiming as we extend the suite to
Claude and the other harnesses.

Claude source docs:

- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- https://docs.anthropic.com/en/api/messages
