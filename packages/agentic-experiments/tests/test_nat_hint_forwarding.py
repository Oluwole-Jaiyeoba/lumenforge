from agentic_experiments.runners.run_hint_benchmark import nat_backend_payload


def test_nat_forwarding_preserves_native_hints_and_adds_only_benchmark_identity():
    native = {"model": "nat-hint-benchmark-model", "messages": [{"role": "user", "content": "test"}],
              "nvext": {"agent_hints": {"priority": 7}}, "max_tokens": 8}
    forwarded = nat_backend_payload(native, model="Qwen/Qwen2.5-Coder-7B-Instruct", request_id="r1")
    assert native == {"model": "nat-hint-benchmark-model", "messages": [{"role": "user", "content": "test"}],
                      "nvext": {"agent_hints": {"priority": 7}}, "max_tokens": 8}
    assert forwarded["nvext"] == native["nvext"]
    assert forwarded["custom_params"]["agentic_kv"]["correlation_id"] == "r1"
    assert forwarded["custom_params"]["agentic_kv"]["request_id"] == "r1"
    assert forwarded["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
