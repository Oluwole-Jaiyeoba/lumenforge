from argparse import Namespace

from scripts import measure_instrumentation_overhead as measure


def test_long_prefix_overhead_sequence_keeps_two_waits_and_three_turns(monkeypatch):
    requests = []
    sleeps = []

    def fake_request(model, port, index, *, prompt, max_tokens, salt):
        requests.append((model, port, index, prompt, max_tokens, salt))
        return {"latency_ms": 100, "prompt_tokens": 4000, "completion_tokens": 16}

    monkeypatch.setattr(measure, "_request", fake_request)
    monkeypatch.setattr(measure.time, "sleep", sleeps.append)
    args = Namespace(model="test", port=30000, prompt_tokens=4090, max_tokens=16, wait_ms=2000)
    result = measure._audit_sequence(args, 7)
    assert len(requests) == 3
    assert sleeps == [2, 2]
    assert result["latency_ms"] == 300
    assert result["prompt_tokens"] == 12000
    assert requests[0][3] in requests[1][3] and requests[1][3] in requests[2][3]
    assert len({request[5] for request in requests}) == 1
