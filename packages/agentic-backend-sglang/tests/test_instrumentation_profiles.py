from agentic_backends.sglang.instrumentation_profiles import profile_flags, validate_installation
from agentic_backends.sglang.hook_registry import resolve_hook
from agentic_backends.sglang.evidence import normalize_trace_event
from agentic_instrumentation import PROFILES


def test_legacy_profile_defaults_remain_unchanged():
    expected = {
        "minimal": ("0", "0", "0", "0", "0", "0", "1"),
        "deadline": ("1", "0", "1", "0", "0", "0", "1"),
        "controller_decision": ("1", "0", "1", "1", "1", "1", "1"),
        "idle_gap": ("1", "0", "1", "1", "1", "1", "1"),
        "cache_debug": ("1", "1", "1", "1", "1", "1", "1"),
        "full_debug": ("1", "1", "1", "1", "1", "1", "1"),
    }
    for name, values in expected.items():
        assert tuple(profile_flags(name).values()) == values
    assert profile_flags("controller_queue")["AGENTIC_KV_COPY_TELEMETRY_ENABLE"] == "0"
    assert set(profile_flags("kv_lifecycle_lean").values()) == {"0"}
    assert PROFILES["kv_lifecycle_lean"].required_signals == PROFILES["kv_lifecycle"].required_signals


def test_missing_hook_fails_loud_and_complete_profile_passes():
    assert not validate_installation("kv_lifecycle", "v0510", {"adapter": "v0510", "installed_hooks": []})["valid"]
    installed = []
    for hook_id in ("kv_gpu_load", "kv_prefix_match"):
        installed.extend(target["target"] for target in resolve_hook(hook_id, "v0510")["targets"])
    assert validate_installation("kv_lifecycle", "v0510", {"adapter": "v0510", "installed_hooks": installed})["valid"]
    assert validate_installation("kv_lifecycle_lean", "v0510", {"adapter": "v0510", "installed_hooks": installed})["valid"]
    assert not validate_installation("kv_lifecycle", "v0511", {"adapter": "v0510", "installed_hooks": installed})["valid"]


def test_batch_completion_is_not_a_request_completion():
    event = normalize_trace_event({"event": "scheduler.process_batch_result.end", "ts_ns": 17})
    assert event is not None
    assert event.signal_id == "batch.completed"
    assert "batch.completed" in PROFILES["controller_queue"].required_signals


def test_semantic_and_per_layer_loads_remain_distinct():
    context = {"agent_session_id": "s", "device_indices": {"values": [5]}}
    semantic = normalize_trace_event({"event": "hicache.load.end", "ts_ns": 20, "kv_context": context})
    layer = normalize_trace_event({"event": "hostpool.load_to_device_per_layer.end", "ts_ns": 19, "kv_context": context})
    assert semantic is not None and semantic.signal_id == "kv.load_gpu"
    assert layer is not None and layer.signal_id == "kv.layer_copy"
    assert layer.payload["kv_event_type"] == "KV_LOAD_GPU"
