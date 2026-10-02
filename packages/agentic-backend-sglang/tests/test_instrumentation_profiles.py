from agentic_backends.sglang.instrumentation_profiles import profile_flags, validate_installation
from agentic_backends.sglang.hook_registry import resolve_hook


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


def test_missing_hook_fails_loud_and_complete_profile_passes():
    assert not validate_installation("kv_lifecycle", "v0510", {"installed_hooks": []})["valid"]
    installed = []
    for hook_id in ("kv_gpu_load", "kv_prefix_match"):
        installed.extend(target["target"] for target in resolve_hook(hook_id, "v0510")["targets"])
    assert validate_installation("kv_lifecycle", "v0510", {"installed_hooks": installed})["valid"]
