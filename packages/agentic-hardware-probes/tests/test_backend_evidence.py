from agentic_hardware_probes.backend_evidence import assess_trial


def _trial():
    return {"request_id": "target", "donor_session_id": "donor",
            "decode": {"request_start_ns": 100, "request_end_ns": 200}}


def _event(signal, time, session="", request=""):
    return {"signal_id": signal, "time_ns": time, "session_id": session, "request_id": request}


def test_direct_overlap_requires_same_donor_copy_during_decode():
    events = [_event("request.accepted", 105, request="target"),
              _event("kv.load_gpu", 150, session="donor"),
              _event("kv.layer_copy", 151, session="donor")]
    result = assess_trial(_trial(), "direct_overlap_reload", events)
    assert result["valid"]
    assert result["donor_layer_copies_during_decode"] == 1
    assert "not GPU-kernel overlap" in result["proof_limit"]
    events[-1]["session_id"] = "another-session"
    assert not assess_trial(_trial(), "direct_overlap_reload", events)["valid"]


def test_nonoverlap_needs_copy_before_target_and_control_needs_only_acceptance():
    events = [_event("request.accepted", 110, request="target"),
              _event("kv.layer_copy", 90, session="donor")]
    assert assess_trial(_trial(), "non_overlap_reload", events)["valid"]
    assert assess_trial(_trial(), "decode_control", events)["valid"]
    assert not assess_trial(_trial(), "direct_overlap_reload", events)["valid"]
