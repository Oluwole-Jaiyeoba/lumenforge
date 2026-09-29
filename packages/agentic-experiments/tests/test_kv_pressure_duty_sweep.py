from types import SimpleNamespace

from agentic_experiments.runners.run_kv_pressure_duty_sweep import validate_pressure


def test_pressure_validation_counts_recycle_and_cuda_share() -> None:
    args = SimpleNamespace(
        load_count=2,
        donor_count=1,
        minimum_host_tokens=512,
        minimum_cuda_load_share_pct=5.0,
    )
    decode = {
        "request_start_ns": 0,
        "request_end_ns": 100_000_000,
        "total_latency_ms": 100.0,
        "chunk_times_ns": [0, 10_000_000, 50_000_000, 100_000_000],
    }
    loads = [
        {"load_sequence": 0, "start_ns": 15_000_000, "finish_ns": 25_000_000, "cuda_duration_ms": 5.0, "loaded_tokens": 1024},
        {"load_sequence": 1, "start_ns": 55_000_000, "finish_ns": 65_000_000, "cuda_duration_ms": 5.0, "loaded_tokens": 1024},
    ]
    regions = {"during": {"interval_count": 2}}
    proof = validate_pressure(args, decode, loads, [{"ok": True}], regions)

    assert proof["valid"]
    assert proof["cuda_load_share_of_decode_pct"] == 10.0
    assert proof["recycle_proofs"] == 1
