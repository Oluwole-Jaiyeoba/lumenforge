from agentic_experiments.runners.run_gpu_copy_contention import window_gaps_ms


def test_window_gaps_exclude_transitions_outside_pressure_window() -> None:
    chunks = [0, 10_000_000, 20_000_000, 35_000_000, 50_000_000]

    assert window_gaps_ms(chunks, 10_000_000, 35_000_000) == [10.0, 15.0]
