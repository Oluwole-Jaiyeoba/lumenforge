from agentic_experiments.runners.run_sustained_kv_pressure import (
    count_load_windows_with_visible_overlap,
    interval_metrics_windows,
    percentile,
)


def test_interval_metrics_separates_reload_windows() -> None:
    chunks = [0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
    result = interval_metrics_windows(
        chunks,
        [(12_000_000, 18_000_000), (32_000_000, 38_000_000)],
    )

    assert result["before"]["interval_count"] == 1
    assert result["during"]["interval_count"] == 2
    assert result["between"]["interval_count"] == 1
    assert result["after"]["interval_count"] == 1


def test_each_load_window_is_checked_even_when_updates_are_coalesced() -> None:
    chunks = [0, 10_000_000, 40_000_000]
    windows = [(12_000_000, 18_000_000), (20_000_000, 30_000_000)]

    assert count_load_windows_with_visible_overlap(chunks, windows) == 2


def test_tail_percentile_uses_nearest_rank() -> None:
    assert percentile([514.0, 1199.0], 0.95) == 1199.0
