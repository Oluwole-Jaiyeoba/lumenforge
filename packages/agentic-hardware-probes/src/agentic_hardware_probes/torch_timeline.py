"""Classify CUDA copy/compute overlap in a PyTorch Chrome timeline.

Application-level request windows are not sufficient evidence of a memory
collision.  This utility inspects GPU activity directly and reports whether
host-to-device copies intersect GPU kernels in the captured timeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Interval = tuple[float, float]


def _merged(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _duration(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in _merged(intervals))


def _intersection(left: list[Interval], right: list[Interval]) -> float:
    total = 0.0
    for left_start, left_end in _merged(left):
        for right_start, right_end in _merged(right):
            total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return total


def _intervals(events: list[dict[str, Any]], predicate: Any) -> list[Interval]:
    intervals: list[Interval] = []
    for event in events:
        if not predicate(event):
            continue
        start = event.get("ts")
        duration = event.get("dur")
        if isinstance(start, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
            intervals.append((float(start), float(start + duration)))
    return intervals


def summarize_timeline(path: Path) -> dict[str, Any]:
    """Return copy/compute overlap measured in microseconds."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    events = list(payload.get("traceEvents") or [])
    h2d = _intervals(
        events,
        lambda event: event.get("cat") == "gpu_memcpy" and "htod" in str(event.get("name", "")).lower(),
    )
    kernels = _intervals(events, lambda event: event.get("cat") == "kernel")
    overlap_us = _intersection(h2d, kernels)
    copy_start = min((start for start, _ in h2d), default=None)
    copy_end = max((end for _, end in h2d), default=None)
    kernel_start = min((start for start, _ in kernels), default=None)
    kernel_end = max((end for _, end in kernels), default=None)
    return {
        "schema_version": "torch_cuda_timeline_overlap.v1",
        "trace": str(path),
        "h2d_copy_count": len(h2d),
        "h2d_copy_active_us": round(_duration(h2d), 3),
        "h2d_copy_window_start_us": copy_start,
        "h2d_copy_window_end_us": copy_end,
        "kernel_count": len(kernels),
        "kernel_active_us": round(_duration(kernels), 3),
        "kernel_window_start_us": kernel_start,
        "kernel_window_end_us": kernel_end,
        "h2d_kernel_overlap_us": round(overlap_us, 3),
        "actual_gpu_collision": overlap_us > 0,
        "classification": "physical_gpu_copy_compute_overlap" if overlap_us > 0 else "serialized_no_gpu_overlap",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_timeline(args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
