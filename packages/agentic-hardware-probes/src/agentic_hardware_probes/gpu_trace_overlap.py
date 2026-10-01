"""Measure GPU-time overlap between a backend kernel trace and copy-worker trace."""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from pathlib import Path
from typing import Iterator


TIMING = re.compile(r'"ts":\s*([0-9.]+),\s*"dur":\s*([0-9.]+)')
START = re.compile(r'"ts":\s*([0-9.]+)')
DURATION = re.compile(r'"dur":\s*([0-9.]+)')
BASE_TIME = re.compile(r'"baseTimeNanoseconds":\s*([0-9]+)')


def base_time(path: Path) -> int:
    with path.open(encoding="utf-8") as trace:
        for _, line in zip(range(100), trace):
            match = BASE_TIME.search(line)
            if match:
                return int(match.group(1))
    raise ValueError(f"No trace clock base found in {path}")


def intervals(path: Path, category: str) -> Iterator[tuple[float, float]]:
    pending = False
    start: float | None = None
    with path.open(encoding="utf-8") as trace:
        for line in trace:
            if '"cat":' in line:
                pending = f'"cat": "{category}"' in line
                start = None
                continue
            if not pending:
                continue
            match = TIMING.search(line)
            if match:
                event_start = float(match.group(1))
                duration = float(match.group(2))
                if duration > 0:
                    yield event_start, event_start + duration
                pending = False
                continue
            start_match = START.search(line)
            if start_match:
                start = float(start_match.group(1))
            duration_match = DURATION.search(line)
            if start is not None and duration_match:
                duration = float(duration_match.group(1))
                if duration > 0:
                    yield start, start + duration
                pending = False


def analyze(
    backend_path: Path, worker_path: Path, witness_path: Path | None = None
) -> dict[str, object]:
    backend_base = base_time(backend_path)
    worker_base = base_time(worker_path)
    if backend_base != worker_base:
        raise ValueError("Backend and worker GPU traces do not share a clock base")
    copies = sorted(intervals(worker_path, "gpu_memcpy"))
    if not copies:
        raise ValueError("The worker trace contains no GPU memcpy events")
    copy_ends = [end for _, end in copies]
    copy_window = (copies[0][0], max(copy_ends))
    kernel_count = 0
    window_kernel_count = 0
    overlapping_kernel_count = 0
    overlap_us = 0.0
    window_kernels: list[tuple[float, float]] = []
    for start, end in intervals(backend_path, "kernel"):
        kernel_count += 1
        if end <= copy_window[0] or start >= copy_window[1]:
            continue
        window_kernel_count += 1
        if witness_path is not None:
            window_kernels.append((start, end))
        first = bisect.bisect_right(copy_ends, start)
        kernel_overlap = 0.0
        for index in range(first, len(copies)):
            copy_start, copy_end = copies[index]
            if copy_start >= end:
                break
            kernel_overlap += max(0.0, min(end, copy_end) - max(start, copy_start))
        if kernel_overlap > 0:
            overlapping_kernel_count += 1
            overlap_us += kernel_overlap
    if not kernel_count:
        raise ValueError("The backend trace contains no GPU kernel events")
    if witness_path is not None:
        witness_path.parent.mkdir(parents=True, exist_ok=True)
        witness = json.dumps({
            "schema_version": "gpu_trace_overlap_witness.v1",
            "trace_base_time_ns": backend_base,
            "copy_gpu_intervals_us": copies,
            "backend_kernel_intervals_us": window_kernels,
        }, separators=(",", ":")) + "\n"
        if witness_path.suffix == ".gz":
            with gzip.open(witness_path, "wt", encoding="utf-8") as output:
                output.write(witness)
        else:
            witness_path.write_text(witness, encoding="utf-8")
    return {
        "schema_version": "gpu_trace_overlap.v1",
        "backend_trace": str(backend_path),
        "worker_trace": str(worker_path),
        "trace_base_time_ns": backend_base,
        "copy_events": len(copies),
        "copy_window_duration_ms": round((copy_window[1] - copy_window[0]) / 1000, 3),
        "total_copy_gpu_time_ms": round(sum(end - start for start, end in copies) / 1000, 3),
        "backend_kernel_events": kernel_count,
        "backend_kernels_in_copy_window": window_kernel_count,
        "backend_kernels_overlapping_copies": overlapping_kernel_count,
        "summed_kernel_copy_overlap_ms": round(overlap_us / 1000, 3),
        "physical_overlap_observed": overlapping_kernel_count > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend_trace", type=Path)
    parser.add_argument("worker_trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--witness-out", type=Path)
    args = parser.parse_args()
    result = analyze(args.backend_trace, args.worker_trace, args.witness_out)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(output, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
