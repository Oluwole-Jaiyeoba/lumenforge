# GH200 Hardware Experiment Transfer

This is a standalone handoff for an agent working on a NVIDIA Grace Hopper
GH200 machine. It does not require a clone of the project that produced the
reference A10G result. Build the small, reusable hardware-experiment lane
described here inside the GH200 project's existing NVIDIA-supported runtime,
then run one controlled experiment: independent Grace-host-to-GPU copies during
a real SGLang decode.

Do not replace the GH200 project's working SGLang, CUDA, PyTorch, or container
stack with older versions just to match the reference machine. Preserve the
experimental controls and metric definitions. Adapt only the backend launcher,
request fields that differ across SGLang versions, and GPU profiler export.

## 1. Question and boundaries

The first question is: while SGLang is decoding one response, can independent
host-to-GPU traffic on the same GPU slow its output? Compare an `idle` worker
with a `copy` worker. Both workers remain alive and hold the same pinned host
source and GPU destination buffer. The copy worker repeatedly transfers the
source into the destination for a fixed window; the idle worker waits for the
same duration. The backend sees one identical-shape target request at a time.

This is a controlled hardware-mechanism microbenchmark, not a claim that a
particular production agent naturally reloads this many KV blocks. The target
is real SGLang inference, but the initial prompt and the competing bytes are
synthetic. A later experiment can replace the copy worker with natural
cross-session SGLang KV load-back. Do not merge those two claims.

The first study must answer four separate questions:

1. Did the worker move bytes from Grace host memory into the selected GPU?
2. How many GB/s did it actually sustain during the window?
3. Did GPU copy operations physically overlap backend decode kernels?
4. Did in-window output cadence or total decode time change versus idle?

Physical overlap plus slowdown does not, by itself, prove that HBM bandwidth
was the sole contended resource. Copy engines, cache hierarchy, scheduling,
clocks, and power can also contribute. Only claim HBM-bandwidth attribution
with suitable device-memory counters and additional controls.

## 2. Portability boundary

Implement these independent components in the GH200 project:

| Component | Responsibility | Version-specific? |
| --- | --- | --- |
| Backend launcher | Start the GH200-supported SGLang server and expose its OpenAI-compatible chat-completions endpoint | Yes |
| Copy worker | Allocate pinned CPU and GPU buffers; run idle or H2D copy windows on a chosen CUDA GPU | No SGLang imports |
| Trial runner | Send streaming requests, trigger windows, record timestamps, compare modes | Only the request payload adapter |
| Trace exporter | Capture GPU copy and backend kernel intervals on one time base | Yes, depends on available profiler |
| Overlap checker and reporter | Validate overlap and publish raw trials plus summaries | No SGLang imports |

Use the GH200 project's existing directory conventions. A possible local layout
is `hardware_experiments/{copy_worker.py,run_copy_trial.py,check_overlap.py}`
with a run-specific `artifacts/<run-id>/` directory. These names are examples,
not an import or path dependency. The host can run the runner; SGLang may stay
in its existing container. The worker may run on the host or in another GPU
container with a suitable PyTorch build. Backend and worker must be separate
processes and must target the same physical GPU. No SGLang patch is needed for
the unprofiled timing experiment.

The runner requires Python 3.10+ and `httpx`. The worker requires a CUDA-enabled
PyTorch build compatible with the GH200 host. Do not install packages into or
alter a working SGLang environment merely to satisfy the runner. The profiler
may use the GH200 project's supported NVIDIA tooling.

## 3. Reference A10G result and exact definitions

The reference was a NVIDIA A10G 24 GB GDDR6, Qwen/Qwen2.5-Coder-7B-Instruct,
SGLang 0.5.10.post1. One request generated up to 320 tokens. After 16 streamed
content chunks, a 5-second window began. The copy worker repeatedly moved a
128 MiB pinned-host buffer to a preallocated GPU buffer. Trial order was
`idle, copy, copy, idle`: two trials per condition, so this is a pilot.

| Metric | Idle | Copy | Difference |
| --- | ---: | ---: | ---: |
| Median time from first content chunk to response end | 10,720.2 ms | 10,919.6 ms | +199.4 ms (+1.9%) |
| Median across trials of each trial's mean in-window content-chunk gap | 33.59 ms | 34.97 ms | +1.38 ms (+4.1%) |
| Median full request time | 10,920.9 ms | 10,961.1 ms | +40.2 ms (+0.4%) |

Across the two copy trials, 999 copies moved 134,083,510,272 bytes in roughly
10 seconds of copy windows, or about 13.4 decimal GB/s. A separate profiled
diagnostic found 499 GPU copy events in one five-second window, with 56,483 of
56,762 backend kernels in that window touching at least one copy interval in
GPU time. Its request timed out under profiler overhead, so its latency was
**not** used in the table. The overlap result is a physical-timeline check, not
a measurement of how much HBM bandwidth each operation consumed.

Definitions that must remain unchanged on GH200:

- `TTFT = first_nonempty_content_chunk_time - request_start_time`.
- `decode_after_first_token = response_end_time - first_nonempty_content_chunk_time`.
- `full_request = response_end_time - request_start_time`.
- A window gap is the elapsed time between **consecutive nonempty streamed
  content chunks when both timestamps are inside the worker's actual window**.
- One trial's `window_mean_chunk_gap` is the arithmetic mean of those gaps.
  The mode summary is the median of trial means, not the mean of all pooled gaps.
- `achieved_copy_GBps = bytes_copied / actual_window_seconds / 1e9`.
- For a metric where smaller is better, `percent_change = 100 *
  (copy_median - idle_median) / idle_median`. Positive means slower.

An HTTP stream content chunk is not guaranteed to equal one model token. Label
the cadence metric as a **content-chunk gap**, not token time. If the backend
also exposes per-token timestamps, report those separately without replacing
the reference definition. Report usage/completion tokens when available.

Do not require the GH200 percentages to exceed the A10G percentages. Different
hardware, achieved transfer bandwidth, clocks, and model execution may produce
smaller, larger, or zero slowdown. Valid evidence matters more than direction.

## 4. Preflight and calibration

Before a performance run, the GH200 agent must record:

- Machine and GPU identity, GPU UUID/index, Grace CPU and GPU-memory variant,
  host/GPU memory capacities, driver, CUDA, PyTorch, SGLang, container image or
  environment identity, model revision, launch flags, and available GPU memory.
- Which physical GPU the backend uses and which the worker uses. Pin both to
  that GPU, including any container GPU mapping. Fail if identity is ambiguous.
- Whether the explicit `source` host buffer is pinned and the `destination`
  buffer is device-resident. Use a real H2D operation; do not mistake coherent
  CPU-memory access or a GPU-to-GPU copy for this experiment.
- Whether the backend accepts the streaming payload. `ignore_eos` and custom
  request metadata are optional SGLang-version-specific fields; this first
  experiment does not need custom metadata or priority hints.
- Whether a short model warm-up works. Confirm no other clients, jobs, or
  profilers are using the GPU during the timing run.

Calibrate output length before collecting paired trials. The decode must
continue after 16 content chunks, last for longer than the entire copy window,
and produce enough chunks **inside** the window for a stable cadence estimate.
Start with the same 5-second window and 128 MiB copy buffer; increase
`max_tokens` if GH200 finishes too soon. Record the final values. Do not alter
the window duration between idle and copy trials. If the chosen model differs
from the A10G reference, label cross-machine comparisons as directional only.

Verify the host-to-GPU route with the platform's supported transfer counters
or profiler. The worker's byte count is necessary but not sufficient to assert
which physical link carried those bytes. Record observed C2C traffic if the
GH200 profiler exposes it; otherwise mark route evidence as unavailable and
avoid a quantified NVLink-C2C bandwidth claim. Report measured GB/s, not the
link's advertised maximum.

## 5. Minimal self-contained copy worker

The following reference worker needs no SGLang import. It serves control
requests on loopback and uses the same allocations in both modes. Launch it in
a separate process with the correct CUDA device visible. The runner and worker
must share the same host monotonic clock domain; ordinary processes and normal
containers on one Linux host do, but verify if the deployment uses special
time namespaces or multiple hosts.

```python
# copy_worker.py
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31992)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--buffer-mib", type=int, default=128)
    p.add_argument("--profile-path", type=Path)
    args = p.parse_args()
    if not 1 <= args.buffer_mib <= 1024:
        p.error("buffer-mib must be in [1, 1024]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the copy worker")

    torch.cuda.set_device(args.device)
    size = args.buffer_mib * 1024 * 1024
    source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    source.fill_(7)
    destination = torch.empty(size, dtype=torch.uint8, device=f"cuda:{args.device}")
    stream = torch.cuda.Stream(device=args.device)
    with torch.cuda.stream(stream):
        destination.copy_(source, non_blocking=True)
    stream.synchronize()
    lock = threading.Lock()
    state = {"status": "ready", "buffer_bytes": size, "device": args.device}

    def window(mode, duration_s):
        profiler = None
        if args.profile_path is not None:
            from torch.profiler import ProfilerActivity, profile
            profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            profiler.__enter__()
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(duration_s * 1_000_000_000)
        copies = 0
        while time.monotonic_ns() < deadline_ns:
            if mode == "copy":
                with torch.cuda.stream(stream):
                    destination.copy_(source, non_blocking=True)
                stream.synchronize()
                copies += 1
            else:
                time.sleep(min(0.01, max(0, (deadline_ns - time.monotonic_ns()) / 1e9)))
        finished_ns = time.monotonic_ns()
        if profiler is not None:
            profiler.__exit__(None, None, None)
            args.profile_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.profile_path))
        with lock:
            state.update(status="finished", mode=mode, started_ns=started_ns,
                         finished_ns=finished_ns, copies=copies,
                         bytes_copied=copies * size)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_json(self, code, body):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/status":
                return self.send_json(404, {"error": "unknown endpoint"})
            with lock:
                return self.send_json(200, dict(state))

        def do_POST(self):
            if self.path != "/start":
                return self.send_json(404, {"error": "unknown endpoint"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                mode, duration_s = body["mode"], float(body["duration_s"])
                if mode not in ("idle", "copy") or not 0 < duration_s <= 60:
                    raise ValueError("invalid mode or duration")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self.send_json(400, {"error": str(exc)})
            with lock:
                if state["status"] == "running":
                    return self.send_json(409, {"error": "window already running"})
                state.update(status="running", mode=mode, copies=0, bytes_copied=0)
                state.pop("started_ns", None)
                state.pop("finished_ns", None)
            threading.Thread(target=window, args=(mode, duration_s), daemon=True).start()
            return self.send_json(202, {"status": "running", "mode": mode})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

`destination.copy_(source, non_blocking=True)` is the experimental operation;
`stream.synchronize()` prevents the Python loop from merely queuing unlimited
asynchronous copies. `started_ns` and `finished_ns` cover the entire worker
window. Copies may slightly overshoot the requested duration because an
in-flight copy must finish; the runner uses the **actual** window duration.

## 6. Minimal self-contained timing runner

This reference runner talks only to an OpenAI-compatible HTTP endpoint and
the worker's two endpoints. Run one trial at a time, not simultaneous target
requests. Use one worker process for all idle and copy trials. Its output is
raw per-trial JSON plus a compact summary; the GH200 project may render the
same fields in its own HTML reporting style.

```python
# run_copy_trial.py
import argparse
import asyncio
import datetime
import json
import statistics
import time
from pathlib import Path

import httpx


async def stream_decode(client, base_url, model, prompt, max_tokens, ready, warmup_chunks,
                        ignore_eos):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    if ignore_eos:
        payload["ignore_eos"] = True  # Keep only if this SGLang version supports it.
    started_ns = time.monotonic_ns()
    chunks, usage = [], {}
    async with client.stream("POST", base_url.rstrip("/") + "/chat/completions",
                             json=payload) as response:
        if response.is_error:
            detail = (await response.aread()).decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"backend HTTP {response.status_code}: {detail}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                content = event.get("choices", [{}])[0].get("delta", {}).get("content")
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                continue
            if content not in (None, ""):
                chunks.append(time.monotonic_ns())
                if len(chunks) >= warmup_chunks:
                    ready.set()
    ended_ns = time.monotonic_ns()
    ready.set()
    if not chunks:
        raise RuntimeError("no nonempty streamed content chunks")
    return {"request_start_ns": started_ns, "first_content_ns": chunks[0],
            "request_end_ns": ended_ns, "content_chunk_ns": chunks,
            "stream_chunks": len(chunks), "usage": usage,
            "ttft_ms": (chunks[0] - started_ns) / 1e6,
            "decode_after_first_token_ms": (ended_ns - chunks[0]) / 1e6,
            "full_request_ms": (ended_ns - started_ns) / 1e6}


async def get_finished_window(client, worker_url, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = await client.get(worker_url.rstrip("/") + "/status")
        response.raise_for_status()
        state = response.json()
        if state.get("status") == "finished":
            return state
        await asyncio.sleep(0.05)
    raise RuntimeError("copy worker did not finish its window")


async def trial(client, args, mode, number):
    ready = asyncio.Event()
    prompt = ("Repeat the word token with a space until the output limit. "
              "Do not add punctuation or stop early. Trial ID: " + f"{number:04d}")
    task = asyncio.create_task(stream_decode(
        client, args.base_url, args.model, prompt, args.max_tokens, ready,
        args.warmup_chunks, args.ignore_eos))
    try:
        await asyncio.wait_for(ready.wait(), args.timeout_s)
        if task.done():
            raise RuntimeError("decode finished before pressure could start")
        response = await client.post(args.worker_url.rstrip("/") + "/start",
                                     json={"mode": mode, "duration_s": args.window_s})
        response.raise_for_status()
        decode = await asyncio.wait_for(task, args.timeout_s)
        window = await get_finished_window(client, args.worker_url, args.window_s + 30)
    except BaseException:
        task.cancel()
        raise
    start, end = window["started_ns"], window["finished_ns"]
    if not (decode["first_content_ns"] < start < end < decode["request_end_ns"]):
        raise RuntimeError("copy window is not fully inside active decode")
    chunks = decode["content_chunk_ns"]
    gaps = [(right - left) / 1e6 for left, right in zip(chunks, chunks[1:])
            if start <= left and right <= end]
    if len(gaps) < args.min_window_gaps:
        raise RuntimeError(f"too few content-chunk gaps inside window: {len(gaps)}")
    if mode == "copy" and (window["copies"] <= 0 or window["bytes_copied"] <= 0):
        raise RuntimeError("copy mode moved no bytes")
    if mode == "idle" and window["bytes_copied"] != 0:
        raise RuntimeError("idle control unexpectedly copied bytes")
    seconds = (end - start) / 1e9
    return {"mode": mode, "trial": number, "decode": decode, "window": window,
            "window_gaps_ms": gaps, "window_mean_chunk_gap_ms": statistics.mean(gaps),
            "window_median_chunk_gap_ms": statistics.median(gaps),
            "actual_window_s": seconds,
            "achieved_copy_GBps": window["bytes_copied"] / seconds / 1e9}


def summarize(rows, field):
    return {mode: statistics.median(
        row["window_mean_chunk_gap_ms"] if field == "window_gap"
        else row["decode"][field] for row in rows if row["mode"] == mode)
        for mode in ("idle", "copy")}


async def run(args):
    timeout = httpx.Timeout(args.timeout_s + 30, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        warmup_ready = asyncio.Event()
        await stream_decode(client, args.base_url, args.model,
                            "Warm up the model by writing a short sentence.", 8,
                            warmup_ready, 1, args.ignore_eos)
        rows = []
        for number, mode in enumerate(["idle", "copy", "copy", "idle"] * args.blocks):
            row = await trial(client, args, mode, number)
            rows.append(row)
            print(f"{number}: {mode}, gap={row['window_mean_chunk_gap_ms']:.3f} ms, "
                  f"H2D={row['achieved_copy_GBps']:.2f} GB/s", flush=True)
    metrics = {name: summarize(rows, name) for name in
               ("window_gap", "decode_after_first_token_ms", "full_request_ms", "ttft_ms")}
    changes = {name: {"absolute_ms": values["copy"] - values["idle"],
                      "percent": 100 * (values["copy"] - values["idle"]) / values["idle"]}
               for name, values in metrics.items() if values["idle"] > 0}
    return {"schema_version": "independent_gpu_copy.v1",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": args.model,
            "configuration": {"max_tokens": args.max_tokens,
                              "warmup_chunks": args.warmup_chunks,
                              "window_s": args.window_s, "blocks": args.blocks,
                              "ignore_eos": args.ignore_eos,
                              "min_window_gaps": args.min_window_gaps,
                              "trial_order": [row["mode"] for row in rows]},
            "trials": rows, "condition_medians": metrics, "changes": changes,
            "copy_GBps_median": statistics.median(
                row["achieved_copy_GBps"] for row in rows if row["mode"] == "copy")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="SGLang URL ending in /v1")
    p.add_argument("--worker-url", default="http://127.0.0.1:31992")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--warmup-chunks", type=int, default=16)
    p.add_argument("--window-s", type=float, default=5.0)
    p.add_argument("--min-window-gaps", type=int, default=10)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.blocks < 1 or args.max_tokens < 64 or args.warmup_chunks < 1:
        p.error("invalid blocks, max-tokens, or warmup-chunks")
    result = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

The GH200 agent may wrap this runner in its existing CLI and report framework.
It must preserve the timing definitions and raw trial fields. If the server
does not support `stream_options.include_usage` or `ignore_eos`, remove only
those optional fields, document the change, and verify the target remains long
enough. Do not silently substitute a non-streaming request or infer token-level
latency from content chunks.

## 7. Execution stages and commands

These are templates, not assumptions about the GH200 project's paths or image.
Start the GH200 project's existing SGLang backend first, make its `/v1/models`
and `/v1/chat/completions` reachable, and ensure no other GPU workload is
running. Set `CUDA_VISIBLE_DEVICES` so the worker's device 0 is that same GPU.

```bash
export CUDA_VISIBLE_DEVICES=<GH200_GPU_INDEX>
export MODEL=<EXACT_MODEL_ID_IN_THE_GH200_BACKEND>
export BASE_URL=http://127.0.0.1:<BACKEND_PORT>/v1

python3 copy_worker.py --device 0 --buffer-mib 128 --port 31992 >copy_worker.log 2>&1 &
WORKER_PID=$!
curl --fail http://127.0.0.1:31992/status

# One ABBA block (two trials per condition) is bring-up only.
python3 run_copy_trial.py --base-url "$BASE_URL" --model "$MODEL" \
  --max-tokens <CALIBRATED_TOKEN_LIMIT> --window-s 5 --blocks 1 \
  --out artifacts/bringup/trials.json

# Repeated unprofiled timing: four ABBA blocks = eight trials per condition.
python3 run_copy_trial.py --base-url "$BASE_URL" --model "$MODEL" \
  --max-tokens <CALIBRATED_TOKEN_LIMIT> --window-s 5 --blocks 4 \
  --out artifacts/timing/trials.json

kill "$WORKER_PID"
```

Use a run-specific artifact directory in the actual implementation. Preserve
all successful and failed trial records; do not hide timeouts or discard a
negative result. A large request can make eight pairs costly, so report the
bring-up separately and never describe two-per-mode as a stable estimate.
If the timing run is too expensive, agree on a smaller `blocks` value and
label its limited statistical power.

The two timing modes should share the same live backend and worker instance.
No profiler should be active. Capture backend logs and GPU telemetry in the
run directory. Record idle and copy results even if their difference is zero.

## 8. Physical overlap evidence: a separate diagnostic

Run a separate copy-only diagnostic with GPU profiling enabled for both the
backend and worker, or use a single supported profiler session that captures
both CUDA processes. Prefer the GH200 project's supported NVIDIA tooling; do
not force a particular old SGLang hook into its upgraded build. The profiler
must identify the same physical GPU and put GPU copy intervals and backend
decode kernel intervals on one **verified common time base**. A mere overlap
between HTTP request and copy-window wall times is not physical proof.

If the profiler emits separate traces, normalize them to a small JSON file:

```json
{
  "schema_version": "gpu_overlap_intervals.v1",
  "backend_gpu_uuid": "<GPU UUID from backend trace>",
  "worker_gpu_uuid": "<GPU UUID from worker trace>",
  "clock_domain": "<verified common GPU timeline>",
  "clock_alignment_evidence": "<how the profiler aligned the two traces>",
  "backend_decode_kernel_intervals_us": [[1000.0, 1010.0]],
  "worker_h2d_copy_intervals_us": [[1005.0, 1015.0]]
}
```

Include only positive-duration GPU events. Backend kernels must be identified
as decode work on the target GPU, not unrelated activity. The normalized
intervals and raw profiler output (or an export command and immutable trace
location) are evidence. The checker below is independent of profiler format.

```python
# check_overlap.py
import argparse
import json
from pathlib import Path


def validate(name, intervals):
    if not intervals:
        raise ValueError(f"no {name} intervals")
    rows = sorted((float(start), float(end)) for start, end in intervals)
    if any(end <= start for start, end in rows):
        raise ValueError(f"invalid {name} interval")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("intervals", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.intervals.read_text(encoding="utf-8"))
    if not data.get("backend_gpu_uuid") or not data.get("worker_gpu_uuid"):
        raise ValueError("both GPU identities must be recorded")
    if data["backend_gpu_uuid"] != data["worker_gpu_uuid"]:
        raise ValueError("backend and worker traces are from different GPUs")
    if not data.get("clock_domain") or not data.get("clock_alignment_evidence"):
        raise ValueError("GPU clock alignment must be proven")
    copies = validate("copy", data["worker_h2d_copy_intervals_us"])
    kernels = validate("decode kernel", data["backend_decode_kernel_intervals_us"])
    count, overlap_us = 0, 0.0
    for kernel_start, kernel_end in kernels:
        this_overlap = 0.0
        for copy_start, copy_end in copies:
            if copy_start >= kernel_end:
                break
            if copy_end > kernel_start:
                this_overlap += max(0.0, min(kernel_end, copy_end) -
                                    max(kernel_start, copy_start))
        if this_overlap > 0:
            count += 1
            overlap_us += this_overlap
    result = {"schema_version": "gpu_copy_overlap_result.v1",
              "physical_overlap_observed": count > 0,
              "gpu_uuid": data["backend_gpu_uuid"],
              "copy_events": len(copies), "decode_kernel_events": len(kernels),
              "decode_kernels_overlapping_copy": count,
              "summed_kernel_copy_overlap_us": overlap_us,
              "clock_alignment_evidence": data["clock_alignment_evidence"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if count == 0:
        raise RuntimeError("no physical GPU overlap; do not claim copy interference")


if __name__ == "__main__":
    main()
```

The diagnostic must not be mixed into timing medians. Profiler overhead caused
the A10G diagnostic request to time out after the overlap window; its trace
was useful for overlap evidence but its request latency was not comparable.
Prefer a complete GH200 diagnostic, and report any partial/timeout status
plainly. Do not treat the sample JSON above as experimental evidence.
For this diagnostic, launch a fresh worker with `--profile-path` and run only
one `copy` window. The worker will export a PyTorch Chrome trace. Collect the
backend's kernel trace with the GH200 stack's supported profiler, then verify
that both traces can be aligned before normalizing them. Merely having two
separate trace files is not enough to establish a common GPU time base.

## 9. Run manifest, report, and acceptance gates

Each GH200 run should produce at least:

```text
artifacts/<run-id>/
  run_manifest.json          machine, GPU UUID, runtime and experiment parameters
  backend.log                server launch and errors
  copy_worker.log            allocation, device and worker errors
  timing/trials.json         every unprofiled trial and derived medians
  profile/gpu_intervals.json normalized GPU intervals, if captured
  profile/overlap.json       physical-overlap check, if captured
  report.html                readable results and explicit limitations
```

The run manifest must record exact SGLang and model revisions, backend image
or environment identity, CUDA/driver/PyTorch versions, GPU memory variant,
backend launch flags, worker GPU UUID, buffer size, window duration, response
limit, warm-up count, trial order, profiling state, and whether extra traffic
was present. Store unavailable counters as `null`/`not_observed`, never zero.
A completed timing run can use this concrete structure (replace every bracketed
string with measured or configured information):

```json
{
  "schema_version": "gpu_copy_run_manifest.v1",
  "run_id": "<unique timestamped run ID>",
  "hardware": {
    "host": "<GH200 machine identifier>",
    "gpu_name": "<exact name>",
    "gpu_uuid": "<UUID>",
    "gpu_memory_type": "<HBM3 or HBM3e, as detected>",
    "grace_memory": "<capacity and type, as detected>",
    "driver": "<driver version>",
    "cuda": "<CUDA runtime version>"
  },
  "runtime": {
    "sglang_version": "<version or commit>",
    "pytorch_version": "<version>",
    "container_or_environment": "<image digest or environment lock ID>",
    "backend_launch_flags": ["<actual flags>"],
    "backend_gpu_uuid": "<UUID>"
  },
  "model": {
    "id": "<served model ID>",
    "revision": "<model revision or checkpoint identity>"
  },
  "worker": {
    "gpu_uuid": "<UUID>",
    "buffer_mib": 128,
    "source_pinned_host_memory": true,
    "destination_device_memory": true
  },
  "experiment": {
    "target_prompt_kind": "short synthetic repeat prompt",
    "max_tokens": 320,
    "warmup_content_chunks": 16,
    "requested_window_s": 5.0,
    "trial_order": ["idle", "copy", "copy", "idle"],
    "profiling_during_timing": false,
    "other_gpu_workloads": false
  },
  "host_to_gpu_route": {
    "verification_method": null,
    "measured_c2c_GBps": null,
    "status": "not_observed"
  }
}
```

The trial order and `max_tokens` above illustrate one bring-up block; the
final manifest must record its actual values. The worker's measured H2D
`achieved_copy_GBps` is distinct from a hardware counter's C2C reading. If
the latter is unavailable, leave `measured_c2c_GBps` as `null` while still
reporting the worker-measured H2D rate.

The report should show raw idle and copy medians, absolute and percentage
changes, achieved copy GB/s (per trial and median), trial count, and simple
variation such as minimum/maximum and per-trial values. Give the in-window
chunk gap prominence; the five-second pressure window may be only part of the
full decode. Show TTFT separately because copying starts *after* the first
token. Label any latency from a profiled run as diagnostic only. Include the
copy-overlap result, the route-evidence status, and all limitations.

This minimal report builder consumes the runner's raw JSON and an optional
overlap result. It does not fabricate missing GPU counters or profiler proof.
The GH200 project may replace the styling while retaining the values and
limitations.

```python
# build_report.py
import argparse
import html
import json
from pathlib import Path


def fmt(value, unit="ms"):
    return "n/a" if value is None else f"{value:,.3f} {unit}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trials", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--overlap", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.trials.read_text(encoding="utf-8"))
    manifest = (json.loads(args.manifest.read_text(encoding="utf-8"))
                if args.manifest else {})
    overlap = (json.loads(args.overlap.read_text(encoding="utf-8"))
               if args.overlap else None)
    rows = []
    labels = {"window_gap": "Mean content-chunk gap in window",
              "decode_after_first_token_ms": "Decode after first content chunk",
              "full_request_ms": "Full request time",
              "ttft_ms": "Time to first content chunk"}
    for key, label in labels.items():
        values = data["condition_medians"][key]
        change = data["changes"].get(key, {})
        rows.append("<tr><td>" + html.escape(label) + "</td><td>" +
                    fmt(values["idle"]) + "</td><td>" + fmt(values["copy"]) +
                    "</td><td>" + fmt(change.get("absolute_ms")) + " (" +
                    fmt(change.get("percent"), "%") + ")</td></tr>")
    trial_rows = []
    for trial in data["trials"]:
        trial_rows.append("<tr><td>" + str(trial["trial"]) + "</td><td>" +
                          html.escape(trial["mode"]) + "</td><td>" +
                          fmt(trial["window_mean_chunk_gap_ms"]) + "</td><td>" +
                          fmt(trial["decode"]["decode_after_first_token_ms"]) +
                          "</td><td>" + fmt(trial["achieved_copy_GBps"], "GB/s") +
                          "</td></tr>")
    overlap_text = ("not observed; timing only" if overlap is None else
                    "observed" if overlap.get("physical_overlap_observed") else
                    "not observed in diagnostic")
    runtime = html.escape(json.dumps(manifest, indent=2))
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Independent GPU Copy Contention</title>
<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;
color:#172331}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}
th,td{padding:.55rem;text-align:left;border-bottom:1px solid #ccd4dc}
th{background:#e8f2f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;
padding:1rem}small{color:#475b69}</style>
<h1>Independent GPU Copy Contention</h1>
<p>Real SGLang decode; independent synthetic pinned-host-to-GPU copies.</p>
<h2>Condition medians</h2><table><tr><th>Metric</th><th>Idle</th><th>Copy</th><th>Change</th></tr>"""
    document += "".join(rows) + "</table><p>Median achieved copy rate: " + \
                fmt(data["copy_GBps_median"], "GB/s") + ".</p>"
    document += "<p>Physical GPU overlap: " + html.escape(overlap_text) + ".</p>"
    document += ("<h2>All trials</h2><table><tr><th>#</th><th>Mode</th><th>Window gap</th>"
                 "<th>Decode after first chunk</th><th>Copy rate</th></tr>" +
                 "".join(trial_rows) + "</table>")
    document += ("<small>Content chunks are not guaranteed to equal tokens. "
                 "Profiler timing is excluded from condition medians. "
                 "Overlap plus slowdown does not isolate HBM bandwidth from other "
                 "shared GPU resources. This synthetic copy test does not measure "
                 "the frequency of natural KV reloads.</small>")
    document += "<h2>Runtime manifest</h2><pre>" + runtime + "</pre></html>"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
```

Run it after timing and, if available, after the separate overlap diagnostic:

```bash
python3 build_report.py artifacts/timing/trials.json \
  --manifest artifacts/run_manifest.json \
  --overlap artifacts/profile/overlap.json \
  --out artifacts/report.html
```

Omit `--overlap` when the diagnostic has not run. A manifest is mandatory for
the completed handoff even though the builder accepts an absent manifest for
local bring-up. Put `not_observed` or `null` in the manifest for hardware
counters the GH200 tooling cannot expose. Never put zero in their place.

Fail the experiment loudly, rather than publishing a slowdown claim, if:

- Worker and backend GPU identity cannot be reconciled.
- Copy mode moves zero bytes, idle mode moves bytes, or achieved copy GB/s is
  missing. Very low GB/s is not an automatic failure, but must be investigated.
- The worker window starts before the first content chunk, ends after the
  response, or has too few in-window content-chunk gaps.
- Backend errors, unexpected early stream termination, missing output, or another GPU job
  invalidates the matched comparison.
- A physical-overlap claim is requested but the two GPU timelines cannot be
  aligned, the events are on different GPUs, or no overlap is seen.

An observed slowdown of zero or less is **not** a failure. Report it honestly.
Also report if the worker ran but the profiler could not verify the route or
overlap: timing may be retained as exploratory data, but the causal hardware
claim remains unproven.

## 10. Bring-up and completion checklist for the GH200 agent

1. Integrate the five components with the GH200 project's existing supported
   SGLang/runtime stack. Keep the runner and analysis backend-neutral.
2. Show the backend and worker resolve to the same GPU UUID. Confirm pinned
   host source, GPU destination, and nonzero measured H2D bytes/GB/s.
3. Calibrate output length so the full five-second window and at least ten
   content-chunk gaps fit inside active decode; record chosen values.
4. Run an ABBA bring-up block. Verify both modes share allocations and all
   trial-level records are valid. Then run repeated unprofiled ABBA blocks.
5. Collect an independent GPU profiler trace. Export aligned decode-kernel
   and worker-copy intervals, run the overlap check, and retain evidence.
6. Publish a report with raw numbers, achieved GB/s, variation, profiler
   status, and a careful comparison to the A10G pilot. State whether results
   support, contradict, or are inconclusive about stronger GH200 interference.

Before calling the lane reusable, check the embedded scripts with
`python3 -m py_compile`, test an idle and a copy worker window in isolation,
and feed the overlap checker both a known-overlap interval pair and a
different-GPU pair. The first must report overlap; the second must fail.
Retain these test outputs with the first GH200 run.

Do not add natural multi-agent requests, controller policies, priority hints,
or SGLang KV-load commands to this first study. Those are separate follow-up
experiments. This first completed report is the portability and hardware
mechanism baseline for that later work.
