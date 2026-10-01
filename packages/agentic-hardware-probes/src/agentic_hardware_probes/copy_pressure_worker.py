"""Independent, copy-only GPU pressure for controlled decode experiments."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def serve(port: int, buffer_mib: int, profile_path: Path | None = None) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the copy-pressure worker")
    nbytes = buffer_mib * 1024 * 1024
    source = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    source.fill_(7)
    target = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        target.copy_(source, non_blocking=True)
    stream.synchronize()
    state: dict[str, object] = {"status": "ready", "buffer_bytes": nbytes}
    lock = threading.Lock()

    def run_window(mode: str, duration_s: float) -> None:
        profiler = None
        if profile_path is not None:
            from torch.profiler import ProfilerActivity, profile

            profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            profiler.__enter__()
        started_ns = time.time_ns()
        deadline = time.monotonic() + duration_s
        copies = 0
        while time.monotonic() < deadline:
            if mode == "copy":
                with torch.cuda.stream(stream):
                    target.copy_(source, non_blocking=True)
                stream.synchronize()
                copies += 1
            else:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        finished_ns = time.time_ns()
        if profiler is not None:
            profiler.__exit__(None, None, None)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(profile_path))
        with lock:
            state.update(
                status="finished", mode=mode, started_ns=started_ns,
                finished_ns=finished_ns, copies=copies, bytes_copied=copies * nbytes,
                profile_path=str(profile_path) if profile_path is not None else None,
            )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def reply(self, status: int, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path != "/status":
                self.reply(404, {"error": "unknown endpoint"})
                return
            with lock:
                self.reply(200, dict(state))

        def do_POST(self) -> None:
            if self.path != "/start":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                mode = request["mode"]
                duration_s = float(request["duration_s"])
                if mode not in {"idle", "copy"} or not 0 < duration_s <= 30:
                    raise ValueError("mode must be idle or copy; duration_s must be in (0, 30]")
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            with lock:
                if state["status"] == "running":
                    self.reply(409, {"error": "a window is already running"})
                    return
                state.clear()
                state.update(status="running", mode=mode, buffer_bytes=nbytes)
            threading.Thread(target=run_window, args=(mode, duration_s), daemon=True).start()
            self.reply(202, {"status": "running", "mode": mode})

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=31992)
    parser.add_argument("--buffer-mib", type=int, default=128)
    parser.add_argument("--profile-path", type=Path)
    args = parser.parse_args()
    if not 1 <= args.buffer_mib <= 1024:
        parser.error("buffer-mib must be between 1 and 1024")
    serve(args.port, args.buffer_mib, args.profile_path)


if __name__ == "__main__":
    main()
