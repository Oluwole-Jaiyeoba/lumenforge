#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

import agentic_kv  # noqa: F401  (puts <repo>/packages/*/src on sys.path when not installed)
from agentic_kv.controller import build_harness_controller_signal

# --- Moved during the SGLang-portability refactor (docs/architecture/README_RESTRUCTURING.md) ---
# Harness hint parsing   -> agentic_harnesses.request_hints
# Client API parsing     -> agentic_gateway.client_api
# Mode/priority/cache translation -> agentic_gateway.translation
# SGLang wire encoding   -> agentic_backends.sglang.lowering
# The names stay importable from this script for backward compatibility.
from agentic_harnesses.request_hints import (  # noqa: F401
    as_dict,
    compact_json,
    is_present,
    CACHE_SIGNAL_KEYS,
    CACHE_IDENTITY_KEYS,
    CACHE_SIGNAL_HEADERS,
    CACHE_IDENTITY_HEADERS,
    payload_nvext_priority,
    emitted_priority_signal,
    iter_structured_signals,
    header_signal_items,
    emitted_cache_signal,
    first_signal_value,
    first_header_signal_value,
    emitted_signal_is_urgent,
)
from agentic_gateway.client_api import (  # noqa: F401
    MARKER,
    parse_b64_json,
    iter_text,
    strip_marker,
    marker_from_payload,
    payload_text_chars,
    payload_stream_requested,
    payload_model,
    request_shape,
    is_bookkeeping_payload,
    text_from_openai_chat,
    text_from_anthropic_messages,
    text_from_responses,
    api_kind_from_path,
)
from agentic_gateway.translation import (  # noqa: F401
    PRIORITY_ENABLED_MODES,
    PRE_HARNESS_PRIORITY_MODE,
    NAT_INFERRED_PRIORITY_MODE,
    CACHE_LOWER_MODE,
    HARNESS_EMITTED_SIGNAL_MODE,
    CONTROLLER_SCHEDULER_PRIORITY_MODE,
    CONTROLLER_DEMOTE_RESTORE_MODE,
    CONTROLLER_PRIORITY_DEMOTE_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE,
    CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE,
    CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE,
    CONTROLLER_DEADLINE_FAIR_MODE,
    CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_MODE,
    CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_ADMISSION_GUARD_MODE,
    CONTROLLER_READY_TIME_GPU_BACKFILL_MODE,
    CONTROLLER_VALUE_AWARE_EVICTION_MODE,
    CONTROLLER_MEMORY_ADMISSION_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MODES,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
    CONTROLLER_ADMISSION_CONTROL_MODE,
    CONTROLLER_FULL_MODE,
    CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
    CONTROLLER_PRIORITY_MODES,
    CACHE_SIGNAL_MODES,
    cache_translation_context,
    metadata_context,
    priority_enabled,
    priority_translation_context,
    resolve_backend_priority,
    translate_request,
)
from agentic_backends.sglang.lowering import lower_translation

# Historical name; the priority is backend-neutral, the function was renamed.
sglang_priority = resolve_backend_priority


def build_sglang_payload(payload: dict[str, Any], meta: dict[str, Any], api_kind: str, model: str) -> dict[str, Any]:
    """Translate a harness request and encode it for SGLang (same JSON as before the refactor)."""

    return lower_translation(translate_request(payload, meta, api_kind), model)


def write_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    row.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def extract_chat_delta(line: str) -> str:
    if not line.startswith("data: "):
        return ""
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return ""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = as_dict(as_dict(choices[0]).get("delta"))
    value = delta.get("content")
    return value if isinstance(value, str) else ""


def call_sglang(target_base: str, payload: dict[str, Any]) -> tuple[str, float | None, float, int, int, int | None, int]:
    start = time.perf_counter()
    forward_started_ns = time.time_ns()
    first = None
    first_content_ns = None
    text_parts: list[str] = []
    chunks = 0
    status = 502
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{target_base.rstrip('/')}/v1/chat/completions", json=payload) as response:
            status = response.status_code
            response.raise_for_status()
            for line in response.iter_lines():
                chunk = extract_chat_delta(line)
                if chunk and first is None:
                    first = time.perf_counter()
                    first_content_ns = time.time_ns()
                if chunk:
                    text_parts.append(chunk)
                if line.startswith("data: ") and line.removeprefix("data: ").strip() != "[DONE]":
                    chunks += 1
    end = time.perf_counter()
    return ("".join(text_parts), (first - start) * 1000.0 if first is not None else None,
            (end - start) * 1000.0, chunks, status, first_content_ns, forward_started_ns)


def fake_chat_response(text: str, model: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl_harness_gateway",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode("utf-8")


def chat_sse(text: str, model: str) -> bytes:
    chunk_id = f"chatcmpl_harness_gateway_{int(time.time() * 1_000_000)}"
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    done = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        f"data: {json.dumps(done, separators=(',', ':'))}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def anthropic_response(text: str, model: str) -> bytes:
    return json.dumps(
        {
            "id": "msg_harness_gateway",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": max(1, len(text.split()))},
        }
    ).encode("utf-8")


def response_object(text: str, model: str) -> dict[str, Any]:
    message_id = f"msg_harness_{int(time.time() * 1_000_000)}"
    return {
        "id": f"resp_harness_{int(time.time() * 1_000_000)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "metadata": {},
        "error": None,
        "incomplete_details": None,
        "reasoning": {"effort": None, "summary": None},
        "usage": {
            "input_tokens": 1,
            "output_tokens": max(1, len(text.split())),
            "total_tokens": 1 + max(1, len(text.split())),
            "output_tokens_details": {"reasoning_tokens": 0},
            "input_tokens_details": {"cached_tokens": 0},
        },
    }


def responses_sse(text: str, model: str) -> bytes:
    response = response_object(text, model)
    item = response["output"][0]
    part = item["content"][0]
    in_progress = dict(response)
    in_progress["status"] = "in_progress"
    in_progress["output"] = []
    events = [
        ("response.created", {"type": "response.created", "sequence_number": 0, "response": in_progress}),
        ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": {**item, "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "sequence_number": 2, "item_id": item["id"], "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 3, "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text}),
        ("response.output_text.done", {"type": "response.output_text.done", "sequence_number": 4, "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text}),
        ("response.content_part.done", {"type": "response.content_part.done", "sequence_number": 5, "item_id": item["id"], "output_index": 0, "content_index": 0, "part": part}),
        ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0, "item": item}),
        ("response.completed", {"type": "response.completed", "sequence_number": 7, "response": response}),
    ]
    return "".join(
        f"event: {event_name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        for event_name, data in events
    ).encode("utf-8")


def make_handler(target_base: str, trace_path: Path | None, log_path: Path | None, model: str,
                 encoder=None, encoding_scope: str = "target_requests"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HarnessSGLangGateway/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(fmt % args, file=sys.stderr)

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/v1/models"):
                models = [model]
                if "sglang-qwen-coder" not in models:
                    models.append("sglang-qwen-coder")
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model"} for model_id in models],
                        "models": models,
                    }
                ).encode()
            else:
                body = json.dumps({"ok": True, "gateway": "harness_sglang_gateway"}).encode()
            self._send_bytes(200, body, "application/json")

        def do_HEAD(self) -> None:
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            started = time.perf_counter()
            received_ns = time.time_ns()
            body = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                payload = {}
            api_kind = api_kind_from_path(self.path)
            payload_dict = as_dict(payload)
            request_headers = {key: value for key, value in self.headers.items()}
            shape = request_shape(body, payload, api_kind)
            write_jsonl(log_path, {"event": "gateway.request_received", "path": self.path, **shape})
            meta = marker_from_payload(payload)
            if is_bookkeeping_payload(payload) or not meta or meta.get("marker_parse_error"):
                write_jsonl(log_path, {"event": "gateway.unmarked_request", "path": self.path, **shape})
                if api_kind == "anthropic":
                    return self._send_bytes(200, anthropic_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")
                if api_kind == "responses":
                    return self._send_bytes(200, responses_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                if payload_stream_requested(payload):
                    return self._send_bytes(200, chat_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                return self._send_bytes(200, fake_chat_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")

            session_id = str(meta.get("session_id") or "")
            phase = str(meta.get("phase") or "")
            mode = str(meta.get("mode") or "")
            harness = str(meta.get("harness") or "")
            label = str(meta.get("label") or f"{session_id}_{phase}")
            seen_labels = getattr(self.server, "seen_request_labels", set())
            if label in seen_labels:
                write_jsonl(
                    log_path,
                    {
                        "event": "gateway.duplicate_marked_request",
                        "path": self.path,
                        "api_kind": api_kind,
                        "harness": harness,
                        "phase": phase,
                        "label": label,
                    },
                )
                if api_kind == "anthropic":
                    return self._send_bytes(200, anthropic_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")
                if api_kind == "responses":
                    return self._send_bytes(200, responses_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                if payload_stream_requested(payload):
                    return self._send_bytes(200, chat_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                return self._send_bytes(200, fake_chat_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")
            seen_labels.add(label)
            setattr(self.server, "seen_request_labels", seen_labels)
            priority = sglang_priority(meta, payload_dict)
            priority_chain = priority_translation_context(meta, payload_dict)
            cache_chain = cache_translation_context(meta, payload_dict, request_headers)
            common = {
                "session_id": session_id,
                "phase": phase,
                "mode": mode,
                "harness": harness,
                "label": label,
                "request_id": label,
                "pressure_level": meta.get("pressure_level", ""),
                "task_index": meta.get("task_index", ""),
                "request_group": meta.get("request_group", ""),
                "prompt_tokens": meta.get("prompt_tokens", ""),
                "max_tokens": meta.get("max_tokens", ""),
                "tool_wait_ms": meta.get("tool_wait_ms", ""),
                "tool_wait_step": meta.get("tool_wait_step", ""),
                "task_replay_steps": meta.get("task_replay_steps", ""),
                "tool_wait_profile": meta.get("tool_wait_profile", ""),
                "tool_wait_class": meta.get("tool_wait_class", ""),
                "agentic_workload_profile": meta.get("agentic_workload_profile", ""),
                "workload_phase_family": meta.get("workload_phase_family", ""),
                "workload_request_kind": meta.get("workload_request_kind", ""),
                "workload_prompt_tokens_target": meta.get("workload_prompt_tokens_target", ""),
                "workload_max_tokens": meta.get("workload_max_tokens", ""),
                "workload_description": meta.get("workload_description", ""),
                "estimated_runtime_ms": meta.get("estimated_runtime_ms", ""),
                "concurrency": meta.get("concurrency", ""),
                "active_background_requests": meta.get("active_background_requests", ""),
                "demotable_background_requests": meta.get("demotable_background_requests", ""),
                "filler_sessions": meta.get("filler_sessions", ""),
                "filler_backlog_mode": meta.get("filler_backlog_mode", ""),
                "filler_backlog_target": meta.get("filler_backlog_target", ""),
                "filler_backlog_total": meta.get("filler_backlog_total", ""),
                "client_submit_seq": meta.get("client_submit_seq", ""),
                "client_sem_capacity": meta.get("client_sem_capacity", ""),
                "client_pending_before_acquire": meta.get("client_pending_before_acquire", ""),
                "client_inflight_before_acquire": meta.get("client_inflight_before_acquire", ""),
                "client_pending_at_submit": meta.get("client_pending_at_submit", ""),
                "client_inflight_at_submit": meta.get("client_inflight_at_submit", ""),
                "client_queue_wait_ms": meta.get("client_queue_wait_ms", ""),
                "prompt_hash": meta.get("prompt_hash", ""),
                "oracle_runtime_key": meta.get("oracle_runtime_key", ""),
                "hint_source": "harness_gateway_intercept",
                "dynamo_agent_priority": "high" if priority is not None and priority > 0 else "low" if priority is not None else "",
                "sglang_priority": priority if priority is not None else "",
                "dynamo_hint_priority": priority if priority is not None else "",
                "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
                "harness_controller_signal": build_harness_controller_signal(
                    {
                        **{key: value for key, value in meta.items() if key != "harness_controller_signal"},
                        **priority_chain,
                        **cache_chain,
                        "sglang_priority": priority if priority is not None else "",
                    }
                ),
                "priority_policy": "harness_gateway_intercepted_sglang_priority" if priority is not None else "none",
                "speculative_prefill": bool(meta.get("speculative_prefill")),
                "speculative_prefill_role": meta.get("speculative_prefill_role", ""),
                "speculative_prefill_strategy": meta.get("speculative_prefill_strategy", ""),
                "controller_decision_id": meta.get("controller_decision_id", ""),
                "controller_command_id": meta.get("controller_command_id", ""),
                "controller_priority_translation": meta.get("controller_priority_translation", ""),
                "controller_replay_rank": meta.get("controller_replay_rank", ""),
                "controller_urgent_replay_count": meta.get("controller_urgent_replay_count", ""),
                "controller_priority_ladder": meta.get("controller_priority_ladder", ""),
                "controller_eviction_policy": meta.get("controller_eviction_policy", ""),
                "controller_eviction_value_score": meta.get("controller_eviction_value_score", ""),
                "controller_eviction_value_class": meta.get("controller_eviction_value_class", ""),
                "controller_eviction_translation": meta.get("controller_eviction_translation", ""),
                "parent_request_id": meta.get("parent_request_id", ""),
                "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
                "warmup_prompt_tokens": meta.get("warmup_prompt_tokens", ""),
                "kv_storage_enabled": meta.get("kv_storage_enabled", ""),
                "kv_storage_backend": meta.get("kv_storage_backend", ""),
                "kv_storage_prefetch_policy": meta.get("kv_storage_prefetch_policy", ""),
                "kv_storage_path": meta.get("kv_storage_path", ""),
                "kv_storage_signal_source": meta.get("kv_storage_signal_source", ""),
                **priority_chain,
                **cache_chain,
                **shape,
            }
            write_jsonl(trace_path, {"event": "m27.request.submitted", **common})
            write_jsonl(trace_path, {"event": "m27.request.start", **common})
            status = 502
            first_content_ns = None
            forward_started_ns = None
            common["gateway_received_ns"] = received_ns
            common["encoding_codec"] = "identity"
            common["encoding_status"] = "disabled"
            text = ""
            ttft_ms = 0.0
            latency_ms = 0.0
            chunks = 0
            error = ""
            try:
                sglang_payload = build_sglang_payload(as_dict(payload), meta, api_kind, model)
                if encoder is not None:
                    eligible = (encoding_scope == "all" or phase == "replay"
                                or (encoding_scope == "target_requests" and phase == "initial_turn"))
                    common["encoding_codec"] = encoder.config.codec
                    common["encoding_config_hash"] = encoder.fingerprint
                    common["encoding_scope"] = encoding_scope
                    if eligible:
                        encoding = encoder.encode(sglang_payload, "openai_chat")
                        sglang_payload = encoding.payload
                        common.update(encoding.evidence())
                    else:
                        common.update(encoding_status="skipped", encoding_reason="out_of_scope")
                    write_jsonl(log_path, {"event": "gateway.prompt_encoding", **common})
                write_jsonl(log_path, {"event": "gateway.forward_start", "path": self.path, **common})
                text, ttft_ms, latency_ms, chunks, status, first_content_ns, forward_started_ns = call_sglang(target_base, sglang_payload)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                text = "gateway-error"
                latency_ms = (time.perf_counter() - started) * 1000.0
                ttft_ms = None
            common.update(first_content_ts_ns=first_content_ns, backend_forward_started_ns=forward_started_ns,
                          first_token_observed=first_content_ns is not None,
                          gateway_to_first_content_ms=((first_content_ns - received_ns) / 1e6
                                                       if first_content_ns is not None else None))
            if not error and first_content_ns is None:
                error = "empty_backend_content"
            write_jsonl(trace_path, {"event": "m27.request.end", **common, "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None, "total_latency_ms": round(latency_ms, 3), "stream_chunks": chunks, "status": status, "error": error})
            write_jsonl(log_path, {"event": "gateway.forwarded_request", "path": self.path, "api_kind": api_kind, **common, "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None, "total_latency_ms": round(latency_ms, 3), "stream_chunks": chunks, "status": status, "error": error})
            if error:
                return self._send_bytes(502, json.dumps({"error": error}).encode("utf-8"), "application/json")
            if api_kind == "anthropic":
                return self._send_bytes(200, anthropic_response(text, str(payload_dict.get("model") or model)), "application/json")
            if api_kind == "responses":
                return self._send_bytes(200, responses_sse(text, str(payload_dict.get("model") or model)), "text/event-stream")
            if payload_stream_requested(payload):
                return self._send_bytes(200, chat_sse(text, str(payload_dict.get("model") or model)), "text/event-stream")
            return self._send_bytes(200, fake_chat_response(text, str(payload_dict.get("model") or model)), "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Codex/Claude/Hatcher harness traffic into SGLang requests.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=31080)
    parser.add_argument("--target-base", default="http://127.0.0.1:30000")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--prompt-codec-config", default=os.environ.get("PROMPT_CODEC_CONFIG", ""))
    parser.add_argument("--encoding-scope", choices=("target_requests", "replay", "all"),
                        default=os.environ.get("PROMPT_ENCODING_SCOPE", "target_requests"))
    args = parser.parse_args()
    encoder = None
    if args.prompt_codec_config:
        from agentic_prompt_codec.config import load_encoder
        encoder = load_encoder(args.prompt_codec_config, args.model)
        write_jsonl(args.log, {"event": "gateway.encoding_config", "encoding_codec": encoder.config.codec,
                              "encoding_config_hash": encoder.fingerprint,
                              "encoding_scope": args.encoding_scope,
                              "tokenizer_id": getattr(encoder.counter, "identity", "")})
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), make_handler(args.target_base, args.trace, args.log, args.model, encoder, args.encoding_scope))
    setattr(server, "seen_request_labels", set())
    print(f"harness gateway listening on http://{args.listen_host}:{args.listen_port} -> {args.target_base}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
