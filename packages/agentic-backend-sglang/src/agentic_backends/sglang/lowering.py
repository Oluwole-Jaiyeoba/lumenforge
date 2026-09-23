"""Encode a backend-neutral ``BackendRequest`` as an SGLang request body.

This is the only place that knows SGLang's OpenAI-compatible wire fields the
project relies on:

- top-level ``priority`` (honoured with ``--enable-priority-scheduling``),
- ``custom_params`` (free-form dict SGLang passes through to the request; our
  trace hooks read ``custom_params.agentic_kv`` for correlation),
- ``cache_salt`` (cache namespace isolation).

The adapter surface lists these as ``ChatCompletionRequest`` fields, so the
static surface check fails loudly if a future SGLang release drops or renames
them.  The produced JSON is byte-identical to the pre-refactor
``build_sglang_payload`` (proven by ``tests/golden``).
"""

from __future__ import annotations

from typing import Any

from agentic_core import BackendRequest


def lower_translation(request: BackendRequest, model: str) -> dict[str, Any]:
    """Build the ``/v1/chat/completions`` JSON body for SGLang."""

    agent_hints = request.agent_hints
    context = request.request_context
    custom_params: dict[str, Any] = {
        "agentic_kv": request.request_metadata,
        "request_context": context,
        "nvext": {"agent_hints": agent_hints, "request_context": context},
        "dynamo_speculative_prefill_bridge": request.speculative_prefill,
    }
    if request.native_cache_bridge is not None:
        custom_params["native_cache_bridge"] = request.native_cache_bridge
        custom_params["nvext"]["cache_control"] = request.native_cache_bridge
    out: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt_text}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream,
        "custom_params": custom_params,
    }
    if request.priority is not None:
        out["priority"] = request.priority
        out["nvext"] = {"agent_hints": agent_hints, "request_context": context}
    if request.cache_salt:
        out["cache_salt"] = request.cache_salt
    return out


class SGLangRequestLowering:
    """``agentic_backend_api.RequestLowering`` implementation for SGLang."""

    def lower(self, request: BackendRequest, *, model: str) -> dict[str, Any]:
        return lower_translation(request, model)
