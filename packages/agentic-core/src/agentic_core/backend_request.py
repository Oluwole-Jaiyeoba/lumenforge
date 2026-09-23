"""Backend-neutral description of one model request after gateway translation.

The gateway decides *what* a request should carry (prompt, priority, cache
salt, correlation metadata).  A backend integration (for example
``agentic_backends.sglang.lowering``) decides *how* that is encoded on the
wire for one specific backend version.  This record is the contract between
the two, so a backend upgrade never requires touching gateway or controller
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import to_primitive


BACKEND_REQUEST_SCHEMA_VERSION = "agentic_backend_request.v1"


@dataclass(frozen=True)
class BackendRequest:
    """Everything a backend needs to build its native request body.

    Field notes:

    - ``priority``: scheduler priority chosen by the gateway/controller, or
      ``None`` when no priority should be sent.  Larger means more urgent in
      this project's convention; backends translate if their scheduler uses
      another convention.
    - ``cache_salt``: optional cache namespace/isolation key.
    - ``request_context``: correlation identity (session, phase, request id).
    - ``agent_hints``: the full hint record the harness/controller produced for
      this request (priority chain, controller decision ids, cache chain ...).
    - ``request_metadata``: flat correlation metadata that backends may carry
      through to their traces (for SGLang this becomes
      ``custom_params.agentic_kv``).
    - ``speculative_prefill``: speculative-prefill bridge metadata.
    - ``native_cache_bridge``: lowered harness cache signal, or ``None`` when
      the gateway did not lower a cache signal.
    """

    prompt_text: str
    max_tokens: int
    priority: int | None = None
    cache_salt: str = ""
    temperature: float = 0
    stream: bool = True
    request_context: dict[str, Any] = field(default_factory=dict)
    agent_hints: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    speculative_prefill: dict[str, Any] = field(default_factory=dict)
    native_cache_bridge: dict[str, Any] | None = None
    schema_version: str = BACKEND_REQUEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
