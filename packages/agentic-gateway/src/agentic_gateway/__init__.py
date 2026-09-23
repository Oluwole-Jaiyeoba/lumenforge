"""Backend-neutral gateway translation.

``translate_request(payload, meta, api_kind) -> agentic_core.BackendRequest``
decides what a model request should carry; a backend package
(``agentic_backends.sglang.lowering``) encodes it for one backend version.

May import: agentic_core, agentic_harnesses.  Must NOT import agentic_backends
(the gateway composes a backend at runtime; see scripts/harness_sglang_gateway.py).
"""

from .client_api import api_kind_from_path, marker_from_payload, request_shape
from .translation import (
    cache_translation_context,
    metadata_context,
    priority_translation_context,
    resolve_backend_priority,
    translate_request,
)

__all__ = [
    "api_kind_from_path",
    "cache_translation_context",
    "marker_from_payload",
    "metadata_context",
    "priority_translation_context",
    "request_shape",
    "resolve_backend_priority",
    "translate_request",
]
