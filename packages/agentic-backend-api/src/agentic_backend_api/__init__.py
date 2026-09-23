"""Portable capability and command interface for inference backends."""

from .models import BACKEND_API_SCHEMA_VERSION, BackendActionResult, BackendCapabilities
from .protocols import BackendAdapter

__all__ = [
    "BACKEND_API_SCHEMA_VERSION",
    "BackendActionResult",
    "BackendAdapter",
    "BackendCapabilities",
]
