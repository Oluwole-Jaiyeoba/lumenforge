from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentic_core import BackendCommand

from .models import BackendActionResult, BackendCapabilities


@runtime_checkable
class BackendAdapter(Protocol):
    def capabilities(self) -> BackendCapabilities:
        ...

    def apply(self, command: BackendCommand) -> BackendActionResult:
        ...
