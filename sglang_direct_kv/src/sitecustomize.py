"""Optional process-wide hooks for the direct KV testbed.

Python imports ``sitecustomize`` automatically when it is present on
``PYTHONPATH``. The SGLang launch scripts add this project's ``src`` directory
to ``PYTHONPATH`` so we can enable trace hooks without editing site-packages.

The hook code itself lives in ``packages/agentic-backend-sglang``
(``agentic_backends.sglang``); importing ``agentic_kv`` first puts
``<repo>/packages/*/src`` on ``sys.path`` when the packages are not installed.

Failure policy (changed in the SGLang-portability refactor): a hook that
fails to install now ALWAYS prints a warning to stderr (previously only with
``AGENTIC_KV_TRACE_DEBUG=1``), and ``AGENTIC_SGLANG_STRICT=1`` re-raises so the
server refuses to start with broken instrumentation.
"""

from __future__ import annotations

import os
import sys


def _strict() -> bool:
    return os.environ.get("AGENTIC_SGLANG_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}


def _report(tag: str, exc: BaseException) -> None:
    print(f"[{tag}] failed to install: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if _strict():
        # site.py swallows ordinary exceptions from sitecustomize; SystemExit
        # is the only way to actually stop the interpreter from starting.
        raise SystemExit(f"[{tag}] AGENTIC_SGLANG_STRICT=1 and hook installation failed: {exc}")


def _sglang_installed() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):
        return False


if os.environ.get("AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE", "0") == "1":
    try:
        import agentic_kv  # noqa: F401  (sys.path fallback for packages/)
        from agentic_backends.sglang.compat import maybe_enable_priority_radix_eviction_choice

        maybe_enable_priority_radix_eviction_choice()
    except Exception as exc:  # pragma: no cover - defensive startup hook
        _report("agentic-kv-compat", exc)

if (
    os.environ.get("AGENTIC_KV_TRACE_ENABLE", "0") == "1"
    or os.environ.get("AGENTIC_RUNTIME_TELEMETRY", "0") == "1"
    or os.environ.get("AGENTIC_RUNTIME_TELEMETRY_ENABLE", "0") == "1"
) and _sglang_installed():  # processes without SGLang have nothing to trace (same as before)
    try:
        import agentic_kv  # noqa: F401  (sys.path fallback for packages/)
        from agentic_backends.sglang.trace import install

        install()
    except Exception as exc:  # pragma: no cover - defensive startup hook
        _report("agentic-kv-trace", exc)
