"""Build and pre-flight SGLang server command lines.

``launch_spec`` turns backend-neutral options into ``python -m
sglang.launch_server ...``.  ``preflight`` statically checks that every
``--flag`` in a command line is accepted by the installed (or given) SGLang
source, so an upgrade that renames a flag fails *before* a 10-minute model
load instead of halfway through an experiment.

CLI (used by ``sglang_direct_kv/scripts/run_sglang_*server.sh``)::

    python -m agentic_backends.sglang.launch preflight -- --enable-priority-scheduling --schedule-policy lpm
    python -m agentic_backends.sglang.launch preflight --strict -- $EXTRA_SERVER_ARGS

Exit status: 0 when all flags are known (or when not strict), 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from agentic_backend_api import LaunchSpec

from .selection import installed_sglang_version, select_adapter


def _flag(name: str) -> str:
    return name if name.startswith("--") else "--" + name.replace("_", "-")


def launch_spec(
    *,
    model: str,
    host: str = "0.0.0.0",
    port: int = 30000,
    options: Mapping[str, Any] | None = None,
    python: str = "python",
    sglang_version: str | None = None,
) -> LaunchSpec:
    """``options`` maps flag names (``enable_priority_scheduling`` or
    ``--enable-priority-scheduling``) to a value; ``True`` means a bare flag,
    ``False``/``None`` omits it."""

    argv: list[str] = [python, "-m", "sglang.launch_server", "--model-path", model, "--host", host, "--port", str(port)]
    for name, value in (options or {}).items():
        if value is None or value is False:
            continue
        argv.append(_flag(name))
        if value is not True:
            argv.append(str(value))
    version = sglang_version if sglang_version is not None else installed_sglang_version()
    selection = select_adapter(version)
    return LaunchSpec(
        backend_name="sglang",
        backend_version=version,
        argv=tuple(argv),
        notes=(f"adapter={selection.adapter.name}", f"selection={selection.status}"),
    )


def unknown_flags(argv: Sequence[str], source_root: str | None = None) -> list[str]:
    """Flags in ``argv`` that the SGLang source does not declare."""

    from .surface import SourceIndex, _flag_present, installed_source_root

    root = source_root or installed_source_root()
    if not root:
        return []
    index = SourceIndex(root)
    missing: list[str] = []
    for token in argv:
        if not token.startswith("--") or token == "--":
            continue
        flag = token.split("=", 1)[0]
        present, _ = _flag_present(index, flag)
        if not present:
            missing.append(flag)
    return missing


def preflight(argv: Sequence[str], *, source_root: str | None = None, strict: bool = False) -> int:
    try:
        missing = unknown_flags(argv, source_root)
    except Exception as exc:  # never block a launch because the checker itself broke
        print(f"[agentic-sglang-preflight] check skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0
    if missing:
        print(
            "[agentic-sglang-preflight] WARNING: SGLang "
            f"{installed_sglang_version() or '?'} does not declare: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1 if strict else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SGLang launch helpers")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="check that server flags exist in the installed SGLang")
    pre.add_argument("--strict", action="store_true")
    pre.add_argument("--sglang-src", default="")
    pre.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    flags = [f for f in args.flags if f != "--"]
    return preflight(flags, source_root=args.sglang_src or None, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
