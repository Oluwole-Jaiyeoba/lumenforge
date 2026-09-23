"""Choose the SGLang adapter for an installed/requested SGLang version.

Selection order (``select_adapter``):

1. ``AGENTIC_SGLANG_ADAPTER=<name>`` forces an adapter (status ``forced``).
2. The adapter whose ``version_range`` contains the version
   (status ``tested`` if the version is in ``tested_versions``, else
   ``in_range``).
3. Otherwise, if an SGLang source tree is available, every adapter is
   statically surface-checked newest-first and the first one with no missing
   *required* surface is used (status ``probed``).
4. Otherwise: with ``strict=True`` (or ``AGENTIC_SGLANG_STRICT=1``) raise
   ``UnsupportedBackendVersion``; without strict, fall back to the legacy
   choice (``v0510`` below 0.5.11, newest adapter above) with status
   ``unverified`` -- and callers are expected to warn loudly.

The old code silently mapped *any* unparsable or unknown version to ``v0510``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from agentic_backend_api import CompatibilityReport, UnsupportedBackendVersion

from .versions import ADAPTERS, AdapterSpec, newest_adapter

STATUS_FORCED = "forced"
STATUS_TESTED = "tested"
STATUS_IN_RANGE = "in_range"
STATUS_PROBED = "probed"
STATUS_UNVERIFIED = "unverified"


def parse_version(version: str | None) -> tuple[int, int, int, int] | None:
    """``"0.5.10.post1"`` -> ``(0, 5, 10, 1)``; returns None when unparsable."""

    if not version:
        return None
    match = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", version)
    if not match:
        return None
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def _in_range(version: tuple[int, int, int, int], adapter: AdapterSpec) -> bool:
    low = parse_version(adapter.version_range[0])
    high = parse_version(adapter.version_range[1])
    if low is None or high is None:
        return False
    return low <= version < high


def adapter_for_version(version: str | None) -> AdapterSpec | None:
    parsed = parse_version(version)
    if parsed is None:
        return None
    for adapter in ADAPTERS.values():
        if _in_range(parsed, adapter):
            return adapter
    return None


def legacy_adapter_name(version: str | None) -> str:
    """Name the pre-refactor selector would have produced, extended to new adapters.

    Kept for ``agentic_kv.sglang_adapters.select_adapter_name`` callers: never
    raises; unparsable versions map to ``v0510`` exactly as before.
    """

    adapter = adapter_for_version(version)
    if adapter is not None:
        return adapter.name
    parsed = parse_version(version)
    if parsed is None or parsed < (0, 5, 11, 0):
        return "v0510"
    return newest_adapter().name


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AdapterSelection:
    adapter: AdapterSpec
    status: str
    sglang_version: str
    reason: str
    report: CompatibilityReport | None = None

    @property
    def trusted(self) -> bool:
        return self.status in (STATUS_FORCED, STATUS_TESTED, STATUS_IN_RANGE, STATUS_PROBED)

    def warning(self) -> str:
        """Human-readable warning, or "" when nothing needs attention."""

        parts: list[str] = []
        if self.status == STATUS_UNVERIFIED:
            parts.append(
                f"SGLang {self.sglang_version or '<unknown>'} is not covered by any adapter; "
                f"using {self.adapter.name} unverified ({self.reason})"
            )
        elif self.adapter.verification != "runtime" and self.status != STATUS_FORCED:
            parts.append(
                f"adapter {self.adapter.name} for SGLang {self.sglang_version} is only "
                f"'{self.adapter.verification}'-verified; run the reference experiment before trusting results"
            )
        if self.report is not None and not self.report.ok:
            parts.append("missing required SGLang surface: " + ", ".join(self.report.broken_features()))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter.name,
            "status": self.status,
            "sglang_version": self.sglang_version,
            "reason": self.reason,
            "verification": self.adapter.verification,
            "warning": self.warning(),
            "surface_ok": None if self.report is None else self.report.ok,
            "broken_features": [] if self.report is None else list(self.report.broken_features()),
        }


def select_adapter(
    version: str | None = None,
    *,
    source_root: str | None = None,
    surface_check: bool = False,
    strict: bool | None = None,
) -> AdapterSelection:
    """Pick an adapter.  See module docstring for the rules.

    ``version`` defaults to the installed SGLang version (read from package
    metadata; SGLang is never imported).  ``surface_check=True`` additionally
    runs the static surface check on the chosen adapter and attaches the
    report; with ``strict`` a failing required surface raises.
    """

    from .surface import SourceIndex, check_adapter, installed_source_root

    if strict is None:
        strict = _truthy(os.environ.get("AGENTIC_SGLANG_STRICT"))
    if version is None:
        version = installed_sglang_version()

    def finish(selection: AdapterSelection) -> AdapterSelection:
        report = None
        if surface_check:
            root = source_root or installed_source_root()
            if root:
                try:
                    report = check_adapter(SourceIndex(root), selection.adapter)
                except Exception:  # a broken checker must never take the server down
                    if strict:
                        raise
                    report = None
        selection = AdapterSelection(
            adapter=selection.adapter,
            status=selection.status,
            sglang_version=selection.sglang_version,
            reason=selection.reason,
            report=report,
        )
        if strict and (not selection.trusted or (report is not None and not report.ok)):
            raise UnsupportedBackendVersion(selection.warning() or "SGLang adapter selection not trusted")
        return selection

    forced = os.environ.get("AGENTIC_SGLANG_ADAPTER", "").strip()
    if forced:
        if forced not in ADAPTERS:
            raise UnsupportedBackendVersion(f"AGENTIC_SGLANG_ADAPTER={forced!r} is not a known adapter: {sorted(ADAPTERS)}")
        return finish(AdapterSelection(ADAPTERS[forced], STATUS_FORCED, version or "", "AGENTIC_SGLANG_ADAPTER"))

    adapter = adapter_for_version(version)
    if adapter is not None:
        status = STATUS_TESTED if version in adapter.tested_versions else STATUS_IN_RANGE
        return finish(AdapterSelection(adapter, status, version or "", f"version in {adapter.version_range}"))

    root = source_root or installed_source_root()
    if root:
        try:
            index = SourceIndex(root)
            for candidate in reversed(list(ADAPTERS.values())):
                report = check_adapter(index, candidate)
                if report.ok:
                    return AdapterSelection(candidate, STATUS_PROBED, version or "", "static surface check passed", report)
        except Exception:
            if strict:
                raise

    fallback = ADAPTERS[legacy_adapter_name(version)]
    return finish(
        AdapterSelection(fallback, STATUS_UNVERIFIED, version or "", "no adapter range matched and no surface probe passed")
    )


def installed_sglang_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return ""
