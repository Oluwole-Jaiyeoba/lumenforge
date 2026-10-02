"""Inspect adapter hooks and fail loudly when required trace evidence is absent."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from agentic_instrumentation import PROFILES, SIGNALS

from .evidence import normalize_trace_event
from .instrumentation_profiles import validate_installation
from .versions import get_adapter


_CONSUMERS = {
    "request.accepted": ("controller_preflight", "controller_report"),
    "batch.scheduled": ("controller_preflight", "controller_report"),
    "batch.completed": ("controller_preflight", "controller_report"),
    "kv.write_host": ("controller_report", "work_audit"),
    "kv.evict_gpu": ("controller_report", "work_audit"),
    "kv.evict_host": ("controller_report", "work_audit"),
    "kv.load_gpu": ("controller_report", "work_audit"),
    "kv.nested_load": ("controller_report",),
    "kv.layer_copy": ("controller_report", "work_audit"),
    "kv.layer_backup": ("controller_report",),
    "kv.detail": ("controller_report",),
    "kv.prefix_match": ("controller_report", "work_audit"),
    "model.forward": ("controller_report",),
}


def hook_inventory(adapter_name: str, observed: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """List every version-adapter hook, including hooks not seen in this run."""
    adapter = get_adapter(adapter_name)
    observed = observed or {}
    result = []
    for target in adapter.hook_targets:
        for method, prefix in target.methods.items():
            name = f"{target.module}.{target.class_name}.{method}"
            sample = observed.get(prefix, {})
            mapped = normalize_trace_event({"event": f"{prefix}.end", "ts_ns": 1}, adapter_name)
            signal_id = mapped.signal_id if mapped and mapped.signal_id in SIGNALS else None
            result.append({
                "target": name,
                "required_target": (target.class_name, method) not in adapter.optional_hooks,
                "source_event_prefix": prefix,
                "signal_id": signal_id,
                "raw_only": signal_id is None,
                "signal_scope": SIGNALS[signal_id].scope if signal_id else None,
                "consumers": list(_CONSUMERS.get(signal_id, ("raw_trace_only",))),
                "required_evidence_fields": list(SIGNALS[signal_id].required_fields) if signal_id else [],
                "trace_cost": SIGNALS[signal_id].cost if signal_id else "unclassified",
                "activation": "AGENTIC_KV_TRACE_SCHEDULER=1" if target.scheduler_required else "backend trace enabled",
                "observed_count": sample.get("count", 0),
                "observed_fields": sample.get("fields", []),
                "observed_context_fields": sample.get("context_fields", []),
            })
    return result


def inspect_trace(rows: Iterable[dict[str, Any]], adapter_name: str) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "fields": set(), "context_fields": set()})
    prefixes = {item["source_event_prefix"] for item in hook_inventory(adapter_name)}
    signal_counts: Counter[str] = Counter()
    valid_fields: set[str] = set()
    invalid_examples: dict[str, list[str]] = {}
    installation: dict[str, Any] | None = None
    for row in rows:
        if row.get("event") == "trace.install.summary":
            installation = {"adapter": row.get("adapter"), "sglang_version": row.get("sglang_version"),
                            "installed_hooks": row.get("installed_hooks") or [],
                            "missing_required_hooks": row.get("missing_required_hooks") or []}
        event = normalize_trace_event(row, adapter_name)
        source = str(row.get("source_event") or row.get("event") or "")
        prefix = source.rsplit(".", 1)[0]
        if prefix not in prefixes:
            continue
        item = observed[prefix]
        item["count"] += 1
        item["fields"].update(row)
        context = row.get("kv_context")
        if isinstance(context, dict):
            item["context_fields"].update(context)
        if event is None or event.signal_id not in SIGNALS:
            continue
        signal_counts[event.signal_id] += 1
        missing = [field for field in SIGNALS[event.signal_id].required_fields
                   if (getattr(event, field, None) if hasattr(event, field) else event.payload.get(field)) in (None, "", {})]
        if not missing:
            valid_fields.add(event.signal_id)
        else:
            invalid_examples.setdefault(event.signal_id, missing)
    return {
        "hooks": hook_inventory(adapter_name, {
            key: {"count": value["count"], "fields": sorted(value["fields"]),
                  "context_fields": sorted(value["context_fields"])}
            for key, value in observed.items()
        }),
        "signal_counts": dict(signal_counts),
        "signals_with_valid_fields": sorted(valid_fields),
        "invalid_field_examples": invalid_examples,
        "embedded_installation": installation,
    }


def validate_bundle(profile: str, adapter_name: str, installation: dict[str, Any], inspection: dict[str, Any]) -> dict[str, Any]:
    install = validate_installation(profile, adapter_name, installation)
    required = PROFILES[profile].required_signals
    counts = inspection["signal_counts"]
    valid_fields = set(inspection["signals_with_valid_fields"])
    missing_events = [name for name in required if not counts.get(name)]
    missing_fields = [name for name in required if counts.get(name) and name not in valid_fields]
    return {
        "profile": profile,
        "adapter": adapter_name,
        "valid": install["valid"] and not missing_events and not missing_fields,
        "installation": install,
        "missing_events": missing_events,
        "invalid_fields": missing_fields,
        "optional_unobserved": [name for name in PROFILES[profile].optional_signals if not counts.get(name)],
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                yield value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--installation", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    inspection = inspect_trace(_read_jsonl(args.trace), args.adapter)
    installation = (json.loads(args.installation.read_text(encoding="utf-8")) if args.installation
                    else inspection["embedded_installation"])
    if not installation:
        parser.error("pass --installation or include trace.install.summary in the trace")
    output = {"schema_version": "agentic.instrumentation.audit.v1", "trace": str(args.trace),
              "installation_report": str(args.installation) if args.installation else "embedded trace.install.summary",
              "inspection": inspection,
              "gate": validate_bundle(args.profile, args.adapter, installation, inspection)}
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)
    return 0 if output["gate"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
