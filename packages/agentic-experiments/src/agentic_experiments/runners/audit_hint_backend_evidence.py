"""Audit a hint run against separately gated backend request evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_harnesses.hint_benchmark.backend_evidence import audit_backend_mappings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hint-observations", type=Path, required=True)
    parser.add_argument("--backend-audit", type=Path, required=True)
    parser.add_argument("--backend-events", type=Path, required=True)
    parser.add_argument("--request-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-same-request", action="store_true",
                        help="Fail unless every mapped request has matching capture and backend IDs.")
    args = parser.parse_args()
    audit = json.loads(args.backend_audit.read_text(encoding="utf-8"))
    gate = audit.get("gate") or {}
    if gate.get("profile") != "request_boundary" or gate.get("validation_level") != "live_evidence" or not gate.get("valid"):
        parser.error("backend audit must pass the live request_boundary gate")
    mapping_data = json.loads(args.request_map.read_text(encoding="utf-8"))
    mappings = mapping_data.get("mappings") if isinstance(mapping_data, dict) else mapping_data
    if not isinstance(mappings, list):
        parser.error("request map must be a list or an object with a mappings list")
    result = audit_backend_mappings(read_jsonl(args.hint_observations), mappings,
                                    read_jsonl(args.backend_events))
    result["require_same_request"] = args.require_same_request
    if args.require_same_request:
        unlinked = [row for row in result["mappings"] if not row["same_request_proven"]]
        if unlinked:
            result["valid"] = False
            result["errors"].append({"reason": "same_request_link_not_proven",
                                     "count": len(unlinked)})
    result["backend_adapter"] = gate.get("adapter")
    result["backend_audit"] = str(args.backend_audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
