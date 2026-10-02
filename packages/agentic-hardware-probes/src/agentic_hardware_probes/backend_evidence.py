"""Join shared backend evidence to one sustained-decode hardware trial.

This proves hook completion during a client-visible decode interval, not HBM
contention or GPU-kernel overlap. Raw backend parsing belongs to its adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess_trial(trial: dict[str, Any], condition: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    request_id = str(trial.get("request_id") or "")
    donor_session = str(trial.get("donor_session_id") or "")
    identity_source = "explicit"
    if not donor_session and request_id.endswith("-target-decode"):
        donor_session = request_id.removesuffix("-target-decode") + "-donor"
        identity_source = "legacy_request_suffix"
    decode = trial.get("decode") or {}
    start = int(decode.get("request_start_ns") or 0)
    finish = int(decode.get("request_end_ns") or 0)
    if not request_id or not donor_session or not 0 < start < finish:
        raise ValueError("trial lacks target request, donor session, or decode timestamps")

    accepted = [row for row in events if row.get("signal_id") == "request.accepted"
                and row.get("request_id") == request_id]
    copies = [row for row in events if row.get("signal_id") == "kv.layer_copy"
              and row.get("session_id") == donor_session]
    loads = [row for row in events if row.get("signal_id") == "kv.load_gpu"
             and row.get("session_id") == donor_session]
    during = [row for row in copies if start <= int(row.get("time_ns") or 0) <= finish]
    before = [row for row in copies if 0 < int(row.get("time_ns") or 0) < start]
    missing = []
    if not accepted:
        missing.append("target_request_accepted")
    if condition == "direct_overlap_reload" and not during:
        missing.append("donor_layer_copy_during_client_decode")
    if condition == "non_overlap_reload" and not before:
        missing.append("donor_layer_copy_before_client_decode")
    return {
        "schema_version": "hardware.backend_evidence.v1",
        "condition": condition,
        "target_request_id": request_id,
        "donor_session_id": donor_session,
        "donor_identity_source": identity_source,
        "target_accepted_count": len(accepted),
        "donor_semantic_load_count": len(loads),
        "donor_layer_copy_count": len(copies),
        "donor_layer_copies_before_decode": len(before),
        "donor_layer_copies_during_decode": len(during),
        "valid": not missing,
        "missing": missing,
        "proof_limit": "Hook timestamps establish client-decode-window coincidence, not GPU-kernel overlap or HBM contention.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    trials = [row for row in probe.get("trials", []) if row.get("sample_id") == args.sample_id]
    if len(trials) != 1:
        parser.error(f"expected one trial for {args.sample_id}, found {len(trials)}")
    events = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = assess_trial(trials[0], str(probe["condition"]), events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
