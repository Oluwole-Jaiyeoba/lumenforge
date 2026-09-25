#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for package_src in (REPO_ROOT / "packages").glob("*/src"):
    sys.path.insert(0, str(package_src))

from agentic_core import RunManifest  # noqa: E402


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an immutable experiment run manifest.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--hardware-profile", default="")
    parser.add_argument("--runtime-contract", type=Path)
    parser.add_argument("--workload-json", default="{}")
    parser.add_argument("--instrumentation", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--completion-status", default="created")
    args = parser.parse_args()

    runtime: dict = {}
    if args.runtime_contract:
        runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))
    artifacts: dict[str, str] = {}
    for item in args.artifact:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error(f"--artifact must be NAME=PATH, got: {item}")
        artifacts[name] = path
    manifest = RunManifest(
        run_id=args.run_id,
        created_at_ms=int(time.time() * 1000),
        experiment=args.experiment,
        git_commit=_git_commit(),
        model=args.model,
        hardware_profile=args.hardware_profile,
        backend_name=str(runtime.get("backend_name") or ""),
        backend_version=str(runtime.get("backend_version") or ""),
        workload=json.loads(args.workload_json),
        enabled_instrumentation=tuple(args.instrumentation),
        artifact_locations=artifacts,
        backend_runtime_contract=runtime,
        completion_status=args.completion_status,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote run manifest to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
