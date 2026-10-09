"""Run independent controller scenarios and freeze their reproduction evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def clean_environment() -> dict[str, str]:
    # Do not inherit old experimental flags or serialize authentication secrets.
    allowed = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH"}
    return {key: value for key, value in os.environ.items() if key in allowed}


def arm_environment(root: Path, spec: dict, scenario: str, trial: int, mode: str,
                    run_id: str, cache: Path, image: str) -> dict[str, str]:
    values = dict(spec["common_env"])
    values.update(spec["scenarios"][scenario]["env"])
    values.update({
        "MODES": mode, "TOOL_WAIT_SEED": str(spec["seeds"][trial - 1]),
        "SGLANG_DOCKER_IMAGE": image, "AGENTIC_MODEL_CACHE": str(cache),
        "RESULTS_ROOT": f"artifacts/results/work_audit/{run_id}/raw",
        "REPORT_LABEL": f"s{scenario}_trial{trial}_{mode}",
        "PYTHONPATH": ":".join(str(p) for p in sorted((root / "packages").glob("*/src"))),
    })
    return values


def assert_idle() -> None:
    for port in (30000, 31080, 31991):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} already occupied; leave that workload untouched")
    processes = command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader")
    if processes:
        raise RuntimeError(f"GPU has active compute processes: {processes}")


def compress_traces(path: Path) -> None:
    for source in path.rglob("*.jsonl"):
        with source.open("rb") as inp, gzip.open(str(source) + ".gz", "wb") as out:
            shutil.copyfileobj(inp, out)
        source.unlink()


def verify_source_archive(root: Path, archive: Path) -> int:
    checked = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            path = (root / member.name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source archive contains an unsafe path")
            with bundle.extractfile(member) as stream:
                expected = hashlib.sha256(stream.read()).hexdigest()
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source differs from frozen archive: {member.name}")
            checked += 1
    if checked < 10:
        raise ValueError("Source archive is incomplete")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, default=Path("configs/experiment_specs/controller_audit_pivots.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--model-cache", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--image")
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("run-id must contain only letters, numbers, underscores and hyphens")
    root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else root / args.spec
    spec = json.loads(spec_path.read_text())
    image = args.image or spec["image"]
    cache = args.model_cache.resolve()
    if not cache.is_dir():
        parser.error("model cache does not exist")
    out = root / "sglang_direct_kv/artifacts/results/work_audit" / args.run_id
    assert_idle()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(spec_path, out / "experiment_spec.json")
    paths = ["packages", "configs", "infra", "scripts", "sglang_direct_kv/scripts", "sglang_direct_kv/configs"]
    if args.source_archive:
        if not args.source_revision or len(args.source_revision) != 40:
            parser.error("source-archive requires its full source-revision")
        revision = args.source_revision
        shutil.copy2(args.source_archive, out / "source.tar.gz")
    else:
        revision = command("git", "rev-parse", "HEAD", cwd=root)
        subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(out / "source.tar.gz"), revision, *paths], cwd=root, check=True)
    verified_files = verify_source_archive(root, out / "source.tar.gz")
    metadata = {
        "schema": "controller_audit_manifest.v1", "run_id": args.run_id,
        "started_ns": time.time_ns(), "source_revision": revision,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256((out / "source.tar.gz").read_bytes()).hexdigest(),
        "verified_source_files": verified_files,
        "model": spec["model"], "image": image,
        "image_inspect": json.loads(command("docker", "image", "inspect", image)),
        "gpu": command("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"),
        "scenarios": args.scenarios, "spec": spec, "arms": [],
        "measurement_boundary": "driver workload start through all session completions; includes initial requests and tool waits; excludes server startup and preflight",
        "matching": "Within a trial: identical model, prompt hashes, output limits, waits, capacities and instrumentation. Tool waits start after each session's preceding completion, so absolute arrivals can change as a consequence of policy.",
    }
    write_json(out / "run_manifest.json", metadata)
    py = root / "sglang_direct_kv/.venv/bin/python"
    (out / "host_dependencies.txt").write_text(command(str(py), "-m", "pip", "freeze") + "\n")
    (out / "container_dependencies.txt").write_text(command("docker", "run", "--rm", "--entrypoint", "python3", image, "-m", "pip", "freeze") + "\n")
    model_dir = cache / "hub" / ("models--" + spec["model"].replace("/", "--"))
    if not model_dir.exists():
        model_dir = cache / ("models--" + spec["model"].replace("/", "--"))
    model_files = {}
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
        for path in sorted(model_dir.glob(f"snapshots/*/{filename}")):
            model_files[str(path.relative_to(model_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not model_files:
        raise RuntimeError("Cannot freeze model snapshot identity from model cache")
    write_json(out / "model_identity.json", model_files)
    status = {"state": "running", "completed_arms": 0, "expected_arms": len(args.scenarios) * 4}
    write_json(out / "status.json", status)
    try:
        for scenario in args.scenarios:
            for trial in (1, 2):
                modes = ["no_prefetch", spec["scenarios"][scenario]["mode"]]
                if trial == 2:
                    modes.reverse()
                for mode in modes:
                    assert_idle()
                    if shutil.disk_usage(out).free < 4 * 1024 ** 3:
                        raise RuntimeError("Less than 4 GiB free; preserve evidence and stop")
                    values = arm_environment(root, spec, scenario, trial, mode, args.run_id, cache, image)
                    label = values["REPORT_LABEL"]
                    status.update(current_arm=label)
                    write_json(out / "status.json", status)
                    arm = {"scenario": scenario, "trial": trial, "mode": mode, "label": label, "env": values,
                           "command": ["bash", "infra/container/run_hybrid_reference.sh", spec["model"]], "started_ns": time.time_ns()}
                    metadata["arms"].append(arm)
                    write_json(out / "run_manifest.json", metadata)
                    print(f"START {label}", flush=True)
                    with (out / f"{label}.log").open("w") as log:
                        subprocess.run(arm["command"], cwd=root, env={**clean_environment(), **values},
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    arm["completed_ns"] = time.time_ns()
                    case_root = out / "raw/runs/controlled" / label
                    compress_traces(case_root)
                    status["completed_arms"] += 1
                    write_json(out / "run_manifest.json", metadata)
                    write_json(out / "status.json", status)
                    print(f"DONE {label}", flush=True)
                    time.sleep(3)
        status["state"] = "complete"
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        write_json(out / "status.json", status)
    print(f"Evidence saved: {out}", flush=True)


if __name__ == "__main__":
    main()
