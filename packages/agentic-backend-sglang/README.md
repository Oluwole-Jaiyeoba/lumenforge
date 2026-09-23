# agentic-backend-sglang (`agentic_backends.sglang`)

The only package in this repository that knows SGLang internals: module paths,
class/method/attribute names, CLI flags and request wire fields. Everything
else (controller, harnesses, gateway translation, reports) talks to it through
`agentic_backend_api` protocols and `agentic_core` records.

- Design, decisions and hand-off notes: [`docs/architecture/README_RESTRUCTURING.md`](../../docs/architecture/README_RESTRUCTURING.md)
- Per-release compatibility: [`COMPATIBILITY.md`](COMPATIBILITY.md)

## Upgrading SGLang (checklist)

1. `pip download "sglang==NEW" --no-deps` and unzip the wheel.
2. `python -m agentic_backends.sglang.surface --sglang-src <dir>`.
3. If required elements are missing, add `versions/vNNNN.py` derived from the
   newest adapter with `replace_hook_targets(...)` (keep **event names**
   unchanged), register it in `versions/__init__.py`, set `version_range`
   and `verification="static"`.
4. `bash scripts/check_portability.sh` (adds nothing SGLang-specific anywhere else).
5. Change the pin in `sglang_direct_kv/requirements.txt`, run one reference
   experiment on a GPU, then set `verification="runtime"` and add the version
   to `tested_versions`.
6. Add the version to the CI matrix in `.github/workflows/portability.yml`
   and regenerate `COMPATIBILITY.md`.
