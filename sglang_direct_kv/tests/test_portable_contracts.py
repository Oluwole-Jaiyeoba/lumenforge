from __future__ import annotations

import re
import unittest
from pathlib import Path

from agentic_backend_api import BackendActionResult, BackendAdapter, BackendCapabilities
from agentic_controller import ControllerPolicy as PortableControllerPolicy
from agentic_core import (
    AttachmentLevel,
    BackendCommand,
    BackendObservation,
    ControllerCommand,
    ControllerDecision,
    ControllerEvent,
    EventType,
    HardwareTelemetry,
    HarnessSignal,
    KVMemoryTier,
    KVResidency,
    RequestEnvelope,
    RunManifest,
    SchedulerAction,
    SchedulerState,
    SessionPhase,
    SignalProvenance,
)
from agentic_harnesses import HarnessControllerSignal as PortableHarnessControllerSignal
from agentic_harnesses.hint_benchmark import evidence_tier_for_mode as portable_evidence_tier_for_mode
from agentic_kv.controller import (
    BackendActionResult as LegacyBackendActionResult,
    BackendCapabilities as LegacyBackendCapabilities,
    ControllerCommand as LegacyControllerCommand,
    ControllerEvent as LegacyControllerEvent,
    ControllerPolicy as LegacyControllerPolicy,
    HarnessControllerSignal as LegacyHarnessControllerSignal,
)
from agentic_kv.hint_benchmark import evidence_tier_for_mode as legacy_evidence_tier_for_mode


class FakeBackend:
    def __init__(self) -> None:
        self.commands: list[BackendCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(priority_queue=True, backend_name="fake")

    def apply(self, command: BackendCommand) -> BackendActionResult:
        self.commands.append(command)
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=True,
            reason="fake backend accepted command",
            backend_name="fake",
        )


class PortableContractTests(unittest.TestCase):
    def test_legacy_controller_imports_reexport_new_contracts(self) -> None:
        self.assertIs(LegacyControllerCommand, ControllerCommand)
        self.assertIs(LegacyControllerEvent, ControllerEvent)
        self.assertIs(LegacyBackendCapabilities, BackendCapabilities)
        self.assertIs(LegacyBackendActionResult, BackendActionResult)
        self.assertIs(LegacyControllerPolicy, PortableControllerPolicy)
        self.assertIs(LegacyHarnessControllerSignal, PortableHarnessControllerSignal)
        self.assertIs(legacy_evidence_tier_for_mode, portable_evidence_tier_for_mode)
        self.assertEqual(BackendCapabilities().schema_version, "agentic_backend_api.v1")
        self.assertNotIn("schema_version", BackendCapabilities().to_dict())

    def test_controller_serialization_remains_compatible(self) -> None:
        event = ControllerEvent(
            event_id="event-1",
            event=EventType.TOOL_STARTED,
            session_id="session-1",
            prefix_id="prefix-1",
            monotonic_ms=100,
        )
        command = ControllerCommand(
            command_id="command-1",
            session_id="session-1",
            prefix_id="prefix-1",
            session_generation=0,
            scheduler_action=SchedulerAction.SET_PRIORITY,
            priority=100,
        )
        decision = ControllerDecision(
            decision_id="decision-1",
            session_id="session-1",
            phase=SessionPhase.READY,
            commands=(command,),
            reason="deadline ready",
        )

        self.assertEqual(event.to_dict()["event"], "tool_started")
        self.assertEqual(command.to_dict()["scheduler_action"], "set_priority")
        self.assertEqual(decision.to_dict()["commands"][0]["priority"], 100)
        self.assertEqual(event.schema_version, "agentic_controller.v1")

    def test_normalized_contracts_are_json_ready(self) -> None:
        signal = HarnessSignal(
            signal_id="signal-1",
            name="priority",
            harness="nemo_agent_toolkit",
            observed_at_ms=100,
            value="high",
            attachment_level=AttachmentLevel.REQUEST,
            provenance=SignalProvenance.NATIVE_HARNESS,
            session_id="session-1",
            request_id="request-1",
        )
        request = RequestEnvelope(
            request_id="request-1",
            session_id="session-1",
            created_at_ms=90,
            ready_at_ms=200,
            deadline_at_ms=250,
        )
        observation = BackendObservation(
            observation_id="observation-1",
            observed_at_ms=110,
            backend_name="fake",
            kind="queue_state",
            payload={"waiting": ("request-1",)},
        )
        scheduler = SchedulerState(
            observed_at_ms=110,
            backend_name="fake",
            waiting_request_ids=("request-1",),
        )
        residency = KVResidency(
            prefix_id="prefix-1",
            observed_at_ms=110,
            tier=KVMemoryTier.HOST,
        )
        telemetry = HardwareTelemetry(
            observed_at_ms=110,
            device_id="gpu-0",
            gpu_utilization_pct=75.0,
        )
        manifest = RunManifest(
            run_id="run-1",
            created_at_ms=1,
            experiment="portable-contract-test",
            enabled_instrumentation=("controller_ledger",),
        )

        self.assertEqual(signal.to_dict()["attachment_level"], "request")
        self.assertEqual(signal.to_dict()["provenance"], "native_harness")
        self.assertEqual(request.to_dict()["deadline_at_ms"], 250)
        self.assertEqual(observation.to_dict()["payload"]["waiting"], ["request-1"])
        self.assertEqual(scheduler.to_dict()["waiting_request_ids"], ["request-1"])
        self.assertEqual(residency.to_dict()["tier"], "host")
        self.assertEqual(telemetry.to_dict()["gpu_utilization_pct"], 75.0)
        self.assertEqual(manifest.to_dict()["enabled_instrumentation"], ["controller_ledger"])
        self.assertEqual(manifest.to_dict()["schema_version"], "agentic_run_manifest.v2")
        self.assertEqual(manifest.to_dict()["completion_status"], "created")

    def test_fake_backend_satisfies_backend_protocol(self) -> None:
        backend = FakeBackend()
        command = BackendCommand(
            command_id="command-1",
            session_id="session-1",
            prefix_id="prefix-1",
            session_generation=0,
        )

        self.assertIsInstance(backend, BackendAdapter)
        self.assertTrue(backend.apply(command).acted)
        self.assertEqual(backend.commands, [command])

    def test_portable_packages_do_not_import_sglang(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        import_pattern = re.compile(r"^\s*(?:from|import)\s+sglang(?:\.|\s|$)", re.MULTILINE)
        offenders: list[str] = []
        for package_name in ("agentic_core", "agentic_backend_api"):
            for path in (source_root / package_name).rglob("*.py"):
                if import_pattern.search(path.read_text(encoding="utf-8")):
                    offenders.append(str(path.relative_to(source_root)))
        self.assertEqual(offenders, [])

    def test_controller_and_harness_packages_are_independent(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        forbidden = {
            "agentic_controller": ("agentic_harnesses", "agentic_kv", "sglang"),
            "agentic_harnesses": ("agentic_controller", "agentic_kv", "sglang"),
        }
        offenders: list[str] = []
        for package_name, forbidden_roots in forbidden.items():
            pattern = re.compile(
                rf"^\s*(?:from|import)\s+(?:{'|'.join(forbidden_roots)})(?:\.|\s|$)",
                re.MULTILINE,
            )
            for path in (source_root / package_name).rglob("*.py"):
                if pattern.search(path.read_text(encoding="utf-8")):
                    offenders.append(str(path.relative_to(source_root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
