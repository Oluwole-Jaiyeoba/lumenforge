import unittest

from agentic_backend_api import (
    BackendActionResult,
    BackendCapabilities,
    BackendRuntimeInfo,
    CompatibilityFinding,
    CompatibilityReport,
    EffectLevel,
)


class ContractsTest(unittest.TestCase):
    def test_effect_level_is_omitted_when_unset(self) -> None:
        legacy = BackendActionResult(command_id="c", accepted=True, acted=False, reason="r")
        self.assertNotIn("effect_level", legacy.to_dict())
        tagged = BackendActionResult("c", True, True, "r", effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY)
        self.assertEqual(tagged.to_dict()["effect_level"], "lowered_at_request_boundary")

    def test_compatibility_report(self) -> None:
        report = CompatibilityReport(
            backend_name="x",
            backend_version="1",
            adapter="a",
            findings=(
                CompatibilityFinding("m:a", "feat.a", True, False),
                CompatibilityFinding("m:b", "feat.b", False, False),
                CompatibilityFinding("m:c", "feat.a", True, True),
            ),
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.broken_features(), ("feat.a",))
        self.assertEqual(report.to_dict()["checked"], 3)

    def test_runtime_info_serializes_normalized_capabilities(self) -> None:
        runtime = BackendRuntimeInfo(
            runtime_profile="nvidia-gh200",
            backend_name="sglang",
            backend_version="0.5.10.post1",
            adapter="v0510",
            probe_ok=True,
            healthy=True,
            health_status="healthy",
            container_image_identity="local-image-id:sha256:abc",
            capabilities=BackendCapabilities(priority_queue=True, backend_name="sglang"),
        )
        row = runtime.to_dict()
        self.assertEqual(row["schema_version"], "agentic_backend_runtime.v1")
        self.assertTrue(row["probe_ok"])
        self.assertEqual(row["container_image_identity"], "local-image-id:sha256:abc")
        self.assertTrue(row["capabilities"]["priority_queue"])


if __name__ == "__main__":
    unittest.main()
