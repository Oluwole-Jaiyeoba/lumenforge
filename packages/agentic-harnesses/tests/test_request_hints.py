import unittest

from agentic_harnesses import request_hints
from agentic_harnesses.emission_emulation import outbound_priority_fields


class RequestHintsTest(unittest.TestCase):
    def test_priority_signal_sources(self) -> None:
        payload = {"service_tier": "priority", "nvext": {"agent_hints": {"priority": 100}}}
        signal = request_hints.emitted_priority_signal(payload)
        self.assertIn("service_tier=priority", signal["harness_emit_priority_signal"])
        self.assertIn("nvext.agent_hints", signal["harness_emit_priority_signal_source"])
        self.assertTrue(request_hints.emitted_signal_is_urgent(payload))

    def test_cache_signal_in_body_and_headers(self) -> None:
        signal = request_hints.emitted_cache_signal({"prompt_cache_key": "k"}, {"X-Session-Affinity": "a"})
        self.assertEqual(signal["harness_native_cache_signal_seen"], "yes")
        self.assertIn("@headers.x-session-affinity", signal["harness_native_cache_identity_source"])

    def test_emulated_urgent_openai_hint(self) -> None:
        fields = outbound_priority_fields({"mode": "pre_harness_priority_hints", "priority_intent": {"class": "urgent"}}, "openai_chat")
        self.assertEqual(fields["service_tier"], "priority")
        self.assertEqual(fields["extra_body"]["agentic_hints"]["priority_class"], "urgent")


if __name__ == "__main__":
    unittest.main()
