import unittest

from agentic_core import BackendRequest
from agentic_gateway import resolve_backend_priority, translate_request


class TranslationTest(unittest.TestCase):
    def test_replay_gets_high_priority_in_e2e_mode(self) -> None:
        self.assertEqual(resolve_backend_priority({"mode": "e2e_priority_hints", "phase": "replay"}), 100)
        self.assertEqual(resolve_backend_priority({"mode": "e2e_priority_hints", "phase": "pressure_filler"}), -100)
        self.assertIsNone(resolve_backend_priority({"mode": "no_prefetch", "phase": "replay"}))

    def test_controller_priority_is_taken_from_meta(self) -> None:
        meta = {"mode": "controller_full", "phase": "replay", "controller_sglang_priority": 321}
        self.assertEqual(resolve_backend_priority(meta), 321)

    def test_translate_request_is_backend_neutral(self) -> None:
        request = translate_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"mode": "e2e_priority_hints", "phase": "replay", "session_id": "s", "label": "s_replay"},
            "openai_chat",
        )
        self.assertIsInstance(request, BackendRequest)
        self.assertEqual(request.prompt_text, "hello")
        self.assertEqual(request.priority, 100)
        self.assertEqual(request.request_context["request_id"], "s_replay")
        self.assertIsNone(request.native_cache_bridge)


if __name__ == "__main__":
    unittest.main()
