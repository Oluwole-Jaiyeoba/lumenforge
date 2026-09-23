import json
import unittest

from agentic_core import BackendRequest


class BackendRequestTest(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        request = BackendRequest(prompt_text="p", max_tokens=3, priority=None, agent_hints={"a": 1})
        row = json.loads(json.dumps(request.to_dict()))
        self.assertEqual(row["schema_version"], "agentic_backend_request.v1")
        self.assertIsNone(row["priority"])
        self.assertEqual(row["agent_hints"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
