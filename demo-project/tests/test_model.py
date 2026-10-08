import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import DecisionError
from model import OpenAIRouter


class ModelTests(unittest.TestCase):
    def make_router(self):
        with patch.dict(os.environ, {"DEMO_ALLOW_PAID_CALLS": "1", "OPENAI_API_KEY": "test-not-a-key"}), patch("openai.OpenAI") as client:
            router = OpenAIRouter()
            self.assertEqual(client.call_args.kwargs["max_retries"], 0)
            self.assertEqual(client.call_args.kwargs["timeout"], 20)
        router.client = Mock()
        return router

    def test_paid_calls_disabled_by_default(self):
        with patch.dict(os.environ, {"DEMO_ALLOW_PAID_CALLS": "0"}):
            with self.assertRaises(DecisionError):
                OpenAIRouter()

    def test_request_limits_and_failure_do_not_retry(self):
        router = self.make_router()
        with patch.dict(os.environ, {"DEMO_ALLOW_PAID_CALLS": "1"}):
            router.client.chat.completions.create.side_effect = RuntimeError("private details")
            with self.assertRaisesRegex(DecisionError, "Provider request failed"):
                router({"utterance": "slots"})
            self.assertEqual(router.client.chat.completions.create.call_count, 1)
            router.calls = 20
            with self.assertRaisesRegex(DecisionError, "limit"):
                router({"utterance": "slots"})
            self.assertEqual(router.client.chat.completions.create.call_count, 1)

    def test_valid_response_and_usage(self):
        router = self.make_router()
        router.client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content='{"action":"availability","service":null,"slot_number":null}'
            ))],
        )
        with patch.dict(os.environ, {"DEMO_ALLOW_PAID_CALLS": "1"}):
            self.assertEqual(router({"utterance": "slots"})["action"], "availability")
        self.assertAlmostEqual(router.estimated_cost_usd(), 0.00056)

    def test_oversized_input_is_blocked_before_request(self):
        router = self.make_router()
        with patch.dict(os.environ, {"DEMO_ALLOW_PAID_CALLS": "1"}):
            with self.assertRaisesRegex(DecisionError, "too large"):
                router({"utterance": "x" * 7000})
        router.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()