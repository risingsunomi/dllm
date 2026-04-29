from __future__ import annotations

import unittest

from dllm.config import Settings
from dllm.server import (
    _chat_stream_events,
    _chat_response,
    _models_response,
    _normalize_chat_payload,
    _normalize_completion_payload,
)


class ServerPayloadTests(unittest.TestCase):
    def test_completion_max_tokens_alias(self) -> None:
        payload = _normalize_completion_payload({"prompt": ["a", "b"], "max_tokens": 3})
        self.assertEqual(payload["prompt"], "a\nb")
        self.assertEqual(payload["max_new_tokens"], 3)

    def test_completion_max_completion_tokens_alias(self) -> None:
        payload = _normalize_completion_payload({"prompt": "a", "max_completion_tokens": 4})
        self.assertEqual(payload["max_new_tokens"], 4)

    def test_chat_requires_messages(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_chat_payload({"messages": []})

        payload = _normalize_chat_payload(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
        )
        self.assertEqual(payload["max_new_tokens"], 2)
        self.assertNotIn("prompt", payload)

    def test_chat_normalizes_legacy_functions(self) -> None:
        payload = _normalize_chat_payload(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
                "function_call": {"name": "lookup"},
            }
        )
        self.assertEqual(payload["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(payload["tool_choice"]["function"]["name"], "lookup")

    def test_models_response_advertises_context_length(self) -> None:
        settings = Settings(model_name="test-model", context_length=32768)
        response = _models_response(settings)
        self.assertEqual(response["data"][0]["id"], "test-model")
        self.assertEqual(response["data"][0]["context_length"], 32768)

    def test_chat_stream_events_are_openai_shaped(self) -> None:
        settings = Settings(model_name="test-model")
        engine_events = iter(
            [
                {"event": "token", "text": "hello"},
                {"event": "done", "generated_tokens": 1, "prompt_tokens": 2},
            ]
        )
        events = list(_chat_stream_events(settings, {}, engine_events))
        self.assertEqual(events[0]["object"], "chat.completion.chunk")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_chat_response_tool_calls(self) -> None:
        settings = Settings(model_name="test-model")
        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ]
        }
        result = {"text": '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'}
        response = _chat_response(settings, payload, result)
        message = response["choices"][0]["message"]
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "lookup")

    def test_chat_stream_tool_calls_are_native_deltas(self) -> None:
        settings = Settings(model_name="test-model")
        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ]
        }
        engine_events = iter(
            [
                {"event": "token", "text": '<tool_call>{"name":"lookup",'},
                {"event": "token", "text": '"arguments":{"q":"x"}}</tool_call>'},
                {"event": "done", "generated_tokens": 4, "prompt_tokens": 8},
            ]
        )
        events = list(_chat_stream_events(settings, payload, engine_events))
        self.assertIn("tool_calls", events[1]["choices"][0]["delta"])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
