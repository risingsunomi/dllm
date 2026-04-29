from __future__ import annotations

import json
import unittest

from dllm.tools import extract_tool_calls, normalize_tools, wants_tool_calls


class ToolCallTests(unittest.TestCase):
    def test_extract_tagged_tool_call(self) -> None:
        content, calls = extract_tool_calls(
            'before <tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
        )
        self.assertEqual(content, "before")
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["path"], "README.md")

    def test_extract_openai_tool_calls_payload(self) -> None:
        _content, calls = extract_tool_calls(
            '{"tool_calls":[{"function":{"name":"search","arguments":"{\\"q\\":\\"x\\"}"}}]}'
        )
        self.assertEqual(calls[0]["function"]["name"], "search")

    def test_wants_tool_calls_respects_none(self) -> None:
        payload = {"tools": [{"type": "function", "function": {"name": "x"}}]}
        self.assertTrue(wants_tool_calls(payload))
        self.assertFalse(wants_tool_calls({**payload, "tool_choice": "none"}))

    def test_normalize_shorthand_tool(self) -> None:
        tools = normalize_tools([{"name": "lookup", "description": "Lookup things"}])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "lookup")


if __name__ == "__main__":
    unittest.main()
