from __future__ import annotations

import json
import re
import uuid
from typing import Any


def wants_tool_calls(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, str) and tool_choice.strip().lower() == "none":
        return False
    return True


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = dict(tool["function"])
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            function.setdefault("description", "")
            function.setdefault("parameters", {"type": "object", "properties": {}})
            normalized.append({"type": "function", "function": function})
            continue

        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return normalized


def tool_system_prompt(tools: list[dict[str, Any]], tool_choice: Any = None) -> str:
    if not tools:
        return ""

    lines = [
        "You have access to tools. When a tool is needed, respond with one or more tool calls and no extra prose.",
        "Use this exact format for each call:",
        '<tool_call>{"name":"tool_name","arguments":{"key":"value"}}</tool_call>',
        "Available tools:",
    ]
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if not isinstance(function, dict):
            continue
        spec = {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {"type": "object", "properties": {}}),
        }
        lines.append(json.dumps(spec, separators=(",", ":"), ensure_ascii=False))

    forced_name = _forced_tool_name(tool_choice)
    if forced_name:
        lines.append(f"You must call the `{forced_name}` tool if a tool call is possible.")
    return "\n".join(lines)


def extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    source = str(text or "")
    tagged_calls, without_tags = _extract_tagged_tool_calls(source)
    calls = [_coerce_tool_call(call) for call in tagged_calls]
    calls = [call for call in calls if call is not None]
    if calls:
        return without_tags.strip(), calls

    parsed = _loads_json_fragment(source.strip())
    calls = _coerce_tool_calls_payload(parsed)
    if calls:
        return "", calls

    fenced = _extract_first_fenced_json(source)
    if fenced is not None:
        calls = _coerce_tool_calls_payload(_loads_json_fragment(fenced))
        if calls:
            return source.replace(fenced, "").strip("`\n "), calls

    return source, []


def _extract_tagged_tool_calls(text: str) -> tuple[list[Any], str]:
    calls: list[Any] = []

    def replace_call(match: re.Match[str]) -> str:
        payload = match.group(1).strip()
        parsed = _loads_json_fragment(payload)
        if parsed is not None:
            calls.append(parsed)
        return ""

    without_single = re.sub(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        replace_call,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_calls(match: re.Match[str]) -> str:
        payload = match.group(1).strip()
        parsed = _loads_json_fragment(payload)
        if isinstance(parsed, list):
            calls.extend(parsed)
        elif parsed is not None:
            calls.append(parsed)
        return ""

    without_group = re.sub(
        r"<tool_calls>\s*(.*?)\s*</tool_calls>",
        replace_calls,
        without_single,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return calls, without_group


def _coerce_tool_calls_payload(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        calls = [_coerce_tool_call(item) for item in payload]
        return [call for call in calls if call is not None]
    if isinstance(payload, dict):
        if isinstance(payload.get("tool_calls"), list):
            calls = [_coerce_tool_call(item) for item in payload["tool_calls"]]
            return [call for call in calls if call is not None]
        call = _coerce_tool_call(payload)
        return [call] if call is not None else []
    return []


def _coerce_tool_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    if isinstance(value.get("function"), dict):
        function = value["function"]
        name = str(function.get("name", value.get("name", ""))).strip()
        arguments = function.get("arguments", value.get("arguments", {}))
    else:
        name = str(
            value.get("name")
            or value.get("tool_name")
            or value.get("function_name")
            or value.get("tool")
            or ""
        ).strip()
        arguments = (
            value.get("arguments")
            if "arguments" in value
            else value.get("parameters", value.get("input", {}))
        )

    if not name:
        return None

    if isinstance(arguments, str):
        argument_text = arguments
    else:
        argument_text = json.dumps(arguments or {}, separators=(",", ":"), ensure_ascii=False)

    return {
        "id": str(value.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": argument_text,
        },
    }


def _loads_json_fragment(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    start_candidates = [idx for idx in (value.find("{"), value.find("[")) if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(value), start, -1):
        fragment = value[start:end].strip()
        if not fragment:
            continue
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            continue
    return None


def _extract_first_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _forced_tool_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, dict):
        return ""
    function = tool_choice.get("function")
    if isinstance(function, dict):
        return str(function.get("name", "")).strip()
    return ""
