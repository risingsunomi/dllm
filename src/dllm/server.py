from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .config import Settings
from .distributed import DistributedInferenceEngine
from .tools import extract_tool_calls, wants_tool_calls


_MAX_HTTP_BODY = 64 * 1024 * 1024


def run_http_server(settings: Settings, engine: DistributedInferenceEngine) -> None:
    handler_cls = _make_handler(settings, engine)
    server = ThreadingHTTPServer((settings.host, settings.port), handler_cls)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _make_handler(settings: Settings, engine: DistributedInferenceEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"dllm/{__version__}"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                query = parse_qs(parsed.query)
                probe = str(query.get("probe", ["false"])[0]).lower() in {"1", "true", "yes"}
                self._send_json(HTTPStatus.OK, engine.health(probe_peers=probe))
                return
            if parsed.path in {"/models", "/v1/models"}:
                self._send_json(HTTPStatus.OK, _models_response(settings))
                return
            if parsed.path == "/peers":
                self._send_json(HTTPStatus.OK, {"peers": engine.refresh_peer_health()})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/generate":
                    result = engine.generate(_normalize_generation_payload(payload))
                    self._send_json(HTTPStatus.OK, result)
                    return
                if parsed.path == "/v1/completions":
                    normalized = _normalize_completion_payload(payload)
                    if _is_streaming_request(payload):
                        self._send_sse(_completion_stream_events(settings, engine.stream(normalized)))
                    else:
                        result = engine.generate(normalized)
                        self._send_json(HTTPStatus.OK, _completion_response(settings, result))
                    return
                if parsed.path == "/v1/chat/completions":
                    normalized = _normalize_chat_payload(payload)
                    if _is_streaming_request(payload):
                        self._send_sse(_chat_stream_events(settings, normalized, engine.stream(normalized)))
                    else:
                        result = engine.generate(normalized)
                        self._send_json(HTTPStatus.OK, _chat_response(settings, normalized, result))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, _error_response(str(exc), "invalid_request_error"))
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_response(str(exc), "server_error"))

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0") or "0")
            if length <= 0:
                return {}
            if length > _MAX_HTTP_BODY:
                raise ValueError("request body is too large")
            raw = self.rfile.read(length)
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValueError("request body must be a JSON object")
            return decoded

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
            self.send_response(int(status))
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, events: Any) -> None:
            self.send_response(int(HTTPStatus.OK))
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.end_headers()
            self.close_connection = True
            try:
                for event in events:
                    data = json.dumps(event, separators=(",", ":"), default=str)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except Exception as exc:
                data = json.dumps(_error_response(str(exc), "server_error"), separators=(",", ":"))
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def _normalize_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "max_tokens" in normalized and "max_new_tokens" not in normalized:
        normalized["max_new_tokens"] = normalized["max_tokens"]
    if "max_completion_tokens" in normalized and "max_new_tokens" not in normalized:
        normalized["max_new_tokens"] = normalized["max_completion_tokens"]
    return normalized


def _normalize_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_generation_payload(payload)
    prompt = normalized.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(item) for item in prompt)
    normalized["prompt"] = str(prompt)
    return normalized


def _normalize_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_generation_payload(payload)
    messages = normalized.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized["messages"] = messages
    if "tools" not in normalized and isinstance(normalized.get("functions"), list):
        normalized["tools"] = [
            {"type": "function", "function": function}
            for function in normalized["functions"]
            if isinstance(function, dict)
        ]
    if "tool_choice" not in normalized and "function_call" in normalized:
        normalized["tool_choice"] = _function_call_to_tool_choice(normalized.get("function_call"))
    normalized.pop("prompt", None)
    return normalized


def _function_call_to_tool_choice(value: Any) -> Any:
    if isinstance(value, str):
        if value in {"none", "auto", "required"}:
            return value
        return {"type": "function", "function": {"name": value}}
    if isinstance(value, dict):
        name = str(value.get("name", "")).strip()
        if name:
            return {"type": "function", "function": {"name": name}}
    return value


def _models_response(settings: Settings) -> dict[str, Any]:
    model = {
        "id": settings.model_name,
        "object": "model",
        "created": 0,
        "owned_by": "dllm",
    }
    if settings.context_length > 0:
        model["context_length"] = settings.context_length
        model["max_context_length"] = settings.context_length
    return {
        "object": "list",
        "data": [model],
    }


def _is_streaming_request(payload: dict[str, Any]) -> bool:
    return bool(payload.get("stream", False))


def _completion_response(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": settings.model_name,
        "choices": [
            {
                "index": 0,
                "text": result.get("text", ""),
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("generated_tokens", 0),
            "total_tokens": int(result.get("prompt_tokens", 0) or 0)
            + int(result.get("generated_tokens", 0) or 0),
        },
        "dllm": result,
    }


def _chat_response(settings: Settings, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    content = str(result.get("text", ""))
    tool_calls: list[dict[str, Any]] = []
    if wants_tool_calls(payload):
        content, tool_calls = extract_tool_calls(content)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["content"] = content or None
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("generated_tokens", 0),
            "total_tokens": int(result.get("prompt_tokens", 0) or 0)
            + int(result.get("generated_tokens", 0) or 0),
        },
        "dllm": result,
    }


def _completion_stream_events(settings: Settings, events: Any):
    now = time.time()
    created = int(now)
    completion_id = f"cmpl-{int(now * 1000)}"
    final: dict[str, Any] = {}
    for event in events:
        if event.get("event") == "token":
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": settings.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": event.get("text", ""),
                        "finish_reason": None,
                    }
                ],
            }
        elif event.get("event") == "done":
            final = event
    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": settings.model_name,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(final),
    }


def _chat_stream_events(settings: Settings, payload: dict[str, Any], events: Any):
    now = time.time()
    created = int(now)
    completion_id = f"chatcmpl-{int(now * 1000)}"
    tools_requested = wants_tool_calls(payload)
    final: dict[str, Any] = {}

    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": settings.model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }

    if tools_requested:
        buffered: list[str] = []
        for event in events:
            if event.get("event") == "token":
                buffered.append(str(event.get("text", "")))
            elif event.get("event") == "done":
                final = event
        content, tool_calls = extract_tool_calls("".join(buffered))
        if tool_calls:
            for index, tool_call in enumerate(tool_calls):
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [_tool_call_delta(index, tool_call)]},
                            "finish_reason": None,
                        }
                    ],
                }
            yield _chat_stream_final_chunk(
                completion_id,
                created,
                settings.model_name,
                "tool_calls",
                final,
            )
            return
        if content:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
        yield _chat_stream_final_chunk(completion_id, created, settings.model_name, "stop", final)
        return

    for event in events:
        if event.get("event") == "token":
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event.get("text", "")},
                        "finish_reason": None,
                    }
                ],
            }
        elif event.get("event") == "done":
            final = event
    yield _chat_stream_final_chunk(completion_id, created, settings.model_name, "stop", final)


def _chat_stream_final_chunk(
    completion_id: str,
    created: int,
    model_name: str,
    finish_reason: str,
    final: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(final),
    }


def _tool_call_delta(index: int, tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    return {
        "index": index,
        "id": str(tool_call.get("id", "")),
        "type": "function",
        "function": {
            "name": str(function.get("name", "")),
            "arguments": str(function.get("arguments", "")),
        },
    }


def _usage(result: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
    completion_tokens = int(result.get("generated_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _error_response(message: str, error_type: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }
