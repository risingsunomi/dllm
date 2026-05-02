from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .model import TorchTransformersEngine
from .sharding import LayerShard


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PeerResponse:
    ok: bool
    payload: dict[str, Any]


class PeerProtocolError(RuntimeError):
    pass


class PeerClient:
    def __init__(self, *, timeout: float = 300.0, connect_timeout: float = 5.0) -> None:
        self.timeout = timeout
        self.connect_timeout = connect_timeout

    def health(self, host: str, port: int) -> dict[str, Any]:
        return self.send(host, port, {"command": "health", "payload": {}}, timeout=self.connect_timeout)

    def load_model(
        self,
        host: str,
        port: int,
        *,
        model_name: str,
        device: str | None = None,
        dtype: str | None = None,
        offline: bool | None = None,
        trust_remote_code: bool | None = None,
        device_map: str | None = None,
        offload_folder: str | None = None,
        attention_implementation: str | None = None,
        language_only: bool | None = None,
        language_weight_prefix: str | None = None,
        weight_key_mapping: str | None = None,
        shard: LayerShard | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_name": model_name}
        if device:
            payload["device"] = device
        if dtype:
            payload["dtype"] = dtype
        if offline is not None:
            payload["offline"] = offline
        if trust_remote_code is not None:
            payload["trust_remote_code"] = trust_remote_code
        if device_map:
            payload["device_map"] = device_map
        if offload_folder:
            payload["offload_folder"] = offload_folder
        if attention_implementation:
            payload["attention_implementation"] = attention_implementation
        if language_only is not None:
            payload["language_only"] = language_only
        if language_weight_prefix:
            payload["language_weight_prefix"] = language_weight_prefix
        if weight_key_mapping:
            payload["weight_key_mapping"] = weight_key_mapping
        if shard is not None:
            payload["shard"] = shard.as_dict()
        return self.send(
            host,
            port,
            {
                "command": "load_model",
                "payload": payload,
            },
            timeout=self.timeout,
        )

    def forward_shard(self, host: str, port: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.send(host, port, {"command": "forward_shard", "payload": payload}, timeout=self.timeout)

    def unload_model(self, host: str, port: int, *, reason: str = "generation_done") -> dict[str, Any]:
        return self.send(
            host,
            port,
            {"command": "unload_model", "payload": {"reason": reason}},
            timeout=self.timeout,
        )

    def send(self, host: str, port: int, message: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        if len(body) > _MAX_MESSAGE_BYTES:
            raise PeerProtocolError("request is too large")

        socket_timeout = self.timeout if timeout is None else timeout
        with socket.create_connection((host, int(port)), timeout=self.connect_timeout) as sock:
            sock.settimeout(socket_timeout)
            sock.sendall(_HEADER.pack(len(body)))
            sock.sendall(body)
            response = _read_message(sock)

        decoded = json.loads(response.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise PeerProtocolError("peer returned a non-object response")
        if decoded.get("ok") is False:
            error = decoded.get("error") or decoded.get("payload", {}).get("error") or "peer request failed"
            raise RuntimeError(str(error))
        return decoded


class InferenceWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = TorchTransformersEngine(
            model_name=settings.model_name,
            device=settings.device,
            dtype=settings.dtype,
            offline=settings.offline,
            trust_remote_code=settings.trust_remote_code,
            device_map=settings.device_map,
            offload_folder=settings.offload_folder,
            attention_implementation=settings.attention_implementation,
            language_only=settings.language_only,
            language_weight_prefix=settings.language_weight_prefix,
            weight_key_mapping=settings.weight_key_mapping,
        )
        self._engine_lock = threading.RLock()
        self._server: _ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self.started_at = time.time()

    def serve_forever(self) -> None:
        server = self._build_server()
        self._server = server
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            server.server_close()

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._server = self._build_server()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"dllm-worker-{self.settings.node_name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        command = str(message.get("command", "")).strip()
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if command == "health":
            return {"ok": True, "payload": self.health()}
        if command == "load_model":
            # The wire command name is kept for compatibility. In shard-first
            # mode it configures the runtime and assigned layer range only; the
            # actual weights load lazily when forward_shard is called.
            return self._handle_load_model(payload)
        if command == "forward_shard":
            return self._handle_forward_shard(payload)
        if command == "unload_model":
            return self._handle_unload_model(payload)
        if command == "shutdown":
            threading.Thread(target=self.stop, daemon=True).start()
            return {"ok": True, "payload": {"stopping": True}}
        return {"ok": False, "error": f"unknown command {command!r}"}

    def health(self) -> dict[str, Any]:
        with self._engine_lock:
            engine_health = self.engine.health()
        return {
            "node_name": self.settings.node_name,
            "role": "worker",
            "uptime_seconds": time.time() - self.started_at,
            "host": self.settings.peer_host,
            "port": self.settings.peer_port,
            "device_info": engine_health.get("device_info"),
            "engine": engine_health,
        }

    def _handle_load_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._engine_lock:
            requested_model = str(payload.get("model_name") or self.engine.model_name).strip()
            requested_device = str(payload.get("device") or self.engine.requested_device).strip()
            requested_dtype = str(payload.get("dtype") or self.engine.dtype_name).strip()
            requested_offline = bool(payload.get("offline", self.engine.offline))
            requested_trust = bool(payload.get("trust_remote_code", self.engine.trust_remote_code))
            requested_device_map = str(payload.get("device_map") or self.engine.device_map_name).strip()
            requested_offload_folder = str(payload.get("offload_folder") or self.engine.offload_folder).strip()
            requested_attention = str(
                payload.get("attention_implementation") or self.engine.attention_implementation
            ).strip()
            requested_language_only = bool(payload.get("language_only", self.engine.language_only))
            requested_language_prefix = str(
                payload.get("language_weight_prefix") or self.engine.language_weight_prefix
            ).strip()
            requested_key_mapping = str(payload.get("weight_key_mapping") or self.engine.weight_key_mapping).strip()
            requested_shard = LayerShard.from_mapping(payload.get("shard"))

            same_runtime = (
                requested_model == self.engine.model_name
                and requested_device == self.engine.requested_device
                and requested_dtype == self.engine.dtype_name
                and requested_offline == self.engine.offline
                and requested_trust == self.engine.trust_remote_code
                and requested_device_map == self.engine.device_map_name
                and requested_offload_folder == self.engine.offload_folder
                and requested_attention == self.engine.attention_implementation
                and requested_language_only == self.engine.language_only
                and requested_language_prefix == self.engine.language_weight_prefix
                and requested_key_mapping == self.engine.weight_key_mapping
                and _shard_dict(requested_shard) == _shard_dict(self.engine.shard)
            )
            if not same_runtime:
                self.engine.unload()
                self.engine = TorchTransformersEngine(
                    model_name=requested_model,
                    device=requested_device,
                    dtype=requested_dtype,
                    offline=requested_offline,
                    trust_remote_code=requested_trust,
                    device_map=requested_device_map,
                    offload_folder=requested_offload_folder,
                    attention_implementation=requested_attention,
                    language_only=requested_language_only,
                    language_weight_prefix=requested_language_prefix,
                    weight_key_mapping=requested_key_mapping,
                    shard=requested_shard,
                )

            start = time.perf_counter()
            if same_runtime and requested_shard is not None:
                self.engine.set_shard(requested_shard)
            elapsed = time.perf_counter() - start
            return {
                "ok": True,
                "payload": {
                    "configured": True,
                    "loaded": self.engine.loaded,
                    "elapsed_seconds": elapsed,
                    "engine": self.engine.health(),
                    "node_name": self.settings.node_name,
                },
            }

    def _handle_forward_shard(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._engine_lock:
            result = self.engine.forward_shard(payload)
        return {
            "ok": True,
            "payload": {
                **result,
                "node_name": self.settings.node_name,
            },
        }

    def _handle_unload_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        reason = str(payload.get("reason") or "generation_done")
        with self._engine_lock:
            start = time.perf_counter()
            self.engine.unload()
            elapsed = time.perf_counter() - start
            engine_health = self.engine.health()
        return {
            "ok": True,
            "payload": {
                "unloaded": True,
                "reason": reason,
                "elapsed_seconds": elapsed,
                "engine": engine_health,
                "node_name": self.settings.node_name,
            },
        }

    def _build_server(self) -> "_ThreadingTCPServer":
        worker = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                try:
                    raw = _read_message(self.request)
                    message = json.loads(raw.decode("utf-8"))
                    if not isinstance(message, dict):
                        raise PeerProtocolError("request must be a JSON object")
                    response = worker.dispatch(message)
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                _write_message(self.request, json.dumps(response, separators=(",", ":")).encode("utf-8"))

        server = _ThreadingTCPServer((self.settings.peer_host, self.settings.peer_port), Handler)
        server.worker = self  # type: ignore[attr-defined]
        return server


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _payload_with_prompt(
    engine: TorchTransformersEngine,
    payload: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    merged = {**defaults, **payload}
    if "prompt" in merged and str(merged.get("prompt", "")):
        return merged

    messages = merged.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            normalized.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": _message_content_to_text(message.get("content", "")),
                }
            )
        tool_choice = merged.get("tool_choice")
        tools = None if isinstance(tool_choice, str) and tool_choice.lower() == "none" else merged.get("tools")
        merged["prompt"] = engine.format_chat_prompt(
            normalized,
            tools=tools,
            tool_choice=tool_choice,
        )
        return merged

    merged["prompt"] = ""
    return merged


def _read_message(sock: socket.socket) -> bytes:
    header = _read_exact(sock, _HEADER.size)
    if len(header) != _HEADER.size:
        raise PeerProtocolError("missing message header")
    size = _HEADER.unpack(header)[0]
    if size > _MAX_MESSAGE_BYTES:
        raise PeerProtocolError("message is too large")
    return _read_exact(sock, size)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip()
            if item_type in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        return json.dumps(content, separators=(",", ":"), default=str)
    return str(content)


def _shard_dict(shard: LayerShard | None) -> dict[str, Any] | None:
    return shard.as_dict() if shard is not None else None


def _write_message(sock: socket.socket, body: bytes) -> None:
    if len(body) > _MAX_MESSAGE_BYTES:
        raise PeerProtocolError("response is too large")
    sock.sendall(_HEADER.pack(len(body)))
    sock.sendall(body)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(65536, remaining))
        if not chunk:
            raise PeerProtocolError("socket closed while reading message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
