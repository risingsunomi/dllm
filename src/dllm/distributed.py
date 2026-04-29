from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import PeerSpec, Settings
from .model import TorchTransformersEngine, request_from_payload
from .peer import PeerClient, _payload_with_prompt


@dataclass
class PeerState:
    spec: PeerSpec
    healthy: bool = False
    loaded: bool = False
    last_error: str = ""
    last_seen: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "healthy": self.healthy,
            "loaded": self.loaded,
            "last_error": self.last_error,
            "last_seen": self.last_seen,
        }


class DistributedInferenceEngine:
    """Coordinator for local or multi-node inference.

    Multiple nodes are used as distributed serving workers. Each request is sent
    to one available worker, while concurrent requests can be spread across the
    configured peer set by the coordinator.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        local_engine: TorchTransformersEngine | None = None,
    ) -> None:
        self.settings = settings
        self.local_engine = local_engine
        if self.local_engine is None and settings.load_local:
            self.local_engine = TorchTransformersEngine(
                model_name=settings.model_name,
                device=settings.device,
                dtype=settings.dtype,
                offline=settings.offline,
                trust_remote_code=settings.trust_remote_code,
                device_map=settings.device_map,
                max_memory=settings.max_memory,
                offload_folder=settings.offload_folder,
                attention_implementation=settings.attention_implementation,
                language_only=settings.language_only,
                language_weight_prefix=settings.language_weight_prefix,
                weight_key_mapping=settings.weight_key_mapping,
            )
        self.peer_client = PeerClient(
            timeout=settings.request_timeout,
            connect_timeout=settings.peer_connect_timeout,
        )
        self.peers = [PeerState(spec=peer) for peer in settings.peers]
        self._rr = 0
        self._lock = threading.RLock()
        self._ready = False

    def ensure_ready(self) -> None:
        with self._lock:
            if self._ready:
                return
            errors: list[str] = []
            if self.local_engine is not None:
                self.local_engine.load()
            for peer in self.peers:
                try:
                    self._load_peer(peer)
                except Exception as exc:
                    errors.append(f"{peer.spec.name}: {exc}")
            if self.local_engine is None and not any(peer.loaded for peer in self.peers):
                detail = "; ".join(errors) if errors else "no local or remote targets configured"
                raise RuntimeError(f"no inference targets are ready: {detail}")
            self._ready = True

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_ready()
        errors: list[str] = []
        for target in self._target_order():
            if target == "local":
                try:
                    return self._generate_local(payload)
                except Exception as exc:
                    errors.append(f"local: {exc}")
                    continue

            assert isinstance(target, PeerState)
            try:
                response = self.peer_client.generate(target.spec.host, target.spec.port, payload)
                target.healthy = True
                target.loaded = True
                target.last_error = ""
                target.last_seen = time.time()
                result = dict(response.get("payload", {}))
                result["target"] = target.spec.name
                result["distributed"] = len(self.peers) > 0
                return result
            except Exception as exc:
                target.healthy = False
                target.last_error = str(exc)
                errors.append(f"{target.spec.name}: {exc}")

        detail = "; ".join(errors) if errors else "no generation targets are configured"
        raise RuntimeError(f"inference failed: {detail}")

    def stream(self, payload: dict[str, Any]):
        self.ensure_ready()
        errors: list[str] = []
        for target in self._target_order():
            if target == "local":
                try:
                    yield from self._stream_local(payload)
                    return
                except Exception as exc:
                    errors.append(f"local: {exc}")
                    continue

            assert isinstance(target, PeerState)
            try:
                for event in self.peer_client.generate_stream(target.spec.host, target.spec.port, payload):
                    event_out = dict(event)
                    if event_out.get("event") == "done":
                        target.healthy = True
                        target.loaded = True
                        target.last_error = ""
                        target.last_seen = time.time()
                        event_out["target"] = target.spec.name
                        event_out["distributed"] = len(self.peers) > 0
                    yield event_out
                return
            except Exception as exc:
                target.healthy = False
                target.last_error = str(exc)
                errors.append(f"{target.spec.name}: {exc}")

        detail = "; ".join(errors) if errors else "no generation targets are configured"
        raise RuntimeError(f"inference stream failed: {detail}")

    def health(self, *, probe_peers: bool = False) -> dict[str, Any]:
        if probe_peers:
            self.refresh_peer_health()
        return {
            "node_name": self.settings.node_name,
            "role": self.settings.role,
            "model_name": self.settings.model_name,
            "device": self.settings.device,
            "ready": self._ready,
            "local": self.local_engine.health() if self.local_engine is not None else None,
            "peers": [peer.as_dict() for peer in self.peers],
        }

    def refresh_peer_health(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for peer in self.peers:
            try:
                response = self.peer_client.health(peer.spec.host, peer.spec.port)
                peer.healthy = True
                peer.last_error = ""
                peer.last_seen = time.time()
                payload = response.get("payload", {})
                engine = payload.get("engine", {}) if isinstance(payload, dict) else {}
                peer.loaded = bool(engine.get("loaded", peer.loaded))
                results.append({"peer": peer.as_dict(), "health": payload})
            except Exception as exc:
                peer.healthy = False
                peer.last_error = str(exc)
                results.append({"peer": peer.as_dict(), "error": str(exc)})
        return results

    def _load_peer(self, peer: PeerState) -> None:
        try:
            response = self.peer_client.load_model(
                peer.spec.host,
                peer.spec.port,
                model_name=self.settings.model_name,
            )
            peer.healthy = True
            peer.loaded = True
            peer.last_error = ""
            peer.last_seen = time.time()
            if response.get("ok") is False:
                raise RuntimeError(str(response.get("error", "peer load failed")))
        except Exception as exc:
            peer.healthy = False
            peer.loaded = False
            peer.last_error = str(exc)
            raise

    def _target_order(self) -> list[str | PeerState]:
        with self._lock:
            remote_targets = [peer for peer in self.peers if peer.healthy or not peer.last_error]
            targets: list[str | PeerState] = []
            if self.local_engine is not None:
                targets.append("local")
            targets.extend(remote_targets)
            if not targets:
                return []
            start = self._rr % len(targets)
            self._rr += 1
            return targets[start:] + targets[:start]

    def _generate_local(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.local_engine is None:
            raise RuntimeError("local inference is disabled")
        request_payload = _payload_with_prompt(
            self.local_engine,
            payload,
            self.settings.generation_defaults(),
        )
        request = request_from_payload(request_payload, self.settings.generation_defaults())
        result = self.local_engine.generate(request).as_dict()
        result["node_name"] = self.settings.node_name
        result["target"] = "local"
        result["distributed"] = len(self.peers) > 0
        return result

    def _stream_local(self, payload: dict[str, Any]):
        if self.local_engine is None:
            raise RuntimeError("local inference is disabled")
        request_payload = _payload_with_prompt(
            self.local_engine,
            payload,
            self.settings.generation_defaults(),
        )
        request = request_from_payload(request_payload, self.settings.generation_defaults())
        for event in self.local_engine.generate_stream(request):
            event_out = dict(event)
            if event_out.get("event") == "done":
                event_out["node_name"] = self.settings.node_name
                event_out["target"] = "local"
                event_out["distributed"] = len(self.peers) > 0
            yield event_out
