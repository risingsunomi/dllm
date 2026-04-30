from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

from . import __version__
from .config import PeerSpec, Settings


_DISCOVERY_MAGIC = "dllm.discovery.v1"


class DiscoveryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._serve, name="dllm-discovery", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _serve(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            sock.bind(("", int(self.settings.discovery_port)))
            sock.settimeout(0.25)
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                payload = _decode_discovery(data)
                if payload.get("type") != "probe":
                    continue
                if str(payload.get("model_name", "")) not in {"", self.settings.model_name}:
                    continue
                response = _announcement_payload(self.settings, target_host=str(addr[0]))
                try:
                    sock.sendto(_encode_discovery(response), addr)
                except OSError:
                    continue
        finally:
            try:
                sock.close()
            except OSError:
                pass


def discover_peers(settings: Settings) -> tuple[PeerSpec, ...]:
    if not settings.peer_discovery or settings.role not in {"server", "both"}:
        return ()

    found: list[PeerSpec] = []
    deadline = time.monotonic() + max(float(settings.discovery_timeout), 0.0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.1)
        probe = _encode_discovery(
            {
                "magic": _DISCOVERY_MAGIC,
                "type": "probe",
                "version": __version__,
                "node_name": settings.node_name,
                "model_name": settings.model_name,
            }
        )
        for host in ("255.255.255.255", "<broadcast>"):
            try:
                sock.sendto(probe, (host, int(settings.discovery_port)))
            except OSError:
                continue

        while time.monotonic() < deadline:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            payload = _decode_discovery(data)
            if payload.get("type") != "announce":
                continue
            if str(payload.get("model_name", "")) != settings.model_name:
                continue
            node_name = str(payload.get("node_name", "") or "").strip()
            if node_name == settings.node_name:
                continue
            host = str(payload.get("host", "") or addr[0]).strip()
            port = _int(payload.get("peer_port"), 0)
            if not host or port <= 0:
                continue
            found.append(PeerSpec(name=node_name or host, host=host, port=port))
    finally:
        sock.close()
    return tuple(found)


def merge_peers(
    manual_peers: tuple[PeerSpec, ...],
    discovered_peers: tuple[PeerSpec, ...],
    *,
    local_node_name: str,
) -> tuple[PeerSpec, ...]:
    merged: list[PeerSpec] = []
    seen: set[tuple[str, int]] = set()
    seen_names: set[str] = {str(local_node_name)}
    for peer in (*manual_peers, *discovered_peers):
        key = (peer.host, int(peer.port))
        if key in seen or peer.name in seen_names:
            continue
        seen.add(key)
        seen_names.add(peer.name)
        merged.append(peer)
    return tuple(merged)


def lan_ip_addresses() -> tuple[str, ...]:
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            address = str(info[4][0])
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    target_ip = _local_ip_for_target("8.8.8.8")
    if target_ip and not target_ip.startswith("127."):
        addresses.add(target_ip)
    return tuple(sorted(addresses))


def _announcement_payload(settings: Settings, *, target_host: str) -> dict[str, Any]:
    return {
        "magic": _DISCOVERY_MAGIC,
        "type": "announce",
        "version": __version__,
        "node_name": settings.node_name,
        "model_name": settings.model_name,
        "host": _advertise_host(settings.peer_host, target_host),
        "peer_port": int(settings.peer_port),
        "role": settings.role,
    }


def _advertise_host(bind_host: str, target_host: str) -> str:
    host = str(bind_host or "").strip()
    if host and host not in {"0.0.0.0", "::"}:
        return host
    return _local_ip_for_target(target_host) or (lan_ip_addresses()[0] if lan_ip_addresses() else "127.0.0.1")


def _local_ip_for_target(target_host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_host, 9))
        return str(sock.getsockname()[0])
    except OSError:
        return ""
    finally:
        sock.close()


def _encode_discovery(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")


def _decode_discovery(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("magic") != _DISCOVERY_MAGIC:
        return {}
    return payload


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
