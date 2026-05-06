from __future__ import annotations

from . import __version__
from .config import PeerSpec, Settings
from .discovery import lan_ip_addresses
from .model.device_info import collect_host_info


_LOGO = r"""
 ######     ##      ##      ##      ##
 ##   ##    ##      ##      ####  ####
 ##    ##   ##      ##      ## #### ##
 ##   ##    ##      ##      ##  ##  ##
 ######     ######  ######  ##      ##
    D I S T R I B U T E D   L L M
"""


def startup_banner(settings: Settings, *, peers: tuple[PeerSpec, ...], title: str) -> str:
    lines = [_LOGO.rstrip(), f"dllm v{__version__} | {title} | shard-first distributed inference"]
    fp16_mode = "on" if settings.fp16_mode else "off"
    lines.append(
        f"node={settings.node_name} role={settings.role} "
        f"device={settings.device} dtype={settings.dtype} fp16_mode={fp16_mode}"
    )
    selected = collect_host_info().get("selected", {})
    if isinstance(selected, dict) and selected:
        label = selected.get("device") or selected.get("kind") or "device"
        name = selected.get("name") or ""
        memory = selected.get("available_vram_gb") or selected.get("available_ram_gb") or selected.get("ram_gb")
        lines.append(f"detected_device={label} {name} available_memory_gb={memory}")
    lines.append(f"model={settings.model_name}")
    if settings.role in {"server", "both"}:
        lines.append(f"http={settings.host}:{settings.port}")
    if settings.role in {"worker", "both"}:
        lines.append(f"peer_tcp={settings.peer_host}:{settings.peer_port}")
    ips = ", ".join(lan_ip_addresses()) or "unknown"
    lines.append(f"lan_ips={ips}")
    lines.append(
        f"discovery={'on' if settings.peer_discovery else 'off'} "
        f"udp={settings.discovery_port} timeout={settings.discovery_timeout:g}s"
    )
    if peers:
        peer_text = ", ".join(f"{peer.name}@{peer.host}:{peer.port}" for peer in peers)
    else:
        peer_text = "none"
    lines.append(f"peers={peer_text}")
    return "\n".join(lines)
