from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class PeerSpec:
    name: str
    host: str
    port: int

    @property
    def address(self) -> tuple[str, int]:
        return self.host, self.port

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "host": self.host, "port": self.port}


@dataclass(frozen=True)
class Settings:
    model_name: str = ""
    node_name: str = field(default_factory=socket.gethostname)
    device: str = "auto"
    role: str = "server"
    host: str = "0.0.0.0"
    port: int = 8000
    peer_host: str = "0.0.0.0"
    peer_port: int = 8765
    peers: tuple[PeerSpec, ...] = ()
    load_local: bool = True
    peer_discovery: bool = True
    discovery_port: int = 8764
    discovery_timeout: float = 1.0
    offline: bool = False
    dtype: str = "auto"
    fp16_mode: bool = False
    trust_remote_code: bool = False
    device_map: str = ""
    offload_folder: str = ""
    attention_implementation: str = ""
    language_only: bool = True
    language_weight_prefix: str = "auto"
    weight_key_mapping: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    context_length: int = 0
    request_timeout: float = 3600.0
    peer_connect_timeout: float = 5.0
    env_file: str = ".env"

    def __post_init__(self) -> None:
        if self.fp16_mode and str(self.dtype or "").strip().lower() not in {"fp16", "float16"}:
            object.__setattr__(self, "dtype", "fp16")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Settings":
        env = os.environ

        def pick(key: str, env_name: str, default: Any = None) -> Any:
            value = values.get(key)
            if value is not None:
                return value
            return env.get(env_name, default)

        role = str(pick("role", "DLLM_ROLE", "server") or "server").strip().lower()
        if role not in {"server", "worker", "both"}:
            raise ValueError("role must be one of: server, worker, both")

        model_name = str(pick("model_name", "DLLM_MODEL_NAME", "") or "").strip()
        peers_value = pick("peers", "DLLM_PEERS", "") or ""

        fp16_mode = _bool(pick("fp16_mode", "DLLM_FP16_MODE", False), False)
        dtype = str(pick("dtype", "DLLM_DTYPE", "auto") or "auto")
        if fp16_mode:
            dtype = "fp16"

        return cls(
            model_name=model_name,
            node_name=str(pick("node_name", "DLLM_NODE_NAME", socket.gethostname()) or socket.gethostname()),
            device=str(pick("device", "DLLM_DEVICE", "auto") or "auto"),
            role=role,
            host=str(pick("host", "DLLM_HOST", "0.0.0.0") or "0.0.0.0"),
            port=_int(pick("port", "DLLM_PORT", 8000), 8000),
            peer_host=str(pick("peer_host", "DLLM_PEER_HOST", "0.0.0.0") or "0.0.0.0"),
            peer_port=_int(pick("peer_port", "DLLM_PEER_PORT", 8765), 8765),
            peers=tuple(parse_peers(peers_value)),
            load_local=_bool(pick("load_local", "DLLM_LOAD_LOCAL", True), True),
            peer_discovery=_bool(pick("peer_discovery", "DLLM_PEER_DISCOVERY", True), True),
            discovery_port=_int(pick("discovery_port", "DLLM_DISCOVERY_PORT", 8764), 8764),
            discovery_timeout=_float(pick("discovery_timeout", "DLLM_DISCOVERY_TIMEOUT", 1.0), 1.0),
            offline=_bool(pick("offline", "DLLM_OFFLINE", False), False),
            dtype=dtype,
            fp16_mode=fp16_mode,
            trust_remote_code=_bool(pick("trust_remote_code", "DLLM_TRUST_REMOTE_CODE", False), False),
            device_map=str(pick("device_map", "DLLM_DEVICE_MAP", "") or ""),
            offload_folder=str(pick("offload_folder", "DLLM_OFFLOAD_FOLDER", "") or ""),
            attention_implementation=str(
                pick("attention_implementation", "DLLM_ATTENTION_IMPLEMENTATION", "") or ""
            ),
            language_only=_bool(pick("language_only", "DLLM_LANGUAGE_ONLY", True), True),
            language_weight_prefix=str(
                pick("language_weight_prefix", "DLLM_LANGUAGE_WEIGHT_PREFIX", "auto") or "auto"
            ),
            weight_key_mapping=str(pick("weight_key_mapping", "DLLM_WEIGHT_KEY_MAPPING", "") or ""),
            max_new_tokens=_int(pick("max_new_tokens", "DLLM_MAX_NEW_TOKENS", 128), 128),
            temperature=_float(pick("temperature", "DLLM_TEMPERATURE", 0.0), 0.0),
            top_p=_float(pick("top_p", "DLLM_TOP_P", 1.0), 1.0),
            top_k=_int(pick("top_k", "DLLM_TOP_K", 0), 0),
            repetition_penalty=_float(
                pick("repetition_penalty", "DLLM_REPETITION_PENALTY", 1.0),
                1.0,
            ),
            context_length=_int(pick("context_length", "DLLM_CONTEXT_LENGTH", 0), 0),
            request_timeout=_float(pick("request_timeout", "DLLM_REQUEST_TIMEOUT", 3600.0), 3600.0),
            peer_connect_timeout=_float(
                pick("peer_connect_timeout", "DLLM_PEER_CONNECT_TIMEOUT", 5.0),
                5.0,
            ),
            env_file=str(values.get("env_file") or env.get("DLLM_ENV_FILE", ".env")),
        )

    def validate_for_runtime(self) -> None:
        if not self.model_name:
            raise ValueError("model name is required; set DLLM_MODEL_NAME or pass --model-name")
        if self.role in {"server", "both"} and not self.load_local:
            raise ValueError("the server node must load the first shard; do not use --no-load-local")

    def generation_defaults(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }


def load_env_file(path: str | os.PathLike[str], *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _unquote_env_value(value.strip())


def parse_peers(value: str | Iterable[str]) -> list[PeerSpec]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        raw_items = [str(item).strip() for item in value]

    peers: list[PeerSpec] = []
    for item in raw_items:
        if not item:
            continue
        name = ""
        address = item
        if "@" in item:
            name, address = item.split("@", 1)
            name = name.strip()
        if ":" not in address:
            raise ValueError(f"peer must be host:port or name@host:port, got {item!r}")
        host, port_text = address.rsplit(":", 1)
        host = host.strip()
        if not host:
            raise ValueError(f"peer host is missing in {item!r}")
        port = _int(port_text, -1)
        if port <= 0:
            raise ValueError(f"peer port is invalid in {item!r}")
        peers.append(PeerSpec(name=name or host, host=host, port=port))
    return peers


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
