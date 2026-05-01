from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import PeerSpec, Settings
from .model import TorchTransformersEngine, request_from_payload
from .model.device_info import collect_host_info, host_weight
from .peer import PeerClient, _payload_with_prompt
from .sharding import LayerShard, build_layer_shards, total_layers_from_config


@dataclass
class PeerState:
    spec: PeerSpec
    healthy: bool = False
    loaded: bool = False
    last_error: str = ""
    last_seen: float = 0.0
    shard: LayerShard | None = None
    device_info: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "healthy": self.healthy,
            "loaded": self.loaded,
            "last_error": self.last_error,
            "last_seen": self.last_seen,
            "shard": self.shard.as_dict() if self.shard is not None else None,
            "device_info": self.device_info,
        }


class DistributedInferenceEngine:
    """Coordinator for local or multi-node inference.

    One request flows through the server's local layer shard and then each peer
    shard. With no peers configured or discovered, the server runs as a single
    full-model node.
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
        self.local_device_info = collect_host_info()
        self.local_shard: LayerShard | None = None
        self.plan_prefill_tokens = 0
        self.plan_decode_tokens = 0
        self._lock = threading.RLock()
        self._ready = False

    def ensure_ready(self, *, prefill_tokens: int = 0, decode_tokens: int = 0) -> None:
        with self._lock:
            if self._ready:
                return
            errors: list[str] = []
            if self.peers:
                self.refresh_peer_health()
            self._plan_shards(prefill_tokens=prefill_tokens, decode_tokens=decode_tokens)
            for peer in self.peers:
                if peer.shard is None:
                    continue
                try:
                    self._load_peer(peer)
                except Exception as exc:
                    errors.append(f"{peer.spec.name}: {exc}")
            if errors:
                raise RuntimeError(f"not all shard peers are ready: {'; '.join(errors)}")
            if self.local_engine is None and not any(peer.healthy for peer in self.peers):
                detail = "; ".join(errors) if errors else "no local or remote targets configured"
                raise RuntimeError(f"no inference targets are ready: {detail}")
            self._ready = True

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload, request, prefill_tokens = self._prepare_request(payload)
        self.ensure_ready(prefill_tokens=prefill_tokens, decode_tokens=request.max_new_tokens)
        if self._using_shards():
            return self._generate_sharded(request_payload, request, prefill_tokens)
        return self._generate_local(request_payload, request)

    def stream(self, payload: dict[str, Any]):
        request_payload, request, prefill_tokens = self._prepare_request(payload)
        self.ensure_ready(prefill_tokens=prefill_tokens, decode_tokens=request.max_new_tokens)
        if self._using_shards():
            yield from self._stream_sharded(request_payload, request, prefill_tokens)
            return
        yield from self._stream_local(request_payload, request)

    def health(self, *, probe_peers: bool = False) -> dict[str, Any]:
        if probe_peers:
            self.refresh_peer_health()
        return {
            "node_name": self.settings.node_name,
            "role": self.settings.role,
            "model_name": self.settings.model_name,
            "device": self.settings.device,
            "ready": self._ready,
            "runtime": "sharded" if self._using_shards() else "single-node",
            "device_info": self.local_device_info,
            "plan_prefill_tokens": self.plan_prefill_tokens,
            "plan_decode_tokens": self.plan_decode_tokens,
            "local_shard": self.local_shard.as_dict() if self.local_shard is not None else None,
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
                device_info = payload.get("device_info") if isinstance(payload, dict) else None
                if not isinstance(device_info, dict):
                    device_info = engine.get("device_info") if isinstance(engine, dict) else None
                peer.device_info = device_info if isinstance(device_info, dict) else peer.device_info
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
                offline=self.settings.offline,
                trust_remote_code=self.settings.trust_remote_code,
                language_only=self.settings.language_only,
                language_weight_prefix=self.settings.language_weight_prefix,
                weight_key_mapping=self.settings.weight_key_mapping,
                shard=peer.shard,
            )
            peer.healthy = True
            payload = response.get("payload", {})
            engine = payload.get("engine", {}) if isinstance(payload, dict) else {}
            peer.loaded = bool(engine.get("loaded", False))
            peer.last_error = ""
            peer.last_seen = time.time()
            if response.get("ok") is False:
                raise RuntimeError(str(response.get("error", "peer load failed")))
        except Exception as exc:
            peer.healthy = False
            peer.loaded = False
            peer.last_error = str(exc)
            raise

    def _plan_shards(self, *, prefill_tokens: int, decode_tokens: int) -> None:
        self.plan_prefill_tokens = max(int(prefill_tokens), 0)
        self.plan_decode_tokens = max(int(decode_tokens), 0)
        if not self.peers or self.local_engine is None:
            self.local_shard = None
            if self.local_engine is not None:
                self.local_engine.set_shard(None)
            for peer in self.peers:
                peer.shard = None
            return

        total_layers = self._total_layers()
        node_names = [self.settings.node_name, *[peer.spec.name for peer in self.peers]]
        plan = build_layer_shards(
            model_name=self.settings.model_name,
            total_layers=total_layers,
            node_names=node_names,
            node_weights=self._node_weights(),
            prefill_tokens=self.plan_prefill_tokens,
            decode_tokens=self.plan_decode_tokens,
        )
        if len(plan) <= 1:
            self.local_shard = None
            self.local_engine.set_shard(None)
            for peer in self.peers:
                peer.shard = None
            return

        self.local_shard = plan[0]
        self.local_engine.set_shard(self.local_shard)
        remote_plan = plan[1:]
        for index, peer in enumerate(self.peers):
            peer.shard = remote_plan[index] if index < len(remote_plan) else None

    def _total_layers(self) -> int:
        try:
            from transformers import AutoConfig
        except Exception as exc:
            raise RuntimeError("transformers is required for sharded layer planning") from exc
        config = AutoConfig.from_pretrained(
            self.settings.model_name,
            local_files_only=self.settings.offline,
            trust_remote_code=self.settings.trust_remote_code,
        )
        total_layers = total_layers_from_config(config)
        if total_layers <= 1:
            raise RuntimeError("model config does not expose num_hidden_layers for sharding")
        return total_layers

    def _node_weights(self) -> list[float]:
        weights = [host_weight(self.local_device_info, self.settings.device)]
        for peer in self.peers:
            weights.append(host_weight(peer.device_info, self.settings.device))
        return weights

    def _using_shards(self) -> bool:
        return self.local_shard is not None and any(peer.shard is not None for peer in self.peers)

    def _shard_peers(self) -> list[PeerState]:
        return [peer for peer in self.peers if peer.shard is not None and (peer.healthy or not peer.last_error)]

    def _prepare_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], Any, int]:
        if self.local_engine is None:
            raise RuntimeError("the server node must load the first shard")
        request_payload = _payload_with_prompt(
            self.local_engine,
            payload,
            self.settings.generation_defaults(),
        )
        request = request_from_payload(request_payload, self.settings.generation_defaults())
        prefill_tokens = self.local_engine.count_prompt_tokens(request.prompt)
        return request_payload, request, prefill_tokens

    def _generate_sharded(self, request_payload: dict[str, Any], request: Any, prefill_tokens: int) -> dict[str, Any]:
        events = list(
            self._run_sharded(
                request_payload,
                request,
                prefill_tokens=prefill_tokens,
                emit_tokens=False,
            )
        )
        done = events[-1] if events else {}
        return dict(done)

    def _stream_sharded(self, request_payload: dict[str, Any], request: Any, prefill_tokens: int):
        yield from self._run_sharded(
            request_payload,
            request,
            prefill_tokens=prefill_tokens,
            emit_tokens=True,
        )

    def _run_sharded(
        self,
        request_payload: dict[str, Any],
        request: Any,
        *,
        prefill_tokens: int,
        emit_tokens: bool,
    ):
        if self.local_engine is None:
            raise RuntimeError("sharded inference requires a local server shard")

        encoded_inputs = self.local_engine.encode_inputs(request.prompt)
        prompt_tokens = int(encoded_inputs.get("prompt_tokens", prefill_tokens))
        seen_tokens = list(encoded_inputs.get("seen_tokens", []))
        token_ids: list[int] = []
        text_parts: list[str] = []
        stop_filter = _ShardStopFilter(request.stop)
        start = time.perf_counter()

        for _ in range(max(int(request.max_new_tokens), 1)):
            stage_payload: dict[str, Any] = {
                **request_payload,
                "input_ids": encoded_inputs["input_ids"],
                "attention_mask": encoded_inputs["attention_mask"],
                "position_ids": encoded_inputs["position_ids"],
                "seen_tokens": seen_tokens,
            }
            result = self.local_engine.forward_shard(stage_payload)
            for peer in self._shard_peers():
                result = self._forward_peer_shard(peer, {**stage_payload, **_next_shard_payload(result)})

            if "token_id" not in result:
                raise RuntimeError("sharded inference ended without a final token")

            token_id = int(result["token_id"])
            token_ids.append(token_id)
            seen_tokens.append(token_id)
            piece = self.local_engine.decode_token_ids([token_id])
            filtered = stop_filter.push(piece)
            if filtered:
                text_parts.append(filtered)
                if emit_tokens:
                    yield {"event": "token", "text": filtered}

            if bool(result.get("end_token")) or stop_filter.stopped:
                break
            encoded_inputs = self.local_engine.append_token_to_inputs(encoded_inputs, token_id)

        trailing = stop_filter.finish()
        if trailing:
            text_parts.append(trailing)
            if emit_tokens:
                yield {"event": "token", "text": trailing}

        elapsed = time.perf_counter() - start
        done = {
            "event": "done",
            "text": "".join(text_parts),
            "generated_tokens": len(token_ids),
            "prompt_tokens": prompt_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": (len(token_ids) / elapsed) if elapsed > 0 else 0.0,
            "model_name": self.settings.model_name,
            "device": self.settings.device,
            "node_name": self.settings.node_name,
            "target": "sharded",
            "distributed": True,
            "shards": self._shard_summary(),
        }
        _log_tokens_per_second(done)
        yield done

    def _forward_peer_shard(self, peer: PeerState, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.peer_client.forward_shard(peer.spec.host, peer.spec.port, payload)
            peer.healthy = True
            peer.loaded = True
            peer.last_error = ""
            peer.last_seen = time.time()
            return dict(response.get("payload", {}))
        except Exception as exc:
            peer.healthy = False
            peer.last_error = str(exc)
            raise

    def _shard_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        if self.local_shard is not None:
            summary.append({"node_name": self.settings.node_name, "shard": self.local_shard.as_dict()})
        for peer in self.peers:
            if peer.shard is not None:
                summary.append({"node_name": peer.spec.name, "shard": peer.shard.as_dict()})
        return summary

    def _generate_local(self, request_payload: dict[str, Any], request: Any) -> dict[str, Any]:
        if self.local_engine is None:
            raise RuntimeError("local inference is disabled")
        result = self.local_engine.generate(request).as_dict()
        result["node_name"] = self.settings.node_name
        result["target"] = "local"
        result["distributed"] = False
        _log_tokens_per_second(result)
        return result

    def _stream_local(self, request_payload: dict[str, Any], request: Any):
        del request_payload
        if self.local_engine is None:
            raise RuntimeError("local inference is disabled")
        for event in self.local_engine.generate_stream(request):
            event_out = dict(event)
            if event_out.get("event") == "done":
                event_out["node_name"] = self.settings.node_name
                event_out["target"] = "local"
                event_out["distributed"] = False
                _log_tokens_per_second(event_out)
            yield event_out


def _next_shard_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("hidden_state", "attention_mask", "position_ids"):
        if key in result:
            payload[key] = result[key]
    return payload


class _ShardStopFilter:
    def __init__(self, stop: tuple[str, ...]) -> None:
        self.stop = tuple(marker for marker in stop if marker)
        self.pending = ""
        self.stopped = False

    def push(self, text: str) -> str:
        if self.stopped:
            return ""
        if not self.stop:
            return text
        self.pending += text
        stop_index = _first_stop_index(self.pending, self.stop)
        if stop_index is not None:
            output = self.pending[:stop_index]
            self.pending = ""
            self.stopped = True
            return output
        keep = _longest_stop_prefix_suffix(self.pending, self.stop)
        if keep <= 0:
            output = self.pending
            self.pending = ""
            return output
        output = self.pending[:-keep]
        self.pending = self.pending[-keep:]
        return output

    def finish(self) -> str:
        if self.stopped:
            return ""
        output = self.pending
        self.pending = ""
        return output


def _first_stop_index(text: str, stop: tuple[str, ...]) -> int | None:
    indexes = [text.find(marker) for marker in stop if marker and text.find(marker) >= 0]
    return min(indexes) if indexes else None


def _longest_stop_prefix_suffix(text: str, stop: tuple[str, ...]) -> int:
    longest = 0
    for marker in stop:
        max_len = min(len(marker) - 1, len(text))
        for size in range(max_len, 0, -1):
            if text.endswith(marker[:size]):
                longest = max(longest, size)
                break
    return longest


def _log_tokens_per_second(result: dict[str, Any]) -> None:
    generated = int(result.get("generated_tokens", 0) or 0)
    elapsed = float(result.get("elapsed_seconds", 0.0) or 0.0)
    tok_s = float(result.get("tokens_per_second", 0.0) or 0.0)
    if tok_s <= 0 and elapsed > 0:
        tok_s = generated / elapsed
    print(
        "generation complete "
        f"target={result.get('target', 'unknown')} "
        f"model={result.get('model_name', '')} "
        f"tokens={generated} elapsed={elapsed:.3f}s tok_s={tok_s:.2f}",
        flush=True,
    )
