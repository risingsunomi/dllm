from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from dllm.model.devices_flop import DeviceFlops


@dataclass(frozen=True)
class LayerShard:
    model_name: str
    start_layer: int
    end_layer: int
    total_layers: int
    node_name: str = ""

    @property
    def transformer_layers(self) -> int:
        return max(int(self.total_layers) - 1, 0)

    @property
    def is_first(self) -> bool:
        return int(self.start_layer) <= 0

    @property
    def is_final(self) -> bool:
        return int(self.end_layer) >= self.transformer_layers

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "start_layer": int(self.start_layer),
            "end_layer": int(self.end_layer),
            "total_layers": int(self.total_layers),
            "node_name": self.node_name,
            "is_first": self.is_first,
            "is_final": self.is_final,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "LayerShard | None":
        if not isinstance(payload, dict):
            return None
        try:
            model_name = str(payload.get("model_name", "") or "")
            start_layer = int(payload.get("start_layer", 0))
            end_layer = int(payload.get("end_layer", 0))
            total_layers = int(payload.get("total_layers", 0))
            node_name = str(payload.get("node_name", "") or "")
        except (TypeError, ValueError):
            return None
        if total_layers <= 1 or end_layer < start_layer:
            return None
        return cls(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            total_layers=total_layers,
            node_name=node_name,
        )


def build_layer_shards(
    *,
    model_name: str,
    total_layers: int,
    node_names: Iterable[str],
    node_weights: Iterable[float] | None = None,
    prefill_tokens: int = 0,
    decode_tokens: int = 0,
) -> list[LayerShard]:
    names = [str(name or "").strip() or f"node-{index + 1}" for index, name in enumerate(node_names)]
    transformer_layers = max(int(total_layers or 0) - 1, 0)
    if transformer_layers <= 1 or len(names) <= 1:
        return []

    names = names[: min(len(names), transformer_layers)]
        
    weights = _normalized_weights(node_weights, len(names))
    if weights:
        spans = _capacity_weighted_spans(transformer_layers=transformer_layers, weights=weights)
    else:
        spans = _prefill_weighted_spans(
            transformer_layers=transformer_layers,
            parts=len(names),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
    shards: list[LayerShard] = []
    start = 0
    for index, name in enumerate(names):
        span = spans[index]
        end = min(start + span, transformer_layers)
        shards.append(
            LayerShard(
                model_name=model_name,
                start_layer=start,
                end_layer=end,
                total_layers=int(total_layers),
                node_name=name,
            )
        )
        start = end
    return shards


def _normalized_weights(values: Iterable[float] | None, parts: int) -> list[float]:
    if values is None:
        return []
    weights: list[float] = []
    for value in values:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = 0.0
        weights.append(max(weight, 0.0))
    if len(weights) < parts:
        weights.extend([1.0] * (parts - len(weights)))
    weights = weights[:parts]
    if parts <= 1 or sum(weights) <= 0:
        return []
    return [weight if weight > 0 else 0.01 for weight in weights]

def _capacity_weighted_spans(
    *,
    transformer_layers: int,
    weights: list[float],
    max_layers: list[int] | None = None,
) -> list[int]:
    parts = len(weights)
    layers = max(int(transformer_layers), 0)
    if parts <= 0:
        return []
    if parts == 1:
        return [layers]
    if layers <= parts:
        return [1 if index < layers else 0 for index in range(parts)]

    if max_layers is None:
        max_layers = [layers] * parts
    else:
        max_layers = [max(int(value), 0) for value in max_layers[:parts]]
        if len(max_layers) < parts:
            max_layers.extend([layers] * (parts - len(max_layers)))

    total_weight = sum(weights)
    raw = [layers * (weight / total_weight) for weight in weights]

    spans = [
        min(max_layers[index], max(1, int(raw[index])))
        for index in range(parts)
    ]

    while sum(spans) > layers:
        candidates = [
            index for index in range(parts)
            if spans[index] > 1
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: (spans[item], -weights[item]))
        spans[index] -= 1

    while sum(spans) < layers:
        candidates = [
            index for index in range(parts)
            if spans[index] < max_layers[index]
        ]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda item: (weights[item], raw[item] - spans[item]),
        )
        spans[index] += 1

    return spans 


def _prefill_weighted_spans(
    *,
    transformer_layers: int,
    parts: int,
    prefill_tokens: int,
    decode_tokens: int,
) -> list[int]:
    if parts <= 0:
        return []
    if parts == 1:
        return [max(int(transformer_layers), 0)]

    layers = max(int(transformer_layers), 0)
    layer_work = max(int(prefill_tokens), 1) + max(int(decode_tokens), 1)
    final_stage_work = max(int(decode_tokens), 1)
    total_work = layers * layer_work + final_stage_work
    target = total_work / parts
    final_target = max(target - final_stage_work, layer_work)
    final_span = max(1, min(layers - (parts - 1), round(final_target / layer_work)))
    remaining = layers - final_span
    base_span, remainder = divmod(remaining, parts - 1)
    spans = [base_span + (1 if index < remainder else 0) for index in range(parts - 1)]
    spans.append(final_span)
    return [int(span) for span in spans]

def _node_flop_weights(
    node_devices: dict[str, str],
    *,
    dtype: str = "fp16",
) -> dict[str, float]:
    result: dict[str, float] = {}
    for node_name, device_name in node_devices.items():
        result[str(node_name)] = max(
            float(DeviceFlops(str(device_name)).get_flops(dtype=dtype)),
            0.01,
        )
    return result


def total_layers_from_config(config: Any) -> int:
    config_dict = _config_dict(config)
    nested = _nested_language_config(config_dict)
    if nested:
        config_dict = nested

    for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        try:
            value = int(config_dict.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value + 1
    return 0


def _nested_language_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "language_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and nested:
            return dict(nested)
    return {}


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}
