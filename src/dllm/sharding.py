from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


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
) -> list[LayerShard]:
    names = [str(name or "").strip() or f"node-{index + 1}" for index, name in enumerate(node_names)]
    transformer_layers = max(int(total_layers or 0) - 1, 0)
    if transformer_layers <= 1 or len(names) <= 1:
        return []

    names = names[: min(len(names), transformer_layers)]
    base_span, remainder = divmod(transformer_layers, len(names))
    shards: list[LayerShard] = []
    start = 0
    for index, name in enumerate(names):
        span = base_span + (1 if index < remainder else 0)
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
