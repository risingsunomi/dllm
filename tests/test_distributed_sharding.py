from __future__ import annotations

import unittest

from dllm.config import Settings, parse_peers
from dllm import peer as peer_module
from dllm.distributed import DistributedInferenceEngine
from dllm.model import (
    GenerationRequest,
    _add_gptq_weight_fallbacks,
    _add_split_bnb_expert_fallbacks,
    _is_bnb_4bit_weight_key,
)
from dllm.peer import InferenceWorker
from dllm.sharding import LayerShard


class _FakeLocalEngine:
    def __init__(self) -> None:
        self.shard = None
        self.load_calls = 0

    def set_shard(self, shard):
        self.shard = shard

    def load(self) -> None:
        self.load_calls += 1

    def health(self) -> dict:
        return {"loaded": False, "shard": self.shard.as_dict() if self.shard else None}

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(prompt.split()) or 1

    def encode_inputs(self, prompt: str) -> dict:
        return {
            "input_ids": {"buffer": "ids"},
            "attention_mask": {"buffer": "mask"},
            "position_ids": {"buffer": "pos"},
            "prompt_tokens": 2,
            "seen_tokens": [1, 2],
        }

    def forward_shard(self, payload: dict) -> dict:
        return {
            "hidden_state": {"buffer": "hidden"},
            "attention_mask": payload["attention_mask"],
            "position_ids": payload["position_ids"],
        }

    def decode_token_ids(self, token_ids: list[int]) -> str:
        return {65: "A"}.get(token_ids[0], "")

    def append_token_to_inputs(self, encoded_inputs: dict, token_id: int) -> dict:
        return encoded_inputs


class _FakePeerClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.load_payloads: list[dict] = []
        self.unload_payloads: list[dict] = []

    def load_model(self, host: str, port: int, **payload) -> dict:
        self.load_payloads.append({"host": host, "port": port, "payload": payload})
        return {"payload": {"configured": True, "loaded": False, "engine": {"loaded": False}}}

    def forward_shard(self, host: str, port: int, payload: dict) -> dict:
        self.payloads.append({"host": host, "port": port, "payload": payload})
        return {"payload": {"token_id": 65, "end_token": True}}

    def unload_model(self, host: str, port: int, *, reason: str = "generation_done") -> dict:
        self.unload_payloads.append({"host": host, "port": port, "reason": reason})
        return {"payload": {"unloaded": True, "engine": {"loaded": False}}}


class _PlanningEngine(DistributedInferenceEngine):
    def _total_layers(self) -> int:
        return 5


class _FakeWorkerEngine:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        dtype: str = "auto",
        offline: bool = False,
        trust_remote_code: bool = False,
        device_map: str = "",
        offload_folder: str = "",
        attention_implementation: str = "",
        language_only: bool = True,
        language_weight_prefix: str = "auto",
        weight_key_mapping: str = "",
        shard: LayerShard | None = None,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.dtype_name = dtype
        self.offline = offline
        self.trust_remote_code = trust_remote_code
        self.device_map_name = device_map
        self.offload_folder = offload_folder
        self.attention_implementation = attention_implementation
        self.language_only = language_only
        self.language_weight_prefix = language_weight_prefix
        self.weight_key_mapping = weight_key_mapping
        self.shard = shard
        self.loaded = False
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self.loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False

    def set_shard(self, shard: LayerShard | None) -> None:
        self.shard = shard

    def health(self) -> dict:
        return {"loaded": self.loaded, "shard": self.shard.as_dict() if self.shard else None}


class DistributedShardingTests(unittest.TestCase):
    def test_gptq_weight_fallback_maps_language_prefixed_qweight(self) -> None:
        target_to_source: dict[str, str] = {}
        weight_map = {
            "model.language_model.layers.0.mlp.down_proj.qweight": "model-00001.safetensors",
            "model.language_model.layers.0.mlp.down_proj.qzeros": "model-00001.safetensors",
            "model.language_model.layers.0.mlp.down_proj.scales": "model-00001.safetensors",
        }
        state_keys = {"model.layers.0.mlp.down_proj.weight"}

        _add_gptq_weight_fallbacks(
            target_to_source,
            weight_map,
            state_keys,
            {r"^model\.language_model\.": "model."},
        )

        self.assertEqual(
            target_to_source["model.layers.0.mlp.down_proj.weight"],
            "model.language_model.layers.0.mlp.down_proj.qweight",
        )

    def test_bitsandbytes_4bit_weight_detection_uses_quant_state_metadata(self) -> None:
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "model.layers.0.self_attn.q_proj.weight.absmax": "model-00001.safetensors",
            "model.layers.0.self_attn.q_proj.weight.nested_absmax": "model-00001.safetensors",
            "model.layers.0.self_attn.q_proj.weight.nested_quant_map": "model-00001.safetensors",
            "model.layers.0.self_attn.q_proj.weight.quant_map": "model-00001.safetensors",
            "model.layers.0.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4": "model-00001.safetensors",
        }

        self.assertTrue(_is_bnb_4bit_weight_key("model.layers.0.self_attn.q_proj.weight", weight_map))
        self.assertFalse(_is_bnb_4bit_weight_key("model.layers.0.self_attn.k_proj.weight", weight_map))

    def test_bitsandbytes_4bit_detection_supports_parameter_without_weight_suffix(self) -> None:
        weight_map = {
            "model.layers.0.mlp.experts.down_proj": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_proj.absmax": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_proj.quant_map": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_proj.quant_state.bitsandbytes__nf4": "model-00001.safetensors",
        }

        self.assertTrue(_is_bnb_4bit_weight_key("model.layers.0.mlp.experts.down_proj", weight_map))

    def test_split_bitsandbytes_expert_fallback_maps_unsloth_gpt_oss_layout(self) -> None:
        target_to_source: dict[str, str] = {}
        state_keys = {
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.down_proj_bias",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.router.bias",
        }
        weight_map = {
            "model.layers.0.mlp.experts.down_projs.0.weight": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_projs.0.weight.absmax": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_projs.0.weight.quant_map": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_projs.0.weight.quant_state.bitsandbytes__nf4": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_projs.0.bias": "model-00001.safetensors",
            "model.layers.0.mlp.experts.gate_up_projs.0.weight": "model-00001.safetensors",
            "model.layers.0.mlp.experts.gate_up_projs.0.weight.absmax": "model-00001.safetensors",
            "model.layers.0.mlp.experts.gate_up_projs.0.weight.quant_map": "model-00001.safetensors",
            "model.layers.0.mlp.experts.gate_up_projs.0.weight.quant_state.bitsandbytes__nf4": "model-00001.safetensors",
            "model.layers.0.mlp.experts.gate_up_projs.0.bias": "model-00001.safetensors",
            "model.layers.0.mlp.router.linear.weight": "model-00001.safetensors",
            "model.layers.0.mlp.router.linear.bias": "model-00001.safetensors",
        }

        _add_split_bnb_expert_fallbacks(target_to_source, weight_map, state_keys)

        self.assertEqual(
            target_to_source["model.layers.0.mlp.experts.down_proj"],
            "model.layers.0.mlp.experts.down_projs.0.weight",
        )
        self.assertEqual(
            target_to_source["model.layers.0.mlp.experts.down_proj_bias"],
            "model.layers.0.mlp.experts.down_projs.0.bias",
        )
        self.assertEqual(
            target_to_source["model.layers.0.mlp.experts.gate_up_proj"],
            "model.layers.0.mlp.experts.gate_up_projs.0.weight",
        )
        self.assertEqual(
            target_to_source["model.layers.0.mlp.experts.gate_up_proj_bias"],
            "model.layers.0.mlp.experts.gate_up_projs.0.bias",
        )
        self.assertEqual(
            target_to_source["model.layers.0.mlp.router.weight"],
            "model.layers.0.mlp.router.linear.weight",
        )
        self.assertEqual(
            target_to_source["model.layers.0.mlp.router.bias"],
            "model.layers.0.mlp.router.linear.bias",
        )

    def test_ensure_ready_plans_shards_without_loading_weights(self) -> None:
        settings = Settings(
            model_name="demo",
            node_name="server",
            peers=tuple(parse_peers("node-b@127.0.0.1:8765")),
        )
        local = _FakeLocalEngine()
        engine = _PlanningEngine(settings, local_engine=local)  # type: ignore[arg-type]
        peer_client = _FakePeerClient()
        engine.peer_client = peer_client  # type: ignore[assignment]

        engine.ensure_ready(prefill_tokens=12, decode_tokens=4)

        self.assertEqual(local.load_calls, 0)
        self.assertIsNotNone(local.shard)
        self.assertEqual(engine.plan_prefill_tokens, 12)
        self.assertEqual(engine.plan_decode_tokens, 4)
        self.assertEqual(len(peer_client.load_payloads), 1)
        self.assertFalse(engine.peers[0].loaded)
        self.assertTrue(engine.peers[0].healthy)

    def test_worker_load_model_configures_shard_without_loading_weights(self) -> None:
        original_engine = peer_module.TorchTransformersEngine
        peer_module.TorchTransformersEngine = _FakeWorkerEngine  # type: ignore[assignment]
        try:
            settings = Settings(model_name="demo", node_name="node-b", role="worker")
            worker = InferenceWorker(settings)
            shard = LayerShard("demo", 2, 4, 5, "node-b")

            response = worker._handle_load_model({"model_name": "demo", "shard": shard.as_dict()})

            self.assertTrue(response["payload"]["configured"])
            self.assertFalse(response["payload"]["loaded"])
            self.assertEqual(worker.engine.load_calls, 0)
            self.assertEqual(worker.engine.shard, shard)
        finally:
            peer_module.TorchTransformersEngine = original_engine  # type: ignore[assignment]

    def test_worker_unload_model_releases_weights(self) -> None:
        original_engine = peer_module.TorchTransformersEngine
        peer_module.TorchTransformersEngine = _FakeWorkerEngine  # type: ignore[assignment]
        try:
            settings = Settings(model_name="demo", node_name="node-b", role="worker")
            worker = InferenceWorker(settings)
            worker.engine.load()

            response = worker._handle_unload_model({"reason": "generation_done"})

            self.assertTrue(response["payload"]["unloaded"])
            self.assertFalse(response["payload"]["engine"]["loaded"])
            self.assertEqual(worker.engine.unload_calls, 1)
        finally:
            peer_module.TorchTransformersEngine = original_engine  # type: ignore[assignment]

    def test_sharded_generation_passes_hidden_state_to_peer(self) -> None:
        settings = Settings(
            model_name="demo",
            node_name="server",
            peers=tuple(parse_peers("node-b@127.0.0.1:8765")),
        )
        engine = DistributedInferenceEngine(settings, local_engine=_FakeLocalEngine())  # type: ignore[arg-type]
        peer_client = _FakePeerClient()
        engine.peer_client = peer_client  # type: ignore[assignment]
        engine.local_shard = LayerShard("demo", 0, 1, 3, "server")
        engine.peers[0].shard = LayerShard("demo", 1, 2, 3, "node-b")
        engine.peers[0].healthy = True

        request_payload = {"prompt": "hi", "max_new_tokens": 1}
        request = GenerationRequest(prompt="hi", max_new_tokens=1)
        result = engine._generate_sharded(request_payload, request, 2)

        self.assertEqual(result["text"], "A")
        self.assertEqual(result["target"], "sharded")
        self.assertIn("hidden_state", peer_client.payloads[0]["payload"])
        self.assertEqual(len(peer_client.unload_payloads), 1)
        self.assertFalse(engine.peers[0].loaded)
        self.assertEqual(result["shards"][0]["shard"]["start_layer"], 0)
        self.assertTrue(result["shards"][1]["shard"]["is_final"])


if __name__ == "__main__":
    unittest.main()
