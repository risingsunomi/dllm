from __future__ import annotations

import unittest

from dllm.config import Settings, parse_peers
from dllm.distributed import DistributedInferenceEngine
from dllm.sharding import LayerShard


class _FakeLocalEngine:
    def __init__(self) -> None:
        self.shard = None

    def set_shard(self, shard):
        self.shard = shard

    def load(self) -> None:
        return None

    def health(self) -> dict:
        return {"loaded": True, "shard": self.shard.as_dict() if self.shard else None}

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

    def forward_shard(self, host: str, port: int, payload: dict) -> dict:
        self.payloads.append({"host": host, "port": port, "payload": payload})
        return {"payload": {"token_id": 65, "end_token": True}}


class DistributedShardingTests(unittest.TestCase):
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
        engine.peers[0].loaded = True

        result = engine._generate_sharded({"prompt": "hi", "max_new_tokens": 1})

        self.assertEqual(result["text"], "A")
        self.assertEqual(result["target"], "sharded")
        self.assertIn("hidden_state", peer_client.payloads[0]["payload"])
        self.assertEqual(result["shards"][0]["shard"]["start_layer"], 0)
        self.assertTrue(result["shards"][1]["shard"]["is_final"])


if __name__ == "__main__":
    unittest.main()
