from __future__ import annotations

import unittest

from dllm.sharding import LayerShard, build_layer_shards, total_layers_from_config


class ShardingTests(unittest.TestCase):
    def test_total_layers_from_transformers_config(self) -> None:
        self.assertEqual(total_layers_from_config({"num_hidden_layers": 32}), 33)
        self.assertEqual(total_layers_from_config({"text_config": {"num_hidden_layers": 28}}), 29)

    def test_build_layer_shards_uses_cheetah_convention(self) -> None:
        shards = build_layer_shards(
            model_name="demo",
            total_layers=17,
            node_names=["server", "node-b"],
        )
        self.assertEqual([shard.as_dict() for shard in shards], [
            {
                "model_name": "demo",
                "start_layer": 0,
                "end_layer": 8,
                "total_layers": 17,
                "node_name": "server",
                "is_first": True,
                "is_final": False,
            },
            {
                "model_name": "demo",
                "start_layer": 8,
                "end_layer": 16,
                "total_layers": 17,
                "node_name": "node-b",
                "is_first": False,
                "is_final": True,
            },
        ])

    def test_layer_shard_from_mapping(self) -> None:
        shard = LayerShard.from_mapping(
            {"model_name": "demo", "start_layer": 1, "end_layer": 2, "total_layers": 3}
        )
        self.assertIsNotNone(shard)
        assert shard is not None
        self.assertFalse(shard.is_first)
        self.assertTrue(shard.is_final)


if __name__ == "__main__":
    unittest.main()
