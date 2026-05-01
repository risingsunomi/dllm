from __future__ import annotations

import os
import tempfile
import unittest

from dllm.config import Settings, load_env_file, parse_peers


class ConfigTests(unittest.TestCase):
    def test_parse_named_and_unnamed_peers(self) -> None:
        peers = parse_peers("node-b@127.0.0.1:8765,127.0.0.2:8766")
        self.assertEqual(peers[0].name, "node-b")
        self.assertEqual(peers[0].address, ("127.0.0.1", 8765))
        self.assertEqual(peers[1].name, "127.0.0.2")

    def test_env_file_and_cli_precedence(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("DLLM_MODEL_NAME=from-env\nDLLM_NODE_NAME=env-node\n")
            path = handle.name
        try:
            old_model = os.environ.pop("DLLM_MODEL_NAME", None)
            old_node = os.environ.pop("DLLM_NODE_NAME", None)
            load_env_file(path)
            settings = Settings.from_mapping({"model_name": "from-cli"})
            self.assertEqual(settings.model_name, "from-cli")
            self.assertEqual(settings.node_name, "env-node")
        finally:
            os.unlink(path)
            os.environ.pop("DLLM_MODEL_NAME", None)
            os.environ.pop("DLLM_NODE_NAME", None)
            if old_model is not None:
                os.environ["DLLM_MODEL_NAME"] = old_model
            if old_node is not None:
                os.environ["DLLM_NODE_NAME"] = old_node

    def test_moe_loading_settings(self) -> None:
        settings = Settings.from_mapping(
            {
                "model_name": "mixtral",
                "device_map": "auto",
                "offload_folder": ".offload",
                "attention_implementation": "sdpa",
                "peer_discovery": False,
                "discovery_port": 9999,
                "discovery_timeout": 0.25,
                "language_only": True,
                "language_weight_prefix": "model.language_model.",
                "weight_key_mapping": r"^layers\.=model.layers.",
            }
        )
        self.assertEqual(settings.device_map, "auto")
        self.assertEqual(settings.offload_folder, ".offload")
        self.assertEqual(settings.attention_implementation, "sdpa")
        self.assertFalse(settings.peer_discovery)
        self.assertEqual(settings.discovery_port, 9999)
        self.assertEqual(settings.discovery_timeout, 0.25)
        self.assertTrue(settings.language_only)
        self.assertEqual(settings.language_weight_prefix, "model.language_model.")
        self.assertEqual(settings.weight_key_mapping, r"^layers\.=model.layers.")

    def test_server_requires_local_first_shard(self) -> None:
        settings = Settings.from_mapping(
            {
                "model_name": "demo",
                "role": "server",
                "load_local": False,
                "peers": "node-b@127.0.0.1:8765",
            }
        )
        with self.assertRaises(ValueError):
            settings.validate_for_runtime()


if __name__ == "__main__":
    unittest.main()
