from __future__ import annotations

import unittest

from dllm.model import (
    _auto_weight_key_mapping,
    _key_mapping_for_prefixes,
    _last_hidden_state,
    _moe_metadata,
    _nested_language_config_dict,
    _parse_device_map,
    _parse_max_memory,
    _parse_weight_key_mapping,
    _weight_index_sample_keys,
)


class ModelLoadingOptionsTests(unittest.TestCase):
    def test_last_hidden_state_unwraps_nested_tuples(self) -> None:
        class FakeTensor:
            shape = (1, 2, 3)
            dtype = "float32"

            def detach(self):  # pragma: no cover - identity marker for tensor-like duck typing
                return self

        tensor = FakeTensor()
        self.assertIs(_last_hidden_state(((tensor,),)), tensor)

    def test_parse_device_map(self) -> None:
        self.assertIsNone(_parse_device_map(""))
        self.assertIsNone(_parse_device_map("none"))
        self.assertEqual(_parse_device_map("auto"), "auto")
        self.assertEqual(_parse_device_map('{"model.embed_tokens":"cuda:0"}'), {"model.embed_tokens": "cuda:0"})

    def test_parse_max_memory(self) -> None:
        self.assertEqual(_parse_max_memory("0=20GiB,cpu=80GiB"), {0: "20GiB", "cpu": "80GiB"})
        self.assertEqual(_parse_max_memory('{"0":"10GiB","cpu":"48GiB"}'), {0: "10GiB", "cpu": "48GiB"})

    def test_moe_metadata(self) -> None:
        meta = _moe_metadata(
            {
                "model_type": "mixtral",
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
            }
        )
        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["num_local_experts"], 8)
        self.assertEqual(meta["num_experts_per_tok"], 2)

    def test_nested_language_config(self) -> None:
        config = {
            "model_type": "vision_language",
            "text_config": {"model_type": "qwen_text", "hidden_size": 1024},
            "vision_config": {"hidden_size": 512},
        }
        self.assertEqual(_nested_language_config_dict(config)["model_type"], "qwen_text")

    def test_language_prefix_key_mapping(self) -> None:
        mapping = _key_mapping_for_prefixes(["model.language_model."])
        self.assertEqual(mapping[r"^model\.language_model\."], "model.")

    def test_parse_weight_key_mapping(self) -> None:
        mapping = _parse_weight_key_mapping(
            r"^layers\.=model.layers.,^head\.weight$=lm_head.weight",
            "",
            offline=True,
        )
        self.assertEqual(mapping[r"^layers\."], "model.layers.")
        self.assertEqual(mapping[r"^head\.weight$"], "lm_head.weight")

    def test_nonstandard_safetensors_index_name(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "weights.safetensors.index.json").write_text(
                '{"weight_map":{"layers.0.self_attn.q_proj.weight":"weights-00001.safetensors"}}',
                encoding="utf-8",
            )
            self.assertEqual(
                _weight_index_sample_keys(str(path), offline=True, trust_remote_code=False),
                ["layers.0.self_attn.q_proj.weight"],
            )
            self.assertEqual(
                _auto_weight_key_mapping(str(path), offline=True),
                {r"^layers\.": "model.layers."},
            )


if __name__ == "__main__":
    unittest.main()
