from __future__ import annotations

import unittest

from dllm.model import (
    _add_mxfp4_expert_fallbacks,
    _auto_weight_key_mapping,
    _cast_floating_tensor,
    _key_mapping_for_prefixes,
    _language_loading_plan,
    _last_hidden_state,
    _moe_metadata,
    _nested_language_config_dict,
    _parse_device_map,
    _parse_weight_key_mapping,
    _weight_index_sample_keys,
)


class ModelLoadingOptionsTests(unittest.TestCase):
    def test_mxfp4_expert_fallback_maps_blocks_to_projection(self) -> None:
        target_to_source: dict[str, str] = {}
        state_keys = {
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        }
        weight_map = {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": "model-00001.safetensors",
            "model.layers.0.mlp.experts.gate_up_proj_scales": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_proj_blocks": "model-00001.safetensors",
            "model.layers.0.mlp.experts.down_proj_scales": "model-00001.safetensors",
        }

        _add_mxfp4_expert_fallbacks(target_to_source, weight_map, state_keys)

        self.assertEqual(
            target_to_source["model.layers.0.mlp.experts.gate_up_proj"],
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
        )
        self.assertEqual(
            target_to_source["model.layers.0.mlp.experts.down_proj"],
            "model.layers.0.mlp.experts.down_proj_blocks",
        )

    def test_cast_floating_tensor_uses_target_dtype(self) -> None:
        class FakeTensor:
            def __init__(self) -> None:
                self.dtype = "float32"
                self.requested_dtype = None

            def is_floating_point(self) -> bool:
                return True

            def to(self, *, dtype=None):
                self.requested_dtype = dtype
                return self

        tensor = FakeTensor()
        self.assertIs(_cast_floating_tensor(tensor, dtype="bfloat16"), tensor)
        self.assertEqual(tensor.requested_dtype, "bfloat16")

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

    def test_language_loading_plan_drops_duplicate_model_type_kwarg(self) -> None:
        class FakeAutoConfig:
            @staticmethod
            def for_model(model_type, **kwargs):
                return {"model_type": model_type, **kwargs}

        class FakeTransformers:
            AutoConfig = FakeAutoConfig

        config = {
            "model_type": "vision_language",
            "text_config": {"model_type": "qwen_text", "hidden_size": 1024},
            "vision_config": {"hidden_size": 512},
        }

        plan = _language_loading_plan(
            FakeTransformers,
            "demo",
            config,
            language_only=True,
            language_weight_prefix="model.language_model.",
            offline=True,
            trust_remote_code=False,
        )

        self.assertEqual(plan["config"], {"model_type": "qwen_text", "hidden_size": 1024})
        self.assertTrue(plan["metadata"]["active"])

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
