from __future__ import annotations

import unittest

from dllm.model import (
    _add_mxfp4_expert_fallbacks,
    _auto_weight_key_mapping,
    _cast_floating_tensor,
    _decode_tensor,
    _encode_tensor,
    _key_mapping_for_prefixes,
    _language_loading_plan,
    _language_weight_prefixes,
    _last_hidden_state,
    _moe_metadata,
    _nested_language_config_dict,
    _parse_device_map,
    _parse_weight_key_mapping,
    _resolve_dtype,
    _resolve_checkpoint_source,
    _safetensor_files,
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

    def test_encode_tensor_can_force_fp16_transport_dtype(self) -> None:
        import torch

        tensor = torch.ones((1, 2), dtype=torch.float32)
        payload = _encode_tensor(tensor, dtype=torch.float16)
        decoded = _decode_tensor(payload, torch, "cpu")

        self.assertEqual(payload["dtype"], "float16")
        self.assertEqual(decoded.dtype, torch.float16)

    def test_encode_tensor_can_return_raw_transport_buffer(self) -> None:
        import torch

        tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        payload = _encode_tensor(tensor, raw=True)
        decoded = _decode_tensor(payload, torch, "cpu")

        self.assertIsInstance(payload["buffer"], bytes)
        self.assertTrue(torch.equal(decoded, tensor))

    def test_resolve_dtype_accepts_bfp16_alias(self) -> None:
        import torch

        self.assertIs(_resolve_dtype("bfp16", "cuda", torch), torch.bfloat16)

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

    def test_language_prefix_detects_nested_language_model_model_layout(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "model.safetensors.index.json").write_text(
                (
                    '{"weight_map":{'
                    '"language_model.model.embed_tokens.weight":"model-00001.safetensors",'
                    '"language_model.model.layers.0.mlp.down_proj.weight":"model-00001.safetensors"'
                    "}}"
                ),
                encoding="utf-8",
            )

            prefixes = _language_weight_prefixes(
                str(path),
                explicit_prefix="auto",
                offline=True,
                trust_remote_code=False,
            )

        self.assertEqual(prefixes[:2], ["language_model.model.", "language_model."])
        mapping = _key_mapping_for_prefixes(prefixes)
        self.assertEqual(
            mapping[r"^language_model\.model\."],
            "model.",
        )

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

    def test_safetensor_files_include_nested_paths(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "weights"
            nested.mkdir()
            (root / "model.safetensors").touch()
            (nested / "part.safetensors").touch()

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in _safetensor_files(root)],
                ["model.safetensors", "weights/part.safetensors"],
            )

    def test_remote_checkpoint_resolver_downloads_safetensors_when_index_missing(self) -> None:
        import sys
        import tempfile
        import types
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            calls: list[tuple[str, ...]] = []

            fake_hub = types.ModuleType("huggingface_hub")

            def snapshot_download(*, repo_id, local_files_only, allow_patterns):
                del repo_id, local_files_only
                calls.append(tuple(str(pattern) for pattern in allow_patterns))
                if len(calls) == 2:
                    (root / "model.safetensors").touch()
                return str(root)

            fake_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
            original_hub = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = fake_hub
            try:
                source = _resolve_checkpoint_source("example/unindexed", offline=False)
            finally:
                if original_hub is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = original_hub

            self.assertEqual(source.root, root)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("*.safetensors", calls[0])
            self.assertIn("*.safetensors", calls[1])


if __name__ == "__main__":
    unittest.main()
