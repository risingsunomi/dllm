from __future__ import annotations

import json
import re
import time
import threading
import base64
import inspect
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

from .sharding import LayerShard
from .tools import normalize_tools, tool_system_prompt


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    prompt_tokens: int
    elapsed_seconds: float
    model_name: str
    device: str

    def as_dict(self) -> dict[str, Any]:
        tok_s = self.generated_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
        return {
            "text": self.text,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens": self.prompt_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": tok_s,
            "model_name": self.model_name,
            "device": self.device,
        }


@dataclass(frozen=True)
class StreamEvent:
    event: str
    text: str = ""
    generated_tokens: int = 0
    prompt_tokens: int = 0
    elapsed_seconds: float = 0.0
    model_name: str = ""
    device: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"event": self.event}
        if self.text:
            payload["text"] = self.text
        if self.event == "done":
            tok_s = self.generated_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
            payload.update(
                {
                    "generated_tokens": self.generated_tokens,
                    "prompt_tokens": self.prompt_tokens,
                    "elapsed_seconds": self.elapsed_seconds,
                    "tokens_per_second": tok_s,
                    "model_name": self.model_name,
                    "device": self.device,
                }
            )
        return payload


class TorchTransformersEngine:
    """Small PyTorch/Transformers causal language model engine.

    The class is intentionally independent from Cheetah. It owns model loading,
    prompt formatting, and synchronized generation for one process/node.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        dtype: str = "auto",
        offline: bool = False,
        trust_remote_code: bool = False,
        device_map: str = "",
        max_memory: str = "",
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
        self.offline = bool(offline)
        self.trust_remote_code = bool(trust_remote_code)
        self.device_map_name = str(device_map or "")
        self.max_memory = str(max_memory or "")
        self.offload_folder = str(offload_folder or "")
        self.attention_implementation = str(attention_implementation or "")
        self.language_only = bool(language_only)
        self.language_weight_prefix = str(language_weight_prefix or "auto")
        self.weight_key_mapping = str(weight_key_mapping or "")
        self.shard = shard
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.model_config: dict[str, Any] = {}
        self.source_config: dict[str, Any] = {}
        self.language_loading: dict[str, Any] = {}
        self.device = "unloaded"
        self.input_device = "cpu"
        self.loaded = False
        self._lock = threading.RLock()

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if not self.model_name:
                raise ValueError("model_name is required before loading the model")

            torch = _import_torch()
            transformers = _import_transformers()
            device = _resolve_device(self.requested_device, torch)
            dtype = _resolve_dtype(self.dtype_name, device, torch)
            device_map = _parse_device_map(self.device_map_name)
            base_config = transformers.AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=self.offline,
                trust_remote_code=self.trust_remote_code,
            )
            language_plan = _language_loading_plan(
                transformers,
                self.model_name,
                base_config,
                language_only=self.language_only,
                language_weight_prefix=self.language_weight_prefix,
                offline=self.offline,
                trust_remote_code=self.trust_remote_code,
            )
            key_mapping = dict(language_plan.get("key_mapping", {}))
            key_mapping.update(
                _parse_weight_key_mapping(
                    self.weight_key_mapping,
                    self.model_name,
                    offline=self.offline,
                )
            )

            tokenizer = self._load_tokenizer(transformers)

            model_kwargs: dict[str, Any] = {
                "local_files_only": self.offline,
                "trust_remote_code": self.trust_remote_code,
                "config": language_plan.get("config") or base_config,
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            if device_map is not None:
                model_kwargs["device_map"] = device_map
                max_memory = _parse_max_memory(self.max_memory)
                if max_memory:
                    model_kwargs["max_memory"] = max_memory
                if self.offload_folder:
                    model_kwargs["offload_folder"] = self.offload_folder
            if self.attention_implementation:
                model_kwargs["attn_implementation"] = self.attention_implementation
            if key_mapping:
                model_kwargs["key_mapping"] = key_mapping

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            if self.shard is not None:
                _apply_loaded_shard(model, self.shard, torch)
            if device_map is None:
                model.to(device)
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            self.source_config = _config_to_dict(base_config)
            self.model_config = _model_config_dict(model)
            self.language_loading = language_plan.get("metadata", {})
            self.input_device = _model_input_device(model, fallback=device)
            self.device = _runtime_device_label(model, fallback=device, device_map=device_map)
            self.loaded = True

    def load_tokenizer(self) -> None:
        with self._lock:
            transformers = _import_transformers()
            self._load_tokenizer(transformers)

    def _load_tokenizer(self, transformers: Any) -> Any:
        if self.tokenizer is not None:
            return self.tokenizer
        if not self.model_name:
            raise ValueError("model_name is required before loading the tokenizer")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.offline,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        return tokenizer

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.tokenizer = None
            self.model_config = {}
            self.source_config = {}
            self.language_loading = {}
            self.loaded = False
            self.device = "unloaded"
            self.input_device = "cpu"
            try:
                _import_torch().cuda.empty_cache()
            except Exception:
                pass

    def set_shard(self, shard: LayerShard | None) -> None:
        with self._lock:
            current = self.shard.as_dict() if self.shard is not None else None
            requested = shard.as_dict() if shard is not None else None
            if current == requested:
                return
            if self.loaded:
                self.unload()
            self.shard = shard

    def format_chat_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> str:
        self.load_tokenizer()
        assert self.tokenizer is not None
        normalized_tools = normalize_tools(tools)
        if getattr(self.tokenizer, "chat_template", None):
            if normalized_tools:
                for tool_payload in (
                    normalized_tools,
                    [tool["function"] for tool in normalized_tools if isinstance(tool.get("function"), dict)],
                ):
                    try:
                        return self.tokenizer.apply_chat_template(
                            messages,
                            tools=tool_payload,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        pass
            return self.tokenizer.apply_chat_template(
                _messages_with_tool_prompt(messages, normalized_tools, tool_choice),
                tokenize=False,
                add_generation_prompt=True,
            )

        rendered: list[str] = []
        for message in _messages_with_tool_prompt(messages, normalized_tools, tool_choice):
            role = str(message.get("role", "user")).strip() or "user"
            content = str(message.get("content", ""))
            rendered.append(f"{role}: {content}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        torch = _import_torch()
        transformers = _import_transformers()

        with self._lock:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
            encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max(int(request.max_new_tokens), 1),
                "repetition_penalty": max(float(request.repetition_penalty), 0.01),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            stopping_criteria = _stop_criteria(
                transformers,
                self.tokenizer,
                prompt_tokens,
                request.stop,
            )
            if stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = stopping_criteria

            temperature = float(request.temperature)
            if temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = min(max(float(request.top_p), 0.0), 1.0)
                if int(request.top_k) > 0:
                    generate_kwargs["top_k"] = int(request.top_k)
            else:
                generate_kwargs["do_sample"] = False

            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = self.model.generate(**encoded, **generate_kwargs)
            elapsed = time.perf_counter() - start

            generated_ids = output_ids[0, prompt_tokens:]
            generated_tokens = int(generated_ids.shape[-1])
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            text = _apply_stop(text, request.stop)

        return GenerationResult(
            text=text,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            elapsed_seconds=elapsed,
            model_name=self.model_name,
            device=self.device,
        )

    def generate_stream(self, request: GenerationRequest):
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        torch = _import_torch()
        transformers = _import_transformers()

        with self._lock:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
            encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            streamer = transformers.TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs: dict[str, Any] = {
                **encoded,
                "streamer": streamer,
                "max_new_tokens": max(int(request.max_new_tokens), 1),
                "repetition_penalty": max(float(request.repetition_penalty), 0.01),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            stopping_criteria = _stop_criteria(
                transformers,
                self.tokenizer,
                prompt_tokens,
                request.stop,
            )
            if stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = stopping_criteria

            temperature = float(request.temperature)
            if temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = min(max(float(request.top_p), 0.0), 1.0)
                if int(request.top_k) > 0:
                    generate_kwargs["top_k"] = int(request.top_k)
            else:
                generate_kwargs["do_sample"] = False

            state: dict[str, Any] = {"output_ids": None, "error": None}

            def run_generate() -> None:
                try:
                    with torch.inference_mode():
                        state["output_ids"] = self.model.generate(**generate_kwargs)
                except Exception as exc:
                    state["error"] = exc
                    try:
                        streamer.on_finalized_text("", stream_end=True)
                    except Exception:
                        pass

            start = time.perf_counter()
            thread = Thread(target=run_generate, name="dllm-generate-stream", daemon=True)
            thread.start()

            full_text_parts: list[str] = []
            stop_filter = _StopFilter(request.stop)
            for piece in streamer:
                filtered = stop_filter.push(str(piece))
                if filtered:
                    full_text_parts.append(filtered)
                    yield StreamEvent(event="token", text=filtered).as_dict()
                if stop_filter.stopped:
                    break

            trailing = stop_filter.finish()
            if trailing:
                full_text_parts.append(trailing)
                yield StreamEvent(event="token", text=trailing).as_dict()

            thread.join()
            if state["error"] is not None:
                raise state["error"]

            elapsed = time.perf_counter() - start
            output_ids = state.get("output_ids")
            if output_ids is not None:
                generated_tokens = int(output_ids[0, prompt_tokens:].shape[-1])
            else:
                generated_tokens = len(self.tokenizer.encode("".join(full_text_parts), add_special_tokens=False))

            yield StreamEvent(
                event="done",
                text="".join(full_text_parts),
                generated_tokens=generated_tokens,
                prompt_tokens=prompt_tokens,
                elapsed_seconds=elapsed,
                model_name=self.model_name,
                device=self.device,
            ).as_dict()

    def encode_inputs(self, prompt: str) -> dict[str, Any]:
        self.load_tokenizer()
        assert self.tokenizer is not None
        torch = _import_torch()

        with self._lock:
            encoded = self.tokenizer(str(prompt), return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.input_device, dtype=torch.long)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = attention_mask.to(self.input_device, dtype=torch.long)
            position_ids = _position_ids_from_attention(attention_mask, torch)
            return {
                "input_ids": _encode_tensor(input_ids),
                "attention_mask": _encode_tensor(attention_mask),
                "position_ids": _encode_tensor(position_ids),
                "prompt_tokens": int(input_ids.shape[-1]),
                "seen_tokens": [int(value) for value in input_ids.detach().cpu().reshape(-1).tolist()],
            }

    def append_token_to_inputs(self, encoded_inputs: dict[str, Any], token_id: int) -> dict[str, Any]:
        torch = _import_torch()
        with self._lock:
            input_ids = _decode_tensor(encoded_inputs.get("input_ids"), torch, self.input_device)
            attention_mask = _decode_tensor(encoded_inputs.get("attention_mask"), torch, self.input_device)
            if input_ids is None or attention_mask is None:
                raise ValueError("encoded input_ids and attention_mask are required")
            next_token = torch.tensor([[int(token_id)]], device=self.input_device, dtype=torch.long)
            input_ids = torch.cat((input_ids.to(dtype=torch.long), next_token), dim=1)
            next_mask = torch.ones((attention_mask.shape[0], 1), device=self.input_device, dtype=torch.long)
            attention_mask = torch.cat((attention_mask.to(dtype=torch.long), next_mask), dim=1)
            position_ids = _position_ids_from_attention(attention_mask, torch)
            return {
                "input_ids": _encode_tensor(input_ids),
                "attention_mask": _encode_tensor(attention_mask),
                "position_ids": _encode_tensor(position_ids),
                "prompt_tokens": int(encoded_inputs.get("prompt_tokens", input_ids.shape[-1])),
                "seen_tokens": [int(value) for value in input_ids.detach().cpu().reshape(-1).tolist()],
            }

    def decode_token_ids(self, token_ids: list[int]) -> str:
        self.load_tokenizer()
        assert self.tokenizer is not None
        return self.tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True)

    def count_prompt_tokens(self, prompt: str) -> int:
        self.load_tokenizer()
        assert self.tokenizer is not None
        with self._lock:
            encoded = self.tokenizer(str(prompt), return_tensors="pt")
            return int(encoded["input_ids"].shape[-1])

    def forward_shard(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        if self.shard is None:
            raise RuntimeError("forward_shard requires an assigned layer shard")

        torch = _import_torch()
        request = request_from_payload(payload, {})
        seen_tokens = _int_list(payload.get("seen_tokens", []))

        with self._lock:
            input_ids = _decode_tensor(payload.get("input_ids"), torch, self.input_device)
            attention_mask = _decode_tensor(payload.get("attention_mask"), torch, self.input_device)
            position_ids = _decode_tensor(payload.get("position_ids"), torch, self.input_device)
            hidden_state = _decode_tensor(payload.get("hidden_state"), torch, self.input_device)

            if input_ids is not None:
                input_ids = input_ids.to(self.input_device, dtype=torch.long)
            if attention_mask is None:
                if input_ids is None:
                    raise ValueError("attention_mask is required when forwarding a hidden_state")
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = attention_mask.to(self.input_device, dtype=torch.long)
            if position_ids is None:
                position_ids = _position_ids_from_attention(attention_mask, torch)
            else:
                position_ids = position_ids.to(self.input_device, dtype=torch.long)
            if hidden_state is not None:
                hidden_state = hidden_state.to(self.input_device)

            start = time.perf_counter()
            with torch.inference_mode():
                output = _run_loaded_shard(
                    self.model,
                    shard=self.shard,
                    input_ids=input_ids,
                    hidden_state=hidden_state,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                elapsed = time.perf_counter() - start

                if not self.shard.is_final:
                    return {
                        "hidden_state": _encode_tensor(output),
                        "attention_mask": _encode_tensor(attention_mask),
                        "position_ids": _encode_tensor(position_ids),
                        "shard": self.shard.as_dict(),
                        "elapsed_seconds": elapsed,
                    }

                token_id = _sample_next_token(
                    output[:, -1, :],
                    request,
                    seen_tokens=seen_tokens,
                    torch=torch,
                )
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                return {
                    "token_id": int(token_id),
                    "end_token": eos_token_id is not None and int(token_id) == int(eos_token_id),
                    "shard": self.shard.as_dict(),
                    "elapsed_seconds": elapsed,
                }

    def health(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "device": self.device,
            "input_device": self.input_device,
            "dtype": self.dtype_name,
            "device_map": self.device_map_name,
            "max_memory": self.max_memory,
            "offload_folder": self.offload_folder,
            "attention_implementation": self.attention_implementation,
            "language_only": self.language_only,
            "language_weight_prefix": self.language_weight_prefix,
            "weight_key_mapping": self.weight_key_mapping,
            "language_loading": self.language_loading,
            "shard": self.shard.as_dict() if self.shard is not None else None,
            "loaded": self.loaded,
            "offline": self.offline,
            "model_type": self.model_config.get("model_type", ""),
            "moe": _moe_metadata(self.model_config),
        }


def request_from_payload(payload: dict[str, Any], defaults: dict[str, Any]) -> GenerationRequest:
    stop_value = payload.get("stop", ())
    if isinstance(stop_value, str):
        stop = (stop_value,)
    elif isinstance(stop_value, list):
        stop = tuple(str(item) for item in stop_value if str(item))
    else:
        stop = ()

    return GenerationRequest(
        prompt=str(payload.get("prompt", "")),
        max_new_tokens=_int(payload.get("max_new_tokens", defaults.get("max_new_tokens", 128)), 128),
        temperature=_float(payload.get("temperature", defaults.get("temperature", 0.0)), 0.0),
        top_p=_float(payload.get("top_p", defaults.get("top_p", 1.0)), 1.0),
        top_k=_int(payload.get("top_k", defaults.get("top_k", 0)), 0),
        repetition_penalty=_float(
            payload.get("repetition_penalty", defaults.get("repetition_penalty", 1.0)),
            1.0,
        ),
        stop=stop,
    )


def _encode_tensor(tensor: Any) -> dict[str, Any]:
    try:
        from safetensors.torch import save
    except Exception as exc:
        raise RuntimeError("safetensors is required for sharded tensor transport") from exc
    payload = save({"tensor": tensor.detach().cpu().contiguous()})
    return {
        "format": "safetensors",
        "buffer": base64.b64encode(payload).decode("ascii"),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def _decode_tensor(payload: Any, torch: Any, device: str) -> Any | None:
    if not isinstance(payload, dict):
        return None
    buffer = payload.get("buffer")
    if not isinstance(buffer, str) or not buffer:
        return None
    if str(payload.get("format", "safetensors")) != "safetensors":
        raise ValueError("unsupported tensor payload format")
    try:
        from safetensors.torch import load
    except Exception as exc:
        raise RuntimeError("safetensors is required for sharded tensor transport") from exc
    tensors = load(base64.b64decode(buffer.encode("ascii")))
    tensor = tensors.get("tensor")
    if tensor is None:
        return None
    return tensor.to(device)


def _position_ids_from_attention(attention_mask: Any, torch: Any) -> Any:
    mask = attention_mask.to(dtype=torch.long)
    return (mask.cumsum(dim=1) - 1).clamp(min=0) * mask


def _run_loaded_shard(
    model: Any,
    *,
    shard: LayerShard,
    input_ids: Any | None,
    hidden_state: Any | None,
    attention_mask: Any,
    position_ids: Any,
) -> Any:
    base_model = _causal_base_model(model)
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    if hidden_state is not None:
        kwargs["inputs_embeds"] = hidden_state
    else:
        if input_ids is None:
            raise ValueError("input_ids is required for the first shard")
        kwargs["input_ids"] = input_ids

    outputs = _call_with_supported_kwargs(base_model, kwargs)
    hidden = _last_hidden_state(outputs)
    if not shard.is_final:
        return hidden

    output_embeddings = _output_embeddings(model)
    if output_embeddings is None:
        raise RuntimeError("model does not expose output embeddings for final shard sampling")
    return output_embeddings(hidden)


def _apply_loaded_shard(model: Any, shard: LayerShard, torch: Any) -> None:
    layer_info = _decoder_layers(model)
    if layer_info is None:
        raise RuntimeError("could not locate decoder layers for sharded loading")
    _, layers = layer_info
    start = max(int(shard.start_layer), 0)
    end = min(int(shard.end_layer), len(layers))
    if start > end:
        raise RuntimeError(f"invalid shard range {start}:{end} for {len(layers)} decoder layers")

    for index in range(len(layers)):
        if start <= index < end:
            continue
        layers[index] = _passthrough_decoder_layer(torch)

    if not shard.is_final:
        norm_info = _final_norm_module(model)
        if norm_info is not None:
            parent, name = norm_info
            setattr(parent, name, torch.nn.Identity())


def _passthrough_decoder_layer(torch: Any) -> Any:
    class PassthroughDecoderLayer(torch.nn.Module):
        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> tuple[Any]:
            del args, kwargs
            return (hidden_states,)

    return PassthroughDecoderLayer()


def _causal_base_model(model: Any) -> Any:
    base_model_prefix = getattr(model, "base_model_prefix", "")
    if base_model_prefix:
        candidate = getattr(model, base_model_prefix, None)
        if candidate is not None:
            return candidate
    for attr in ("model", "transformer", "gpt_neox", "base_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate is not model:
            return candidate
    return model


def _decoder_layers(model: Any) -> tuple[Any, Any] | None:
    for path in (
        "model.layers",
        "model.decoder.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "base_model.layers",
    ):
        resolved = _resolve_parent_attr(model, path)
        if resolved is None:
            continue
        parent, name = resolved
        layers = getattr(parent, name, None)
        if layers is not None and hasattr(layers, "__len__") and hasattr(layers, "__setitem__"):
            return parent, layers
    return None


def _final_norm_module(model: Any) -> tuple[Any, str] | None:
    for path in (
        "model.norm",
        "model.decoder.final_layer_norm",
        "model.final_layernorm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "base_model.norm",
    ):
        resolved = _resolve_parent_attr(model, path)
        if resolved is None:
            continue
        parent, name = resolved
        if getattr(parent, name, None) is not None:
            return parent, name
    return None


def _resolve_parent_attr(root: Any, path: str) -> tuple[Any, str] | None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        return None
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    return parent, parts[-1]


def _call_with_supported_kwargs(module: Any, kwargs: dict[str, Any]) -> Any:
    forward = getattr(module, "forward", module)
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return module(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return module(**kwargs)
    accepted = {name for name in signature.parameters}
    return module(**{key: value for key, value in kwargs.items() if key in accepted})


def _last_hidden_state(outputs: Any) -> Any:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise RuntimeError("shard forward did not return hidden states")


def _output_embeddings(model: Any) -> Any | None:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output = getter()
        if output is not None:
            return output
    for attr in ("lm_head", "embed_out", "output"):
        output = getattr(model, attr, None)
        if output is not None:
            return output
    return None


def _sample_next_token(
    logits: Any,
    request: GenerationRequest,
    *,
    seen_tokens: list[int],
    torch: Any,
) -> int:
    scores = logits[0].detach().float().clone()
    penalty = max(float(request.repetition_penalty), 0.01)
    if penalty != 1.0 and seen_tokens:
        vocab_size = int(scores.shape[-1])
        for token_id in set(seen_tokens):
            if 0 <= int(token_id) < vocab_size:
                if scores[token_id] < 0:
                    scores[token_id] *= penalty
                else:
                    scores[token_id] /= penalty

    temperature = float(request.temperature)
    if temperature <= 0:
        return int(torch.argmax(scores).item())

    scores = scores / max(temperature, 1e-5)
    top_k = int(request.top_k)
    if top_k > 0 and top_k < int(scores.shape[-1]):
        threshold = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    top_p = min(max(float(request.top_p), 0.0), 1.0)
    if 0 < top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = float("-inf")

    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
        return int(torch.argmax(logits[0]).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


class _StopFilter:
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


def _messages_with_tool_prompt(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: Any,
) -> list[dict[str, str]]:
    prompt = tool_system_prompt(tools, tool_choice)
    if not prompt:
        return messages
    return [{"role": "system", "content": prompt}, *messages]


def _apply_stop(text: str, stop: tuple[str, ...]) -> str:
    if not stop:
        return text
    cut = len(text)
    for marker in stop:
        if not marker:
            continue
        index = text.find(marker)
        if index >= 0:
            cut = min(cut, index)
    return text[:cut]


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


def _stop_criteria(
    transformers: Any,
    tokenizer: Any,
    prompt_tokens: int,
    stop: tuple[str, ...],
) -> Any | None:
    markers = tuple(marker for marker in stop if marker)
    if not markers:
        return None

    class StopOnStrings(transformers.StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            del scores, kwargs
            try:
                generated = input_ids[0, int(prompt_tokens) :]
                text = tokenizer.decode(generated, skip_special_tokens=True)
            except Exception:
                return False
            return any(marker in text for marker in markers)

    return transformers.StoppingCriteriaList([StopOnStrings()])


def _resolve_device(requested: str, torch: Any) -> str:
    device = str(requested or "cpu").strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "metal":
        return "mps"
    return device


def _resolve_dtype(dtype_name: str, device: str, torch: Any) -> Any | None:
    name = str(dtype_name or "auto").strip().lower()
    if name in {"", "auto"}:
        if device.startswith("cuda"):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32
    if name in {"none", "default"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {dtype_name!r}; use auto, fp32, fp16, or bf16")
    return mapping[name]


def _parse_device_map(value: str) -> Any | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "false", "off", "no"}:
        return None
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("device_map JSON must be an object")
        return parsed
    return text


def _parse_max_memory(value: str) -> dict[Any, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("max_memory JSON must be an object")
        return {_memory_key(key): str(amount) for key, amount in parsed.items()}

    result: dict[Any, str] = {}
    for chunk in text.replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("max_memory entries must use key=value syntax")
        key, amount = item.split("=", 1)
        result[_memory_key(key.strip())] = amount.strip()
    return result


def _parse_weight_key_mapping(value: str, model_name: str, *, offline: bool) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    if text.lower() == "auto":
        return _auto_weight_key_mapping(model_name, offline=offline)
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("weight_key_mapping JSON must be an object")
        return {str(source): str(target) for source, target in parsed.items()}

    mapping: dict[str, str] = {}
    for chunk in text.replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("weight_key_mapping entries must use source=target syntax")
        source, target = item.split("=", 1)
        mapping[source.strip()] = target.strip()
    return mapping


def _auto_weight_key_mapping(model_name: str, *, offline: bool) -> dict[str, str]:
    keys = _weight_index_sample_keys(model_name, offline=offline, trust_remote_code=False)
    if not keys:
        return {}

    mapping: dict[str, str] = {}
    if any(key.startswith("layers.") for key in keys) and not any(key.startswith("model.layers.") for key in keys):
        mapping[r"^layers\."] = "model.layers."
    if any(key.startswith("embed.") for key in keys):
        mapping[r"^embed\."] = "model.embed_tokens."
    if any(key == "embed.weight" for key in keys):
        mapping[r"^embed\.weight$"] = "model.embed_tokens.weight"
    if any(key.startswith("tok_embeddings.") for key in keys):
        mapping[r"^tok_embeddings\."] = "model.embed_tokens."
    if any(key.startswith("norm.") for key in keys) and not any(key.startswith("model.norm.") for key in keys):
        mapping[r"^norm\."] = "model.norm."
    if any(key == "head.weight" for key in keys):
        mapping[r"^head\.weight$"] = "lm_head.weight"
    if any(key == "output.weight" for key in keys):
        mapping[r"^output\.weight$"] = "lm_head.weight"
    return mapping


def _language_loading_plan(
    transformers: Any,
    model_name: str,
    base_config: Any,
    *,
    language_only: bool,
    language_weight_prefix: str,
    offline: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    base_config_dict = _config_to_dict(base_config)
    text_config_dict = _nested_language_config_dict(base_config)
    if not language_only or not text_config_dict:
        return {
            "config": None,
            "key_mapping": {},
            "metadata": {
                "active": False,
                "reason": "no nested language config" if language_only else "disabled",
                "source_model_type": base_config_dict.get("model_type", ""),
            },
        }

    text_model_type = str(text_config_dict.get("model_type", "") or "").strip()
    if not text_model_type:
        return {
            "config": None,
            "key_mapping": {},
            "metadata": {
                "active": False,
                "reason": "nested language config has no model_type",
                "source_model_type": base_config_dict.get("model_type", ""),
            },
        }

    language_config = transformers.AutoConfig.for_model(text_model_type, **text_config_dict)
    prefixes = _language_weight_prefixes(
        model_name,
        explicit_prefix=language_weight_prefix,
        offline=offline,
        trust_remote_code=trust_remote_code,
    )
    key_mapping = _key_mapping_for_prefixes(prefixes)
    return {
        "config": language_config,
        "key_mapping": key_mapping,
        "metadata": {
            "active": True,
            "source_model_type": base_config_dict.get("model_type", ""),
            "language_model_type": text_model_type,
            "weight_prefixes": prefixes,
            "ignored_vision": bool(base_config_dict.get("vision_config")),
        },
    }


def _nested_language_config_dict(config: Any) -> dict[str, Any]:
    for attr in ("text_config", "language_config", "llm_config"):
        nested = getattr(config, attr, None)
        if nested is None and isinstance(config, dict):
            nested = config.get(attr)
        nested_dict = _config_to_dict(nested)
        if nested_dict:
            return nested_dict
    config_dict = _config_to_dict(config)
    for key in ("text_config", "language_config", "llm_config"):
        nested = config_dict.get(key)
        if isinstance(nested, dict) and nested:
            return dict(nested)
    return {}


def _language_weight_prefixes(
    model_name: str,
    *,
    explicit_prefix: str,
    offline: bool,
    trust_remote_code: bool,
) -> list[str]:
    normalized = str(explicit_prefix or "auto").strip()
    if normalized and normalized.lower() not in {"auto", "none", "false", "off"}:
        return [_ensure_trailing_dot(normalized)]
    if normalized.lower() in {"none", "false", "off"}:
        return []

    keys = _weight_index_sample_keys(
        model_name,
        offline=offline,
        trust_remote_code=trust_remote_code,
    )
    candidates = (
        "model.language_model.",
        "language_model.",
        "model.text_model.",
        "text_model.",
        "model.llm.",
        "llm.",
    )
    found = [prefix for prefix in candidates if any(key.startswith(prefix) for key in keys)]
    return found or ["model.language_model.", "language_model."]


def _weight_index_sample_keys(
    model_name: str,
    *,
    offline: bool,
    trust_remote_code: bool,
    limit: int = 4096,
) -> list[str]:
    del trust_remote_code
    index_path = _resolve_weight_index_path(model_name, offline=offline)
    if index_path is None:
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return []
    return [str(key) for key in list(weight_map.keys())[:limit]]


def _resolve_weight_index_path(model_name: str, *, offline: bool) -> Path | None:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return _preferred_weight_index_path(candidate)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception:
        return None
    try:
        preferred_path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors.index.json",
            local_files_only=offline,
        )
        return Path(preferred_path)
    except Exception:
        if offline:
            return None
    try:
        repo_files = list_repo_files(repo_id=model_name)
    except Exception:
        return None
    index_name = _preferred_weight_index_name(repo_files)
    if index_name is None:
        return None
    try:
        path = hf_hub_download(
            repo_id=model_name,
            filename=index_name,
            local_files_only=offline,
        )
    except Exception:
        return None
    return Path(path)


def _preferred_weight_index_path(folder: Path) -> Path | None:
    preferred = folder / "model.safetensors.index.json"
    if preferred.exists():
        return preferred
    candidates = [path for path in folder.glob("*.safetensors.index.json") if path.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.name != "model.safetensors.index.json", path.name))[0]


def _preferred_weight_index_name(repo_files: list[str]) -> str | None:
    preferred = "model.safetensors.index.json"
    if preferred in repo_files:
        return preferred
    candidates = [name for name in repo_files if name.endswith(".safetensors.index.json")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: ("/" in name, name))[0]


def _key_mapping_for_prefixes(prefixes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for prefix in prefixes:
        clean_prefix = _ensure_trailing_dot(prefix)
        for source, target in _prefix_mapping_entries(clean_prefix):
            mapping[source] = target
    return mapping


def _prefix_mapping_entries(prefix: str) -> list[tuple[str, str]]:
    escaped = re.escape(prefix)
    return [
        (f"^{escaped}", "model."),
    ]


def _ensure_trailing_dot(value: str) -> str:
    text = str(value or "").strip()
    if text and not text.endswith("."):
        return f"{text}."
    return text


def _memory_key(value: Any) -> Any:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def _model_config_dict(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return _config_to_dict(config)


def _config_to_dict(config: Any) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return {}
    if isinstance(config, dict):
        return dict(config)
    return {}


def _model_input_device(model: Any, *, fallback: str) -> str:
    try:
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None and str(device) != "meta":
            return str(device)
    except Exception:
        pass

    try:
        first_parameter = next(model.parameters())
        device = getattr(first_parameter, "device", None)
        if device is not None and str(device) != "meta":
            return str(device)
    except Exception:
        pass
    return fallback


def _runtime_device_label(model: Any, *, fallback: str, device_map: Any | None) -> str:
    if device_map is None:
        return fallback
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        values = sorted({str(value) for value in hf_device_map.values()})
        return f"device_map:{','.join(values)}"
    return f"device_map:{device_map}"


def _moe_metadata(config: dict[str, Any]) -> dict[str, Any]:
    expert_keys = (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "num_experts_per_token",
        "n_shared_experts",
        "moe_intermediate_size",
        "router_aux_loss_coef",
    )
    values = {key: config.get(key) for key in expert_keys if key in config}
    model_type = str(config.get("model_type", "") or "").lower()
    is_moe = bool(values) or "moe" in model_type or "mixtral" in model_type
    return {
        "enabled": is_moe,
        "model_type": config.get("model_type", ""),
        **values,
    }


def _import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("PyTorch is required for inference; install torch first") from exc


def _import_transformers() -> Any:
    try:
        import transformers

        return transformers
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("Transformers is required for inference; install transformers first") from exc


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
