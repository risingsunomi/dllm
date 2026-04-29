from __future__ import annotations

import json
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

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

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.offline,
                trust_remote_code=self.trust_remote_code,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

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

    def format_chat_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> str:
        self.load()
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
