# dllm

Standalone distributed PyTorch inference server. It does not import Cheetah at runtime; the implementation is a small server and peer engine in this repository.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

If you already have a Python environment with PyTorch and Transformers installed, activate it and install this package there:

```bash
python -m pip install -e .
python -m dllm.cli serve --model-name Qwen/Qwen2.5-0.5B-Instruct --device cpu
```

## Single Node

```bash
cp .env.example .env
python -m dllm.cli serve --model-name Qwen/Qwen2.5-0.5B-Instruct --node-name node-a --device cpu
```

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Write a concise test sentence.","max_new_tokens":32}'
```

## MoE Models

MoE architectures are supported through PyTorch + Transformers `AutoModelForCausalLM`. The model implementation handles expert routing, while `dllm` handles serving, streaming, tool calls, and worker distribution.

For large MoE models, use Transformers device mapping and optional CPU/disk offload:

```bash
python -m dllm.cli serve \
  --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --node-name moe-node \
  --device cuda \
  --dtype bf16 \
  --device-map auto \
  --max-memory '0=20GiB,cpu=80GiB' \
  --offload-folder .dllm-offload \
  --attention-implementation sdpa
```

Equivalent `.env` values:

```bash
DLLM_MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
DLLM_DEVICE=cuda
DLLM_DTYPE=bf16
DLLM_DEVICE_MAP=auto
DLLM_MAX_MEMORY=0=20GiB,cpu=80GiB
DLLM_OFFLOAD_FOLDER=.dllm-offload
DLLM_ATTENTION_IMPLEMENTATION=sdpa
```

Use `--trust-remote-code` only for models that require custom model code. `GET /health` reports MoE metadata when the loaded model config exposes expert fields such as `num_local_experts`, `num_experts`, or `num_experts_per_tok`.

## Vision-Language Repos As Language Models

Some current image/video-text repos package a vision tower and a language model in one checkpoint. `dllm` defaults to language-only loading when a repo exposes a nested language config such as `text_config`, `language_config`, or `llm_config`. This avoids loading the vision tower for text-only serving.

For repos whose safetensor keys put the language stack under a prefix like `model.language_model.`, keep the default auto prefix detection:

```bash
python -m dllm.cli serve \
  --model-name Qwen/Qwen3.6-27B \
  --device cuda \
  --dtype bf16 \
  --device-map auto \
  --language-only \
  --language-weight-prefix auto
```

If a repo uses a different prefix, set it explicitly:

```bash
python -m dllm.cli serve \
  --model-name vendor/vision-language-model \
  --language-only \
  --language-weight-prefix language_model.
```

For nonstandard text checkpoints whose safetensor keys do not use the usual `model.*` prefix, leave them alone when the model's own Transformers implementation expects that layout. If you need to adapt such weights to a standard CausalLM class, pass an explicit Transformers key map. The left side is a Python regular expression pattern and the right side is its replacement:

```bash
python -m dllm.cli serve \
  --model-name example/nonstandard-text-model \
  --weight-key-mapping '^layers\.=model.layers.,^embed\.weight$=model.embed_tokens.weight,^head\.weight$=lm_head.weight'
```

`--weight-key-mapping auto` can infer common `layers.*`, `embed.*`, `tok_embeddings.*`, `norm.*`, and `head.weight` mappings from safetensors index files such as `model.safetensors.index.json` or another `*.safetensors.index.json` filename.

## OpenAI-Compatible API

The server exposes an OpenAI-compatible chat/completions surface for Hermes Agent, OpenAI SDK clients, and similar frameworks:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Example:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Write a concise test sentence."}],
    "max_tokens": 32
  }'
```

Token-by-token streaming uses standard server-sent events:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 32,
    "stream": true
  }'
```

Native tool calling accepts OpenAI `tools` and returns OpenAI `tool_calls`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is in README.md?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a local text file.",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }]
  }'
```

For Hermes Agent, choose the custom endpoint provider or add a config like:

```yaml
model:
  provider: custom
  default: Qwen/Qwen2.5-0.5B-Instruct
  base_url: http://localhost:8000/v1
  context_length: 32768
```

Local servers can skip the API key. If you want Hermes to auto-detect context length from `/v1/models`, start `dllm` with `--context-length 32768` or set `DLLM_CONTEXT_LENGTH=32768`.

## Multiple Nodes

Start worker nodes:

```bash
python -m dllm.cli serve --role worker --model-name Qwen/Qwen2.5-0.5B-Instruct --node-name node-b --device cuda --peer-host 0.0.0.0 --peer-port 8765
python -m dllm.cli serve --role worker --model-name Qwen/Qwen2.5-0.5B-Instruct --node-name node-c --device cuda --peer-host 0.0.0.0 --peer-port 8765
```

Start a coordinator:

```bash
python -m dllm.cli serve \
  --role server \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --node-name coordinator \
  --device cpu \
  --no-load-local \
  --peers node-b@192.168.0.5:8765,node-c@192.168.0.6:8765
```

The coordinator exposes:

- `GET /health`
- `GET /peers`
- `GET /v1/models`
- `POST /generate`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Requests are load-balanced across available workers. A single request is generated by one worker; multiple nodes increase serving capacity and provide failover. Streaming works through worker nodes as well as local generation. This is request-level distribution, not tensor/model sharding inside one request.

## License

Apache License 2.0. See `LICENSE`.
