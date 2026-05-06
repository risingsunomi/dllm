<div align="center">
<pre>
 ######     ##      ##      ##      ##
 ##   ##    ##      ##      ####  ####
 ##    ##   ##      ##      ## #### ##
 ##   ##    ##      ##      ##  ##  ##
 ######     ######  ######  ##      ##
    D I S T R I B U T E D   L L M
</pre>
</div>

# dllm

Standalone shard-first distributed inference server using [PyTorch](https://github.com/pytorch/pytorch). Supports MoE, non-MoE, quantized safetensor checkpoints, and language-only loading for text use of vision-language repositories.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

For bitsandbytes 4-bit checkpoints, install the optional quantized dependency on each node that may load those shards:

```bash
pip install -e '.[quantized]'
```

If you already have a Python environment with [PyTorch](https://github.com/pytorch/pytorch) and [Transformers](https://github.com/huggingface/transformers) installed, activate it and install this package there:

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

## Basic LLM Models

For standard decoder-only LLMs, `dllm` serves text generation through [PyTorch](https://github.com/pytorch/pytorch) and the model's [Transformers](https://github.com/huggingface/transformers) causal language model implementation. On one machine, it behaves like a lightweight local inference server with `/generate` and OpenAI-compatible `/v1/chat/completions` endpoints.

When peers are configured, the same model is split by contiguous transformer layer ranges. The server node keeps the tokenizer, prompt formatting, HTTP API, and first layer shard. Worker nodes load later layer shards. During generation, hidden states move from shard to shard, and the final shard applies the last norm/lm_head and samples the next token. Each node loads only the weights needed for its assigned shard.

Use `/v1/chat/completions` for instruct/chat models so the tokenizer chat template is applied. `/generate` sends the prompt as raw text and is better for simple completion-style tests.

## MoE Models

MoE models are supported in two different runtime modes.

On a single node, `dllm` uses the model's [Transformers](https://github.com/huggingface/transformers) implementation through `AutoModelForCausalLM`. That path is useful when one machine can hold the model, or when you want [Transformers](https://github.com/huggingface/transformers) to place modules across local GPU/CPU/disk with `device_map` and `offload_folder`.

On multiple nodes, `dllm` uses shard-native loading. The server builds a layer plan, each node constructs the model on `meta`, drops layers outside its assigned range, and loads only the checkpoint tensors required by that shard. Expert routing stays inside the model layers loaded on each shard. For packed MoE checkpoints such as GPT-OSS MXFP4 experts, `dllm` maps packed expert tensors to the shard parameters during load.

`device_map` does not distribute a model across machines. It is a local [Transformers](https://github.com/huggingface/transformers) placement control. Multi-node distribution uses device-info probing and shard planning, and requires a server plus one or more worker peers, as shown in the Multiple Nodes section.

Single-node MoE example with local [Transformers](https://github.com/huggingface/transformers) placement:

```bash
python -m dllm.cli serve \
  --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --node-name moe-node \
  --device cuda \
  --dtype bf16 \
  --device-map auto \
  --offload-folder .dllm-offload \
  --attention-implementation sdpa
```

Equivalent `.env` values:

```bash
DLLM_MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
DLLM_DEVICE=cuda
DLLM_DTYPE=bf16
DLLM_DEVICE_MAP=auto
DLLM_OFFLOAD_FOLDER=.dllm-offload
DLLM_ATTENTION_IMPLEMENTATION=sdpa
```

For an FP16-only accuracy and throughput test, enable:

```bash
DLLM_FP16_MODE=true
```

This forces the runtime dtype to `fp16` even when `DLLM_DTYPE=auto` would choose BF16, and
sharded hidden states are serialized as FP16 for peer transport. The server also sends the FP16
mode to workers during shard assignment so all nodes use the same setting.

Distributed MoE example:

```bash
# node-b
python3 -m dllm.cli serve \
  --role worker \
  --model-name openai/gpt-oss-20b \
  --node-name node-b \
  --device auto \
  --peer-host 0.0.0.0 \
  --peer-port 8765 \
  --no-peer-discovery \
  --request-timeout 3600

# node-a
python3 -m dllm.cli serve \
  --role server \
  --model-name openai/gpt-oss-20b \
  --node-name node-a \
  --device auto \
  --host 0.0.0.0 \
  --port 8000 \
  --peers node-b@192.168.0.5:8765 \
  --no-peer-discovery \
  --request-timeout 3600
```

On first use, each node resolves the checkpoint index, determines which safetensor files contain its shard's parameters, and downloads only those files. `GET /health` reports MoE metadata when the loaded model config exposes expert fields such as `num_local_experts`, `num_experts`, or `num_experts_per_tok`, and also reports shard-loading details after weights are loaded. Use `--trust-remote-code` only for models that require custom model code.

## Quantized Checkpoints

Shard-native loading supports common safetensor quantized layouts by decoding only the parameters assigned to the local shard.

GPTQ int4 checkpoints such as [Qwen/Qwen3.5-27B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4) and [Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4) store linear weights as `qweight`, `qzeros`, `scales`, and sometimes `g_idx`. `dllm` maps those packed tensors back to the full parameter name for the assigned shard, including language-only prefixes such as `model.language_model.`, then dequantizes that shard's tensor before loading it into the model. Repos with a safetensor index download only the shard files needed by the assigned layers. Repos that publish a single unindexed `model.safetensors` file must download that file before `dllm` can scan its keys.

bitsandbytes 4-bit checkpoints such as [unsloth/gpt-oss-20b-unsloth-bnb-4bit](https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit) store packed `weight` tensors with sidecar metadata like `weight.absmax`, `weight.quant_map`, `weight.nested_quant_map`, and `weight.quant_state.bitsandbytes__nf4`. For GPT-OSS MoE variants that split experts into keys such as `experts.down_projs.0.weight`, `dllm` reassembles those per-expert tensors into the fused `experts.down_proj` and `experts.gate_up_proj` parameters expected by the model. [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) is installed by default to be used to load these weights into a tensor.

Examples:

```bash
python3 -m dllm.cli serve \
  --role server \
  --model-name Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --node-name node-a \
  --device auto \
  --host 0.0.0.0 \
  --port 8000 \
  --peers node-b@192.168.0.5:8765 \
  --no-peer-discovery \
  --request-timeout 3600 \
  --trust-remote-code
```

```bash
python3 -m dllm.cli serve \
  --role server \
  --model-name unsloth/gpt-oss-20b-unsloth-bnb-4bit \
  --node-name node-a \
  --device auto \
  --host 0.0.0.0 \
  --port 8000 \
  --peers node-b@192.168.0.5:8765 \
  --no-peer-discovery \
  --request-timeout 3600
```

Quantized support is a compatibility path for shard-native loading. It avoids loading the full checkpoint on every node, but the assigned packed tensors are dequantized into the runtime parameter dtype for compute.

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

For nonstandard text checkpoints whose safetensor keys do not use the usual `model.*` prefix, leave them alone when the model's own [Transformers](https://github.com/huggingface/transformers) implementation expects that layout. If you need to adapt such weights to a standard CausalLM class, pass an explicit [Transformers](https://github.com/huggingface/transformers) key map. The left side is a Python regular expression pattern and the right side is its replacement:

```bash
python -m dllm.cli serve \
  --model-name example/nonstandard-text-model \
  --weight-key-mapping '^layers\.=model.layers.,^embed\.weight$=model.embed_tokens.weight,^head\.weight$=lm_head.weight'
```

`--weight-key-mapping auto` can infer common `layers.*`, `embed.*`, `tok_embeddings.*`, `norm.*`, and `head.weight` mappings from safetensors index files such as `model.safetensors.index.json` or another `*.safetensors.index.json` filename.

## OpenAI-Compatible API

The server exposes an OpenAI-compatible chat/completions surface for [Hermes Agent](https://github.com/nousresearch/hermes-agent), OpenAI SDK clients, and similar frameworks:

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

For use with [OpenClaw](https://github.com/openclaw/openclaw), configure `dllm` as an OpenAI-compatible local proxy:

```js
{
  agents: {
    defaults: {
      model: { primary: "dllm/Qwen/Qwen2.5-0.5B-Instruct" },
    },
  },
  models: {
    mode: "merge",
    providers: {
      dllm: {
        baseUrl: "http://127.0.0.1:8000/v1",
        apiKey: "sk-local",
        api: "openai-completions",
        models: [
          {
            id: "Qwen/Qwen2.5-0.5B-Instruct",
            name: "Qwen/Qwen2.5-0.5B-Instruct",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

Use `api: "openai-completions"` because `dllm` exposes `/v1/chat/completions` and `/v1/completions`.

For use with [Hermes Agent](https://github.com/nousresearch/hermes-agent) choose the custom endpoint provider or add a config like:

```yaml
model:
  provider: custom
  default: Qwen/Qwen2.5-0.5B-Instruct
  base_url: http://localhost:8000/v1
  context_length: 32768
```

Local servers can skip the API key. If you want Hermes to auto-detect context length from `/v1/models`, start `dllm` with `--context-length 32768` or set `DLLM_CONTEXT_LENGTH=32768`.

## Multiple Nodes

Multi-node execution is layer sharded. The server node is the first shard and owns the HTTP API; peer nodes load later contiguous transformer layer ranges. During generation, hidden states flow from the server shard through each peer shard, and the final shard applies norm/lm_head and samples the next token.

Startup does not load model weights on either server or worker nodes. On the first inference request, the server formats the prompt, counts prefill tokens, probes peer hardware, builds a shard plan from detected CPU/GPU memory, and sends each worker only its shard assignment. Each node then builds the [Transformers](https://github.com/huggingface/transformers) model on `meta` and loads only the safetensor keys needed for its assigned shard.

`--device-map` is a single-process [Transformers](https://github.com/huggingface/transformers) placement option. It is useful for one node, but it does not distribute a model across machines. For multi-node runs, start workers and pass them to the server as peers; memory-aware shard sizing comes from device-info probing.

Start peer nodes first:

```bash
python3 -m dllm.cli serve \
  --role worker \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --node-name node-b \
  --device auto \
  --peer-host 0.0.0.0 \
  --peer-port 8765 \
  --no-peer-discovery

python3 -m dllm.cli serve \
  --role worker \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --node-name node-c \
  --device auto \
  --peer-host 0.0.0.0 \
  --peer-port 8765 \
  --no-peer-discovery
```

Start the server node. Keep local loading enabled; this node runs the first shard:

```bash
python3 -m dllm.cli serve \
  --role server \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --node-name node-a \
  --device auto \
  --host 0.0.0.0 \
  --port 8000 \
  --peers node-b@192.168.0.5:8765,node-c@192.168.0.6:8765 \
  --no-peer-discovery
```

Shard convention:

- `start_layer` is inclusive.
- `end_layer` is exclusive over transformer blocks.
- `total_layers = num_hidden_layers + 1`.
- The extra virtual layer is the final norm/lm_head stage on the last shard.
- Prefill token count and requested decode length are used in the first-request shard plan.

The server exposes:

- `GET /health`
- `GET /peers`
- `GET /v1/models`
- `POST /generate`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Disable LAN discovery with `--no-peer-discovery` or `DLLM_PEER_DISCOVERY=false`. Startup prints the DLLM banner, resolved LAN IPs, model, role, bind addresses, discovery settings, and peer list. Generation logs include elapsed time and tokens/sec.

Before sending a generation request, verify that the server sees peer hardware. After the first request, `/health` also shows the assigned shards:

```bash
curl -s 'http://127.0.0.1:8000/health?probe=true' | python3 -m json.tool
curl -s http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Write one short sentence.","max_new_tokens":16}' | python3 -m json.tool
```

`--preload` only prepares tokenizer/runtime metadata. It does not load full model weights or assign shards at startup because shard planning depends on the first request's prefill and decode sizes.

## License

Apache License 2.0. See `LICENSE`.
