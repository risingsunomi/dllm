# dllm v1.0

## Summary

dllm v1.0 is a shard-first distributed PyTorch inference server for local and LAN-hosted LLM serving. The server node owns the HTTP/OpenAI-compatible API and runs the first model shard; peer nodes are discovered or configured explicitly and run later contiguous transformer layer shards. Hidden states are passed through the peer chain for each generation step, and the final shard samples tokens.

## Highlights

- Shard-first distributed inference with Cheetah-style layer ranges: `start_layer` inclusive, `end_layer` exclusive, `total_layers = num_hidden_layers + 1`.
- LAN peer discovery over UDP probes, with manual `--peers node@host:port` still supported and merged with discovered peers.
- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints.
- Token-by-token SSE streaming and native OpenAI-style tool calling.
- PyTorch + Transformers backend with MoE model support through model-native expert routing.
- Language-only loading for VLM repositories with nested text configs and separate language checkpoint prefixes.
- Safetensors-based hidden-state transport between shards.
- Startup banner with node role, bind addresses, LAN IPs, model, discovery settings, and peer list.
- Generation logs include elapsed time and tokens/sec.
- Apache 2.0 licensed.

## Breaking Changes

- Multi-node serving is always sharded. The earlier replica/load-balancing mode and `--distribution-mode` option were removed.
- Server nodes must keep local loading enabled because the server runs the first shard.

## Quick Start

Start peer nodes:

```bash
python -m dllm.cli serve --role worker --model-name Qwen/Qwen2.5-0.5B-Instruct --node-name node-b --device cuda --peer-host 0.0.0.0 --peer-port 8765
```

Start the server node:

```bash
python -m dllm.cli serve --role server --model-name Qwen/Qwen2.5-0.5B-Instruct --node-name node-a --device cuda
```

Manual peers can be supplied when LAN discovery is not enough:

```bash
python -m dllm.cli serve --role server --model-name Qwen/Qwen2.5-0.5B-Instruct --node-name node-a --device cuda --peers node-b@192.168.0.5:8765
```

## Verification

Validated locally with:

```bash
python3 -m py_compile src/dllm/*.py tests/*.py
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m dllm.cli serve --help
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```
