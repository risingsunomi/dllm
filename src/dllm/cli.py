from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from .config import Settings, load_env_file
from .distributed import DistributedInferenceEngine
from .peer import InferenceWorker
from .server import run_http_server


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", default=".env")
    env_args, _ = env_parser.parse_known_args(argv)
    load_env_file(env_args.env_file, override=False)

    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.error("missing command")

    settings = Settings.from_mapping(vars(args))
    try:
        settings.validate_for_runtime()
    except ValueError as exc:
        parser.error(str(exc))

    return _serve(settings, preload=bool(args.preload))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone distributed PyTorch inference server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start an HTTP server, worker, or both")
    serve.add_argument("--env-file", default=".env", help="path to .env file")
    serve.add_argument("--model-name", default=None, help="Hugging Face model id or local path")
    serve.add_argument("--node-name", default=None, help="stable name for this node")
    serve.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps, metal, or auto")
    serve.add_argument("--dtype", default=None, help="auto, fp32, fp16, bf16, or default")
    serve.add_argument(
        "--device-map",
        default=None,
        help="Transformers device_map for large/MoE models, e.g. auto, balanced, or a JSON map",
    )
    serve.add_argument(
        "--max-memory",
        default=None,
        help='Transformers max_memory, e.g. "0=20GiB,cpu=80GiB" or JSON',
    )
    serve.add_argument(
        "--offload-folder",
        default=None,
        help="folder for CPU/disk offload when using device_map",
    )
    serve.add_argument(
        "--attention-implementation",
        default=None,
        help="Transformers attn_implementation, e.g. sdpa, flash_attention_2, or eager",
    )
    serve.add_argument(
        "--language-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="for multimodal/VLM repos, load only the nested language model when possible",
    )
    serve.add_argument(
        "--language-weight-prefix",
        default=None,
        help="checkpoint prefix to strip for language-only loading, e.g. model.language_model. or auto",
    )
    serve.add_argument(
        "--weight-key-mapping",
        default=None,
        help="extra Transformers regex key_mapping as JSON, source=target pairs, or auto",
    )
    serve.add_argument("--role", choices=("server", "worker", "both"), default=None)
    serve.add_argument("--host", default=None, help="HTTP bind host")
    serve.add_argument("--port", type=int, default=None, help="HTTP bind port")
    serve.add_argument("--peer-host", default=None, help="TCP worker bind host")
    serve.add_argument("--peer-port", type=int, default=None, help="TCP worker bind port")
    serve.add_argument("--peers", default=None, help="comma-separated name@host:port entries")
    serve.add_argument(
        "--load-local",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include this process as a local generation target",
    )
    serve.add_argument(
        "--distribution-mode",
        choices=("shard", "replica"),
        default=None,
        help="shard one request across server+peers, or replica-load-balance full requests",
    )
    serve.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="only use locally cached model files",
    )
    serve.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow custom Transformers model code",
    )
    serve.add_argument("--max-new-tokens", type=int, default=None)
    serve.add_argument("--temperature", type=float, default=None)
    serve.add_argument("--top-p", type=float, default=None)
    serve.add_argument("--top-k", type=int, default=None)
    serve.add_argument("--repetition-penalty", type=float, default=None)
    serve.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="optional context length advertised from /v1/models",
    )
    serve.add_argument("--request-timeout", type=float, default=None)
    serve.add_argument("--peer-connect-timeout", type=float, default=None)
    serve.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load local and remote models before accepting requests",
    )
    return parser


def _serve(settings: Settings, *, preload: bool) -> int:
    worker: InferenceWorker | None = None

    if settings.role in {"worker", "both"}:
        worker = InferenceWorker(settings)
        if preload:
            print(
                f"loading worker model={settings.model_name} node={settings.node_name} device={settings.device}",
                flush=True,
            )
            worker.engine.load()
        if settings.role == "worker":
            print(
                f"worker listening node={settings.node_name} tcp={settings.peer_host}:{settings.peer_port}",
                flush=True,
            )
            try:
                worker.serve_forever()
            except KeyboardInterrupt:
                worker.stop()
            return 0
        worker.start_background()
        print(
            f"worker listening node={settings.node_name} tcp={settings.peer_host}:{settings.peer_port}",
            flush=True,
        )

    local_engine = worker.engine if worker is not None and settings.load_local else None
    engine = DistributedInferenceEngine(settings, local_engine=local_engine)

    if preload:
        print(
            f"preparing coordinator model={settings.model_name} node={settings.node_name} "
            f"http={settings.host}:{settings.port}",
            flush=True,
        )
        engine.ensure_ready()

    print(
        f"http listening node={settings.node_name} role={settings.role} "
        f"http={settings.host}:{settings.port}",
        flush=True,
    )
    try:
        run_http_server(settings, engine)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            worker.stop()
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
