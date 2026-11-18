#!/usr/bin/env python3
"""Run online learning or parameter-memory operations on a locally loaded model.

Example:
  python tools/online_learning.py --backend mlx --target bonsai2 \
      --model-path /models/bonsai2 --scope project --scope-key my-project \
      --request request.json --output result.json

Requests are data, not Python: built-in verifiers have a validated schema and
the CLI never evaluates an expression or loads a callable supplied in JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def read_request(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_REQUEST_BYTES + 1)
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Use a source file for large documents; request JSON is limited to 2 MiB")
    request = json.loads(payload)
    if not isinstance(request, dict) or not isinstance(request.get("method"), str):
        raise ValueError("Request JSON must be an object with a method")
    return request


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model-path", required=True, help="Existing local model directory or native artifact")
    result.add_argument("--model-name", default="Local online-learning model")
    result.add_argument("--backend", choices=("mlx", "torch", "cpu", "cuda", "mps", "gguf", "bitnet", "controller"), required=True)
    result.add_argument("--target", choices=("bonsai2", "bonsai2-crack"))
    result.add_argument("--model-info", help="Optional local JSON with exact runtime and training-parent metadata")
    result.add_argument("--scope", choices=("model", "session", "project"), default="model")
    result.add_argument("--scope-key", default="")
    tasks = result.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--request", help="Online method request JSON")
    tasks.add_argument("--capabilities", action="store_true")
    result.add_argument("--output", help="Write the measured result as JSON")
    result.add_argument("--state-dir", help="Optional independent application state directory")
    result.add_argument("--context-tokens", type=int, default=8192)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 256 <= args.context_tokens <= 2_097_152:
        raise ValueError("context-tokens must be between 256 and 2097152")
    selected = Path(args.model_path).expanduser().resolve()
    if not selected.exists():
        raise ValueError("The selected model must already be present locally")
    if args.state_dir:
        os.environ["SMARTAI_DATA_DIR"] = str(Path(args.state_dir).expanduser().resolve())
    # This runner never starts an implicit model download.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO_ROOT))
    from config.settings import get_settings
    from core import ProReasoningEngine
    from core.memory_scopes import LearningScope

    info = {}
    if args.target:
        from tools.evaluate_persistent_memory import TARGETS
        if args.backend != "mlx":
            raise ValueError("Bonsai 2 and CRACK target presets require --backend mlx")
        info.update(TARGETS[args.target])
    if args.model_info:
        with Path(args.model_info).open("rb") as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Model metadata is too large")
        supplied = json.loads(raw)
        if not isinstance(supplied, dict):
            raise ValueError("Model metadata must be an object")
        info.update(supplied)
    settings_options = dict(backend=args.backend, learning_implementation="managed", use_mock=False, live_mode=True,
                            auto_download=False, lora_adapter_path=None)
    if args.state_dir:
        settings_options["database_path"] = str(Path(os.environ["SMARTAI_DATA_DIR"]) / "learning.sqlite3")
    settings = get_settings(**settings_options)
    engine = ProReasoningEngine(settings=settings)
    engine.set_learning_scope(LearningScope(args.scope, args.scope_key))
    engine._context_budget_tokens = args.context_tokens
    cancelled = threading.Event()

    def stop(_signum, _frame):
        cancelled.set()
        engine.cancel_speed_tuning()

    previous = signal.signal(signal.SIGINT, stop)
    try:
        loaded = engine.load_model(info.get("name", args.model_name), model_path=str(selected),
                                   backend=args.backend, model_info=info)
        if loaded.get("status") != "loaded":
            raise RuntimeError("Model load failed: " + json.dumps(loaded, ensure_ascii=False))
        request = {"method": "capabilities"} if args.capabilities else read_request(args.request)
        result = engine.run_online_learning(request, cancel_event=cancelled,
            progress=lambda *parts, **details: print(json.dumps({"progress": parts, **details}, default=str), file=sys.stderr))
        result = {"model": str(selected), "backend": loaded.get("backend"), **result}
        payload = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if result.get("status") in {"completed", "success", "unchanged", "paused"} else 2
    finally:
        signal.signal(signal.SIGINT, previous)
        unloaded = engine.unload_model()
        if isinstance(unloaded, dict) and unloaded.get("status") == "error":
            print("Model cleanup incomplete: " + str(unloaded.get("error")), file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
