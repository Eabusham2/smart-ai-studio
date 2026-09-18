"""Isolated llama-cpp-python worker for PrismML private GGUF kernels."""
from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import re
import sys
from typing import Any, Dict, List


def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _data_url(raw: bytes, mime: str) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")


def _video_urls(path: str, max_frames: int = 8) -> List[str]:
    import imageio.v3 as iio
    from PIL import Image

    arrays = []
    try:
        meta = iio.immeta(path)
        total = int(meta.get("nframes") or meta.get("n_images") or 0)
    except Exception:
        total = 0

    if total > 0:
        indices = sorted({
            int(round(i * max(0, total - 1) / max(1, max_frames - 1)))
            for i in range(max_frames)
        })
        for idx in indices:
            try:
                arrays.append(iio.imread(path, index=idx))
            except Exception:
                pass
    else:
        try:
            for idx, frame in enumerate(iio.imiter(path)):
                if idx >= max_frames:
                    break
                arrays.append(frame)
        except Exception:
            pass

    if not arrays:
        raise RuntimeError("Video contained no decodable frames")

    urls = []
    for array in arrays:
        buf = io.BytesIO()
        Image.fromarray(array).convert("RGB").save(buf, format="JPEG", quality=90)
        urls.append(_data_url(buf.getvalue(), "image/jpeg"))
    return urls


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("worker requires one config path")
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    # These imports happen only after the parent has injected Prism native-library paths.
    from llama_cpp import Llama

    chat_handler = None
    mmproj = cfg.get("mmproj_path")
    if mmproj and os.path.isfile(mmproj):
        try:
            from llama_cpp.llama_chat_format import MTMDChatHandler
            chat_handler = MTMDChatHandler(
                clip_model_path=mmproj,
                verbose=bool(cfg.get("verbose", False)),
            )
        except Exception:
            try:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(
                    clip_model_path=mmproj,
                    verbose=bool(cfg.get("verbose", False)),
                )
            except Exception:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(
                    clip_model_path=mmproj,
                    verbose=bool(cfg.get("verbose", False)),
                )

    model = Llama(
        model_path=str(cfg["model_path"]),
        n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
        n_ctx=int(cfg.get("n_ctx", 32768)),
        chat_handler=chat_handler,
        lora_path=cfg.get("adapter_path") if cfg.get("adapter_path") and os.path.isfile(cfg["adapter_path"]) else None,
        logits_all=True,
        no_perf=True,
        verbose=bool(cfg.get("verbose", False)),
    )

    stops = ["<|im_end|>", "</s>", "\nuser", "\nHuman:"]
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            op = str(request.get("op") or "")
            if op == "shutdown":
                _emit({"status": "ok"})
                return 0
            if op == "ping":
                _emit({"status": "ok"})
                continue
            if op == "n_ctx":
                _emit({"status": "ok", "n_ctx": int(model.n_ctx())})
                continue
            if op == "tokenize":
                tokens = model.tokenize(str(request.get("text") or "").encode("utf-8"))
                _emit({"status": "ok", "tokens": [int(x) for x in tokens]})
                continue
            if op == "generate":
                result = model(
                    str(request.get("prompt") or ""),
                    max_tokens=int(request.get("max_tokens") or 1536),
                    temperature=float(request.get("temperature") or 0.0),
                    top_p=float(request.get("top_p") or 0.92),
                    stop=stops,
                )
                text = str(result.get("choices", [{}])[0].get("text", "")).strip()
                _emit({"status": "ok", "text": text})
                continue
            if op == "stream":
                stream = model(
                    str(request.get("prompt") or ""),
                    max_tokens=int(request.get("max_tokens") or 1536),
                    temperature=float(request.get("temperature") or 0.0),
                    top_p=float(request.get("top_p") or 0.92),
                    stream=True,
                    stop=stops,
                )
                for chunk in stream:
                    token = str(chunk.get("choices", [{}])[0].get("text", ""))
                    if token:
                        _emit({"event": "token", "text": token})
                _emit({"event": "done"})
                continue
            if op == "entropy":
                tokens = model.tokenize(str(request.get("prompt") or "").encode("utf-8"))
                if not tokens:
                    _emit({"status": "ok", "entropy": 0.35})
                    continue
                model.eval(tokens[-32:])
                logits = model._scores[-1]
                max_l = max(logits)
                exp_l = [math.exp(float(x) - float(max_l)) for x in logits]
                total = sum(exp_l)
                probs = [x / total for x in exp_l if x > 0]
                entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
                max_ent = math.log(max(2, len(logits)))
                _emit({"status": "ok", "entropy": round(entropy / max_ent, 4)})
                continue
            if op == "review_media":
                if chat_handler is None:
                    raise RuntimeError("Prism multimodal projector is not loaded")
                path = os.path.abspath(str(request.get("path") or ""))
                kind = str(request.get("kind") or "").lower()
                if not os.path.isfile(path):
                    raise RuntimeError("Media input file is missing")
                prompt = str(request.get("prompt") or "").strip() or f"Describe the supplied {kind}."
                review = (
                    prompt
                    + "\nReturn JSON only with keys description, score, reasoning. "
                      "score must be 0 to 100 and judge only the supplied media."
                )
                content: List[Dict[str, Any]] = [{"type": "text", "text": review}]
                if kind == "image":
                    with open(path, "rb") as handle:
                        data = handle.read()
                    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                    content.append({"type": "image_url", "image_url": {"url": _data_url(data, mime)}})
                elif kind == "video":
                    for url in _video_urls(path):
                        content.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    raise RuntimeError(f"Unsupported Prism media kind: {kind}")

                result = model.create_chat_completion(
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    max_tokens=768,
                )
                text = str(result.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
                match = re.search(r"\{[\s\S]*\}", text)
                parsed: Dict[str, Any] = {}
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        parsed = {}
                response: Dict[str, Any] = {
                    "status": "ok",
                    "perception_available": True,
                    "description": str(parsed.get("description") or text),
                    "analysis": str(parsed.get("reasoning") or text),
                }
                score = parsed.get("score")
                if isinstance(score, (int, float)):
                    response["score"] = max(0.0, min(100.0, float(score)))
                else:
                    response["reason"] = "Model inspected the media but did not return a numeric self-grade."
                _emit(response)
                continue
            raise RuntimeError(f"Unknown Prism worker operation: {op}")
        except Exception as exc:
            _emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
