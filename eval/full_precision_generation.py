"""Force native/full-precision KV for RSI and Phase-4 generation.

This is a narrow replacement for the generation function installed by
rsi_generation_memory_hardening. Search, prompts, temperatures, scoring, RSI rounds,
and Phase-4 routing stay unchanged. The only policy change is that KV cache
quantization is forbidden. OOM recovery may reduce prefill chunk size, clear caches,
and retry the failed branch, but it may never lower KV precision, truncate context,
or shorten the requested generation allowance.
"""
from __future__ import annotations

import contextlib
import gc
import time
from typing import Any, Dict, List


_OOM_MARKERS = (
    "insufficient memory",
    "outofmemory",
    "out of memory",
    "kiogpucommandbuffercallbackerroroutofmemory",
)


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold().replace(" ", "")
    return any(marker.replace(" ", "") in text for marker in _OOM_MARKERS)


def _clear(mx: Any) -> None:
    gc.collect(2)
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _close(iterator: Any) -> None:
    try:
        fn = getattr(iterator, "close", None)
        if callable(fn):
            fn()
    except Exception:
        pass


def install(phase4_module, live_module) -> None:
    if getattr(phase4_module, "_full_precision_generation_installed", False):
        return

    def generate_full_precision(
        self,
        formatted_prompt: str,
        temperatures: List[float],
        max_tokens: int,
        top_p: float = 0.92,
    ) -> List[str]:
        if (
            not phase4_module.MLX_AVAILABLE
            or self.engine.model is None
            or self.engine.tokenizer is None
        ):
            raise RuntimeError("MLX model/tokenizer unavailable for RSI/Pro branching")

        mlx_lm = phase4_module.mlx_lm
        mx = phase4_module.mx
        stream_generate = getattr(mlx_lm, "stream_generate", None)
        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            make_sampler = None
        if not callable(stream_generate) or make_sampler is None:
            raise RuntimeError("Full-precision RSI/Pro generation requires MLX-LM stream_generate + make_sampler")

        lock = getattr(phase4_module, "METAL_STREAM_LOCK", None)

        def lock_context():
            return lock if hasattr(lock, "__enter__") else contextlib.nullcontext()

        branches: List[str] = []
        total = len(temperatures)

        def one(temp: float, idx: int, prefill_step_size: int) -> str:
            sampler = make_sampler(temp=float(temp), top_p=top_p)
            label = (
                f"live branch {idx}/{total} | T={float(temp):.2f} | "
                f"native/full-precision KV | prefill={prefill_step_size}"
            )
            live_module._write_live_header(self, formatted_prompt, branch_label=label)
            iterator = None
            response = None
            pieces: List[str] = []
            streamed_tokens = 0
            decode_started = None
            started = time.perf_counter()
            _clear(mx)
            try:
                # Deliberately omit kv_bits, kv_group_size, quantized_kv_start, and
                # max_kv_size. MLX-LM therefore uses its native/full-precision cache
                # and retains the complete context.
                kwargs: Dict[str, Any] = {
                    "prompt": formatted_prompt,
                    "max_tokens": max(1, int(max_tokens)),
                    "sampler": sampler,
                    "prefill_step_size": int(prefill_step_size),
                }
                with lock_context():
                    iterator = stream_generate(
                        self.engine.model,
                        self.engine.tokenizer,
                        **kwargs,
                    )
                    for response in iterator:
                        now = time.perf_counter()
                        streamed_tokens += 1
                        if decode_started is None:
                            decode_started = now
                        chunk = getattr(response, "text", None)
                        if chunk is None:
                            chunk = str(response)
                        chunk = str(chunk)
                        pieces.append(chunk)
                        live_module._append_live_text(chunk)

                        # Reuse the existing honest TPS helper if available.
                        try:
                            phase4_module._publish_inflight_tps(
                                self, response, streamed_tokens, decode_started
                            )
                        except Exception:
                            pass
                text = "".join(pieces)
                elapsed = max(0.001, time.perf_counter() - started)
                try:
                    tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
                except Exception:
                    tps = 0.0
                if tps <= 0.0 and streamed_tokens > 0:
                    tps = streamed_tokens / elapsed
                if tps > 0.0:
                    self.last_tok_per_sec = tps
                return text
            finally:
                response = None
                _close(iterator)
                iterator = None
                sampler = None
                pieces.clear()
                _clear(mx)

        for idx, temp in enumerate(temperatures, 1):
            try:
                branches.append(one(float(temp), idx, 256))
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                # Accuracy-preserving OOM recovery: same model, same prompt, same
                # max_tokens, same sampler policy, same full-precision KV. Only the
                # prefill work chunk is smaller to reduce transient peak memory.
                _clear(mx)
                try:
                    live_module._append_live_text(
                        "\n[RSI MEMORY RECOVERY] Metal OOM: retrying this branch with "
                        "smaller prefill chunks. KV precision, full context, and token "
                        "allowance are unchanged.\n"
                    )
                except Exception:
                    pass
                branches.append(one(float(temp), idx, 64))

        return branches

    phase4_module._generate_branches_same_model = generate_full_precision
    phase4_module._full_precision_generation_installed = True
