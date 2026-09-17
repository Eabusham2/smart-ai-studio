"""Fast Phase-1/Learn single-pass generation on the native MLX-LM loop.

Prompt construction, benchmark scoring, checkpointing, stage ownership, Context
accounting, model weights and full-precision KV remain unchanged. The decoder uses
the optimized MLX-LM stream rather than the old per-token Python ``.item()`` loop.

Sampling policy is intentionally tuned for this native low-bit/ternary model:
eval single-pass uses T=0.55. This is a deliberate behavioral/quality choice, not a
claim of bit-identical output to the older greedy baseline.
"""
from __future__ import annotations

import gc
import time
from typing import Any, List

import psutil

from core.temperature_policy import EVAL_N1_TEMPERATURE


def _memory_pressure() -> bool:
    try:
        proc_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        return proc_gb >= 12.5 or avail_gb <= 0.75
    except Exception:
        return False


def _reclaim_if_needed(mx: Any, *, force: bool = False) -> None:
    if not force and not _memory_pressure():
        return
    gc.collect(2)
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def install(runtime_module, live_module, cls) -> None:
    if getattr(cls, "_lossless_baseline_speedup_installed", False):
        return

    def native_single_pass_generate(self, prompt, max_tokens=16384, stream=False):
        if not runtime_module.MLX_AVAILABLE:
            raise RuntimeError("MLX/MLX-LM unavailable: refusing to fake/offline benchmark generation")
        if self.engine.model is None or self.engine.tokenizer is None:
            raise RuntimeError("27B model/tokenizer not loaded: refusing to evaluate without real generation")

        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        tok = self.engine.tokenizer
        model = self.engine.model
        mx = runtime_module.mx
        iterator = None
        response = None
        pieces: List[str] = []
        prompt_tokens = len(tok.encode(prompt))
        self.last_prompt_tokens = prompt_tokens

        requested = max(1, int(max_tokens))
        ctx = runtime_module._model_context_limit(self.engine)
        if ctx is not None:
            remaining = int(ctx) - prompt_tokens
            if remaining <= 0:
                raise RuntimeError(
                    f"Prompt requires {prompt_tokens} tokens but model context is {ctx}; refusing context truncation."
                )
            requested = min(requested, remaining)

        live_module._write_live_header(
            self,
            prompt,
            branch_label=f"primary | native sampled T={EVAL_N1_TEMPERATURE:.2f}/full-KV",
        )
        sampler = make_sampler(temp=EVAL_N1_TEMPERATURE, top_p=0.92)
        started = time.perf_counter()
        last_live_write = started
        pending: List[str] = []
        generated_responses = 0

        try:
            iterator = mlx_lm.stream_generate(
                model,
                tok,
                prompt=prompt,
                max_tokens=requested,
                sampler=sampler,
            )
            for response in iterator:
                generated_responses += 1
                chunk = getattr(response, "text", None)
                if chunk is None:
                    token = getattr(response, "token", None)
                    if token is not None:
                        chunk = tok.decode([int(token)])
                    else:
                        chunk = str(response)
                chunk = str(chunk)
                pieces.append(chunk)
                pending.append(chunk)

                try:
                    tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
                except Exception:
                    tps = 0.0
                if tps > 0.0:
                    self.last_tok_per_sec = tps

                now = time.perf_counter()
                if pending and (now - last_live_write >= 0.5 or len(pending) >= 8):
                    live_module._append_live_text("".join(pending))
                    pending.clear()
                    last_live_write = now

            if pending:
                live_module._append_live_text("".join(pending))
                pending.clear()

            text = "".join(pieces)
            elapsed = max(0.001, time.perf_counter() - started)
            try:
                generation_tokens = int(getattr(response, "generation_tokens", 0) or 0)
            except Exception:
                generation_tokens = 0
            if generation_tokens <= 0:
                try:
                    generation_tokens = len(tok.encode(text))
                except Exception:
                    generation_tokens = generated_responses

            try:
                tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
            except Exception:
                tps = 0.0
            if tps <= 0.0 and generation_tokens:
                tps = generation_tokens / elapsed

            self.last_tok_per_sec = float(tps)
            self.live_generated_tokens = generation_tokens
            self.last_output_tokens = generation_tokens
            self.last_generation_seconds = elapsed
            self.last_generation_error = None
            return text

        except Exception as exc:
            self.last_output_tokens = 0
            self.last_generation_seconds = 0.0
            self.last_generation_error = f"{type(exc).__name__}: {exc}"
            live_module._append_live_text(f"\n\n[GENERATION ERROR] {self.last_generation_error}\n")
            _reclaim_if_needed(mx, force=True)
            raise RuntimeError(f"Real MLX generation failed: {self.last_generation_error}") from exc
        finally:
            response = None
            try:
                closer = getattr(iterator, "close", None)
                if callable(closer):
                    closer()
            except Exception:
                pass
            iterator = None
            pieces.clear()
            _reclaim_if_needed(mx)

    cls._fast_generate = native_single_pass_generate
    runtime_module.fast_generate = native_single_pass_generate
    cls._lossless_baseline_speedup_installed = True
