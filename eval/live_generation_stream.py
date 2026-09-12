"""Real-time current-item generation log for the 4,014-item benchmark.

This intentionally does NOT contain GitHub/CI monitoring.  It exists only so a
second Terminal window can run one simple shell line and see the exact current
model prompt plus raw model output (including <think>...</think>) while tokens are
still being generated.

The historical append-only raw_model_outputs.log remains unchanged.  This module
adds eval_results/live_generation.log, which is truncated when a new item/branch
starts and continuously appended during decode.
"""
from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List


LIVE_GENERATION_LOG = os.path.join("eval_results", "live_generation.log")


def _live_path() -> str:
    return LIVE_GENERATION_LOG


def _meta(self) -> tuple[str, str, str]:
    return (
        str(getattr(self, "_current_phase", "unknown")),
        str(getattr(self, "_current_split", "unknown")),
        str(getattr(self, "_current_item_id", "unknown")),
    )


def _write_live_header(self, formatted_prompt: str, branch_label: str = "primary") -> None:
    """Start the current-item file with the exact prompt sent to the tokenizer."""
    path = _live_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    phase, split, item_id = _meta(self)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("SMART AI — LIVE MODEL GENERATION\n")
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')} | "
            f"{phase} | {split} | {item_id} | {branch_label}\n"
        )
        f.write("=" * 110 + "\n")
        f.write("FULL FORMATTED MODEL PROMPT (EXACT TEXT SENT TO TOKENIZER):\n")
        f.write("-" * 110 + "\n")
        f.write(str(formatted_prompt).rstrip() + "\n")
        f.write("-" * 110 + "\n")
        f.write("RAW MODEL OUTPUT — LIVE (INCLUDING <think> EXACTLY AS EMITTED):\n")
        f.flush()
    os.replace(tmp, path)


def _append_live_text(text: str) -> None:
    if not text:
        return
    path = _live_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


def _decode_piece(tokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(token_ids))
    except Exception:
        parts = []
        for token_id in token_ids:
            try:
                parts.append(str(tokenizer.decode([token_id])))
            except Exception:
                pass
        return "".join(parts)


def _write_final_snapshot(self, formatted_prompt: str, raw_output: str) -> None:
    """Replace incremental display with the exact final decode and measured metrics."""
    _write_live_header(self, formatted_prompt, branch_label="final selected output")
    _append_live_text((raw_output or "").rstrip())
    prompt_tokens = int(getattr(self, "last_prompt_tokens", 0) or 0)
    output_tokens = int(getattr(self, "last_output_tokens", 0) or 0)
    speed = float(getattr(self, "last_tok_per_sec", 0.0) or 0.0)
    seconds = float(getattr(self, "last_generation_seconds", 0.0) or 0.0)
    _append_live_text(
        "\n\nMETRICS AFTER RAW OUTPUT:\n"
        f"Speed: {speed:.3f} t/s\n"
        f"Prompt tokens: {prompt_tokens}\n"
        f"Output tokens: {output_tokens}\n"
        f"Total tokens: {prompt_tokens + output_tokens}\n"
        f"Decode time: {seconds:.3f} s\n"
    )


def _append_live_result(ok: bool) -> None:
    _append_live_text(f"RESULT: {'PASS' if ok else 'FAIL'}\n" + "=" * 110 + "\n")


def install_baseline_stream(runtime_module, cls) -> None:
    """Install a byte-for-byte-equivalent fused decoder plus low-overhead live logging."""
    if getattr(cls, "_live_generation_stream_installed", False):
        return

    MLX_AVAILABLE = runtime_module.MLX_AVAILABLE
    mx = getattr(runtime_module, "mx", None)
    make_prompt_cache = getattr(runtime_module, "make_prompt_cache", None)
    metal_lock = runtime_module.METAL_STREAM_LOCK
    model_context_limit = runtime_module._model_context_limit

    original_raw_log = runtime_module._append_raw_generation_log
    original_result_log = runtime_module._append_result_log

    def live_fast_generate(self, prompt, max_tokens=16384, stream=False):
        """Same fused MLX decode path as the recovered runtime, with live raw-output taps."""
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX/MLX-LM unavailable: refusing to fake/offline benchmark generation")
        if self.engine.model is None or self.engine.tokenizer is None:
            raise RuntimeError("27B model/tokenizer not loaded: refusing to evaluate without real generation")

        tok = self.engine.tokenizer
        model = self.engine.model
        cache = inp = logits = step = None

        try:
            ids = tok.encode(prompt)
            self.last_prompt_tokens = len(ids)
            _write_live_header(self, prompt)

            ctx = model_context_limit(self.engine)
            if ctx is not None:
                max_tokens = min(int(max_tokens), max(1, ctx - len(ids)))

            cache = make_prompt_cache(model)

            eos = set()
            e = getattr(tok, "eos_token_id", None)
            if e is not None:
                eos.update(e if isinstance(e, (list, tuple, set)) else [e])

            for name in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|endoftext|>", "</s>", "<eos>"):
                try:
                    encoded = tok.encode(name, add_special_tokens=False)
                    if len(encoded) == 1:
                        eos.add(int(encoded[0]))
                    if hasattr(tok, "convert_tokens_to_ids"):
                        tid = tok.convert_tokens_to_ids(name)
                        if isinstance(tid, int) and tid >= 0:
                            eos.add(tid)
                except Exception:
                    pass

            pending_live: List[int] = []
            with metal_lock:
                inp = mx.array([ids])
                logits = model(inp, cache=cache)
                next_arr = mx.argmax(logits[0, -1])
                mx.eval(next_arr)
                nxt = int(next_arr.item())

                out: List[int] = []
                if nxt not in eos:
                    out.append(nxt)
                    pending_live.append(nxt)

                # Preserve the recovered timing definition: pure decode starts after prefill.
                decode = 0
                t0 = time.perf_counter()
                live_update = t0

                for _ in range(max(0, max_tokens - 1)):
                    if not out or out[-1] in eos:
                        break

                    step = model(mx.array([[out[-1]]]), cache=cache)
                    next_arr = mx.argmax(step[0, -1])
                    mx.eval(next_arr)
                    nxt = int(next_arr.item())

                    if nxt in eos:
                        break

                    out.append(nxt)
                    pending_live.append(nxt)
                    decode += 1

                    now = time.perf_counter()
                    # Roughly twice per second at the current ~5 t/s.  Decode only
                    # the newly generated handful of tokens so monitoring does not
                    # repeatedly re-decode the whole response or hurt throughput.
                    if pending_live and (now - live_update >= 0.5 or len(pending_live) >= 8):
                        _append_live_text(_decode_piece(tok, pending_live))
                        pending_live.clear()
                        self.last_tok_per_sec = decode / max(0.001, now - t0)
                        self.live_generated_tokens = len(out)
                        live_update = now

                if pending_live:
                    _append_live_text(_decode_piece(tok, pending_live))
                    pending_live.clear()

                dt = max(0.001, time.perf_counter() - t0)

            self.last_tok_per_sec = decode / dt if decode else 0.0
            self.live_generated_tokens = len(out)
            self.last_output_tokens = len(out)
            self.last_generation_seconds = dt
            self.last_generation_error = None
            return tok.decode(out)

        except Exception as exc:
            self.last_tok_per_sec = 0.0
            self.last_output_tokens = 0
            self.last_generation_seconds = 0.0
            self.last_generation_error = f"{type(exc).__name__}: {exc}"
            _append_live_text(f"\n\n[GENERATION ERROR] {self.last_generation_error}\n")
            raise RuntimeError(f"Real MLX generation failed: {self.last_generation_error}") from exc

        finally:
            cache = inp = logits = step = None
            if MLX_AVAILABLE:
                try:
                    if hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                        mx.metal.clear_cache()
                except Exception:
                    pass
            gc.collect()

    def raw_log_with_live_snapshot(self, formatted_prompt: str, user_prompt: str, raw_output: str):
        original_raw_log(self, formatted_prompt, user_prompt, raw_output)
        _write_final_snapshot(self, formatted_prompt, raw_output)

    def result_log_with_live_result(self, ok: bool):
        original_result_log(self, ok)
        _append_live_result(ok)

    cls._fast_generate = live_fast_generate
    runtime_module.fast_generate = live_fast_generate
    runtime_module._append_raw_generation_log = raw_log_with_live_snapshot
    runtime_module._append_result_log = result_log_with_live_result
    cls._live_generation_stream_installed = True


def install_phase4_stream(phase4_module) -> None:
    """Make RSI/Phase-4 branch generation visible live without changing its sampling policy."""
    if getattr(phase4_module, "_live_generation_stream_installed", False):
        return

    original = phase4_module._generate_branches_same_model

    def live_generate_branches_same_model(
        self,
        formatted_prompt: str,
        temperatures: List[float],
        max_tokens: int,
        top_p: float = 0.92,
    ) -> List[str]:
        mlx_lm = phase4_module.mlx_lm
        mx = phase4_module.mx
        if not phase4_module.MLX_AVAILABLE or self.engine.model is None or self.engine.tokenizer is None:
            raise RuntimeError("MLX model/tokenizer unavailable for RSI/Pro branching")

        stream_generate = getattr(mlx_lm, "stream_generate", None)
        if not callable(stream_generate):
            # Older mlx-lm: preserve original behavior.  The final selected output
            # is still written by the runtime snapshot hook.
            return original(self, formatted_prompt, temperatures, max_tokens, top_p)

        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            make_sampler = None

        branches: List[str] = []
        total_branches = len(temperatures)
        for branch_idx, temp in enumerate(temperatures, 1):
            gc.collect(1)
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass

            kwargs: Dict[str, Any] = {"max_tokens": max(1, int(max_tokens)), "verbose": False}
            if make_sampler is not None:
                try:
                    kwargs["sampler"] = make_sampler(temp=float(temp), top_p=top_p)
                except Exception:
                    kwargs["temp"] = float(temp)
                    kwargs["top_p"] = top_p
            else:
                kwargs["temp"] = float(temp)
                kwargs["top_p"] = top_p

            _write_live_header(
                self,
                formatted_prompt,
                branch_label=f"live branch {branch_idx}/{total_branches} | T={float(temp):.2f}",
            )
            pieces: List[str] = []
            try:
                iterator = stream_generate(
                    self.engine.model,
                    self.engine.tokenizer,
                    prompt=formatted_prompt,
                    **kwargs,
                )
            except TypeError:
                kwargs.pop("sampler", None)
                kwargs["temp"] = float(temp)
                kwargs["top_p"] = top_p
                iterator = stream_generate(
                    self.engine.model,
                    self.engine.tokenizer,
                    prompt=formatted_prompt,
                    **kwargs,
                )

            for response in iterator:
                chunk = getattr(response, "text", None)
                if chunk is None:
                    chunk = str(response)
                chunk = str(chunk)
                pieces.append(chunk)
                _append_live_text(chunk)

            branches.append("".join(pieces))

        return branches

    phase4_module._generate_branches_same_model = live_generate_branches_same_model
    phase4_module._live_generation_stream_installed = True
