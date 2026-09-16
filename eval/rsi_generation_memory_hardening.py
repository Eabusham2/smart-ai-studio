"""Memory-safe RSI/Phase-4 branch generation without reducing reasoning length.

The current RSI search/reward algorithm is left untouched. This layer restores the
strong branch-lifecycle behavior from the much older MLX implementation around the
current live-stream generator, while using modern MLX-LM KV controls.

Kept from the old implementation because it is still useful:
- branches remain strictly sequential;
- Python/Metal caches are reclaimed before AND after every branch;
- one branch owns the model/Metal lock at a time.

Modernized for the current implementation:
- 4-bit KV-cache quantization reduces long-generation unified-memory growth;
- smaller prefill chunks reduce peak prompt-prefill memory;
- a Metal OOM retries only the failed branch with earlier KV quantization;
- iterator/response references are explicitly torn down after every streamed branch;
- in-flight and completed branches publish real measured generation TPS for stage telemetry.

The generation allowance is NOT shortened. ``max_tokens`` is passed through
unchanged, so the benchmark/RSI ceiling remains 16,384 tokens. The normal path also
keeps the full context (no sliding ``max_kv_size`` window). The OOM retry keeps the
full context too; it becomes more aggressive by quantizing KV earlier, not by
throwing old reasoning tokens away.
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


def _is_metal_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold().replace(" ", "")
    return any(marker.replace(" ", "") in text for marker in _OOM_MARKERS)


def _clear_runtime_memory(mx: Any) -> None:
    """Release transient graphs and MLX's cached Metal allocations."""
    gc.collect(2)
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _memory_policy(*, aggressive: bool = False) -> Dict[str, int]:
    """Return current MLX-LM controls without imposing a sliding context window."""
    return {
        "prefill_step_size": 128 if aggressive else 256,
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantized_kv_start": 128 if aggressive else 2048,
    }


def _close_iterator(iterator: Any) -> None:
    try:
        closer = getattr(iterator, "close", None)
        if callable(closer):
            closer()
    except Exception:
        pass


def _publish_measured_tps(self, response: Any, text: str, elapsed: float) -> None:
    """Publish honest completed-branch TPS for the stage telemetry layer.

    MLX-LM's final GenerationResponse owns the authoritative decode TPS when it is
    available. Older builds fall back to exact generated-token count over measured
    wall time. A missing measurement is left unchanged instead of fabricating 0.
    """
    tps = 0.0
    try:
        tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
    except Exception:
        tps = 0.0

    if tps <= 0.0 and elapsed > 0.001:
        generated_tokens = 0
        try:
            generated_tokens = int(getattr(response, "generation_tokens", 0) or 0)
        except Exception:
            generated_tokens = 0
        if generated_tokens <= 0:
            try:
                generated_tokens = len(self.engine.tokenizer.encode(str(text)))
            except Exception:
                generated_tokens = 0
        if generated_tokens > 0:
            tps = generated_tokens / elapsed

    if tps > 0.0:
        self.last_tok_per_sec = float(tps)


def _publish_inflight_tps(
    self,
    response: Any,
    streamed_tokens: int,
    decode_started: float | None,
) -> None:
    """Publish live decode TPS while a long RSI branch is still generating.

    Prefer MLX-LM's own cumulative ``generation_tps`` whenever the streamed
    response exposes it. On older builds, measure decode throughput from the first
    emitted token onward so prompt-prefill time is not mislabeled as decode TPS.
    """
    tps = 0.0
    try:
        tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
    except Exception:
        tps = 0.0

    if tps <= 0.0 and decode_started is not None and streamed_tokens > 1:
        elapsed = max(0.0, time.perf_counter() - float(decode_started))
        if elapsed > 0.001:
            # The first streamed token establishes the post-prefill decode clock.
            tps = float(streamed_tokens - 1) / elapsed

    if tps > 0.0:
        self.last_tok_per_sec = float(tps)


def install(phase4_module, live_module, legacy_generate) -> None:
    """Wrap only branch execution; never replace RSI search/reward/training logic."""
    if getattr(phase4_module, "_rsi_generation_memory_hardening_installed", False):
        return

    def safe_generate_branches_same_model(
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

        use_stream = callable(stream_generate) and make_sampler is not None
        branches: List[str] = []
        total_branches = len(temperatures)
        lock = getattr(phase4_module, "METAL_STREAM_LOCK", None)

        def lock_context():
            return lock if hasattr(lock, "__enter__") else contextlib.nullcontext()

        def legacy_one(temp: float) -> str:
            _clear_runtime_memory(mx)
            out = None
            started = time.perf_counter()
            try:
                with lock_context():
                    out = legacy_generate(
                        self,
                        formatted_prompt,
                        [temp],
                        max_tokens,
                        top_p,
                    )
                text = str(out[0]) if out else ""
                _publish_measured_tps(self, None, text, time.perf_counter() - started)
                return text
            finally:
                out = None
                _clear_runtime_memory(mx)

        def streamed_one(temp: float, branch_idx: int, *, aggressive: bool) -> str:
            policy = _memory_policy(aggressive=aggressive)
            sampler = make_sampler(temp=float(temp), top_p=top_p)
            label = (
                f"live branch {branch_idx}/{total_branches} | T={float(temp):.2f} | "
                f"KV4@{policy['quantized_kv_start']} | prefill={policy['prefill_step_size']}"
            )
            live_module._write_live_header(self, formatted_prompt, branch_label=label)

            iterator = None
            response = None
            pieces: List[str] = []
            streamed_tokens = 0
            decode_started = None
            started = time.perf_counter()
            _clear_runtime_memory(mx)
            try:
                kwargs: Dict[str, Any] = {
                    "prompt": formatted_prompt,
                    # Deliberately unchanged: RSI still receives 16,384 here.
                    "max_tokens": max(1, int(max_tokens)),
                    "sampler": sampler,
                    **policy,
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
                        # Heartbeat telemetry runs concurrently. Publish while the
                        # branch is alive instead of waiting for branch completion.
                        _publish_inflight_tps(
                            self,
                            response,
                            streamed_tokens,
                            decode_started,
                        )
                text = "".join(pieces)
                _publish_measured_tps(self, response, text, time.perf_counter() - started)
                return text
            finally:
                response = None
                _close_iterator(iterator)
                iterator = None
                sampler = None
                pieces.clear()
                _clear_runtime_memory(mx)

        for branch_idx, temp in enumerate(temperatures, 1):
            if not use_stream:
                branches.append(legacy_one(float(temp)))
                continue

            try:
                branches.append(streamed_one(float(temp), branch_idx, aggressive=False))
            except TypeError:
                branches.append(legacy_one(float(temp)))
            except RuntimeError as exc:
                if not _is_metal_oom(exc):
                    raise
                _clear_runtime_memory(mx)
                try:
                    live_module._append_live_text(
                        "\n[RSI MEMORY RECOVERY] Metal OOM: retrying only this branch "
                        "with earlier 4-bit KV quantization; 16,384-token allowance and "
                        "full context are unchanged.\n"
                    )
                except Exception:
                    pass
                branches.append(streamed_one(float(temp), branch_idx, aggressive=True))

        return branches

    phase4_module._generate_branches_same_model = safe_generate_branches_same_model
    phase4_module._rsi_generation_memory_hardening_installed = True
