"""Release MLX/Metal branch-generation state between Pro/RSI branches.

This does not change RSI policy, prompts, branch count, rounds, temperatures,
selection, token ceiling, training, or scoring. It only fixes the lifetime of
stream_generate branch-local objects so completed KV/prompt-cache state can be
reclaimed before the next branch begins.
"""
from __future__ import annotations

import gc
from typing import List


def _release_branch_state(mx) -> None:
    """Synchronize, collect Python refs, then release reclaimable MLX/Metal cache."""
    try:
        sync = getattr(mx, "synchronize", None)
        if callable(sync):
            sync()
    except Exception:
        pass

    # stream_generate owns prompt/KV cache objects reachable from its generator.
    # Python references must be dropped before this collection/Metal cache clear.
    gc.collect()
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def install(phase4_module, live_module) -> None:
    """Replace only the live branch streamer with an equivalent memory-safe version."""
    if getattr(phase4_module, "_metal_branch_memory_hardening_installed", False):
        return

    previous = phase4_module._generate_branches_same_model

    def memory_safe_live_branches(
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
            return previous(self, formatted_prompt, temperatures, max_tokens, top_p)

        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            make_sampler = None
        if make_sampler is None:
            return previous(self, formatted_prompt, temperatures, max_tokens, top_p)

        branches: List[str] = []
        total_branches = len(temperatures)

        # Important: iterator/response are explicitly nulled in each finally block
        # BEFORE _release_branch_state(). The old wrapper kept the completed iterator
        # referenced until the next assignment, so its KV/prompt cache could survive
        # the next branch's pre-clear and accumulate across long RSI items.
        for branch_idx, temp in enumerate(temperatures, 1):
            _release_branch_state(mx)
            iterator = None
            response = None
            sampler = None
            pieces = None
            fallback = None
            try:
                sampler = make_sampler(temp=float(temp), top_p=top_p)
                live_module._write_live_header(
                    self,
                    formatted_prompt,
                    branch_label=f"live branch {branch_idx}/{total_branches} | T={float(temp):.2f}",
                )
                pieces = []
                iterator = stream_generate(
                    self.engine.model,
                    self.engine.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max(1, int(max_tokens)),
                    sampler=sampler,
                )
                for response in iterator:
                    chunk = getattr(response, "text", None)
                    if chunk is None:
                        chunk = str(response)
                    chunk = str(chunk)
                    pieces.append(chunk)
                    live_module._append_live_text(chunk)
                branches.append("".join(pieces))
            except TypeError:
                # Preserve the existing older-mlx-lm fallback semantics for this
                # one branch. No different prompt, temperature, token cap, or policy.
                iterator = None
                response = None
                pieces = None
                sampler = None
                _release_branch_state(mx)
                fallback = previous(self, formatted_prompt, [temp], max_tokens, top_p)
                branches.extend(fallback)
                if fallback:
                    live_module._append_live_text(str(fallback[0]))
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                # Drop all objects that can retain generator/cache tensors before
                # asking MLX/Metal to return reclaimable memory.
                response = None
                iterator = None
                sampler = None
                pieces = None
                fallback = None
                _release_branch_state(mx)

        return branches

    phase4_module._generate_branches_same_model = memory_safe_live_branches
    phase4_module._metal_branch_memory_hardening_installed = True
