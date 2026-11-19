"""Measure the exact native VLM inputs without tokenizing the request twice.

The native processor expands templates and any special input markers. Its
prepared arrays are both the source of the count and the arrays submitted to
generation, so a cached prefix or a different plain-text tokenizer cannot make
the Context display undercount the actual request.
"""
from __future__ import annotations

import importlib
import ast
import functools
import inspect
import textwrap


@functools.lru_cache(maxsize=8)
def _accepts_prepared_inputs(stream):
    """Recognize the native bypass, not merely an unrestricted **kwargs sink."""
    try:
        stream = inspect.unwrap(stream)
        parameters = inspect.signature(stream).parameters
        if "input_ids" in parameters:
            return True
        source = ast.parse(textwrap.dedent(inspect.getsource(stream)))
        for node in ast.walk(source):
            if not isinstance(node, ast.If):
                continue
            # Audited mlx-vlm API: one branch consumes the three provided
            # inputs from kwargs, and the other invokes prepare_inputs.
            consumed = set()
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                    continue
                call = statement.value
                if (isinstance(call.func, ast.Attribute) and call.func.attr == "pop"
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == "kwargs"
                        and call.args and isinstance(call.args[0], ast.Constant)):
                    consumed.add(call.args[0].value)
            if {"input_ids", "pixel_values", "mask"}.issubset(consumed):
                return True
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError):
        pass
    return False


def prepare_vlm_inputs(model, processor, prepared_prompt):
    """Return ``(native_kwargs, count)`` for one already-templated text request.

    Older native runtimes without the prepared-input API remain supported, but
    their count stays unknown until they emit authoritative request usage.
    This helper never guesses from words or from an unprocessed text prompt.
    """
    generation = importlib.import_module("mlx_vlm.generate")
    prepare = getattr(generation, "prepare_inputs", None)
    if not callable(prepare) or not _accepts_prepared_inputs(getattr(generation, "stream_generate", None)):
        return {}, None

    config = getattr(model, "config", None)
    config_value = lambda name: (config.get(name) if isinstance(config, dict)
                                 else getattr(config, name, None))
    # Match stream_generate's own preprocessing policy. The same returned
    # input_ids bypass its second preprocessing pass via the native API.
    add_special_tokens = (getattr(processor, "chat_template", None) is None
                          if config_value("model_type") in {"gemma3", "gemma3n", "gemma4"}
                          else True)
    inputs = prepare(processor, images=[], audio=None, videos=None,
                     prompts=prepared_prompt,
                     image_token_index=config_value("image_token_index"),
                     resize_shape=None, add_special_tokens=add_special_tokens)
    input_ids = inputs.get("input_ids")
    shape = getattr(input_ids, "shape", ())
    if len(shape) != 2 or int(shape[0]) != 1 or int(shape[-1]) <= 0:
        raise RuntimeError("Native VLM processor did not return one nonempty tokenized request")
    kwargs = {key: value for key, value in inputs.items()
              if key not in {"attention_mask", "pixel_values"}}
    kwargs["pixel_values"] = inputs.get("pixel_values")
    kwargs["mask"] = inputs.get("attention_mask")
    return kwargs, int(shape[-1])


class PreparedTextInputMemo:
    """One exact native text input, retained only as bounded immutable CPU bytes.

    This supplies the tokenizer-memo option for VLMs, whose processor bypasses
    the ordinary MLX-LM tokenizer path. Hits rebuild fresh request arrays; no
    model, processor, Metal array, KV cache, or generated token is retained.
    Unknown processor configuration or non-text inputs keep native preparation.
    Integrations changing hidden processor state must invalidate prompt reuse or
    increment processor._smartai_revision, as for the text tokenizer memo.
    """
    def __init__(self, max_bytes=256 * 1024):
        self.max_bytes = max(0, int(max_bytes))
        self.hits = self.misses = 0
        self.last_reason = "not requested"
        self.clear()

    def clear(self):
        self._identity = self._prompt = self._snapshot = None
        self.nbytes = 0
        self.last_hit = False

    @staticmethod
    def _signature(model, processor):
        import hashlib
        import json
        from core.prompt_reuse import tokenizer_signature

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("processor has no stable text configuration")
        config = getattr(model, "config", None)
        value = lambda key: config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        # ProcessorMixin.to_dict deep-copies its decoder/tokenizer helpers before
        # filtering them. Reading the public text policy avoids those copies and
        # excludes generation-only state which changes after every response.
        runtime_only = {"tokenizer", "detokenizer", "stopping_criteria", "thinking_budget_criteria",
                        "image_processor", "video_processor", "audio_processor", "feature_extractor"}
        public = {key: val for key, val in vars(processor).items()
                  if not key.startswith("_") and key not in runtime_only and not callable(val)}
        for name in ("chat_template", "text_kwargs", "common_kwargs", "return_mm_token_type_ids",
                     "return_text_replacement_offsets", "unused_input_names", "skip_tensor_conversion"):
            public[name] = getattr(processor, name, None)
        details = {"processor": public,
                   "defaults": getattr(getattr(processor, "valid_processor_kwargs", None), "_defaults", None),
                   "revision": getattr(processor, "_smartai_revision", None),
                   "tokenizer_padding": getattr(tokenizer, "padding_side", None),
                   "tokenizer_truncation": getattr(tokenizer, "truncation_side", None),
                   "model_max_length": getattr(tokenizer, "model_max_length", None),
                   "model_type": value("model_type"), "image_token_index": value("image_token_index")}
        policy = json.dumps(details, sort_keys=True, allow_nan=False).encode()
        if len(policy) > 64 * 1024:
            raise ValueError("processor text policy is too large to memoize")
        signature = hashlib.sha256(policy).hexdigest()
        generation = importlib.import_module("mlx_vlm.generate")
        call = getattr(processor, "__call__", None)
        call = getattr(call, "__func__", call)
        return (id(model), id(processor), id(call), tokenizer_signature(tokenizer), signature,
                id(getattr(generation, "prepare_inputs", None)), id(generation.stream_generate))

    def prepare(self, model, processor, prepared_prompt, *, enabled=True):
        """Reuse only the same fully formatted text under the same native policy."""
        import sys
        identity = None
        self.last_hit = False
        if enabled and self.max_bytes:
            try:
                identity = self._signature(model, processor)
            except Exception:
                self.last_reason = "native processor configuration is not cacheable"
        if identity != self._identity or not enabled:
            self.clear()
        if identity is not None and prepared_prompt == self._prompt and self._snapshot is not None:
            try:
                import mlx.core as mx
                import numpy as np
                restored = {}
                for key, snapshot in self._snapshot.items():
                    if snapshot is None:
                        restored[key] = None
                    else:
                        dtype, shape, data = snapshot
                        restored[key] = mx.array(np.frombuffer(data, dtype=dtype).reshape(shape))
                self.hits += 1
                self.last_hit = True
                self.last_reason = "exact native text input reused"
                return restored, int(restored["input_ids"].shape[-1])
            except Exception:
                self.clear()

        self.misses += 1
        inputs, count = prepare_vlm_inputs(model, processor, prepared_prompt)
        # Replacing a prompt releases the prior bytes before making a new copy.
        self.clear()
        if not enabled:
            self.last_reason = "disabled"
            return inputs, count
        if count is None:
            self.last_reason = "native runtime has no prepared text input API"
        if identity is None or count is None:
            return inputs, count
        try:
            import mlx.core as mx
            import numpy as np
            if (not isinstance(prepared_prompt, str) or inputs.get("pixel_values") is not None
                    or set(inputs) - {"input_ids", "mask", "pixel_values", "token_type_ids", "mm_token_type_ids"}):
                raise ValueError("native input contains media or unsupported fields")
            size = sys.getsizeof(prepared_prompt) + 512
            snapshot = {}
            for key, array in inputs.items():
                if array is None:
                    snapshot[key] = None
                    continue
                if type(array) is not mx.array or size + int(array.nbytes) > self.max_bytes:
                    raise ValueError("native text input exceeds the CPU memo bound")
                cpu = np.asarray(array)
                if cpu.dtype.kind not in "biu" or cpu.shape != (1, count):
                    raise ValueError("native text input has an unsupported layout")
                data = cpu.tobytes()
                size += sys.getsizeof(data) + 128
                if size > self.max_bytes:
                    raise ValueError("native text input exceeds the CPU memo bound")
                snapshot[key] = (cpu.dtype.str, cpu.shape, data)
            # Native text preparation may initialize pad_token from eos_token.
            # Save its post-preparation policy so the next identical call can hit.
            identity = self._signature(model, processor)
            self._identity, self._prompt, self._snapshot = identity, prepared_prompt, snapshot
            self.nbytes = size
            self.last_reason = "exact native text input saved in CPU memo"
        except Exception:
            self.last_reason = "native text input not eligible for bounded CPU memo"
        return inputs, count

    def telemetry(self):
        return {"vlm_input_memo_hits": self.hits, "vlm_input_memo_misses": self.misses,
                "vlm_input_memo_eligible": self._snapshot is not None,
                "vlm_input_memo_used": self.last_hit,
                "vlm_input_memo_bytes": self.nbytes, "vlm_input_memo_reason": self.last_reason}
