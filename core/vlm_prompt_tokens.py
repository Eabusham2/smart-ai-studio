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
