"""Real Transformers streaming with observed IDs and cooperative boundary stop."""
from __future__ import annotations

from contextvars import copy_context
from queue import Empty
import threading

from core.generation_metrics import TokenMeter
from core.tool_boundary_stop import TokenBoundaryCriteria, ToolBoundaryStopper


def stream_transformers(model, tokenizer, inputs, options, *, owner=None,
                        stop_at_tool_boundary=False, cancelled=None):
    from transformers import TextIteratorStreamer, StoppingCriteriaList
    import torch
    meter = TokenMeter(owner, "transformers")
    prompt = inputs.get("input_ids") if hasattr(inputs, "get") else None
    encoder_decoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    prefix = 1 if encoder_decoder else int(prompt.shape[-1]) if prompt is not None else 0
    stopped = threading.Event()
    is_cancelled = lambda: stopped.is_set() or (callable(cancelled) and cancelled())
    criteria = TokenBoundaryCriteria(tokenizer, prefix, meter,
                                    enabled=stop_at_tool_boundary, cancelled=is_cancelled)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.1)
    errors = []
    kwargs = dict(options)
    previous = list(kwargs.pop("stopping_criteria", ()) or ())
    kwargs.update(streamer=streamer, stopping_criteria=StoppingCriteriaList([*previous, criteria]))

    def generate():
        try:
            with torch.no_grad():
                model.generate(**inputs, **kwargs)
        except BaseException as exc:
            errors.append(exc)
            streamer.end()
        finally:
            meter.finish()

    context = copy_context()
    worker = threading.Thread(target=lambda: context.run(generate), daemon=True)
    worker.start()
    detector = ToolBoundaryStopper() if stop_at_tool_boundary else None
    try:
        while True:
            try:
                piece = next(streamer)
            except Empty:
                if not worker.is_alive():
                    break
                continue
            except StopIteration:
                break
            if is_cancelled():
                break
            stop_here = False
            if detector is not None:
                before = len(detector.text)
                stop_here = detector.feed(piece)
                piece = detector.text[before:]
            if piece:
                yield piece
            if stop_here:
                break
        if errors:
            raise errors[0]
    finally:
        stopped.set()
        # Never release the owning model lock while generate still uses it.
        worker.join()
        meter.finish()
