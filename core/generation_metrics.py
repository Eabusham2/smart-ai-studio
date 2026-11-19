"""Observed generation telemetry. Text chunks and words are never token counts."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import functools
import math
import time
import uuid

_observer = ContextVar("smartai_generation_observer", default=None)


def valid_count(value):
    return type(value) is int and value >= 0


def valid_rate(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except (OverflowError, ValueError):
        return False


@contextmanager
def observe_generations(callback):
    previous = _observer.get()
    if previous is not None and previous is not callback:
        def notify(sample):
            previous(sample)
            callback(sample)
    else:
        notify = callback
    # The GUI and per-solve metadata can observe the same real invocation;
    # nesting must not hide live GUI measurements during a Pro search.
    token = _observer.set(notify)
    try:
        yield
    finally:
        _observer.reset(token)


class TokenMeter:
    """One real backend invocation; its decode clock starts on the first token."""
    def __init__(self, owner=None, source="model", clock=None, *, prompt_tokens=None):
        self.owner = owner
        self.source = source
        self.clock = clock or time.perf_counter
        self.started = self.clock()
        self.first = None
        self.last = None
        self.first_count = 0
        self.tokens = None
        # This is the complete logical input, including cached prefix tokens.
        # When the adapter already has the exact submitted token IDs, keep that
        # count: some runtimes report only the uncached suffix in responses.
        self.prompt_tokens = prompt_tokens if valid_count(prompt_tokens) else None
        self._prompt_from_inputs = valid_count(prompt_tokens)
        self.reported_rate = None
        self.id = uuid.uuid4().hex
        self.closed = False
        self.callback = _observer.get()
        if owner is not None:
            owner.last_prompt_token_count = self.prompt_tokens
            owner.last_generation_tokens = None
            owner.last_tok_per_sec = None
            owner.last_generation_seconds = None
        self._publish()

    def observe(self, *, count=None, delta=None, rate=None, text=False,
                prompt_tokens=None, prompt_from_inputs=False):
        now = self.clock()
        old = self.tokens
        if valid_count(prompt_tokens) and (prompt_from_inputs or not self._prompt_from_inputs):
            self.prompt_tokens = prompt_tokens
            self._prompt_from_inputs = self._prompt_from_inputs or bool(prompt_from_inputs)
        if valid_count(count):
            # A final backend usage record may exclude EOS although the raw
            # sampler observed it. Reconcile to that authoritative count.
            self.tokens = count
        elif valid_count(delta):
            self.tokens = (old or 0) + delta
        if text or (self.tokens is not None and self.tokens > (old or 0)):
            if self.first is None:
                self.first = now
                self.first_count = self.tokens or 0
            self.last = now
        if valid_rate(rate):
            self.reported_rate = float(rate)
        self._publish()

    def snapshot(self):
        end = self.last if self.closed and self.last is not None else self.clock()
        seconds = max(0.0, end - self.first) if self.first is not None else 0.0
        decode = self.reported_rate
        if decode is None and self.tokens is not None and self.tokens > self.first_count and seconds > 0:
            decode = (self.tokens - self.first_count) / seconds
        return {
            "id": self.id, "source": self.source, "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "first_token_at": self.first, "last_token_at": self.last,
            "ttft_s": None if self.first is None else self.first - self.started,
            "decode_tps": decode, "decode_seconds": seconds,
            "decode_intervals": None if self.tokens is None else max(0, self.tokens - self.first_count),
            "wall_s": max(0.0, self.clock() - self.started), "finished": self.closed,
        }

    def _publish(self):
        snap = self.snapshot()
        if self.owner is not None:
            self.owner.last_prompt_token_count = self.prompt_tokens
            self.owner.last_generation_tokens = self.tokens
            self.owner.last_tok_per_sec = snap["decode_tps"]
            self.owner.last_generation_seconds = snap["wall_s"]
            self.owner.last_generation_metrics = snap
        if callable(self.callback):
            self.callback(snap)

    def finish(self):
        if not self.closed:
            self.closed = True
            self._publish()


class RequestMetrics:
    """Aggregate all candidate invocations; never count display replay as decode."""
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.started = self.clock()
        self.samples = {}

    def update(self, sample):
        self.samples[sample["id"]] = dict(sample)
        return self.snapshot()

    def snapshot(self):
        samples = list(self.samples.values())
        measured = [s for s in samples if valid_count(s.get("tokens"))]
        complete = bool(samples) and len(measured) == len(samples)
        tokens = sum(s["tokens"] for s in measured) if complete else None
        decode_s = sum(s.get("decode_seconds", 0) for s in measured)
        intervals = sum(s.get("decode_intervals") or 0 for s in measured)
        # A failed older invocation must not replace the most recently started
        # request when that newer request finishes. Each sample is scoped to its
        # own real invocation; aggregate branch cost is kept separately below.
        live = samples[-1] if samples else {}
        first = [s["first_token_at"] for s in samples if s.get("first_token_at") is not None]
        return {
            "tokens": tokens,
            "decode_tps": intervals / decode_s if complete and intervals and decode_s > 0 else live.get("decode_tps"),
            "wall_tps": tokens / max(1e-9, self.clock() - self.started) if tokens is not None else None,
            "ttft_s": min(first) - self.started if first else None,
            "invocations": len(samples), "all_counts_available": complete,
            # Context is occupied by the current native invocation. It is not
            # the sum of sampled tokens from losing Pro branches, a prior tool
            # stage, or a display replay of the selected answer.
            "context": {
                "id": live.get("id"), "source": live.get("source"),
                "prompt_tokens": live.get("prompt_tokens"),
                "completion_tokens": live.get("tokens"),
                "finished": bool(live.get("finished")),
            } if live else None,
        }


@contextmanager
def observe_llama_tokens(model, meter):
    """Observe llama-cpp-python's sampled IDs, not its buffered UTF-8 chunks.

    The caller already owns the model's existing inference lock. Restoration is
    unconditional, including cancellation and a generator close at a tool call.
    """
    original = getattr(model, "generate", None)
    if not callable(original):
        yield False
        return
    own = getattr(model, "__dict__", {})
    had_own = "generate" in own
    previous = own.get("generate")

    @functools.wraps(original)
    def observed(*args, **kwargs):
        # llama-cpp constructs its full prompt (BOS/template/special IDs) before
        # calling generate(). Count that exact submitted sequence, not a second
        # encode with potentially different special-token defaults.
        prompt_ids = kwargs.get("tokens", args[0] if args else None)
        if isinstance(prompt_ids, (list, tuple)) and all(type(token) is int for token in prompt_ids):
            meter.observe(prompt_tokens=len(prompt_ids), prompt_from_inputs=True)
        iterator = original(*args, **kwargs)
        try:
            for token_id in iterator:
                meter.observe(delta=1)
                yield token_id
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    try:
        setattr(model, "generate", observed)
    except (AttributeError, TypeError):
        yield False
        return
    try:
        yield True
    finally:
        if had_own:
            setattr(model, "generate", previous)
        else:
            delattr(model, "generate")


def observe_mlx_response(meter, response):
    """MLX emits generation_tokens/token IDs; a string fallback is unmeasured."""
    count = getattr(response, "generation_tokens", None)
    token = getattr(response, "token", None)
    rate = getattr(response, "generation_tps", None)
    meter.observe(count=count if valid_count(count) else None,
                  delta=1 if not valid_count(count) and type(token) is int else None,
                  rate=rate, text=bool(getattr(response, "text", "")),
                  prompt_tokens=getattr(response, "prompt_tokens", None))


def observe_server_event(meter, event):
    """llama-compatible servers supply cumulative timing counts or token IDs."""
    timing = event.get("timings") if isinstance(event.get("timings"), dict) else {}
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    count = event.get("tokens_predicted", timing.get("predicted_n", usage.get("completion_tokens")))
    tokens = event.get("tokens")
    delta = len(tokens) if isinstance(tokens, list) and all(type(t) is int for t in tokens) else None
    meter.observe(count=count if valid_count(count) else None,
                  delta=delta if not valid_count(count) else None,
                  rate=timing.get("predicted_per_second"),
                  text=bool(event.get("content") or event.get("text")),
                  prompt_tokens=usage.get("prompt_tokens", event.get("tokens_evaluated", timing.get("prompt_n"))))
