"""Stop a candidate at its first *complete*, valid tool-call boundary.

Pure text inspection: this module never dispatches tools. In particular,
an unfinished JSON tag, fenced example, or <think> block is not executable.
"""
from core.best_work import split_selected_action


class ToolBoundaryStopper:
    """Incremental detector for autoregressive model text events."""
    def __init__(self):
        self.text = ""
        self.stopped = False

    def feed(self, chunk):
        if self.stopped:
            return True
        self.text += str(chunk or "")
        _before, selected, tail = split_selected_action(self.text)
        if selected is not None:
            # End exactly at the closing </tool_call> / </media_call>.
            # Never expose candidate guesses after a real tool request.
            self.text = self.text[:-len(tail)] if tail else self.text
            self.stopped = True
        return self.stopped


class TokenBoundaryCriteria:
    """Transformers stopping criterion; no sampler, weights, or score changes."""
    def __init__(self, tokenizer, prompt_length, meter, *, enabled=False, cancelled=None):
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.meter = meter
        self.enabled = bool(enabled)
        self.cancelled = cancelled
        self.detector = ToolBoundaryStopper()

    def __call__(self, input_ids, scores=None, **kwargs):
        if callable(self.cancelled) and self.cancelled():
            return True
        ids = input_ids[0][self.prompt_length:]
        count = len(ids)
        self.meter.observe(count=count)
        if not self.enabled or count == 0:
            return False
        # Check a short trailing window first. Decode the full candidate only
        # when a closing tag appears, avoiding quadratic per-token decoding.
        tail = self.tokenizer.decode(ids[-64:], skip_special_tokens=False)
        if "</tool_call>" not in tail.casefold() and "</media_call>" not in tail.casefold():
            return False
        full = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.detector = ToolBoundaryStopper()
        return self.detector.feed(full)
