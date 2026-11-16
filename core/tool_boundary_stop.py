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
