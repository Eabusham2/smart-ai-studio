"""Model-free tests proving a branch stops at a completed tool call."""
import threading

from core.tool_boundary_stop import ToolBoundaryStopper
from core.engines.gguf_engine import GGUFReasoningBackend


def test_split_valid_call_across_token_events_without_running_any_action():
    stop = ToolBoundaryStopper()
    assert not stop.feed("Before " + "<tool_call>{\"name\":")
    assert not stop.feed("\"read_file\",\"arguments\":{\"path\":\"a\"}}")
    assert stop.feed("</tool_call>DO NOT RUN THIS")
    assert stop.stopped
    assert stop.text.endswith("</tool_call>")
    assert "DO NOT RUN" not in stop.text


def test_partial_and_fenced_code_examples_do_not_stop():
    stop = ToolBoundaryStopper()
    assert not stop.feed("Example:\n" + chr(96) * 3 + "\n")
    assert not stop.feed('<tool_call>{"name":"read_file","arguments":{}}</tool_call>')
    assert not stop.feed("\n" + chr(96) * 3 + "\nFinal response.")
    assert not stop.stopped


def test_complete_media_call_stop_even_when_chunk_contains_tail():
    stop = ToolBoundaryStopper()
    assert stop.feed('<media_call>{"name":"media_generate","arguments":{"kind":"image"}}</media_call>FAKE RESULT')
    assert stop.text.endswith("</media_call>")
    assert "FAKE RESULT" not in stop.text


def test_gguf_branch_closes_stream_before_late_output():
    calls = []

    class FakeStream:
        def __init__(self):
            self.events = iter([
                {"choices": [{"text": "First. "}]},
                {"choices": [{"text": '<tool_call>{"name":"read_file","arguments":{}}</tool_call>'}]},
                {"choices": [{"text": "WRONG: made-up result"}]},
            ])
            self.closed = False
            self.read = 0

        def __iter__(self):
            return self

        def __next__(self):
            result = next(self.events)
            self.read += 1
            return result

        def close(self):
            self.closed = True

    def fake_model(prompt, **kwargs):
        assert kwargs["stream"] is True
        stream = FakeStream()
        calls.append(stream)
        return stream

    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = fake_model
    ticks = []
    result = backend.generate_branches(
        "prompt", branch_count=1, stop_at_tool_boundary=True,
        progress_callback=lambda *args, **kwargs: ticks.append((args, kwargs)),
    )
    assert result == ['First. <tool_call>{"name":"read_file","arguments":{}}</tool_call>']
    assert calls[0].closed and calls[0].read == 2
    assert ticks[-1][0][0] == 1


def test_plain_gguf_generation_preserves_nonstream_path():
    calls = []

    def model(prompt, **kwargs):
        calls.append(kwargs)
        return {"choices": [{"text": "Answer"}], "usage": {"completion_tokens": 5}}

    backend = GGUFReasoningBackend.__new__(GGUFReasoningBackend)
    backend.model = model
    counts = []
    assert backend.generate_branches("hi", progress_callback=lambda d,t,n,s: counts.append(n)) == ["Answer"]
    assert counts == [5]
    assert "stream" not in calls[0]
