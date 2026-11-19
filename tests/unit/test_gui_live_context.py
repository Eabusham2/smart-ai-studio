"""Real Tk paints actual per-invocation context; stale owners cannot repaint it."""
from types import SimpleNamespace
import threading

from core.generation_metrics import TokenMeter
from tests.unit.test_best_work_chat_ui import work_app
from tests.unit.test_strata_gui import app as harness_app


def test_context_updates_on_prefill_and_decode_then_labels_completed_invocation(work_app):
    app = work_app
    app.is_model_loaded = True
    app.engine._context_budget_tokens = 2048
    app._set_generating_state(True)
    snapshots = []
    done = threading.Event()
    failures = []

    def capture():
        painted = threading.Event()
        def record():
            snapshots.append((app.total_tokens_used, app.lbl_context.cget("text")))
            painted.set()
        app._event_queue.put(("ui_callback", record))
        assert painted.wait(3), "Tk did not process queued live context updates"

    def stream(prompt, **kwargs):
        # The first branch is real compute cost but its sampled output is not
        # part of the second native invocation's occupied context.
        earlier = TokenMeter(prompt_tokens=90)
        earlier.observe(count=7)
        earlier.finish()
        current = TokenMeter(prompt_tokens=111)
        capture()  # The new prompt must paint even inside the decode throttle.
        current.observe(count=1)
        yield "Visible "
        capture()
        current.observe(count=4)
        yield "answer."
        current.finish()

    app.multimodal.stream_solve = stream
    def run():
        try:
            app._process_message_thread("question", "question", None, None, False)
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()
    worker = threading.Thread(target=run, daemon=True)
    def poll():
        if done.is_set():
            app.root.after(60, app.root.quit)
        else:
            app.root.after(10, poll)
    app.root.after(10, worker.start)
    app.root.after(20, poll)
    app.root.after(9000, app.root.quit)
    app.root.mainloop()
    worker.join(timeout=2)
    assert done.is_set() and not failures
    assert snapshots[0][0] == 111 and "Live prompt: 111" in snapshots[0][1]
    assert snapshots[1][0] == 112 and "Live context: 112" in snapshots[1][1]
    assert not app.is_generating and app.total_tokens_used == 115
    assert "Last invocation: 115" in app.lbl_context.cget("text")
    assert app._live_generation_snapshot["tokens"] == 11
    assert app._last_context_snapshot["prompt_tokens"] == 111
    assert app._last_context_snapshot["completion_tokens"] == 4


def test_queued_context_callbacks_cannot_mutate_replacement_model(harness_app):
    app = harness_app
    app._live_generation_snapshot = {"replacement": True}
    app._live_context_snapshot = None
    def stream(prompt, **kwargs):
        meter = TokenMeter(prompt_tokens=712)
        meter.observe(count=5)
        yield "old model output"
        meter.finish()
    app.multimodal.stream_solve = stream
    app._process_message_thread("question", "question")
    # Simulate a completed unload/reload before the Tk owner drains this
    # request's updates. The tab may have the same label, but the engine differs.
    app.engine = SimpleNamespace(active_backend="torch")
    app.total_tokens_used = 33
    app.lbl_context.configure(text="replacement model context")
    app.root.drain()
    assert app._live_generation_snapshot == {"replacement": True}
    assert app._live_context_snapshot is None
    assert app.total_tokens_used == 33
    assert app.lbl_context.cget("text") == "replacement model context"
