"""Worker-to-GUI posting never enters Tk before the owner drains the queue."""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from core.gui_dispatch import post_ui


def test_worker_posts_callbacks_and_delays_without_touching_tk():
    owner = threading.get_ident()
    calls, ready = [], []

    def after(delay, callback):
        assert threading.get_ident() == owner, "A worker called Tk.after"
        calls.append(delay)
        ready.append(callback)

    app = SimpleNamespace(root=SimpleNamespace(after=after), _event_queue=queue.Queue(), _ui_thread_ident=owner)
    observed = []
    errors = []

    def worker():
        try:
            post_ui(app, lambda: observed.append(threading.get_ident()))
            post_ui(app, lambda: observed.append('delayed'), delay_ms=75)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start(); thread.join(timeout=2)
    assert not thread.is_alive() and not errors
    assert not calls and not observed
    kind, callback = app._event_queue.get_nowait()
    assert kind == 'ui_callback'
    callback()
    assert observed == [owner]
    _, callback = app._event_queue.get_nowait()
    callback()
    assert calls == [75] and observed == [owner]
    ready.pop()()
    assert observed == [owner, 'delayed']


def test_owner_retains_after_scheduling_and_closed_window_discards_pending_callbacks():
    app = SimpleNamespace(root=SimpleNamespace(after=Mock(return_value='after-1')), _event_queue=queue.Queue())
    callback = Mock()
    assert post_ui(app, callback, delay_ms=12) == 'after-1'
    app.root.after.assert_called_once_with(12, callback)
    thread = threading.Thread(target=lambda: post_ui(app, callback))
    thread.start(); thread.join(timeout=2)
    _, dispatched = app._event_queue.get_nowait()
    app._ui_poll_closed = True
    dispatched()
    assert post_ui(app, callback) is None
    callback.assert_not_called()
    assert app.root.after.call_count == 1
