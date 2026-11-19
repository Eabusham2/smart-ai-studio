"""Post worker results without calling the Tk interpreter from that worker."""
from __future__ import annotations

import threading


def post_ui(app, callback, *, delay_ms=0):
    """The GUI owner drains queued work; delayed work is scheduled there too."""
    if getattr(app, "_ui_poll_closed", False):
        return None
    owner = getattr(app, "_ui_thread_ident", threading.main_thread().ident)
    if threading.get_ident() == owner:
        return app.root.after(delay_ms, callback)
    events = getattr(app, "_event_queue", None)
    if events is None:
        raise RuntimeError("A background GUI callback requires the application event queue")

    def dispatch():
        if getattr(app, "_ui_poll_closed", False):
            return
        if delay_ms:
            app.root.after(delay_ms, callback)
        else:
            callback()

    events.put(("ui_callback", dispatch))
    return None
