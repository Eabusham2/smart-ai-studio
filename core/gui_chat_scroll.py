"""Keep streamed output following the tail only while the reader wants it."""
from __future__ import annotations


def install_chat_scroll(app, scrollbar):
    stream = app.chat_stream
    app._chat_follow_tail = True
    app._chat_scroll_epoch = 0

    def finish_scroll():
        if stream.winfo_exists():
            app._chat_follow_tail = float(stream.yview()[1]) >= 0.995

    def user_scroll(_event=None):
        # Stop pending tail callbacks immediately, before Tk's class binding
        # applies a wheel/key event. Resume only if that user event reaches end.
        app._chat_follow_tail = False
        app._chat_scroll_epoch += 1
        stream.after_idle(finish_scroll)

    def move_scrollbar(*args):
        user_scroll()
        stream.yview(*args)
        finish_scroll()

    scrollbar.configure(command=move_scrollbar)
    for event in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Prior>", "<Next>",
                  "<Home>", "<End>", "<Up>", "<Down>"):
        stream.bind(event, user_scroll, add="+")


def follow_chat_tail(app, *, force=False):
    if force:
        app._chat_follow_tail = True
    if not getattr(app, "_chat_follow_tail", True):
        return
    stream = app.chat_stream
    epoch = getattr(app, "_chat_scroll_epoch", 0)
    tab = app.active_tab_id

    def move():
        if (getattr(app, "_chat_follow_tail", True)
                and epoch == getattr(app, "_chat_scroll_epoch", 0)
                and tab == app.active_tab_id and stream.winfo_exists()):
            stream.see("end")
            stream.yview_moveto(1.0)

    move()
    # Text layout changes after an insertion; guard the second pass against a
    # user scroll or model switch which arrived between the two callbacks.
    app.root.after_idle(move)
