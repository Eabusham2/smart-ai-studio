"""Keep streamed output following the tail only while the reader wants it."""
from __future__ import annotations

import re


def capture_reader_position(stream, replaced_from):
    """Keep a bounded content anchor before replacing streamed Markdown."""
    position = stream.index("@0,0")
    if stream.compare(position, "<", replaced_from):
        return {"index": position}  # Earlier messages are not being replaced.
    line, column = map(int, position.split("."))
    origin_line = int(stream.index(replaced_from).split(".")[0])
    excerpt = stream.get(position, f"{position}+160c").split("\n", 1)[0]
    # Plain text survives unchanged; formatting delimiters may disappear.
    # Keep a few literal fragments rather than copying/diffing a long reply.
    fragments = [(0, excerpt)] if excerpt.strip() else []
    parts = []
    for match in re.finditer(r"[^*_`~#]+", excerpt):
        raw = match.group()
        value = raw.strip()
        if len(value) >= 8 and value != excerpt:
            parts.append((match.start() + len(raw) - len(raw.lstrip()), value))
    fragments.extend(sorted(parts, key=lambda part: -len(part[1]))[:3])
    return {"line": line - origin_line, "column": column, "fragments": fragments}


def restore_reader_position(stream, replaced_from, anchor):
    """Restore the same visible text where possible, without following the tail."""
    if "index" in anchor:
        stream.yview(anchor["index"])
        return
    start, end = stream.index(replaced_from), stream.index("end-1c")
    expected_line = int(start.split(".")[0]) + anchor["line"]
    expected = stream.index(f"{expected_line}.{anchor['column']}")
    repeated = []
    chosen = None
    for offset, text in anchor["fragments"]:
        first = stream.search(text, start, stopindex=end, exact=True)
        if not first:
            continue
        last = stream.search(text, end, stopindex=start, backwards=True, exact=True)
        if not last or first == last:
            chosen = (offset, first)
            break
        repeated.append((offset, text))
    if chosen is None:
        matches = []
        for offset, text in repeated:
            for backwards in (False, True):
                found = stream.search(text, expected, stopindex=start if backwards else end,
                                      backwards=backwards, exact=True)
                if found:
                    line, column = map(int, found.split("."))
                    distance = (abs(line - expected_line), abs(column - offset - anchor["column"]))
                    matches.append((distance, offset, found))
        if matches:
            _, offset, found = min(matches)
            chosen = (offset, found)
    target = expected  # The line/column fallback stays inside the rewritten reply.
    if chosen is not None:
        offset, found = chosen
        line, column = map(int, found.split("."))
        target = stream.index(f"{line}.{max(0, column - offset)}")
    if stream.compare(target, "<", start):
        target = start
    stream.yview(target)


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
