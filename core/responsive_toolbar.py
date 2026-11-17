"""Wrap the existing Tk header controls without grid/pack conflicts or stretched columns.

The controls retain their original Tk parents, callbacks, and variables. `place`
is used only for their inner layout; the two full-width header groups continue
to stack with `pack`. Each row is laid out independently, so long badges never
widen a shared grid column and push controls outside the visible window.
"""
from __future__ import annotations


def install_responsive_toolbar(app, left, right):
    header = app.hud_bar
    groups = ((left, list(left.pack_slaves())), (right, list(right.pack_slaves())))
    pro = getattr(app, "chk_pro_branches", None)
    if pro is not None and pro in groups[1][1]:
        controls = groups[1][1]
        controls.remove(pro)
        controls.insert(0, pro)

    for frame, widgets in groups:
        for widget in widgets:
            widget.pack_forget()
        frame.pack_forget()
        frame.pack(side="top", anchor="w", fill="x", padx=3, pady=1)
        frame.pack_propagate(False)

    previous = None
    pending = False

    def reflow():
        nonlocal previous, pending
        pending = False
        if not header.winfo_exists():
            return
        available = max(160, header.winfo_width() - 22)
        # Do not skip resizing for a newly changed label/selector width.
        requirements = tuple(w.winfo_reqwidth() for _, widgets in groups for w in widgets)
        if previous == (available, requirements):
            return
        previous = (available, requirements)
        for frame, widgets in groups:
            x, y, row_height = 3, 3, 0
            for widget in widgets:
                if widget is getattr(app, "lbl_model_status", None):
                    widget.configure(wraplength=max(96, min(260, available - 24)))
                if widget is getattr(app, "btn_model_menu", None):
                    # Extremely long imported model names remain readable
                    # on narrow windows instead of clipping inside the menu.
                    widget.configure(wraplength=max(130, available - 28))
                width = min(widget.winfo_reqwidth(), available - 8)
                height = widget.winfo_reqheight()
                if x > 3 and x + width + 3 > available:
                    y += row_height + 4
                    x, row_height = 3, 0
                widget.place(x=x, y=y, width=width, height=height)
                x += width + 6
                row_height = max(row_height, height)
            frame.configure(height=y + row_height + 3)

    def schedule(event=None):
        nonlocal pending
        if event is not None and event.widget is not header:
            return
        if not pending:
            pending = True
            header.after_idle(reflow)

    header.bind("<Configure>", schedule, add="+")
    selector = getattr(app, "model_var", None)
    if selector is not None and hasattr(selector, "trace_add"):
        selector.trace_add("write", lambda *_: schedule())

    def watch_requested_sizes():
        # Status badges and load buttons change labels without a <Configure>
        # event because their place size is held fixed. Poll their requested
        # sizes cheaply, and reflow only if something actually changed.
        if not header.winfo_exists():
            return
        available = max(160, header.winfo_width() - 22)
        sizes = tuple(w.winfo_reqwidth() for _, widgets in groups for w in widgets)
        if previous != (available, sizes):
            schedule()
        header.after(160, watch_requested_sizes)

    schedule()
    header.after(160, watch_requested_sizes)
