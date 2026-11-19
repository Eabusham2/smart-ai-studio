"""Wrap the existing Tk header controls without grid/pack conflicts or stretched columns.

The controls retain their original Tk parents, callbacks, and variables. `place`
is used only for their inner layout; the two full-width header groups continue
to stack with `pack`. Each row is laid out independently, so long badges never
widen a shared grid column and push controls outside the visible window.
"""
from __future__ import annotations


def install_wrapping_controls(frame):
    """Wrap every existing action in a dialog/pane toolbar within its width."""
    import tkinter as tk

    controls = tuple(frame.pack_slaves())
    original_wrap = {widget: widget.cget("wraplength") for widget in controls
                     if "wraplength" in widget.keys()}
    for widget in controls:
        widget.pack_forget()
    frame.pack_propagate(False)
    pending = None
    previous = None
    timer_id = None

    def reflow():
        nonlocal pending, previous
        pending = None
        if not frame.winfo_exists():
            return
        # Before Tk allocates the parent's width this is 1px. Reflowing then
        # permanently wrapped every label at 60px and could fill a whole dialog
        # vertically. Its first real Configure event will schedule layout.
        if frame.winfo_width() <= 1:
            return
        available = max(1, frame.winfo_width() - 8)
        sizes = tuple((widget.winfo_reqwidth(), widget.winfo_reqheight()) for widget in controls)
        if previous == (available, sizes):
            return
        x, y, row_height = 4, 4, 0
        for widget in controls:
            if widget in original_wrap:
                # Reconsider natural width on every resize/text change. A
                # narrow window must not leave tall labels behind when widened.
                widget.configure(wraplength=original_wrap[widget])
                if widget.winfo_reqwidth() > available:
                    widget.configure(wraplength=max(1, available - 20))
            width = min(widget.winfo_reqwidth(), available)
            height = widget.winfo_reqheight()
            if x > 4 and x + width > available + 4:
                y += row_height + 5
                x, row_height = 4, 0
            widget.place(x=x, y=y, width=width, height=height)
            x += width + 6
            row_height = max(row_height, height)
        frame.configure(height=y + row_height + 4)
        previous = (available, tuple((widget.winfo_reqwidth(), widget.winfo_reqheight()) for widget in controls))

    def schedule(event=None):
        nonlocal pending
        if event is not None and event.widget is not frame:
            return
        if pending is None:
            pending = frame.after_idle(reflow)

    def destroy(event):
        nonlocal pending, timer_id
        if event.widget is not frame:
            return
        for callback in (pending, timer_id):
            if callback is None:
                continue
            try:
                frame.after_cancel(callback)
            except tk.TclError:
                pass
        pending = timer_id = None

    def watch_sizes():
        nonlocal timer_id
        if not frame.winfo_exists():
            return
        state = (max(1, frame.winfo_width() - 8),
                 tuple((widget.winfo_reqwidth(), widget.winfo_reqheight()) for widget in controls))
        if state != previous:
            schedule()
        timer_id = frame.after(160, watch_sizes)

    frame.bind("<Configure>", schedule, add="+")
    frame.bind("<Destroy>", destroy, add="+")
    schedule()
    timer_id = frame.after(160, watch_sizes)
    return schedule


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

    def set_control_visible(widget, visible=True, *, before=None):
        """Register late-added actions with the SAME responsive hit-test layout.

        Eval and other additive controls are created after this header. If
        they use ``pack`` over a placed row, they can steal clicks from Reset.
        """
        nonlocal previous
        if widget.master not in (left, right):
            raise ValueError("Responsive toolbar can only manage its own controls")
        widgets = groups[0][1] if widget.master is left else groups[1][1]
        if widget in widgets:
            widgets.remove(widget)
        # Never let the old geometry manager keep an invisible overlay on top
        # of the model's Load/Reset/Pro buttons.
        widget.pack_forget()
        widget.place_forget()
        if visible:
            index = widgets.index(before) if before in widgets else len(widgets)
            widgets.insert(index, widget)
        previous = None
        schedule()

    app._responsive_toolbar_set_visible = set_control_visible
    header.bind("<Configure>", schedule, add="+")
    selector = getattr(app, "model_var", None)
    if selector is not None and hasattr(selector, "trace_add"):
        selector.trace_add("write", lambda *_: schedule())

    timer_id = None

    def watch_requested_sizes():
        nonlocal timer_id
        # Status badges and load buttons change labels without a <Configure>
        # event because their place size is held fixed. Poll their requested
        # sizes cheaply, and reflow only if something actually changed.
        if not header.winfo_exists():
            return
        available = max(160, header.winfo_width() - 22)
        sizes = tuple(w.winfo_reqwidth() for _, widgets in groups for w in widgets)
        if previous != (available, sizes):
            schedule()
        timer_id = header.after(160, watch_requested_sizes)

    def destroy(event):
        nonlocal timer_id
        if event.widget is header and timer_id is not None:
            try:
                header.after_cancel(timer_id)
            except tk.TclError:
                pass
            timer_id = None

    import tkinter as tk
    header.bind("<Destroy>", destroy, add="+")
    schedule()
    timer_id = header.after(160, watch_requested_sizes)
