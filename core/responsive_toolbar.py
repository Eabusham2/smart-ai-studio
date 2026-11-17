"""Compact left-aligned responsive header for the existing Tk controls.

Each group occupies its own full-width row and wraps internally.  Never
request a grid column wider than the available header width or stretch chips
across the window. All existing widgets, callbacks, and Tk variables survive.
"""
from __future__ import annotations


def install_responsive_toolbar(app, left, right):
    parent = app.hud_bar
    groups = (
        (left, list(left.pack_slaves())),
        (right, list(right.pack_slaves())),
    )
    pro = getattr(app, "chk_pro_branches", None)
    if pro is not None and pro in groups[1][1]:
        controls = groups[1][1]
        controls.remove(pro)
        controls.insert(0, pro)

    for frame, widgets in groups:
        for widget in widgets:
            widget.pack_forget()
        frame.pack_forget()
        # Separate rows: a wide model selector can never push the action bar
        # beyond the right edge or cause the other controls to justify.
        frame.pack(side="top", fill="x", anchor="w", padx=2, pady=1)
        frame.grid_columnconfigure(0, weight=0)

    previous_width = None
    pending = False

    def reflow():
        nonlocal pending, previous_width
        pending = False
        if not parent.winfo_exists():
            return
        # Use the actual header width (not the root window width). Include
        # its outer padding and allow enough room for borders on each row.
        available = max(160, parent.winfo_width() - 24)
        if available == previous_width:
            return
        previous_width = available
        for frame, widgets in groups:
            for widget in widgets:
                widget.grid_forget()
            row, column, used = 0, 0, 0
            for widget in widgets:
                if widget is getattr(app, "lbl_model_status", None):
                    widget.configure(wraplength=max(90, min(260, available - 30)))
                width = widget.winfo_reqwidth() + 10
                if column and used + width > available:
                    row += 1
                    column = 0
                    used = 0
                widget.grid(row=row, column=column, sticky="w", padx=3, pady=2)
                used += width
                column += 1

    def schedule(event=None):
        nonlocal pending
        if event is not None and event.widget is not parent:
            return
        if not pending:
            pending = True
            parent.after_idle(reflow)

    parent.bind("<Configure>", schedule, add="+")
    schedule()
