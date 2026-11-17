"""Tk top bar reflows existing controls, never hides Pro mode when resized."""
from __future__ import annotations


def install_responsive_toolbar(app, left, right):
    parent = app.hud_bar
    left_widgets = list(left.pack_slaves())
    right_widgets = list(right.pack_slaves())
    # A model can be generating while the window is resized. Preserve widgets,
    # commands and Tk variables; reposition only (no re-creation).
    pro = getattr(app, "chk_pro_branches", None)
    if pro in right_widgets:
        right_widgets.remove(pro)
        right_widgets.insert(0, pro)
    for widget in left_widgets + right_widgets:
        widget.pack_forget()
    for box in (left, right):
        box.pack_forget()

    previous = None
    scheduled = False

    def flow(widgets, available):
        row = col = used = 0
        for widget in widgets:
            # Wrap long status badges, rather than clipping their text.
            if widget is getattr(app, "lbl_model_status", None):
                widget.configure(wraplength=max(130, min(360, available - 28)))
            width = widget.winfo_reqwidth() + 12
            if col and used + width > available:
                row += 1
                col = used = 0
            widget.grid(row=row, column=col, padx=3, pady=2, sticky="w")
            used += width
            col += 1

    def reflow():
        nonlocal previous, scheduled
        scheduled = False
        if not parent.winfo_exists():
            return
        available = max(240, app.root.winfo_width() - 38)
        if previous is not None and abs(previous - available) < 9:
            return
        previous = available
        for widget in left_widgets + right_widgets:
            widget.grid_forget()
        for box in (left, right):
            box.grid_forget()
        # Left controls flow by themselves if even the brand/model row is long.
        flow(left_widgets, available)
        left_width = sum(w.winfo_reqwidth() + 12 for w in left_widgets)
        right_width = sum(w.winfo_reqwidth() + 12 for w in right_widgets)
        together = left_width + right_width <= available
        left.grid(row=0, column=0, sticky="w", padx=2, pady=1)
        right.grid(row=0 if together else 1, column=1 if together else 0,
                   columnspan=1 if together else 2, sticky="w", padx=2, pady=1)
        flow(right_widgets, max(240, available - left_width - 8) if together else available)
        parent.grid_columnconfigure(0, weight=1 if together else 0)

    def changed(event=None):
        nonlocal scheduled
        if event is not None and event.widget is not app.root:
            return
        if not scheduled:
            scheduled = True
            app.root.after_idle(reflow)

    app.root.bind("<Configure>", changed, add="+")
    changed()
