"""Recolor the complete live Tk tree, including hidden panels and open dialogs.

Remember each option's palette role so repeated light/dark switches do not lose
the original dark shades when several roles share one light color. Widget and
tag identities, content, focus, scroll positions and commands are untouched.
"""
from __future__ import annotations

import tkinter as tk


_OPTIONS = (
    "background", "foreground", "activebackground", "activeforeground",
    "disabledbackground", "disabledforeground", "selectbackground", "inactiveselectbackground",
    "selectforeground", "selectcolor", "insertbackground",
    "highlightbackground", "highlightcolor", "troughcolor",
)


def _role_for(color, option, old, widget):
    matches = [key for key, value in old.items() if str(value).lower() == str(color).lower()]
    if not matches:
        return None
    preferences = []
    if option in {"foreground", "activeforeground", "disabledforeground", "insertbackground", "selectforeground"}:
        preferences = ["text_main", "text_muted", "btn_fg", "code_fg"]
    elif option in {"highlightbackground", "highlightcolor"}:
        preferences = ["border", "border_focus"]
    elif option == "selectbackground":
        preferences = ["bg_card_hover", "border_focus", "bg_inline_code"]
    elif isinstance(widget, (tk.Entry, tk.Text)):
        preferences = ["bg_input_inner", "code_bg", "bg_chat", "bg_card"]
    else:
        preferences = ["bg_hud", "bg_app", "bg_card", "btn_bg"]
    return next((role for role in preferences if role in matches), matches[0])


def _replacement(widget, cache, key, value, option, old, new, explicit=None):
    prior = cache.get(key)
    role = explicit or (prior[0] if prior and str(value) == prior[1] else None)
    if role is None:
        role = _role_for(value, option, old, widget)
    if role is None:
        return None
    updated = str(new[role])
    cache[key] = (role, updated)
    return updated


def apply_widget_palette(root, old, new, overrides=None):
    """Apply mapped colors to every existing widget without rebuilding the UI."""
    overrides = overrides or {}

    def visit(widget):
        cache = getattr(widget, "_studio_theme_roles", None)
        if cache is None:
            cache = widget._studio_theme_roles = {}
        explicit = overrides.get(widget, {})
        options = set(widget.keys())
        updates = {}
        for option in _OPTIONS:
            if option not in options:
                continue
            value = widget.cget(option)
            updated = _replacement(widget, cache, ("option", option), value, option, old, new, explicit.get(option))
            if updated is not None:
                # SolidButton intercepts bg/fg to keep its hover/normal state
                # synchronized with the colors currently rendered by Tk.
                updates[{"background": "bg", "foreground": "fg"}.get(option, option)] = updated
        if updates:
            widget.configure(**updates)
        for option, attribute in (("activebackground", "_hover_bg"), ("activeforeground", "_hover_fg")):
            if option not in options and hasattr(widget, attribute):
                value = getattr(widget, attribute)
                updated = _replacement(widget, cache, ("hover", option), value, option, old, new, explicit.get(option))
                if updated is not None:
                    widget.configure(**{option: updated})
        if isinstance(widget, tk.Text):
            tags = set(widget.tag_names())
            for key in tuple(cache):
                if key[0] == "tag" and key[1] not in tags:
                    del cache[key]
            for tag in tags:
                changes = {}
                for option in ("background", "foreground"):
                    value = widget.tag_cget(tag, option)
                    updated = _replacement(widget, cache, ("tag", tag, option), value, option, old, new)
                    if updated is not None:
                        changes[option] = updated
                if changes:
                    widget.tag_configure(tag, **changes)
        elif isinstance(widget, tk.Menu):
            last = widget.index("end")
            for index in range(int(last) + 1 if last is not None else 0):
                available = widget.entryconfigure(index)
                changes = {}
                for option in ("background", "foreground", "activebackground", "activeforeground"):
                    if option not in available:
                        continue
                    value = widget.entrycget(index, option)
                    updated = _replacement(widget, cache, ("menu", index, option), value, option, old, new)
                    if updated is not None:
                        changes[option] = updated
                if changes:
                    widget.entryconfigure(index, **changes)
        elif isinstance(widget, tk.Canvas):
            items = set(widget.find_all())
            for key in tuple(cache):
                if key[0] == "canvas" and key[1] not in items:
                    del cache[key]
            for item in items:
                available = widget.itemconfigure(item)
                changes = {}
                for option in ("fill", "outline"):
                    if option not in available:
                        continue
                    value = widget.itemcget(item, option)
                    updated = _replacement(widget, cache, ("canvas", item, option), value, "foreground", old, new)
                    if updated is not None:
                        changes[option] = updated
                if changes:
                    widget.itemconfigure(item, **changes)
        for child in widget.winfo_children():
            visit(child)

    visit(root)
