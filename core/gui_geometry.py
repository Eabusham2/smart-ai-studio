"""Keep newly opened desktop windows within the current screen's usable area."""
from __future__ import annotations


def fit_window_to_screen(window, width, height, *, min_width=400, min_height=300, parent=None):
    """Size and center a window without placing its controls beyond the screen.

    Reserve room for desktop chrome. Do not impose a maximum size: users may
    move the window to a larger monitor or enlarge it after opening it.
    """
    screen_width, screen_height = window.winfo_screenwidth(), window.winfo_screenheight()
    available_width = max(1, screen_width - 32)
    available_height = max(1, screen_height - 100)
    minimum_width = min(min_width, available_width)
    minimum_height = min(min_height, available_height)
    width = max(minimum_width, min(width, available_width))
    height = max(minimum_height, min(height, available_height))
    window.minsize(minimum_width, minimum_height)
    if parent is None:
        parent = getattr(window, "master", None)
    if parent is not None and parent.winfo_width() > 1:
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    else:
        x, y = (screen_width - width) // 2, (screen_height - height) // 2
    x = max(16, min(x, screen_width - width - 16))
    y = max(32, min(y, screen_height - height - 56))
    window.geometry(f"{width}x{height}+{x}+{y}")
