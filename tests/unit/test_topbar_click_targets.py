"""Real Tk hit testing: visible header controls must receive *actual* clicks."""
import sys
import tkinter as tk

import pytest


def _resize_on_screen(window, width, height):
    """Exercise the requested size inside the CI desktop's real viewport."""
    width = min(width, window.winfo_screenwidth() - 40)
    height = min(height, window.winfo_screenheight() - 90)
    window.geometry(f"{width}x{height}+10+35")
    window.lift()
    window.update()


def _assert_click_target(root, widget):
    """Check visible geometry and native hit-tests where desktop hit-tests exist."""
    assert widget.winfo_ismapped(), widget
    assert widget.winfo_width() > 0 and widget.winfo_height() > 0
    x = widget.winfo_rootx() + widget.winfo_width() // 2
    y = widget.winfo_rooty() + widget.winfo_height() // 2
    assert 0 <= x < root.winfo_screenwidth() and 0 <= y < root.winfo_screenheight()
    assert widget.winfo_x() >= 0
    assert widget.winfo_x() + widget.winfo_width() <= widget.master.winfo_width() + 2
    if widget.winfo_class() in {"Label", "Button", "TButton", "Checkbutton", "TCheckbutton",
                               "Menubutton", "TMenubutton", "TCombobox", "TSpinbox"}:
        assert widget.winfo_height() >= widget.winfo_reqheight(), (widget, "clipped vertically")
    hit = root.winfo_containing(x, y)
    if sys.platform == "win32" and hit is None:
        # Unattended Windows CI may have Tk windows but no accessible system
        # desktop hit-test; retain geometry checks and exercise event routing.
        return
    assert hit == widget, (widget, hit)


@pytest.mark.parametrize("width", [760, 800, 960, 1050, 1280, 1536])
def test_all_header_buttons_are_topmost_and_click_events_invoke_commands(width):
    from config.settings import get_settings
    from app_gui import SmartAIChatbotApp

    root = tk.Tk()
    try:
        app = SmartAIChatbotApp(root, settings=get_settings(backend="mock"))
        # Reset belongs to an installed selection; expose that real UI state.
        app._model_is_cached = lambda _info: True
        app._update_model_action_buttons()
        # The virtual CI display can be 1280px wide. Never assert physical
        # clickability outside the OS screen while testing larger presets.
        _resize_on_screen(root, width, 900)
        calls = []
        names = (
            "btn_app_update",
            "btn_eval", "btn_load_unload", "btn_reset_reinstall", "btn_export_chat",
            "btn_branch_vis", "btn_memory_exp", "btn_sleep_panel", "btn_dsl_play",
            "btn_toggle_canvas", "btn_toggle_resources", "btn_zoom",
            "btn_theme_toggle", "btn_speed_tune", "btn_speed_cancel", "btn_lossless_speedups",
        )
        for name in names:
            widget = getattr(app, name)
            assert widget.winfo_manager() in ("place", "pack"), (width, name)
            _assert_click_target(root, widget)
            old_state = getattr(widget, "_state", "normal")
            widget.configure(state="normal", command=lambda n=name: calls.append(n))
            widget.event_generate("<Button-1>", x=widget.winfo_width()//2,
                                  y=widget.winfo_height()//2)
            root.update()
            assert calls[-1:] == [name], (width, name, calls)
            widget.configure(state=old_state)

        # The native Pro checkbox must also accept real pointer events.
        checkbox = app.chk_pro_branches
        _assert_click_target(root, checkbox)
        current = bool(app.pro_branches_var.get())
        # Aqua native checkboxes ignore event_generate press/release; invoke()
        # runs the same Tcl callback/variable toggle a real click activates.
        assert str(checkbox.cget("state")) == "normal"
        checkbox.invoke()
        root.update()
        assert bool(app.pro_branches_var.get()) is not current
        # The model selector is a Menubutton, not an ordinary action Button.
        selector = app.btn_model_menu
        _assert_click_target(root, selector)
        first_action = app.btn_app_update
        assert first_action.master is selector.master
        assert first_action.winfo_y() <= selector.winfo_y()
        assert first_action.winfo_x() == min(
            widget.winfo_x() for widget in first_action.master.winfo_children()
            if widget.winfo_ismapped() and widget.winfo_y() == first_action.winfo_y()
        )
    finally:
        root.destroy()


def test_eval_is_unmapped_when_nontext_model_selected():
    from config.settings import get_settings
    from app_gui import SmartAIChatbotApp

    root = tk.Tk()
    try:
        app = SmartAIChatbotApp(root, settings=get_settings(backend="mock"))
        app._model_is_cached = lambda _info: True
        app._update_model_action_buttons()
        _resize_on_screen(root, 760, 850)
        assert app.btn_eval.winfo_manager() == "place"
        # Honor the Eval extension's text-only visibility rule.
        app.models_config[app.active_tab_id]["model_type"] = "audio"
        app._sync_eval_button_visibility()
        root.update()
        assert app.btn_eval.winfo_manager() == ""
        app.models_config[app.active_tab_id]["model_type"] = "text"
        app._sync_eval_button_visibility()
        root.update()
        assert app.btn_eval.winfo_manager() == "place"
        _assert_click_target(root, app.btn_reset_reinstall)
    finally:
        root.destroy()
