"""Explicit, observable installation of the desktop's additive controls.

The packaged launcher executes app_gui.py as ``__main__``.  An installer must
therefore use the module passed by the launcher, rather than assuming that an
``app_gui`` import exists. Failed registration is rolled back before another
extension wraps the class, and is reported in both the log and the window.
"""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import sys
import traceback


EXTENSIONS = (
    ("Context Limit", "core.gui_generation_cap", "install_gui_generation_cap", "_generation_cap_ui_installed"),
    ("Eval", "core.gui_eval_panel", "install_gui_eval_panel", "_eval_panel_installed"),
    ("Auto speed", "core.gui_speed_tuning", "install_gui_speed_tuning", "_speed_tuning_ui_installed"),
    ("Speedups", "core.gui_lossless_speedups", "install_gui_lossless_speedups", "_lossless_speedups_ui_installed"),
)


def resolve_gui_module(module=None):
    if module is None:
        module = sys.modules.get("app_gui")
        if module is None:
            main = sys.modules.get("__main__")
            if os.path.basename(str(getattr(main, "__file__", ""))) == "app_gui.py":
                module = main
    if module is None or not hasattr(module, "SmartAIChatbotApp"):
        raise RuntimeError("GUI controls require the module defining SmartAIChatbotApp")
    if getattr(module, "tk", None) is None:
        raise RuntimeError("GUI controls require tkinter on the application module")
    return module


def report_gui_failure(module, feature, error, app=None):
    """Keep a durable diagnostic and a visible notice; never silently omit UI."""
    message = f"{type(error).__name__}: {error}"
    failures = getattr(module, "GUI_EXTENSION_FAILURES", None)
    if failures is None:
        failures = module.GUI_EXTENSION_FAILURES = []
    failure = {"feature": feature, "message": message}
    if failure not in failures:
        failures.append(failure)
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        logging.getLogger(__name__).error("%s control failed: %s\n%s", feature, message, detail)
        get_directory = getattr(module, "get_portable_data_dir", None)
        if callable(get_directory):
            try:
                log = Path(get_directory()) / "logs" / "gui-extensions.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(f"{feature}: {message}\n{detail}\n")
                module.GUI_EXTENSION_LOG = str(log)
            except OSError:
                logging.getLogger(__name__).exception("Could not save the GUI diagnostic log")
    if app is not None:
        show_gui_failures(app, module)


def install_gui_extensions(module):
    module = resolve_gui_module(module)
    cls = module.SmartAIChatbotApp
    module.GUI_EXTENSION_FAILURES = []
    for feature, path, name, flag in EXTENSIONS:
        before = dict(vars(cls))
        try:
            getattr(importlib.import_module(path), name)(module)
            if not getattr(cls, flag, False):
                raise RuntimeError(f"{name} returned without registering its controls")
        except Exception as error:
            # An exception halfway through registration must not leave wrappers
            # stacked on a partial installation or duplicate them on a retry.
            for attribute in set(vars(cls)) - set(before):
                delattr(cls, attribute)
            for attribute, value in before.items():
                if vars(cls).get(attribute) is not value:
                    setattr(cls, attribute, value)
            report_gui_failure(module, feature, error)
    return module.GUI_EXTENSION_FAILURES


def show_gui_failures(app, module):
    failures = getattr(module, "GUI_EXTENSION_FAILURES", ())
    if not failures:
        return
    detail = "; ".join(f"{row['feature']}: {row['message']}" for row in failures)
    log = getattr(module, "GUI_EXTENSION_LOG", "the application log")
    text = f"Some controls could not be initialized. {detail}\nDetails: {log}"
    label = getattr(app, "_gui_failure_notice", None)
    if label is None:
        label = app._gui_failure_notice = module.tk.Label(
            app.main_container, anchor="w", justify="left", padx=8, pady=6,
            bg=app.C["bg_hud"], fg=app.C["accent_red"],
        )
        label.pack(fill="x", before=app.hud_bar, padx=10, pady=2)
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(160, event.width - 20)))
    label.configure(text=text)
