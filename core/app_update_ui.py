"""Background release checks and the desktop's first Update button."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk
import webbrowser

from core.gui_dispatch import post_ui
from core.gui_geometry import fit_window_to_screen


_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000


def _new_updater():
    from core.app_updates import AppUpdater
    return AppUpdater()


class AppUpdateUI:
    """Keep network/install work off Tk and restart only when work is idle."""

    def __init__(self, app, *, updater_factory=None):
        self.app = app
        self._updater_factory = updater_factory or _new_updater
        self._updater = None
        self._worker = None
        self._startup_worker = None
        self._cleanup_workers = []
        self._timer = None
        self._ack_timer = None
        self._closed = False
        self._busy = False
        self._phase = "idle"
        self._checked = False
        self._current_label = "Reading installed version…"
        self._message = "Updates are checked automatically in the background."
        self._release = None
        self._prepared = None
        self._helper_owned = False
        self._progress_value = 0
        self._progress_time = 0.0
        self._dialog = None
        self._widgets = {}

    def create_button(self, parent, *, font):
        colors = self.app.C
        button = tk.Button(
            parent, text="↑ Update", font=font, command=self.show,
            bg=colors["btn_bg"], fg=colors["btn_fg"],
            activebackground=colors["btn_hover"], activeforeground=colors["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
        )
        button.pack(side="left", padx=(0, 6))
        self.app.btn_app_update = button
        return button

    def start(self):
        self.app.root.bind("<Destroy>", self._on_destroy, add="+")
        # Wait until all constructor wrappers finish and Tk processes events.
        self._ack_timer = self.app.root.after_idle(self._start_acknowledgement)
        offline = os.environ.get("SMARTAI_TEST_OFFLINE", "0").lower() in {"1", "true", "yes"}
        if getattr(self.app.settings, "backend", None) != "mock" and not offline:
            self._timer = self.app.root.after(1500, self._automatic_check)

    def _start_acknowledgement(self):
        self._ack_timer = None
        if self._closed:
            return
        self._startup_worker = threading.Thread(
            target=self._acknowledge_startup, name="app-update-startup", daemon=True,
        )
        self._startup_worker.start()

    @staticmethod
    def _acknowledge_startup():
        try:
            from core.update_installer import acknowledge_startup
            acknowledge_startup(Path(__file__).resolve().parents[1])
        except Exception:
            logging.getLogger(__name__).exception("Could not acknowledge application update startup")

    def _on_destroy(self, event):
        if event.widget is not self.app.root:
            return
        self._closed = True
        for timer in (self._timer, self._ack_timer):
            if timer is None:
                continue
            try:
                self.app.root.after_cancel(timer)
            except tk.TclError:
                pass
        self._timer = self._ack_timer = None
        if self._prepared is not None and not self._busy and not self._helper_owned:
            prepared, self._prepared = self._prepared, None
            self._discard_prepared(prepared)

    def _discard_prepared(self, prepared):
        """Remove an unused verified staging directory without blocking Tk."""
        if prepared is None:
            return
        updater = self._updater

        def discard():
            try:
                updater.discard(prepared)
            except Exception:
                logging.getLogger(__name__).exception("Could not remove unused application update")

        # Let an already-started cleanup finish when normal GUI shutdown exits.
        worker = threading.Thread(target=discard, name="app-update-cleanup", daemon=False)
        self._cleanup_workers = [thread for thread in self._cleanup_workers if thread.is_alive()]
        self._cleanup_workers.append(worker)
        worker.start()

    def _automatic_check(self):
        self._timer = None
        if self._closed:
            return
        if not self._busy:
            self.check()
        self._timer = self.app.root.after(_CHECK_INTERVAL_MS, self._automatic_check)

    def _get_updater(self):
        if self._updater is None:
            self._updater = self._updater_factory()
        return self._updater

    def check(self):
        if self._closed or self._busy:
            return
        self._busy, self._phase = True, "checking"
        self._message = "Checking GitHub Releases…"
        self._render()

        def work():
            current = None
            try:
                updater = self._get_updater()
                current = updater.current_label
                result = updater.check()
            except Exception as error:
                post_ui(self.app, lambda error=str(error), current=current: self._finish_error(error, current))
            else:
                post_ui(self.app, lambda: self._finish_check(current, result))

        self._worker = threading.Thread(target=work, name="app-update-check", daemon=True)
        self._worker.start()

    def _finish_check(self, current, result):
        self._busy, self._phase, self._checked = False, "idle", True
        self._current_label = current
        self._message = result.message
        release = result.release if result.status == "available" else None
        if release != self._release and not self._helper_owned:
            prepared, self._prepared = self._prepared, None
            self._discard_prepared(prepared)
        self._release = release
        self._render()

    def _finish_error(self, error, current=None):
        self._busy, self._phase, self._checked = False, "idle", True
        if current is not None:
            self._current_label = current
        self._message = f"Update could not finish: {error}\nYou can try again; the current app is unchanged."
        self._render()

    def _application_busy(self):
        app = self.app
        engine = getattr(app, "engine", None)
        return bool(
            any(getattr(app, flag, False) for flag in (
                "is_generating", "_model_load_in_progress", "_speed_ui_running",
                "_sleep_consolidating", "_model_unload_in_progress", "_model_unload_pending",
                "_strata_unload_pending", "_native_unload_pending",
            ))
            or getattr(getattr(app, "multimodal", None), "busy", False)
            or getattr(engine, "speed_tuning_active", False)
            or getattr(getattr(engine, "awake_consolidator", None), "is_consolidating", False)
            or getattr(app, "prompt_queue", ())
            or any((getattr(app, "_model_prompt_queues", None) or {}).values())
        )

    def update_and_restart(self):
        if self._closed or self._busy or self._release is None:
            return
        if self._application_busy():
            self._message = ("Finish the current model operation and queued prompts, then choose "
                             "Update & Restart. Your work will not be interrupted.")
            self._render()
            return
        if self._prepared is not None:
            self._begin_install()
            return
        self._busy, self._phase = True, "preparing"
        self._message = "Downloading and verifying the update…"
        self._progress_value = 0
        self._progress_time = 0.0
        release = self._release
        self._render()

        def work():
            try:
                prepared = self._get_updater().prepare(release, self._on_progress)
            except Exception as error:
                post_ui(self.app, lambda error=str(error): self._finish_error(error))
            else:
                post_ui(self.app, lambda: self._finish_prepare(prepared))

        self._worker = threading.Thread(target=work, name="app-update-download", daemon=True)
        self._worker.start()

    def _on_progress(self, message, done, total):
        now = time.monotonic()
        if now - self._progress_time < .15 and (not total or done < total):
            return
        self._progress_time = now
        percent = max(0, min(100, done * 100 / total)) if total else 0
        post_ui(self.app, lambda: self._show_progress(message, percent))

    def _show_progress(self, message, percent):
        self._message, self._progress_value = message, percent
        self._render()

    def _finish_prepare(self, prepared):
        self._busy, self._phase = False, "idle"
        self._prepared = prepared
        self._progress_value = 100
        if self._application_busy():
            self._message = ("Update downloaded and verified. Finish the current work, then choose "
                             "Update & Restart when you are ready.")
            self._render()
            return
        self._begin_install()

    def _begin_install(self):
        self._busy, self._phase = True, "installing"
        # Protect staging throughout handoff, including before its result has
        # made it back to the GUI thread.
        self._helper_owned = True
        self._message = "Preparing to restart with the verified update…"
        self._render()

        def work():
            try:
                helper = self._get_updater().install(self._prepared)
            except Exception as error:
                post_ui(self.app, lambda error=str(error): self._finish_install_error(error))
            else:
                post_ui(self.app, lambda: self._finish_install(helper))

        self._worker = threading.Thread(target=work, name="app-update-install", daemon=True)
        self._worker.start()

    def _finish_install_error(self, error):
        self._helper_owned = False
        self._finish_error(error)

    def _finish_install(self, helper):
        if self._application_busy():
            # The helper is still waiting for this GUI to exit. Abandon that
            # handoff if new work started during its creation.
            def stop_helper():
                try:
                    helper.terminate()
                    try:
                        helper.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        helper.kill()
                        helper.wait(timeout=10)
                except Exception:
                    logging.getLogger(__name__).exception("Could not stop waiting update helper")
                    post_ui(self.app, self._handoff_stop_failed)
                else:
                    post_ui(self.app, self._defer_restart)

            self._worker = threading.Thread(target=stop_helper, name="app-update-defer", daemon=True)
            self._worker.start()
            return
        self._message = "Restarting Smart AI Studio…"
        self._render()
        self.app._on_close_app()

    def _handoff_stop_failed(self):
        # Keep this handoff exclusive, and keep its files, until the app exits.
        self._message = ("The waiting update helper could not be stopped. Your current work continues; "
                         "the update is still scheduled for when the app closes.")
        self._render()

    def _defer_restart(self):
        self._busy, self._phase = False, "idle"
        self._helper_owned = False
        self._message = ("New work started, so the restart was deferred. The verified update is ready; "
                         "choose Update & Restart after the current work finishes.")
        self._render()

    def show(self):
        if self._closed:
            return
        if self._dialog is not None and self._dialog.winfo_exists():
            self._dialog.lift()
            return
        colors = self.app.C
        dialog = self._dialog = tk.Toplevel(self.app.root)
        dialog.title("Smart AI Studio Updates")
        fit_window_to_screen(dialog, 620, 410, min_width=420, min_height=310)
        dialog.configure(bg=colors["bg_hud"])
        dialog.transient(self.app.root)
        body = tk.Frame(dialog, bg=colors["bg_hud"], padx=18, pady=16)
        body.pack(fill="both", expand=True)
        self._widgets = {}
        for name in ("current", "available", "status"):
            label = tk.Label(body, text="", anchor="w", justify="left", wraplength=560,
                             bg=colors["bg_hud"], fg=colors["text_main"])
            label.pack(fill="x", pady=(0, 12))
            label.bind("<Configure>", lambda event, item=label: item.configure(
                wraplength=max(180, event.width - 8)))
            self._widgets[name] = label
        self._widgets["progress"] = ttk.Progressbar(body, mode="determinate", maximum=100)
        self._widgets["progress"].pack(fill="x", pady=(0, 14))
        actions = tk.Frame(body, bg=colors["bg_hud"])
        actions.pack(fill="x", side="bottom")
        for name, label, callback in (
            ("check", "Check Again", self.check),
            ("notes", "Release Notes", self._open_release_notes),
            ("install", "Update & Restart", self.update_and_restart),
            ("close", "Close", dialog.destroy),
        ):
            button = tk.Button(actions, text=label, command=callback, bg=colors["btn_bg"],
                               fg=colors["btn_fg"], activebackground=colors["btn_hover"],
                               activeforeground=colors["btn_fg"], padx=10, pady=6)
            button.pack(side="left", padx=(0, 6))
            self._widgets[name] = button
        from core.responsive_toolbar import install_wrapping_controls
        install_wrapping_controls(actions)
        self._render()
        if not self._checked and not self._busy:
            self.check()

    def _open_release_notes(self):
        if self._release is not None:
            webbrowser.open(self._release.html_url)

    def _render(self):
        button = getattr(self.app, "btn_app_update", None)
        if button is not None:
            label = {"checking": "↑ Checking…", "preparing": "↑ Updating…", "installing": "↑ Restarting…"}.get(self._phase)
            button.configure(text=label or ("↑ Update available" if self._release else "↑ Update"))
        if self._dialog is None or not self._dialog.winfo_exists():
            return
        release = self._release
        self._widgets["current"].configure(text=f"Installed: {self._current_label}")
        available = (f"{release.tag} ({release.commit[:12]}) · {release.total_bytes / (1024 * 1024):.1f} MiB"
                     if release else "No update selected")
        self._widgets["available"].configure(text=f"Available: {available}")
        self._widgets["status"].configure(text=self._message)
        self._widgets["progress"].configure(value=self._progress_value)
        self._widgets["check"].configure(state="disabled" if self._busy else "normal")
        self._widgets["notes"].configure(state="normal" if release else "disabled")
        self._widgets["install"].configure(state="normal" if release and not self._busy else "disabled")
