"""Additive desktop controls for measured, native auto speed profiles.

All Tk work stays on the UI thread. Calibration uses the engine's exclusive
runtime API and a queue; its synthetic prompts never enter chat or learning.
"""
from __future__ import annotations

import functools
import json
import math
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any


_PREF_LOCK = threading.RLock()


def format_speed_status(result: Any) -> str:
    """Show measured test results without implying a universal speed gain."""
    if not isinstance(result, dict):
        return "Speed tuning is ready."
    message = str(result.get("message") or result.get("status") or "Speed tuning is ready.")

    def positive(value):
        if type(value) not in (int, float):
            return None
        try:
            number = float(value)
        except (ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    selected = result.get("selected")
    selected = selected if isinstance(selected, dict) else {}
    rate = positive(selected.get("decode_tokens_per_second"))
    prefix = "Saved profile · " if result.get("cached") is True else ""
    if result.get("status") == "improved":
        gain = positive(result.get("gain_pct"))
        if gain is None:
            return message
        text = f"{prefix}Synthetic speed test: +{gain:.1f}%"
        if rate is not None:
            text += f" · {rate:.3g} tokens/s decode"
        return text + "."
    if result.get("status") == "baseline":
        baseline = result.get("baseline")
        if rate is None and isinstance(baseline, dict):
            rate = positive(baseline.get("decode_tokens_per_second"))
        if rate is not None:
            return f"{prefix}Baseline retained after speed test · {rate:.3g} tokens/s decode."
    return message


def load_auto_speed_preference(path: str, default: bool = True) -> bool:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        value = data.get("speed_tuning", {}).get("enabled")
        return value if isinstance(value, bool) else bool(default)
    except (OSError, ValueError, AttributeError, TypeError):
        return bool(default)


def save_auto_speed_preference(path: str, enabled: bool) -> None:
    """Merge one preference without replacing memory limits or other settings."""
    target = Path(path)
    with _PREF_LOCK:
        data: dict[str, Any] = {}
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("App preferences are not a JSON object")
        previous = data.get("speed_tuning", {})
        data["speed_tuning"] = {
            **(previous if isinstance(previous, dict) else {}),
            "enabled": bool(enabled),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent,
                prefix=target.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = handle.name
                json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


def speed_tuning_active(app: Any) -> bool:
    return bool(
        getattr(app, "_speed_ui_running", False)
        or getattr(getattr(app, "engine", None), "speed_tuning_active", False)
    )


def speed_busy_reason(app: Any) -> str:
    if speed_tuning_active(app):
        return "Speed tuning is running. Cancel it or wait for it to finish."
    if getattr(app, "_model_load_in_progress", False):
        return "Wait for model loading to finish before tuning."
    if getattr(app, "is_generating", False):
        return "Finish the current chat or learning task before tuning."
    if getattr(getattr(app, "multimodal", None), "busy", False):
        return "Finish the current media task before tuning."
    if getattr(app, "_sleep_consolidating", False):
        return "Wait for sleep consolidation to finish before tuning."
    consolidator = getattr(getattr(app, "engine", None), "awake_consolidator", None)
    if getattr(consolidator, "is_consolidating", False):
        return "Wait for the current memory update before tuning."
    proc = getattr(app, "_eval_proc", None)
    if proc is not None and proc.poll() is None:
        return "Finish or cancel evaluation before tuning."
    return ""


def install_gui_speed_tuning() -> None:
    mod = sys.modules.get("app_gui")
    if mod is None:
        main = sys.modules.get("__main__")
        if os.path.basename(str(getattr(main, "__file__", ""))) == "app_gui.py":
            mod = main
    cls = getattr(mod, "SmartAIChatbotApp", None) if mod is not None else None
    tk = getattr(mod, "tk", None)
    if cls is None or tk is None or getattr(cls, "_speed_tuning_ui_installed", False):
        return

    original_build = cls._build_single_top_bar
    original_theme = cls._apply_theme_colors
    original_close = cls._on_close_app

    def _status(self, text: str) -> None:
        self._speed_status_text = str(text)
        label = getattr(self, "lbl_speed_status", None)
        if label is not None:
            label.configure(text=self._speed_status_text)

    def _is_text(self) -> bool:
        info = getattr(self, "models_config", {}).get(getattr(self, "active_tab_id", ""), {})
        return str(info.get("model_type", "text") or "text").lower() == "text"

    def _sync_memory_limit(self) -> None:
        self.engine.speed_memory_limit_gb = (
            float(getattr(self, "_memory_limit_gb", 0.0) or 0.0)
            if getattr(self, "_memory_limit_enabled", False) else None
        )

    def _save_preference(self) -> None:
        try:
            save_auto_speed_preference(
                mod.APP_PREFS_FILE, bool(self.engine.auto_speed_enabled),
            )
        except (OSError, ValueError, TypeError) as exc:
            _status(self, f"Speed setting applied; preference could not be saved: {exc}")

    def _cancel(self) -> None:
        self._speed_cancel_event.set()
        cancel = getattr(self.engine, "cancel_speed_tuning", None)
        if callable(cancel):
            cancel()
        _status(self, "Stopping speed test at the next safe token boundary…")

    # Capture and restore existing widget states; do not enable an action that
    # was disabled by the model loader, eval, or another pre-existing policy.
    control_names = (
        "btn_send", "btn_model_menu", "btn_load_unload", "btn_reset_reinstall",
        "btn_eval", "btn_sleep_panel", "btn_steer", "btn_learn",
        "btn_apply_generation_cap", "ent_generation_cap", "_eval_start_button",
        "chk_memory_limit", "entry_memory_limit",
    )

    def _lock_controls(self, active: bool) -> None:
        saved = self._speed_control_states
        if active:
            if self._speed_locked_loaded is None:
                self._speed_locked_loaded = bool(getattr(self, "is_model_loaded", False))
            for name in control_names:
                widget = getattr(self, name, None)
                if widget is None:
                    continue
                try:
                    if name not in saved:
                        saved[name] = (widget, widget.cget("state"))
                    widget.configure(state="disabled")
                except Exception:
                    pass
        elif saved:
            for widget, state in list(saved.values()):
                try:
                    widget.configure(state=state)
                except Exception:
                    pass
            saved.clear()
            # Only refresh when loading actually changed state. Manual tuning
            # must preserve disabled controls belonging to another app policy.
            if self._speed_locked_loaded != bool(getattr(self, "is_model_loaded", False)):
                for method in ("_update_model_action_buttons", "_update_input_lock_state"):
                    callback = getattr(self, method, None)
                    if callable(callback):
                        callback()
            self._speed_locked_loaded = None

    def _run(self, *, enabled: bool, force: bool) -> None:
        _sync_memory_limit(self)
        reason = speed_busy_reason(self)
        if reason:
            self.auto_speed_var.set(bool(self.engine.auto_speed_enabled))
            _status(self, reason)
            return
        if not _is_text(self):
            self.auto_speed_var.set(bool(self.engine.auto_speed_enabled))
            _status(self, "Auto speed applies to supported text models.")
            return
        if not getattr(self, "is_model_loaded", False):
            self.engine.auto_speed_enabled = bool(enabled)
            _status(self, "Auto speed is ready for the next model load." if enabled else "Auto speed is off.")
            _save_preference(self)
            return
        configure = getattr(self.engine, "configure_auto_speed", None)
        if not callable(configure):
            _status(self, "This engine does not expose speed tuning.")
            return

        self._speed_ui_running = True
        self._speed_cancel_event.clear()
        _lock_controls(self, True)
        self.chk_auto_speed.configure(state="disabled")
        self.btn_speed_tune.configure(state="disabled")
        self.btn_speed_cancel.configure(state="normal")
        _status(self, "Measuring speed…" if enabled else "Restoring default execution settings…")

        def worker():
            try:
                result = configure(
                    enabled=enabled, force=force,
                    progress=lambda text: self._speed_events.put(("progress", str(text))),
                    cancelled=self._speed_cancel_event.is_set,
                )
            except Exception as exc:
                result = {"status": "error", "message": f"Speed test failed: {type(exc).__name__}: {exc}"}
            self._speed_events.put(("done", result))

        threading.Thread(target=worker, daemon=True, name="SmartAI-SpeedTune").start()

    def _poll(self):
        if getattr(self, "_speed_ui_closed", False):
            return
        _sync_memory_limit(self)
        try:
            while True:
                kind, value = self._speed_events.get_nowait()
                if kind == "progress":
                    _status(self, value)
                elif kind == "done":
                    self._speed_ui_running = False
                    self.auto_speed_var.set(bool(self.engine.auto_speed_enabled))
                    if isinstance(value, dict):
                        _status(self, format_speed_status(value))
                        if value.get("status") not in {"busy", "cancelled", "failed", "error"}:
                            _save_preference(self)
                    self._speed_last_engine_status = dict(getattr(self.engine, "speed_status", {}) or {})
        except queue.Empty:
            pass

        active = speed_tuning_active(self)
        _lock_controls(self, active or bool(getattr(self, "_model_load_in_progress", False)))
        enabled = bool(getattr(self.engine, "auto_speed_enabled", True))
        busy = bool(speed_busy_reason(self))
        self.chk_auto_speed.configure(state="disabled" if busy or not _is_text(self) else "normal")
        self.btn_speed_tune.configure(
            state="normal" if enabled and not busy and _is_text(self) and self.is_model_loaded else "disabled",
        )
        self.btn_speed_cancel.configure(state="normal" if active else "disabled")
        current = dict(getattr(self.engine, "speed_status", {}) or {})
        if not self._speed_ui_running and current != self._speed_last_engine_status:
            self._speed_last_engine_status = current
            if current.get("message"):
                _status(self, format_speed_status(current))
        try:
            if self.root.winfo_exists():
                self._speed_poll_id = self.root.after(250, lambda: _poll(self))
        except Exception:
            pass

    def build_with_speed(self):
        original_build(self)
        if hasattr(self, "speed_tuning_bar"):
            return
        self._speed_ui_running = False
        self._speed_ui_closed = False
        self._speed_events = queue.Queue()
        self._speed_cancel_event = threading.Event()
        self._speed_control_states = {}
        self._speed_locked_loaded = None
        self._speed_last_engine_status = {}
        enabled = load_auto_speed_preference(
            mod.APP_PREFS_FILE, bool(getattr(self.engine, "auto_speed_enabled", True)),
        )
        self.engine.auto_speed_enabled = enabled
        _sync_memory_limit(self)
        self._speed_last_engine_status = dict(getattr(self.engine, "speed_status", {}) or {})
        self.auto_speed_var = tk.BooleanVar(value=enabled)
        self.speed_tuning_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"])
        self.speed_tuning_bar.pack(fill="x", side="top", padx=10, pady=(0, 4))
        font = getattr(mod, "_FONT_TINY_BOLD")
        self.chk_auto_speed = tk.Checkbutton(
            self.speed_tuning_bar, text="Auto speed", variable=self.auto_speed_var,
            command=lambda: _run(self, enabled=bool(self.auto_speed_var.get()), force=False),
            font=font, bg=self.C["bg_hud"], fg=self.C["text_main"],
            activebackground=self.C["bg_hud"], activeforeground=self.C["text_main"],
            selectcolor=self.C["btn_bg"], highlightthickness=0,
        )
        self.chk_auto_speed.pack(side="left", padx=(2, 6))
        button_style = dict(
            font=font, bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            highlightthickness=0, relief="flat", bd=0, padx=7, pady=2,
        )
        self.btn_speed_tune = tk.Button(
            self.speed_tuning_bar, text="Retune", state="disabled",
            command=lambda: _run(self, enabled=True, force=True), **button_style,
        )
        self.btn_speed_tune.pack(side="left", padx=(0, 5))
        self.btn_speed_cancel = tk.Button(
            self.speed_tuning_bar, text="Cancel", state="disabled",
            command=lambda: _cancel(self), **button_style,
        )
        self.btn_speed_cancel.pack(side="left", padx=(0, 10))
        self.lbl_speed_status = tk.Label(
            self.speed_tuning_bar, font=getattr(mod, "_FONT_TINY"),
            bg=self.C["bg_hud"], fg=self.C["text_muted"], anchor="w", justify="left",
        )
        self.lbl_speed_status.pack(side="left", fill="x", expand=True)
        self.lbl_speed_status.bind(
            "<Configure>", lambda event: self.lbl_speed_status.configure(wraplength=max(120, event.width - 6)),
        )
        _status(self, "Auto speed is ready for the next model load." if enabled else "Auto speed is off.")
        self._speed_poll_id = self.root.after(250, lambda: _poll(self))

    def theme_with_speed(self):
        result = original_theme(self)
        if hasattr(self, "speed_tuning_bar"):
            self.speed_tuning_bar.configure(bg=self.C["bg_hud"])
            self.chk_auto_speed.configure(
                bg=self.C["bg_hud"], fg=self.C["text_main"],
                activebackground=self.C["bg_hud"], activeforeground=self.C["text_main"],
                selectcolor=self.C["btn_bg"],
            )
            self.lbl_speed_status.configure(bg=self.C["bg_hud"], fg=self.C["text_muted"])
            for button in (self.btn_speed_tune, self.btn_speed_cancel):
                button.configure(
                    bg=self.C["btn_bg"], fg=self.C["btn_fg"],
                    activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
                )
        return result

    def close_with_speed(self):
        if hasattr(self, "_speed_cancel_event"):
            _cancel(self)
        if speed_tuning_active(self) or getattr(self, "_model_load_in_progress", False):
            _status(self, "Stopping speed test before closing…")
            self.root.after(100, lambda: close_with_speed(self))
            return
        self._speed_ui_closed = True
        poll_id = getattr(self, "_speed_poll_id", None)
        if poll_id is not None:
            try:
                self.root.after_cancel(poll_id)
            except Exception:
                pass
        return original_close(self)

    def guard(method):
        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            if speed_tuning_active(self) or getattr(self, "_model_load_in_progress", False):
                _status(self, "Speed tuning is running. Cancel it or wait before using the model.")
                return None
            _sync_memory_limit(self)
            return method(self, *args, **kwargs)
        return guarded

    # Guard direct handlers too, so keyboard shortcuts and already-open menus
    # cannot bypass disabled buttons while calibration owns the runtime.
    for name in (
        "_on_send_message", "_on_steer_button_pressed", "_execute_steered_generation",
        "_on_toggle_load_unload", "_on_switch_model_tab", "_on_open_sleep_consolidation_panel",
        "_on_reset_and_reinstall_single_confirm", "_on_import_custom_model_dialog",
        "_on_memory_limit_toggle", "_on_memory_limit_entry_commit", "_run_queued_task",
    ):
        method = getattr(cls, name, None)
        if callable(method):
            setattr(cls, name, guard(method))

    cls._build_single_top_bar = build_with_speed
    cls._apply_theme_colors = theme_with_speed
    cls._on_close_app = close_with_speed
    cls._speed_tuning_ui_installed = True
