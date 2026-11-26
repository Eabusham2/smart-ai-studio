"""Small desktop control for serving the resident model to local clients."""
from __future__ import annotations

import functools
import os
import secrets
import threading


def api_serving(app):
    controller = getattr(app, "_local_api_ui", None)
    return bool(controller is not None and controller.reserves_model)


def desktop_busy_reason(app):
    for flag in ("is_generating", "_model_load_in_progress", "_model_unload_in_progress",
                 "_model_unload_pending", "_strata_unload_pending", "_native_unload_pending",
                 "_sleep_consolidating", "_speed_ui_running"):
        if getattr(app, flag, False):
            return "Finish or cancel the current model operation before starting the API server."
    engine = getattr(app, "engine", None)
    if (getattr(engine, "speed_tuning_active", False)
            or getattr(getattr(engine, "awake_consolidator", None), "is_consolidating", False)
            or getattr(getattr(app, "multimodal", None), "busy", False)):
        return "Finish the current model, learning, or media task first."
    worker = getattr(app, "_generation_worker", None)
    if worker is not None and worker.is_alive():
        return "Wait for the model worker to finish."
    evaluation = getattr(app, "_eval_proc", None)
    if evaluation is not None and evaluation.poll() is None:
        return "Finish or cancel evaluation first."
    if getattr(app, "prompt_queue", ()) or any((getattr(app, "_model_prompt_queues", None) or {}).values()):
        return "Finish or clear queued prompts first."
    return ""


class LocalAPIUI:
    def __init__(self, app, *, server_factory=None, adapter_factory=None):
        self.app = app
        self.server_factory = server_factory
        self.adapter_factory = adapter_factory
        self.server = None
        self._dialog = None
        self._worker = None
        self._stopping = False
        self._status = "Load a text model, then start serving it to other programs."
        self.host = os.getenv("SMARTAI_API_HOST", "127.0.0.1")
        self.port = os.getenv("SMARTAI_API_PORT", "1234")
        self.api_key = os.getenv("SMARTAI_API_KEY", "")

    @property
    def reserves_model(self):
        return bool(self._stopping or (self.server is not None
                    and (self.server.is_running or self.server.active_requests)))

    def create_button(self, parent, font):
        import tkinter as tk
        colors = self.app.C
        self.app.btn_local_api = tk.Button(
            parent, text="API Server", command=self.open, font=font,
            bg=colors["btn_bg"], fg=colors["btn_fg"], activebackground=colors["btn_hover"],
            activeforeground=colors["btn_fg"], highlightbackground=colors["btn_bg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2", highlightthickness=0,
        )
        self.app.btn_local_api.pack(side="left", padx=2)

    def start(self, *, host=None, port=None, api_key=None):
        if self.reserves_model:
            return False
        reason = desktop_busy_reason(self.app)
        if reason:
            self._status = reason
            self._render()
            return False
        from core.local_api_engine import LoadedModelAdapter
        from core.local_api import LocalModelAPIServer
        factory = self.adapter_factory or LoadedModelAdapter
        # Read Tk preferences on its owner thread, not the HTTP worker.
        pro_var = getattr(self.app, "pro_branches_var", None)
        settings = getattr(self.app, "settings", None)
        pro_defaults = {"enabled": bool(pro_var is not None and pro_var.get()),
                        "best_work_passes": getattr(settings, "best_work_passes", 2)}
        adapter = factory(self.app.engine,
                          selected_model=lambda: self.app.models_config.get(self.app.active_tab_id, {}),
                          busy=lambda: desktop_busy_reason(self.app), pro_defaults=pro_defaults)
        if not adapter.model_info():
            self._status = "Select and load a text model with the Load button first."
            self._render()
            return False
        try:
            self.host = str(self.host if host is None else host).strip()
            self.port = str(self.port if port is None else port).strip()
            self.api_key = self.api_key if api_key is None else str(api_key)
            server = (self.server_factory or LocalModelAPIServer)(
                adapter.generate, adapter.model_info, host=self.host,
                port=int(self.port), api_key=self.api_key, count_tokens=adapter.count_tokens,
                busy=lambda: desktop_busy_reason(self.app),
            )
            server.start()
        except Exception as error:
            self._status = f"Could not start API server: {error}"
            self._render()
            return False
        self.server = server
        self._status = f"Serving {adapter.model_info()['name']} at {server.base_url}"
        self._render()
        return True

    def stop(self):
        if self.server is None or self._stopping:
            return
        self._stopping = True
        self._status = "Stopping server and cancelling the active request…"
        self._render()
        server = self.server

        def work():
            from core.gui_dispatch import post_ui
            try:
                stopped = server.stop(timeout=2)
            except Exception as error:
                post_ui(self.app, lambda error=str(error): self._stop_failed(error))
            else:
                post_ui(self.app, lambda: self._finish_stop(server, stopped))

        self._worker = threading.Thread(target=work, name="smartai-api-stop", daemon=True)
        self._worker.start()

    def _stop_failed(self, error):
        self._stopping = False
        self._status = f"Server shutdown is incomplete: {error}. Choose Stop to retry."
        self._render()

    def _finish_stop(self, server, stopped):
        if self.server is not server:
            return
        if not stopped or server.active_requests:
            self._status = "Request cancelled; waiting for the model's current operation to finish…"
            self._render()
            self.app.root.after(200, lambda: self._finish_stop(server, not server.active_requests))
            return
        self.server = None
        self._stopping = False
        self._status = "API server stopped. The model is still loaded and ready in Studio."
        self._render()

    def shutdown_ready(self):
        if not self.reserves_model:
            return True
        self.stop()
        return False

    def _render(self):
        active = self.reserves_model
        button = getattr(self.app, "btn_local_api", None)
        if button is not None:
            button.configure(text="API • Serving" if active else "API Server")
        if self._dialog is None:
            return
        try:
            if not self._dialog.winfo_exists():
                self._dialog = None
                return
            self._status_var.set(self._status)
            self._start_button.configure(state="disabled" if active else "normal")
            self._stop_button.configure(state="normal" if active and not self._stopping else "disabled")
            self._key_button.configure(state="disabled" if active else "normal")
            for entry in self._entries:
                entry.configure(state="disabled" if active else "normal")
            self._endpoint_var.set(self.server.base_url if self.server else "http://127.0.0.1:1234/v1")
        except Exception:
            self._dialog = None

    def open(self):
        import tkinter as tk
        from tkinter import ttk
        from core.gui_geometry import fit_window_to_screen
        if self._dialog is not None and self._dialog.winfo_exists():
            self._dialog.lift()
            return
        dialog = self._dialog = tk.Toplevel(self.app.root)
        dialog.title("Local model API")
        dialog.transient(self.app.root)
        fit_window_to_screen(dialog, 650, 480, min_width=540, min_height=430)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Use your loaded model in other apps", font=("Arial", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text="OpenAI Chat Completions and Responses · Anthropic Messages",
                  wraplength=580).pack(anchor="w", pady=(5, 12))
        self._host_var = tk.StringVar(value=self.host)
        self._port_var = tk.StringVar(value=self.port)
        self._key_var = tk.StringVar(value=self.api_key)
        self._entries = []
        for label, variable, secret in (("Listen address", self._host_var, False),
                                        ("Port", self._port_var, False),
                                        ("API key (optional on localhost)", self._key_var, True)):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=29).pack(side="left")
            entry = ttk.Entry(row, textvariable=variable, show="•" if secret else "")
            entry.pack(side="left", fill="x", expand=True)
            self._entries.append(entry)
        ttk.Label(frame, text="127.0.0.1 serves this computer. For your LAN, use 0.0.0.0 and set an API key.",
                  wraplength=580).pack(anchor="w", pady=5)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=10)
        self._start_button = ttk.Button(actions, text="Start server", command=lambda: self.start(
            host=self._host_var.get(), port=self._port_var.get(), api_key=self._key_var.get()))
        self._start_button.pack(side="left")
        self._stop_button = ttk.Button(actions, text="Stop server", command=self.stop)
        self._stop_button.pack(side="left", padx=6)
        self._key_button = ttk.Button(actions, text="Generate key", command=lambda: self._key_var.set(secrets.token_urlsafe(32)))
        self._key_button.pack(side="right")
        self._endpoint_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value=self._status)
        endpoint_row = ttk.Frame(frame)
        endpoint_row.pack(fill="x", pady=5)
        ttk.Entry(endpoint_row, textvariable=self._endpoint_var, state="readonly").pack(side="left", fill="x", expand=True)

        def copy_url():
            self.app.root.clipboard_clear()
            self.app.root.clipboard_append(self._endpoint_var.get())

        ttk.Button(endpoint_row, text="Copy URL", command=copy_url).pack(side="left", padx=(6, 0))
        ttk.Label(frame, text="Model ID: smart-ai-studio\nClaude Code base URL: omit /v1 from the URL above.",
                  wraplength=580).pack(anchor="w", pady=5)
        ttk.Label(frame, textvariable=self._status_var, wraplength=580, justify="left").pack(anchor="w", pady=8)
        ttk.Label(frame, text="API inherits Pro branches and Best Work passes when started; clients can override smartai.pro. Stop the server to use desktop chat or switch models.",
                  wraplength=580, justify="left").pack(anchor="w", pady=5)
        self._render()


def install_local_api_ui(module=None):
    from core.gui_extensions import resolve_gui_module
    mod = resolve_gui_module(module)
    cls = mod.SmartAIChatbotApp
    if getattr(cls, "_local_api_ui_installed", False):
        return

    def guard(method, *, queued=False):
        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            if api_serving(self):
                if queued:
                    self.root.after(250, lambda: guarded(self, *args, **kwargs))
                else:
                    self._local_api_ui._status = "Stop the API server before using this model in Studio."
                    self._local_api_ui.open()
                    self._local_api_ui._render()
                return None
            return method(self, *args, **kwargs)
        return guarded

    for name in ("_on_send_message", "_on_steer_button_pressed", "_execute_steered_generation",
                 "_on_toggle_load_unload", "_on_switch_model_tab", "_on_open_sleep_consolidation_panel",
                 "_on_reset_and_reinstall_single_confirm", "_remove_hf_fetched_model",
                 "_on_download_hf_model", "_on_import_custom_model_dialog", "_on_fetch_custom_hf_model_dialog",
                 "_on_memory_limit_toggle", "_on_memory_limit_entry_commit", "_run_queued_task",
                 "_open_eval_window", "_open_lossless_speedups", "_on_new_chat", "_on_clear_chat",
                 "_learning_implementation_changed"):
        method = getattr(cls, name, None)
        if callable(method):
            setattr(cls, name, guard(method, queued=name == "_run_queued_task"))

    close = cls._on_close_app

    @functools.wraps(close)
    def close_after_api(self, *args, **kwargs):
        controller = getattr(self, "_local_api_ui", None)
        if controller is not None and not controller.shutdown_ready():
            self.root.after(100, lambda: close_after_api(self, *args, **kwargs))
            return
        return close(self, *args, **kwargs)

    emergency = getattr(cls, "_on_memory_pressure_emergency", None)
    if callable(emergency):
        @functools.wraps(emergency)
        def emergency_after_api(self, *args, **kwargs):
            controller = getattr(self, "_local_api_ui", None)
            if controller is not None and not controller.shutdown_ready():
                self.root.after(100, lambda: emergency_after_api(self, *args, **kwargs))
                return
            return emergency(self, *args, **kwargs)
        cls._on_memory_pressure_emergency = emergency_after_api
    cls._on_close_app = close_after_api
    cls._local_api_ui_installed = True
