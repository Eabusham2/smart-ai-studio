"""Discoverable, model-free desktop preferences for execution speedups.

The dialog edits preferences only. Existing engine guards apply changes at an
idle boundary; native server/model selection takes effect on the next load.
No action here loads a model, runs a benchmark, or starts inference.
"""
from __future__ import annotations

import functools
import os
import sys

from core.lossless_speed_control import (
    DEFAULT_OPTIONS, load_preferences, normalize_options, save_preferences,
)
from core.gui_speed_tuning import speed_busy_reason


_BOOLEAN_FIELDS = (
    ("enabled", "Enable speedups"),
    ("prompt_reuse", "Reuse matching prompt state"),
    ("branch_prefill", "Share prompt preparation across reasoning branches"),
    ("tokenizer_memo", "Reuse exact tokenization results"),
    ("persistent_cache", "Save prompt state locally between sessions (opt-in)"),
    ("metal_warmup", "Warm the GPU runtime after idle (opt-in)"),
)
_PATH_FIELDS = (
    ("cache_dir", "Prompt cache folder", "directory"),
    ("speculative_server_path", "Native server executable", "file"),
    ("draft_model_path", "Compatible draft / head file", "file"),
    ("speculative_runtime_receipt_path", "Runtime receipt (optional)", "file"),
)
_MODES = ("none", "auto", "lookup", "mtp", "dflash2")


def options_from_variables(variables, *, base=None):
    """Convert the form without touching an engine, model, or filesystem."""
    selected = {key: bool(variables[key].get()) for key, _ in _BOOLEAN_FIELDS}
    selected.update({key: str(variables[key].get()).strip() or None for key, _, _ in _PATH_FIELDS})
    selected["speculative_mode"] = str(variables["speculative_mode"].get())
    raw_count = str(variables["max_draft_tokens"].get()).strip()
    if raw_count:
        if not raw_count.isascii() or not raw_count.isdecimal():
            raise ValueError("Draft tokens must be blank for automatic, or a whole number from 1 to 64.")
        selected["max_draft_tokens"] = int(raw_count)
    else:
        selected["max_draft_tokens"] = None
    raw_branches = str(variables["native_branch_count"].get()).strip()
    if not raw_branches.isascii() or not raw_branches.isdecimal():
        raise ValueError("Native branches must be a whole number from 1 to 64.")
    branches = int(raw_branches)
    if not 1 <= branches <= 64:
        raise ValueError("Native branches must be a whole number from 1 to 64.")
    selected["native_branch_count"] = branches
    return normalize_options(selected, base=base)


def apply_dialog_options(app, selected, preferences_path):
    """Recheck current work before applying and atomically saving preferences."""
    if getattr(app, "_speed_close_requested", False):
        return {"status": "busy", "message": "Studio is closing; settings were not changed."}
    reason = speed_busy_reason(app)
    if reason:
        return {"status": "busy", "message": reason.replace("before tuning", "before changing speedups")}
    selected = normalize_options(selected, base=getattr(app.engine, "lossless_speed_options", None))
    configure = getattr(app.engine, "configure_lossless_speedups", None)
    if not callable(configure):
        return {"status": "error", "message": "This engine does not expose speedup settings."}
    result = configure(selected)
    if not isinstance(result, dict):
        return {"status": "error", "message": "The engine did not confirm the speedup settings."}
    if result.get("status") in {"busy", "error", "failed", "cancelled"}:
        return result
    try:
        save_preferences(preferences_path, selected)
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "save_error", "message": f"Settings applied for this session, but could not be saved: {exc}"}
    text = str(result.get("message") or "Speedup settings applied.")
    if "saved" not in text.lower():
        text = "Saved. " + text
    return {**result, "message": text}


def install_gui_lossless_speedups(module=None) -> None:
    """Install once after the existing speed-tuning HUD installer."""
    from core.gui_extensions import resolve_gui_module
    mod = resolve_gui_module(module)
    cls = mod.SmartAIChatbotApp
    tk = mod.tk
    if getattr(cls, "_lossless_speedups_ui_installed", False):
        return
    original_build = cls._build_single_top_bar
    original_theme = getattr(cls, "_apply_theme_colors", None)
    original_close = getattr(cls, "_on_close_app", None)

    def load_before_use(self):
        if getattr(self, "_speedups_preferences_loaded", False):
            return
        selected = load_preferences(mod.APP_PREFS_FILE,
                                    defaults=getattr(self.engine, "lossless_speed_options", None))
        self.engine.lossless_speed_options = selected
        self._speedups_preferences_loaded = True
        self._speedups_status_text = "Speedup settings are ready for the next model load."

    def status(self, message):
        self._speedups_status_text = str(message)
        label = getattr(self, "_speedups_dialog_status", None)
        if label is not None:
            try:
                label.configure(text=self._speedups_status_text)
            except Exception:
                pass

    def close_dialog(self):
        dialog, self._speedups_dialog = getattr(self, "_speedups_dialog", None), None
        self._speedups_dialog_status = None
        self._speedups_dialog_widgets = []
        self._speedups_dialog_labels = []
        self._speedups_variables = {}
        if dialog is not None:
            try:
                dialog.destroy()
            except Exception:
                pass

    def button_style(self):
        return dict(font=getattr(mod, "_FONT_TINY_BOLD", ("Arial", 10, "bold")),
                    bg=self.C["btn_bg"], fg=self.C["btn_fg"],
                    activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
                    highlightthickness=0, relief="flat", bd=0, padx=9, pady=4)

    def open_dialog(self):
        if getattr(self, "_speed_close_requested", False):
            return
        load_before_use(self)
        existing = getattr(self, "_speedups_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    return
            except Exception:
                pass
        selected = normalize_options(getattr(self.engine, "lossless_speed_options", DEFAULT_OPTIONS))
        dialog = tk.Toplevel(self.root)
        self._speedups_dialog = dialog
        dialog.title("Speedups")
        dialog.transient(self.root)
        dialog.configure(bg=self.C["bg_hud"])
        from core.gui_geometry import fit_window_to_screen
        fit_window_to_screen(dialog, 780, 780, min_width=420, min_height=420)
        dialog.protocol("WM_DELETE_WINDOW", lambda: close_dialog(self))
        dialog.bind("<Escape>", lambda event: close_dialog(self))

        footer = tk.Frame(dialog, bg=self.C["bg_hud"])
        footer.pack(side="bottom", fill="x", padx=14, pady=(6, 12))
        canvas = tk.Canvas(dialog, bg=self.C["bg_hud"], highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        dialog.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
            -1 if event.delta > 0 else 1, "units") if event.delta else None)
        dialog.bind("<Button-4>", lambda event: canvas.yview_scroll(-1, "units"))
        dialog.bind("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))
        body = tk.Frame(canvas, bg=self.C["bg_hud"])
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        content = tk.Frame(body, bg=self.C["bg_hud"])
        content.pack(fill="both", expand=True, padx=14, pady=12)
        font = getattr(mod, "_FONT_TINY", ("Arial", 10))
        bold = getattr(mod, "_FONT_TINY_BOLD", ("Arial", 10, "bold"))
        labels = []
        themed = []
        checkboxes = []

        def label(text, *, bold_text=False):
            widget = tk.Label(content, text=text, font=bold if bold_text else font,
                              bg=self.C["bg_hud"], fg=self.C["text_main"] if bold_text else self.C["text_muted"],
                              anchor="w", justify="left", wraplength=680)
            widget.pack(fill="x", pady=(0, 6))
            labels.append(widget)
            return widget

        label("Prompt and GPU speedups", bold_text=True)
        label("Use supported runtime shortcuts while keeping the selected model, full prompt and sampling settings.")
        variables = {key: tk.BooleanVar(value=selected[key]) for key, _ in _BOOLEAN_FIELDS}
        variables.update({key: tk.StringVar(value=selected[key] or "") for key, _, _ in _PATH_FIELDS})
        variables["speculative_mode"] = tk.StringVar(value=selected["speculative_mode"])
        count = selected["max_draft_tokens"]
        variables["max_draft_tokens"] = tk.StringVar(value="" if count is None else str(count))
        variables["native_branch_count"] = tk.StringVar(value=str(selected.get("native_branch_count", 1)))
        self._speedups_variables = variables
        for key, text in _BOOLEAN_FIELDS:
            check = tk.Checkbutton(content, text=text, variable=variables[key], anchor="w",
                                   font=bold if key == "enabled" else font, bg=self.C["bg_hud"],
                                   fg=self.C["text_main"], activebackground=self.C["bg_hud"],
                                   activeforeground=self.C["text_main"], selectcolor=self.C["btn_bg"],
                                   highlightthickness=0)
            check.pack(fill="x", pady=1)
            checkboxes.append(check)
            themed.append((check, "check"))
        label("Local persistence stores prompt tokens and state in a bounded cache. It is off by default. "
              "GPU warmup primes the runtime after idle; it does not generate text or hold clocks up continuously.")

        def browse(key, kind):
            try:
                chooser = getattr(mod, "filedialog", None)
                if chooser is None:
                    from tkinter import filedialog as chooser
                kwargs = {"parent": dialog, "title": "Choose " + dict((k, text) for k, text, _ in _PATH_FIELDS)[key]}
                picked = chooser.askdirectory(**kwargs) if kind == "directory" else chooser.askopenfilename(**kwargs)
                if picked:
                    variables[key].set(picked)
            except Exception as exc:
                status(self, f"The file picker could not open: {exc}")

        def path_row(key, text, kind):
            row = tk.Frame(content, bg=self.C["bg_hud"])
            row.pack(fill="x", pady=(2, 7))
            name = tk.Label(row, text=text, font=font, bg=self.C["bg_hud"], fg=self.C["text_main"], anchor="w")
            name.pack(fill="x")
            entry = tk.Entry(row, textvariable=variables[key], font=font,
                             bg=self.C["btn_bg"], fg=self.C["text_main"], insertbackground=self.C["text_main"],
                             relief="flat", highlightthickness=1)
            entry.pack(side="left", fill="x", expand=True, ipady=4)
            choose = tk.Button(row, text="Browse…", command=lambda: browse(key, kind), **button_style(self))
            choose.pack(side="right", padx=(6, 0))
            themed.extend(((row, "frame"), (name, "label"), (entry, "entry"), (choose, "button")))

        path_row(*_PATH_FIELDS[0])
        label("Leave the cache folder blank to use Studio's app data folder.")
        label("Native target-verified drafting", bold_text=True)
        label("For supported text models with compatible local artifacts. Lookup drafts from the prompt; "
              "MTP and DFlash2 require a compatible draft model or head. Auto selects an available native method.")
        mode_row = tk.Frame(content, bg=self.C["bg_hud"])
        mode_row.pack(fill="x", pady=(0, 8))
        mode_label = tk.Label(mode_row, text="Native mode", font=font, bg=self.C["bg_hud"], fg=self.C["text_main"])
        mode_label.pack(side="left", padx=(0, 10))
        mode = tk.OptionMenu(mode_row, variables["speculative_mode"], *_MODES)
        mode.configure(**button_style(self))
        mode.pack(side="left")
        themed.extend(((mode_row, "frame"), (mode_label, "label"), (mode, "button")))
        for field in _PATH_FIELDS[1:]:
            path_row(*field)
        draft_row = tk.Frame(content, bg=self.C["bg_hud"])
        draft_row.pack(fill="x", pady=(2, 8))
        draft_label = tk.Label(draft_row, text="Maximum draft tokens (blank = automatic, 1–64)", font=font,
                               bg=self.C["bg_hud"], fg=self.C["text_main"])
        draft_entry = tk.Entry(draft_row, textvariable=variables["max_draft_tokens"], width=6, font=font,
                              bg=self.C["btn_bg"], fg=self.C["text_main"], insertbackground=self.C["text_main"])
        draft_entry.pack(side="right")
        draft_label.pack(side="left", fill="x", expand=True)
        themed.extend(((draft_row, "frame"), (draft_label, "label"), (draft_entry, "entry")))
        branch_row = tk.Frame(content, bg=self.C["bg_hud"])
        branch_row.pack(fill="x", pady=(2, 8))
        branch_label = tk.Label(branch_row, text="Native branches (1–64; common choices: 1, 8, 16)", font=font,
                                bg=self.C["bg_hud"], fg=self.C["text_main"])
        branch_entry = tk.Entry(branch_row, textvariable=variables["native_branch_count"], width=6, font=font,
                               bg=self.C["btn_bg"], fg=self.C["text_main"], insertbackground=self.C["text_main"])
        branch_entry.pack(side="right")
        branch_label.pack(side="left", fill="x", expand=True)
        themed.extend(((branch_row, "frame"), (branch_label, "label"), (branch_entry, "entry")))
        label("Native branches applies only to the native text server. The same branch count is used with drafting "
              "off or on; automatic model-entropy routing is unavailable there.")
        label("Save applies supported prompt options when the runtime is idle. Reload the model to apply native "
              "mode, server, draft, receipt or draft-count changes. Saving does not load or run a model.")

        self._speedups_dialog_status = tk.Label(footer, text="", font=font, bg=self.C["bg_hud"],
                                                fg=self.C["text_muted"], anchor="w", justify="left", wraplength=680)
        self._speedups_dialog_status.pack(fill="x", pady=(0, 8))
        status(self, speed_busy_reason(self) or self._speedups_status_text)

        def save():
            try:
                options = options_from_variables(variables, base=getattr(self.engine, "lossless_speed_options", None))
                result = apply_dialog_options(self, options, mod.APP_PREFS_FILE)
                status(self, result.get("message", "Speedup settings updated."))
            except Exception as exc:
                status(self, f"Settings were not saved: {exc}")

        self._speedups_save_button = tk.Button(footer, text="Save settings", command=save, **button_style(self))
        self._speedups_save_button.pack(side="right", padx=(6, 0))
        close = tk.Button(footer, text="Close", command=lambda: close_dialog(self), **button_style(self))
        close.pack(side="right")
        themed.extend(((dialog, "frame"), (footer, "frame"), (canvas, "frame"), (body, "frame"),
                       (content, "frame"), (self._speedups_dialog_status, "label"),
                       (self._speedups_save_button, "button"), (close, "button")))
        self._speedups_dialog_widgets = themed
        self._speedups_dialog_labels = labels
        def wrap_content(event):
            for item in labels:
                item.configure(wraplength=max(260, event.width - 12))
            for item in checkboxes:
                item.configure(wraplength=max(240, event.width - 35), justify="left")
            for item in (draft_label, branch_label):
                item.configure(wraplength=max(180, event.width - 90), justify="left", anchor="w")
        content.bind("<Configure>", wrap_content)
        self._speedups_dialog_status.bind("<Configure>", lambda event:
            self._speedups_dialog_status.configure(wraplength=max(260, event.width - 12))
            if self._speedups_dialog_status is not None else None)

    @functools.wraps(original_build)
    def build(self, *args, **kwargs):
        load_before_use(self)
        result = original_build(self, *args, **kwargs)
        if not hasattr(self, "btn_lossless_speedups"):
            parent = getattr(self, "speed_tuning_bar", None)
            if parent is None:
                parent = tk.Frame(self.main_container, bg=self.C["bg_hud"])
                parent.pack(fill="x", side="top", padx=10, pady=(0, 4))
                self._lossless_speedups_bar = parent
            self.btn_lossless_speedups = tk.Button(parent, text="Speedups…",
                                                   command=lambda: open_dialog(self), **button_style(self))
            # Keep this action next to Retune/Cancel. The long status label
            # gets the remaining width rather than pushing Speedups offscreen.
            status_label = getattr(self, "lbl_speed_status", None)
            pack_options = {"side": "left", "padx": (0, 10)}
            if status_label is not None and parent is getattr(self, "speed_tuning_bar", None):
                pack_options["before"] = status_label
            self.btn_lossless_speedups.pack(**pack_options)
        return result

    def theme(self, *args, **kwargs):
        result = original_theme(self, *args, **kwargs) if callable(original_theme) else None
        button = getattr(self, "btn_lossless_speedups", None)
        if button is not None:
            button.configure(**button_style(self))
        bar = getattr(self, "_lossless_speedups_bar", None)
        if bar is not None:
            bar.configure(bg=self.C["bg_hud"])
        for widget, kind in getattr(self, "_speedups_dialog_widgets", []):
            try:
                if kind == "button":
                    widget.configure(**button_style(self))
                elif kind == "entry":
                    widget.configure(bg=self.C["btn_bg"], fg=self.C["text_main"], insertbackground=self.C["text_main"])
                elif kind == "check":
                    widget.configure(bg=self.C["bg_hud"], fg=self.C["text_main"], selectcolor=self.C["btn_bg"],
                                     activebackground=self.C["bg_hud"], activeforeground=self.C["text_main"])
                else:
                    widget.configure(bg=self.C["bg_hud"], **({"fg": self.C["text_main"]} if kind == "label" else {}))
            except Exception:
                pass
        for widget in getattr(self, "_speedups_dialog_labels", []):
            try:
                widget.configure(bg=self.C["bg_hud"], fg=self.C["text_muted"])
            except Exception:
                pass
        return result

    def close(self, *args, **kwargs):
        close_dialog(self)
        return original_close(self, *args, **kwargs) if callable(original_close) else None

    # Initial load occurs before building the top bar, plus this defensive hook
    # covers direct/keyboard load handlers invoked before a UI rebuild.
    load_handler = getattr(cls, "_on_toggle_load_unload", None)
    if callable(load_handler):
        @functools.wraps(load_handler)
        def load(self, *args, **kwargs):
            load_before_use(self)
            return load_handler(self, *args, **kwargs)
        cls._on_toggle_load_unload = load
    cls._build_single_top_bar = build
    cls._apply_theme_colors = theme
    cls._on_close_app = close
    cls._open_lossless_speedups = open_dialog
    cls._lossless_speedups_ui_installed = True
