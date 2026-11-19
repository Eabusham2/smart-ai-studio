"""Learning controls opened from Memory Explorer; all model work stays off Tk."""
from __future__ import annotations

import json
import queue
import sqlite3
import threading


TASKS = {
    "Read a document": {"method": "ingest", "source": "", "strategy": "stream", "max_chunks": 8},
    "Study relevant evidence": {"method": "ingest", "source": "", "strategy": "evidence", "question": "", "max_chunks": 8},
    "Practice examples": {"method": "few_shot", "demonstrations": [{"prompt": "", "completion": ""}, {"prompt": "", "completion": ""}], "query": "", "votes": 3},
    "Adapt to a question": {"method": "query_adapt", "context": "", "query": ""},
    "Learn from a checked answer": {"method": "reinforce", "prompt": "", "mode": "verifier", "verifier": {"kind": "exact", "expected": ""}, "rollouts": 4, "rounds": 1},
    "Learn from agreement": {"method": "reinforce", "prompt": "", "mode": "consensus", "rollouts": 4, "rounds": 1},
    "Discover a better solution": {"method": "discover", "prompt": "Write a Python function sum_squares(values) that returns the sum of squares.",
        "verifier": {"kind": "python_tests", "function": "sum_squares",
                     "cases": [{"args": [[1, 2, 3]], "expected": 14}, {"args": [[]], "expected": 0}]},
        "rounds": 3, "rollouts": 4, "max_tokens": 128, "archive_size": 16},
    "Consolidate candidate solutions": {"method": "consolidate_candidates", "prompt": "", "mode": "consensus", "rollouts": 4},
    "Save a document memory": {"method": "bank_ingest", "source": "", "key": "", "title": "", "strategy": "stream", "max_chunks": 8},
    "Answer using saved memory": {"method": "bank_answer", "query": "", "routing": "model", "top_k": 3},
    "List saved memories": {"method": "bank_list", "limit": 100},
    "Check model support": {"method": "capabilities"},
}


def show_online_learning(app):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from core.memory_scopes import LearningScope

    modal = tk.Toplevel(app.root)
    modal.title("Learning and saved memory")
    modal.geometry("860x720")
    modal.minsize(640, 560)
    modal.transient(app.root)
    colors = app.C
    modal.configure(bg=colors["bg_hud"])
    frame = ttk.Frame(modal, padding=14)
    frame.pack(fill="both", expand=True)
    # Reserve actions before the resizable editors so small windows cannot
    # squeeze Run/Cancel/Close out of the visible area.
    controls = ttk.Frame(frame)
    controls.pack(side="bottom", fill="x", pady=(10, 0))
    ttk.Label(frame, text="Learn from documents, examples and feedback", font=("Arial", 16, "bold")).pack(anchor="w")
    ttk.Label(frame, text=("Learning persists in the selected memory. Temporary practice restores the previous weights afterward.\n"
                          "A paused document job resumes when you run the same request. Agreement is not a correctness check."),
              wraplength=810, justify="left").pack(anchor="w", pady=(6, 12))

    implementation_bar = ttk.Frame(frame)
    implementation_bar.pack(fill="x", pady=(0, 8))
    implementation_var = tk.StringVar(value=getattr(app.settings, "learning_implementation", "legacy"))
    ttk.Label(implementation_bar, text="Learning implementation:").pack(side="left")
    ttk.Combobox(implementation_bar, textvariable=implementation_var, values=("legacy", "managed", "hybrid"),
                 width=12, state="readonly").pack(side="left", padx=5)
    ttk.Label(frame, text="legacy: original learning. managed: persistent context learning.\n"
              "hybrid: original updates with persistent memory and validation.\n"
              "Changing implementation reloads a resident model. Each implementation keeps its own saved weights and chat.",
              wraplength=810, justify="left").pack(anchor="w", pady=(0, 8))

    scope_bar = ttk.Frame(frame)
    scope_bar.pack(fill="x")
    current = app.engine.learning_scope
    scope_var = tk.StringVar(value=current.kind if current.kind in {"model", "session", "project"} else "model")
    key_var = tk.StringVar(value=current.key if current.kind in {"session", "project"} else "")
    ttk.Label(scope_bar, text="Memory:").pack(side="left")
    ttk.Combobox(scope_bar, textvariable=scope_var, values=("model", "session", "project"), width=12,
                 state="readonly").pack(side="left", padx=5)
    ttk.Entry(scope_bar, textvariable=key_var).pack(side="left", fill="x", expand=True, padx=5)

    method_bar = ttk.Frame(frame)
    method_bar.pack(fill="x", pady=(12, 6))
    task_var = tk.StringVar(value="Read a document")
    chooser = ttk.Combobox(method_bar, textvariable=task_var, values=list(TASKS), width=36, state="readonly")
    chooser.pack(side="left")
    temporary_var = tk.BooleanVar(value=False)
    temporary_button = ttk.Checkbutton(method_bar, text="Temporary practice", variable=temporary_var)
    temporary_button.pack(side="left", padx=8)
    editors = ttk.Panedwindow(frame, orient="vertical")
    editors.pack(fill="both", expand=True)
    request_panel = ttk.Frame(editors)
    editors.add(request_panel, weight=1)
    request_editor = tk.Text(request_panel, height=8, wrap="word", undo=True,
                             bg=colors["bg_input_inner"], fg=colors["text_main"],
                             insertbackground=colors["text_main"])
    request_editor.pack(fill="both", expand=True, pady=(0, 8))
    request_editor.insert("1.0", json.dumps(TASKS[task_var.get()], indent=2))

    def choose(_event=None):
        request_editor.delete("1.0", "end")
        request_editor.insert("1.0", json.dumps(TASKS[task_var.get()], ensure_ascii=False, indent=2))
        method = TASKS[task_var.get()]["method"]
        can_practice = not method.startswith("bank_") and method != "capabilities"
        temporary_button.configure(state="normal" if can_practice else "disabled")
        if not can_practice:
            temporary_var.set(False)

    chooser.bind("<<ComboboxSelected>>", choose)

    def attach():
        selected = filedialog.askopenfilename(parent=modal, title="Choose a learning source")
        if selected:
            try:
                request = json.loads(request_editor.get("1.0", "end"))
                request["source"] = selected
                request_editor.delete("1.0", "end")
                request_editor.insert("1.0", json.dumps(request, ensure_ascii=False, indent=2))
            except (ValueError, TypeError):
                messagebox.showerror("Request", "Correct the request JSON before choosing a source.", parent=modal)

    ttk.Button(method_bar, text="Choose source…", command=attach).pack(side="right")
    output_panel = ttk.Frame(editors)
    editors.add(output_panel, weight=1)
    ttk.Label(output_panel, text="Result and progress").pack(anchor="w")
    output = tk.Text(output_panel, height=6, wrap="word", state="disabled",
                     bg=colors["code_bg"], fg=colors["code_fg"], insertbackground=colors["code_fg"])
    output.pack(fill="both", expand=True)
    events = queue.Queue()
    running = False

    def log(message):
        output.configure(state="normal")
        output.insert("end", message + "\n")
        size = int(output.count("1.0", "end", "chars")[0])
        if size > 160_000:
            output.delete("1.0", f"1.0+{size - 120_000}c")
        output.see("end")
        output.configure(state="disabled")

    def start(operation, *, changed_scope=False, changed_implementation=False):
        nonlocal running
        worker = getattr(app, "_generation_worker", None)
        evaluation = getattr(app, "_eval_proc", None)
        if (running or app.is_generating or getattr(app, "_model_load_in_progress", False)
                or (worker is not None and worker.is_alive())
                or (evaluation is not None and evaluation.poll() is None)
                or getattr(app, "_sleep_consolidating", False)
                or getattr(app, "_model_unload_in_progress", False)
                or getattr(app, "_model_unload_pending", False)
                or getattr(app, "_strata_unload_pending", False)
                or getattr(app, "_native_unload_pending", False)
                or getattr(app, "_speed_ui_running", False) or getattr(app.engine, "speed_tuning_active", False)
                or getattr(app.multimodal, "busy", False)
                or getattr(app.engine.awake_consolidator, "is_consolidating", False)):
            messagebox.showinfo("Learning", "Finish or cancel the current model operation first.", parent=modal)
            return
        running = True
        app._sleep_consolidating = True
        app.cancel_event.clear()
        app._set_generating_state(True)
        run_button.configure(state="disabled")
        scope_button.configure(state="disabled")
        implementation_button.configure(state="disabled")

        def work():
            try:
                result = operation()
                events.put(("done", (result, changed_scope, changed_implementation)))
            except Exception as error:
                events.put(("done", ({"status": "error", "error": str(error)}, changed_scope, changed_implementation)))

        def drain():
            nonlocal running
            while True:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    break
                if kind == "progress":
                    log(str(payload))
                else:
                    result, scope_changed, implementation_changed = payload
                    running = False
                    app._sleep_consolidating = False
                    app._set_generating_state(False)
                    run_button.configure(state="normal")
                    scope_button.configure(state="normal")
                    implementation_button.configure(state="normal")
                    log(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                    app.is_model_loaded = bool(app.engine.is_model_loaded)
                    if implementation_changed:
                        implementation_var.set(app.settings.learning_implementation)
                        selected_scope = app.engine.learning_scope
                        scope_var.set(selected_scope.kind if selected_scope.kind in {"model", "session", "project"} else "model")
                        key_var.set(selected_scope.key if selected_scope.kind in {"session", "project"} else "")
                    if implementation_changed and result.get("status") in {"loaded", "selected"}:
                        from core.memory_preferences import save_learning_implementation
                        try:
                            save_learning_implementation(app.settings.database_path, result["learning_implementation"],
                                                         scope=selected_scope)
                        except (ValueError, OSError, sqlite3.Error) as error:
                            log(f"Implementation changed, but its startup selection could not be saved: {error}")
                        app._learning_implementation_changed(result["previous_implementation"])
                    if scope_changed and result.get("status") in {"loaded", "selected", "unchanged"}:
                        from core.memory_preferences import save_memory_preference
                        try:
                            save_memory_preference(app.settings.database_path, app.engine.learning_scope,
                                                   implementation=app.settings.learning_implementation)
                        except (ValueError, OSError, sqlite3.Error) as error:
                            log(f"Memory changed, but its startup selection could not be saved: {error}")
                        for tab in app.chat_history:
                            if app.models_config.get(tab, {}).get("model_type", "text") == "text":
                                app.chat_history[tab] = app._open_tab_history(tab)
                        app._show_transcript_page()
                    app._update_telemetry()
                    app._update_input_lock_state()
                    app._check_and_run_next_queue()
            if running:
                modal.after(80, drain)

        app._generation_worker = threading.Thread(target=work, name="studio-online-learning", daemon=True)
        app._generation_worker.start()
        modal.after(40, drain)

    def apply_scope():
        try:
            selected = LearningScope(scope_var.get(), "" if scope_var.get() == "model" else key_var.get().strip())
        except ValueError as error:
            messagebox.showerror("Memory", str(error), parent=modal)
            return
        start(lambda: app.engine.set_learning_scope(selected), changed_scope=True)

    def run():
        try:
            request = json.loads(request_editor.get("1.0", "end"))
            if not isinstance(request, dict):
                raise ValueError("The request must be a JSON object")
            if temporary_var.get():
                request["temporary"] = True
        except ValueError as error:
            messagebox.showerror("Learning request", str(error), parent=modal)
            return
        start(lambda: app.engine.run_online_learning(request, cancel_event=app.cancel_event,
            progress=lambda *parts, **details: events.put(("progress", {"message": parts, **details}))))

    scope_button = ttk.Button(scope_bar, text="Apply memory", command=apply_scope)
    scope_button.pack(side="right")
    run_button = ttk.Button(controls, text="Run learning task", command=run)
    run_button.pack(side="left")
    def apply_implementation():
        selected = implementation_var.get()
        def switch():
            from core.memory_preferences import load_memory_preference
            preferred = load_memory_preference(app.settings.database_path, implementation=selected)
            return app.engine.set_learning_implementation(selected, cancel_event=app.cancel_event,
                                                          preferred_scope=preferred)
        start(switch, changed_implementation=True)

    implementation_button = ttk.Button(implementation_bar, text="Apply implementation", command=apply_implementation)
    implementation_button.pack(side="right")
    ttk.Button(controls, text="Cancel", command=app.cancel_event.set).pack(side="left", padx=8)
    ttk.Button(controls, text="Close", command=lambda: None if running else modal.destroy()).pack(side="right")
    modal.protocol("WM_DELETE_WINDOW", lambda: None if running else modal.destroy())
    from core.responsive_toolbar import install_wrapping_controls
    for bar in (implementation_bar, scope_bar, method_bar):
        install_wrapping_controls(bar)
    def wrap_notes(event):
        if event.widget is frame:
            for widget in frame.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.configure(wraplength=max(200, event.width - 32))
    frame.bind("<Configure>", wrap_notes, add="+")
