"""Desktop Eval button/window integration.

Additive GUI patch: the existing chat UI, model selector, inference flow, and memory
watchdog are not replaced. Eval runs in a separate process using the canonical suite.
"""
from __future__ import annotations

import codecs
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def install_gui_eval_panel() -> None:
    mod = sys.modules.get("app_gui")
    cls = getattr(mod, "SmartAIChatbotApp", None) if mod is not None else None
    if cls is None or getattr(cls, "_eval_panel_installed", False):
        return

    tk = getattr(mod, "tk", None)
    messagebox = getattr(mod, "messagebox", None)
    if tk is None or messagebox is None:
        return

    original_build = cls._build_single_top_bar
    original_switch = cls._on_switch_model_tab
    original_theme = cls._apply_theme_colors
    original_close = cls._on_close_app

    def _active_text_info(self):
        info = dict(self.models_config.get(self.active_tab_id, {}) or {})
        if str(info.get("model_type", "text") or "text").lower() != "text":
            return None
        return info

    def _eval_alive(self) -> bool:
        proc = getattr(self, "_eval_proc", None)
        return proc is not None and proc.poll() is None

    def _sync_eval_button(self):
        btn = getattr(self, "btn_eval", None)
        if btn is None:
            return
        is_text = _active_text_info(self) is not None
        try:
            mapped = bool(btn.winfo_ismapped())
        except Exception:
            mapped = False
        if is_text and not mapped:
            try:
                btn.pack(side="left", padx=2, before=self.btn_load_unload)
            except Exception:
                btn.pack(side="left", padx=2)
        elif not is_text and mapped:
            btn.pack_forget()

    def build_with_eval(self):
        original_build(self)
        if hasattr(self, "btn_eval"):
            _sync_eval_button(self)
            return
        parent = self.btn_load_unload.master
        self.btn_eval = tk.Button(
            parent,
            text="Eval",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["btn_bg"],
            fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"],
            activeforeground=self.C["btn_fg"],
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            highlightthickness=0,
            command=lambda: _open_eval_window(self),
        )
        self._eval_proc = None
        self._eval_output_queue = queue.Queue()
        self._eval_run_dir = None
        self._eval_peak_gb = 0.0
        self._eval_paused = False
        self._eval_memory_tripped = False
        self._eval_modal = None
        self._eval_log_widget = None
        self._eval_state_label = None
        self._eval_ram_label = None
        self._eval_elapsed_label = None
        self._eval_start_time = None
        _sync_eval_button(self)

    def switch_with_eval(self, target_tab_id: str):
        result = original_switch(self, target_tab_id)
        try:
            _sync_eval_button(self)
        except Exception:
            pass
        return result

    def theme_with_eval(self):
        result = original_theme(self)
        btn = getattr(self, "btn_eval", None)
        if btn is not None:
            try:
                btn.configure(
                    bg=self.C["btn_bg"],
                    fg=self.C["btn_fg"],
                    activebackground=self.C["btn_hover"],
                    activeforeground=self.C["btn_fg"],
                    highlightbackground=self.C["btn_bg"],
                )
            except Exception:
                pass
        return result

    def _append_eval_log(self, text: str):
        widget = getattr(self, "_eval_log_widget", None)
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
            widget.configure(state="normal")
            widget.insert("end", str(text))
            widget.see("end")
            widget.configure(state="disabled")
        except Exception:
            pass

    def _process_tree_metrics(self):
        proc = getattr(self, "_eval_proc", None)
        if proc is None or proc.poll() is not None:
            return 0.0, 0.0
        try:
            import psutil
            root = psutil.Process(proc.pid)
            procs = [root] + root.children(recursive=True)
            rss = 0
            cpu = 0.0
            for item in procs:
                try:
                    rss += int(item.memory_info().rss)
                    cpu += float(item.cpu_percent(interval=None))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return rss / (1024.0 ** 3), cpu
        except Exception:
            return 0.0, 0.0

    def _kill_eval_tree(self):
        proc = getattr(self, "_eval_proc", None)
        if proc is None:
            return
        try:
            import psutil
            root = psutil.Process(proc.pid)
            children = root.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except Exception:
                    pass
            try:
                root.kill()
            except Exception:
                pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _signal_eval_cancel(self, reason: str = "user"):
        proc = getattr(self, "_eval_proc", None)
        if proc is None or proc.poll() is not None:
            return
        run_dir = getattr(self, "_eval_run_dir", None)
        if run_dir:
            try:
                Path(run_dir, "cancel.flag").write_text(reason, encoding="utf-8")
            except Exception:
                pass
        _append_eval_log(self, f"\n[UI] Cancel requested ({reason}); saving/rolling back at the current safe handler…\n")
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        def _force_if_needed():
            p = getattr(self, "_eval_proc", None)
            if p is not None and p.poll() is None:
                _append_eval_log(self, "[UI] Graceful cancel timed out; terminating eval process tree.\n")
                _kill_eval_tree(self)

        try:
            self.root.after(8000, _force_if_needed)
        except Exception:
            pass

    def _cancel_button(self):
        if not _eval_alive(self):
            return
        confirmed = messagebox.askyesno(
            "Cancel Evaluation",
            "Cancel the running evaluation?\n\n"
            "The runner will save the current checkpoint where supported and roll back "
            "an in-flight transactional consolidation before exiting.",
            parent=getattr(self, "_eval_modal", None),
        )
        if confirmed:
            _signal_eval_cancel(self, "user")

    def _toggle_pause(self):
        if not _eval_alive(self):
            return
        run_dir = getattr(self, "_eval_run_dir", None)
        if not run_dir:
            return
        flag = Path(run_dir) / "pause.flag"
        if not getattr(self, "_eval_paused", False):
            try:
                flag.write_text("pause", encoding="utf-8")
            except Exception:
                return
            self._eval_paused = True
            _append_eval_log(self, "\n[UI] Pause requested; it will stop at the next safe eval boundary.\n")
        else:
            try:
                flag.unlink(missing_ok=True)
            except Exception:
                pass
            self._eval_paused = False
            _append_eval_log(self, "\n[UI] Resume requested.\n")
        btn = getattr(self, "_eval_pause_button", None)
        if btn is not None:
            try:
                btn.configure(text="▶ Resume" if self._eval_paused else "⏸ Pause")
            except Exception:
                pass

    def _reader(self, proc, logfile: Path):
        """Read raw pipe chunks so token-stream output appears before newline/EOS."""
        try:
            with logfile.open("a", encoding="utf-8") as handle:
                stream = proc.stdout
                if stream is None:
                    return
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                fd = stream.fileno()
                while True:
                    raw = os.read(fd, 4096)
                    if not raw:
                        break
                    text = decoder.decode(raw)
                    if text:
                        handle.write(text)
                        handle.flush()
                        self._eval_output_queue.put(text)
                tail = decoder.decode(b"", final=True)
                if tail:
                    handle.write(tail)
                    handle.flush()
                    self._eval_output_queue.put(tail)
        except Exception as exc:
            self._eval_output_queue.put(f"\n[UI reader error] {type(exc).__name__}: {exc}\n")

    def _poll_eval(self):
        modal = getattr(self, "_eval_modal", None)
        proc = getattr(self, "_eval_proc", None)

        try:
            while True:
                line = self._eval_output_queue.get_nowait()
                _append_eval_log(self, line)
        except queue.Empty:
            pass

        alive = proc is not None and proc.poll() is None
        if alive:
            ram_gb, cpu = _process_tree_metrics(self)
            self._eval_peak_gb = max(float(getattr(self, "_eval_peak_gb", 0.0)), ram_gb)
            elapsed = max(0.0, time.time() - float(self._eval_start_time or time.time()))

            try:
                if self._eval_ram_label is not None:
                    self._eval_ram_label.configure(
                        text=f"RAM tree: {ram_gb:.2f} GB • peak {self._eval_peak_gb:.2f} GB • CPU {cpu:.0f}%"
                    )
                if self._eval_elapsed_label is not None:
                    self._eval_elapsed_label.configure(text=f"Elapsed: {int(elapsed)}s")
                if self._eval_state_label is not None:
                    state = "Paused / waiting for safe boundary" if self._eval_paused else "Running"
                    self._eval_state_label.configure(text=state)
            except Exception:
                pass

            try:
                enabled = bool(self._eval_mem_enabled_var.get())
                limit = float(str(self._eval_mem_gb_var.get()).strip())
            except Exception:
                enabled, limit = False, 0.0

            if enabled and limit > 0 and ram_gb >= limit and not self._eval_memory_tripped:
                self._eval_memory_tripped = True
                _append_eval_log(
                    self,
                    f"\n[MEM WATCH] Process tree reached {ram_gb:.2f} GB >= {limit:.2f} GB; "
                    "requesting graceful cancellation.\n",
                )
                _signal_eval_cancel(self, "memory-limit")

            try:
                self.root.after(350, lambda: _poll_eval(self))
            except Exception:
                pass
            return

        if proc is not None:
            code = proc.poll()
            try:
                if self._eval_state_label is not None:
                    if code == 0:
                        label = "Completed"
                    elif code == 130:
                        label = "Cancelled"
                    else:
                        label = f"Stopped (exit {code})"
                    self._eval_state_label.configure(text=label)
                if getattr(self, "_eval_start_button", None) is not None:
                    self._eval_start_button.configure(state="normal", text="▶ Start Eval")
                if getattr(self, "_eval_pause_button", None) is not None:
                    self._eval_pause_button.configure(state="disabled", text="⏸ Pause")
                if getattr(self, "_eval_cancel_button", None) is not None:
                    self._eval_cancel_button.configure(state="disabled")
                if getattr(self, "_eval_mem_check", None) is not None:
                    self._eval_mem_check.configure(state="normal")
                if getattr(self, "_eval_mem_entry", None) is not None:
                    self._eval_mem_entry.configure(state="normal")
                if getattr(self, "_eval_deepswe_check", None) is not None:
                    self._eval_deepswe_check.configure(state="normal")
            except Exception:
                pass
            self._eval_proc = None
            self._eval_paused = False

        if modal is not None:
            try:
                if modal.winfo_exists():
                    self.root.after(700, lambda: _poll_eval(self))
            except Exception:
                pass

    def _start_eval(self):
        if _eval_alive(self):
            return
        info = _active_text_info(self)
        if info is None:
            messagebox.showinfo("Text Models Only", "The evaluation suite is available only for text models.")
            return

        try:
            memory_enabled = bool(self._eval_mem_enabled_var.get())
            memory_gb = float(str(self._eval_mem_gb_var.get()).strip())
            if memory_enabled and memory_gb < 1.0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid Memory Limit", "Enter a memory limit of at least 1.0 GB.")
            return

        # Preserve the exact resolved runtime artifact before unloading chat. This
        # prevents a second copy of a 27B model from being resident during eval.
        eval_info = dict(info)
        active_path = str(getattr(getattr(self, "engine", None), "active_model_path", "") or "")
        if active_path and os.path.exists(active_path):
            eval_info["model_path"] = active_path

        if getattr(self, "is_model_loaded", False):
            try:
                self.engine.unload_model()
            except Exception:
                pass
            self.is_model_loaded = False
            try:
                self._sync_memory_watchdog(force_stop=True)
                self._update_model_action_buttons()
                self._update_input_lock_state()
                self.lbl_model_status.configure(
                    text="○ Chat model unloaded for exclusive Eval",
                    fg=self.C["accent_yellow"],
                )
            except Exception:
                pass

        base = Path(getattr(mod, "get_portable_data_dir")()) / "eval_runs"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = base / f"{stamp}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "launcher.json"
        config = {
            "run_dir": str(run_dir),
            "model_info": eval_info,
            "memory_limit_enabled": memory_enabled,
            "memory_limit_gb": memory_gb,
            "deepswe": bool(self._eval_deepswe_var.get()),
            "max_duration_hours": 72.0,
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        for flag in ("pause.flag", "cancel.flag"):
            try:
                (run_dir / flag).unlink(missing_ok=True)
            except Exception:
                pass

        repo_root = Path(mod.__file__).resolve().parent
        cmd = [sys.executable, "-u", "-m", "eval.app_eval_runner", "--config", str(config_path)]
        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=False,
                bufsize=0,
                creationflags=creationflags,
            )
        except Exception as exc:
            messagebox.showerror("Eval Launch Failed", f"{type(exc).__name__}: {exc}")
            return

        self._eval_proc = proc
        self._eval_run_dir = str(run_dir)
        self._eval_peak_gb = 0.0
        self._eval_memory_tripped = False
        self._eval_paused = False
        self._eval_start_time = time.time()
        _append_eval_log(
            self,
            f"[UI] Started PID {proc.pid}\n[UI] Run folder: {run_dir}\n"
            f"[UI] Model: {eval_info.get('name') or eval_info.get('short_name')}\n",
        )
        try:
            self._eval_start_button.configure(state="disabled", text="Running…")
            self._eval_pause_button.configure(state="normal", text="⏸ Pause")
            self._eval_cancel_button.configure(state="normal")
            self._eval_mem_check.configure(state="disabled")
            self._eval_mem_entry.configure(state="disabled")
            self._eval_deepswe_check.configure(state="disabled")
        except Exception:
            pass

        threading.Thread(
            target=_reader,
            args=(self, proc, run_dir / "live_console.log"),
            daemon=True,
            name="SmartAI-EvalOutput",
        ).start()
        _poll_eval(self)

    def _open_eval_window(self):
        info = _active_text_info(self)
        if info is None:
            messagebox.showinfo("Text Models Only", "Select a text model to use the evaluation suite.")
            return

        modal = getattr(self, "_eval_modal", None)
        try:
            if modal is not None and modal.winfo_exists():
                modal.deiconify()
                modal.lift()
                modal.focus_force()
                return
        except Exception:
            pass

        modal = tk.Toplevel(self.root)
        self._eval_modal = modal
        modal.title("Smart AI • 4,014 Evaluation")
        modal.geometry("1000x720")
        modal.minsize(760, 560)
        modal.configure(bg=self.C["bg_hud"])
        modal.transient(self.root)

        header = tk.Frame(modal, bg=self.C["bg_hud"])
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            header,
            text="🧪 4,014 Benchmark Evaluation",
            font=getattr(mod, "_FONT_H2"),
            bg=self.C["bg_hud"],
            fg=self.C["accent_cyan"],
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                f"Text model: {info.get('name') or info.get('short_name')} • "
                "same suite, isolated eval state • Bonsai 2 is the default main model"
            ),
            font=getattr(mod, "_FONT_SMALL"),
            bg=self.C["bg_hud"],
            fg=self.C["text_muted"],
        ).pack(anchor="w", pady=(2, 0))

        controls = tk.Frame(modal, bg=self.C["bg_card"])
        controls.pack(fill="x", padx=16, pady=6)

        current_mem_on = bool(getattr(self, "_memory_limit_enabled", True))
        current_mem_gb = float(getattr(self, "_memory_limit_gb", 12.5) or 12.5)
        self._eval_mem_enabled_var = tk.BooleanVar(value=current_mem_on)
        self._eval_mem_gb_var = tk.StringVar(value=f"{current_mem_gb:.1f}")
        self._eval_deepswe_var = tk.BooleanVar(value=False)

        self._eval_mem_check = tk.Checkbutton(
            controls,
            text="Memory watcher",
            variable=self._eval_mem_enabled_var,
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_card"],
            fg=self.C["text_main"],
            activebackground=self.C["bg_card"],
            activeforeground=self.C["text_main"],
            selectcolor=self.C["btn_bg"],
            highlightthickness=0,
        )
        self._eval_mem_check.pack(side="left", padx=(10, 4), pady=8)
        self._eval_mem_entry = tk.Entry(
            controls,
            textvariable=self._eval_mem_gb_var,
            width=6,
            justify="right",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_input_inner"],
            fg=self.C["text_main"],
            insertbackground=self.C["text_main"],
            relief="flat",
            bd=0,
        )
        self._eval_mem_entry.pack(side="left", ipady=3)
        tk.Label(
            controls,
            text="GB process-tree limit",
            font=getattr(mod, "_FONT_TINY"),
            bg=self.C["bg_card"],
            fg=self.C["text_muted"],
        ).pack(side="left", padx=(3, 12))

        self._eval_deepswe_check = tk.Checkbutton(
            controls,
            text="DeepSWE flagship (113, very slow)",
            variable=self._eval_deepswe_var,
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_card"],
            fg=self.C["text_main"],
            activebackground=self.C["bg_card"],
            activeforeground=self.C["text_main"],
            selectcolor=self.C["btn_bg"],
            highlightthickness=0,
        )
        self._eval_deepswe_check.pack(side="left", padx=6)

        self._eval_start_button = tk.Button(
            controls,
            text="▶ Start Eval",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["btn_primary_bg"],
            fg=self.C["btn_primary_fg"],
            relief="flat",
            bd=0,
            padx=10,
            pady=4,
            command=lambda: _start_eval(self),
        )
        self._eval_start_button.pack(side="right", padx=(4, 10))

        self._eval_pause_button = tk.Button(
            controls,
            text="⏸ Pause",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["btn_bg"],
            fg=self.C["btn_fg"],
            relief="flat",
            bd=0,
            padx=10,
            pady=4,
            state="normal" if _eval_alive(self) else "disabled",
            command=lambda: _toggle_pause(self),
        )
        self._eval_pause_button.pack(side="right", padx=4)

        self._eval_cancel_button = tk.Button(
            controls,
            text="⏹ Cancel",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["btn_bg"],
            fg=self.C["btn_fg"],
            relief="flat",
            bd=0,
            padx=10,
            pady=4,
            state="normal" if _eval_alive(self) else "disabled",
            command=lambda: _cancel_button(self),
        )
        self._eval_cancel_button.pack(side="right", padx=4)

        telemetry = tk.Frame(modal, bg=self.C["bg_hud"])
        telemetry.pack(fill="x", padx=16, pady=(4, 6))
        self._eval_state_label = tk.Label(
            telemetry,
            text="Running" if _eval_alive(self) else "Ready",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_hud"],
            fg=self.C["accent_green"],
        )
        self._eval_state_label.pack(side="left", padx=(0, 14))
        self._eval_ram_label = tk.Label(
            telemetry,
            text=f"RAM tree: 0.00 GB • peak {float(getattr(self, '_eval_peak_gb', 0.0)):.2f} GB • CPU 0%",
            font=getattr(mod, "_FONT_TINY"),
            bg=self.C["bg_hud"],
            fg=self.C["text_main"],
        )
        self._eval_ram_label.pack(side="left")
        self._eval_elapsed_label = tk.Label(
            telemetry,
            text="Elapsed: 0s",
            font=getattr(mod, "_FONT_TINY"),
            bg=self.C["bg_hud"],
            fg=self.C["text_muted"],
        )
        self._eval_elapsed_label.pack(side="right")

        output_frame = tk.Frame(modal, bg=self.C["code_bg"])
        output_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        scroll = tk.Scrollbar(output_frame, orient="vertical")
        scroll.pack(side="right", fill="y")
        self._eval_log_widget = tk.Text(
            output_frame,
            bg=self.C["code_bg"],
            fg=self.C["code_fg"],
            insertbackground=self.C["code_fg"],
            font=getattr(mod, "_FONT_MONO"),
            wrap="word",
            bd=0,
            padx=10,
            pady=10,
            yscrollcommand=scroll.set,
            state="disabled",
        )
        self._eval_log_widget.pack(fill="both", expand=True)
        scroll.configure(command=self._eval_log_widget.yview)
        _append_eval_log(
            self,
            "Ready. Start Eval runs the canonical suite in a separate process.\n"
            "Pause waits for a safe item/phase boundary. Cancel always asks for confirmation.\n",
        )
        _poll_eval(self)

    def close_with_eval_cleanup(self):
        if _eval_alive(self):
            _signal_eval_cancel(self, "app-close")
            proc = getattr(self, "_eval_proc", None)
            if proc is not None:
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    _kill_eval_tree(self)
        return original_close(self)

    cls._build_single_top_bar = build_with_eval
    cls._on_switch_model_tab = switch_with_eval
    cls._apply_theme_colors = theme_with_eval
    cls._on_close_app = close_with_eval_cleanup
    cls._open_eval_window = _open_eval_window
    cls._sync_eval_button_visibility = _sync_eval_button
    cls._eval_panel_installed = True
