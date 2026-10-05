"""Small GUI integration for one total prompt+output Context budget.

The desktop app keeps the existing GUI intact and adds one compact second header row.
"Context" means the complete active model window: prompt/history plus generated tokens.
There is no separate output-token limit on native MLX generation; after old dialogue is
consolidated, the model may use every remaining Context token until natural EOS.

The same hook also repairs the later streaming GUI's stale Instant-N=1 SQLite label
when historical Pro actually ran, and shows MLX-LM's measured raw decode TPS rather
than confusing complete Pro/prefill wall time with decoder throughput.
"""
from __future__ import annotations

import json
import sqlite3
import sys

from core.platform import get_auto_context_window_size


def install_gui_generation_cap() -> None:
    mod = sys.modules.get("app_gui")
    cls = getattr(mod, "SmartAIChatbotApp", None) if mod is not None else None
    if cls is None or getattr(cls, "_generation_cap_ui_installed", False):
        return

    tk = getattr(mod, "tk", None)
    if tk is None:
        return

    original_build_top = cls._build_single_top_bar
    original_set_generating = cls._set_generating_state
    original_process = cls._process_message_thread
    original_apply_theme = cls._apply_theme_colors

    def _set_reason(self, text: str) -> None:
        label = getattr(self, "lbl_generation_cap_reason", None)
        if label is not None:
            try:
                label.configure(text=str(text))
            except Exception:
                pass

    def _advertised_context(self) -> int:
        try:
            info = self.models_config.get(self.active_tab_id, {})
            return int(info.get("max_context", 0) or 0)
        except Exception:
            return 0

    def _current_context(self) -> int:
        engine = getattr(self, "engine", None)
        selected = int(getattr(engine, "_context_budget_tokens", 0) or 0)
        if selected > 0:
            return selected
        auto = int(get_auto_context_window_size())
        advertised = _advertised_context(self)
        selected = min(auto, advertised) if advertised else auto
        selected = max(1024, selected)
        if engine is not None:
            engine._context_budget_tokens = selected
            backend = getattr(engine, "mlx_backend", None)
            if backend is not None:
                backend.context_budget_tokens = selected
        return selected

    def _apply_context(self, event=None):
        raw = str(getattr(self, "generation_cap_var", None).get()).strip()
        try:
            requested = int(raw.replace(",", ""))
        except Exception:
            requested = _current_context(self)
        requested = max(1024, requested)

        engine = getattr(self, "engine", None)
        if engine is not None:
            engine._context_budget_tokens = requested
            backend = getattr(engine, "mlx_backend", None)
            if backend is not None:
                backend.context_budget_tokens = requested
        self.generation_cap_var.set(f"{requested}")

        advertised = _advertised_context(self)
        if advertised and requested > advertised:
            _set_reason(
                self,
                f"Context {requested:,} requested; selected model advertises {advertised:,}. "
                "Runtime uses the physical model limit. Prompt + history + output all share that one window.",
            )
        else:
            _set_reason(
                self,
                f"Context {requested:,} total. At 80% active prompt/history, old completed dialogue is "
                "consolidated into weights; generation may use every remaining token until EOS.",
            )
        return "break" if event is not None else None

    def build_top_with_cap(self):
        original_build_top(self)
        if hasattr(self, "generation_cap_bar"):
            return

        context_value = _current_context(self)
        self.generation_cap_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"])
        self.generation_cap_bar.pack(fill="x", side="top", padx=10, pady=(0, 3))
        inner = tk.Frame(self.generation_cap_bar, bg=self.C["bg_hud"])
        inner.pack(side="right", padx=4)

        tk.Label(
            inner,
            text="Context Limit",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_hud"],
            fg=self.C["text_main"],
        ).pack(side="left", padx=(0, 5))

        self.generation_cap_var = tk.StringVar(value=str(context_value))
        self.ent_generation_cap = tk.Entry(
            inner,
            textvariable=self.generation_cap_var,
            width=8,
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_input_inner"],
            fg=self.C["text_main"],
            insertbackground=self.C["text_main"],
            relief="flat",
            bd=0,
            highlightbackground=self.C["border"],
            highlightthickness=1,
            justify="right",
        )
        self.ent_generation_cap.pack(side="left", padx=(0, 4), ipady=2)
        self.ent_generation_cap.bind("<Return>", lambda e: _apply_context(self, e))
        self.ent_generation_cap.bind("<FocusOut>", lambda e: _apply_context(self, e))

        self.btn_apply_generation_cap = tk.Button(
            inner,
            text="Apply",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["btn_bg"],
            fg=self.C["btn_fg"],
            relief="flat",
            bd=0,
            padx=6,
            pady=2,
            command=lambda: _apply_context(self),
        )
        self.btn_apply_generation_cap.pack(side="left", padx=(0, 8))

        self.lbl_generation_cap_reason = tk.Label(
            self.generation_cap_bar,
            text=(
                f"Context {context_value:,} total = prompt/history + output. "
                "Old dialogue consolidates at 80%; output uses all remaining room until EOS."
            ),
            font=getattr(mod, "_FONT_TINY"),
            bg=self.C["bg_hud"],
            fg=self.C["text_muted"],
            anchor="e",
        )
        self.lbl_generation_cap_reason.pack(side="right", padx=(4, 10))

    def set_generating_with_cap_reason(self, generating: bool):
        result = original_set_generating(self, generating)
        if not generating:
            engine = getattr(self, "engine", None)
            backend = getattr(engine, "mlx_backend", None)
            reason = str(getattr(backend, "last_generation_cap_reason", "") or "")
            consolidation = str(getattr(engine, "_last_context_consolidation_note", "") or "")
            combined = " • ".join(x for x in (consolidation, reason) if x)
            if combined:
                _set_reason(self, combined)
        return result

    def process_with_real_pro_metadata(
        self,
        full_msg: str,
        user_prompt: str,
        *args,
        **kwargs,
    ):
        # Preserve newer Learn/attachment arguments while adding only telemetry.
        result = original_process(self, full_msg, user_prompt, *args, **kwargs)
        engine = getattr(self, "engine", None)
        backend = getattr(engine, "mlx_backend", None)

        try:
            raw_decode_tps = float(getattr(backend, "last_tok_per_sec", 0.0) or 0.0)
        except Exception:
            raw_decode_tps = 0.0
        if raw_decode_tps > 0.0:
            try:
                self.root.after(0, lambda t=raw_decode_tps: self._update_telemetry(t))
            except Exception:
                pass

        meta = getattr(engine, "_last_stream_pro_meta", None)
        if not isinstance(meta, dict) or int(meta.get("branch_count", 1) or 1) <= 1:
            return result

        try:
            self.last_metadata = dict(meta)
        except Exception:
            pass

        # The streaming GUI historically wrote a hard-coded Instant N=1 row after
        # every response. Correct only the newest row for this exact prompt when Pro ran.
        try:
            raw_branches = meta.get("raw_branches")
            with sqlite3.connect(self.db.db_path) as conn:
                row = conn.execute(
                    "SELECT id FROM interactions WHERE prompt=? AND mode='Instant Stream (N=1)' "
                    "ORDER BY id DESC LIMIT 1",
                    (full_msg,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE interactions SET raw_branches=?, verified_reward=?, surprise_score=?, "
                        "mode=?, entropy=?, winning_branch=?, winning_temp=? WHERE id=?",
                        (
                            json.dumps(raw_branches) if raw_branches else None,
                            float(meta.get("verified_reward", 0.0) or 0.0),
                            float(meta.get("surprise_score", 0.0) or 0.0),
                            str(meta.get("mode", "Pro Search")),
                            float(meta.get("entropy", 0.0) or 0.0),
                            int(meta.get("winning_branch", 0) or 0),
                            float(meta.get("winning_temp", 0.20) or 0.20),
                            int(row[0]),
                        ),
                    )
                    conn.commit()
        except Exception:
            pass
        return result

    def apply_theme_with_cap(self):
        result = original_apply_theme(self)
        if hasattr(self, "generation_cap_bar"):
            try:
                self.generation_cap_bar.configure(bg=self.C["bg_hud"])
                for child in self.generation_cap_bar.winfo_children():
                    try:
                        child.configure(bg=self.C["bg_hud"])
                    except Exception:
                        pass
                self.lbl_generation_cap_reason.configure(
                    bg=self.C["bg_hud"], fg=self.C["text_muted"]
                )
                self.ent_generation_cap.configure(
                    bg=self.C["bg_input_inner"],
                    fg=self.C["text_main"],
                    insertbackground=self.C["text_main"],
                    highlightbackground=self.C["border"],
                )
            except Exception:
                pass
        return result

    cls._build_single_top_bar = build_top_with_cap
    cls._set_generating_state = set_generating_with_cap_reason
    cls._process_message_thread = process_with_real_pro_metadata
    cls._apply_theme_colors = apply_theme_with_cap
    cls._generation_cap_ui_installed = True
