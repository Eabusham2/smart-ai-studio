"""Small GUI integration for the existing max_new_tokens setting.

The desktop app already has a user-facing generation setting in Settings but no GUI
control for it. Install a second compact header row without rewriting app_gui.py.
The MLX backend remains authoritative for the effective cap:

    min(user max_new_tokens, model context - packed prompt tokens)

Old complete dialogue pairs may first be synchronously consolidated into real weights
when doing so is required to preserve the requested output room. The same hook repairs
the later streaming GUI's stale SQLite label when the historical Pro path actually ran
N>1 and replaces the GUI's approximate end-to-end t/s badge with measured MLX decode TPS.
"""
from __future__ import annotations

import json
import sqlite3
import sys


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

    def _apply_cap(self, event=None):
        raw = str(getattr(self, "generation_cap_var", None).get()).strip()
        try:
            requested = int(raw.replace(",", ""))
        except Exception:
            requested = int(getattr(self.settings, "max_new_tokens", 1536) or 1536)
        requested = max(1, requested)
        self.settings.max_new_tokens = requested
        self.generation_cap_var.set(f"{requested}")

        info = self.models_config.get(self.active_tab_id, {})
        advertised = int(info.get("max_context", 0) or 0)
        if advertised and requested >= advertised:
            _set_reason(
                self,
                f"Requested {requested:,}; selected model context is {advertised:,}. "
                "Runtime consolidates old completed dialogue when possible, then caps output to the real tokens remaining.",
            )
        else:
            _set_reason(
                self,
                f"User max {requested:,}. Runtime preserves/consolidates history first; effective output still cannot exceed real model context.",
            )
        return "break" if event is not None else None

    def build_top_with_cap(self):
        original_build_top(self)
        if hasattr(self, "generation_cap_bar"):
            return

        self.generation_cap_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"])
        self.generation_cap_bar.pack(fill="x", side="top", padx=10, pady=(0, 3))
        inner = tk.Frame(self.generation_cap_bar, bg=self.C["bg_hud"])
        inner.pack(side="right", padx=4)

        tk.Label(
            inner,
            text="Max output tokens",
            font=getattr(mod, "_FONT_TINY_BOLD"),
            bg=self.C["bg_hud"],
            fg=self.C["text_main"],
        ).pack(side="left", padx=(0, 5))

        self.generation_cap_var = tk.StringVar(
            value=str(int(getattr(self.settings, "max_new_tokens", 1536) or 1536))
        )
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
        self.ent_generation_cap.bind("<Return>", lambda e: _apply_cap(self, e))
        self.ent_generation_cap.bind("<FocusOut>", lambda e: _apply_cap(self, e))

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
            command=lambda: _apply_cap(self),
        )
        self.btn_apply_generation_cap.pack(side="left", padx=(0, 8))

        self.lbl_generation_cap_reason = tk.Label(
            self.generation_cap_bar,
            text=(
                f"User max {int(getattr(self.settings, 'max_new_tokens', 1536) or 1536):,}. "
                "Old complete dialogue is consolidated if needed; real model context remains the hard ceiling."
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

    def process_with_real_pro_metadata(self, full_msg: str, user_prompt: str):
        result = original_process(self, full_msg, user_prompt)
        engine = getattr(self, "engine", None)
        backend = getattr(engine, "mlx_backend", None)

        # app_gui.py historically estimates t/s from words divided by complete request
        # wall time. Prefer MLX-LM's measured raw decode rate when available so the HUD
        # does not confuse Pro/search/prefill latency with the decoder's actual speed.
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

        # Restore the historical Pro metadata for visualizers/debugging as well as DB.
        try:
            self.last_metadata = dict(meta)
        except Exception:
            pass

        # The streaming GUI historically wrote a hard-coded Instant N=1 row after
        # every response. Correct only the newest row for this exact prompt and only
        # when it still has that stale label.
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
