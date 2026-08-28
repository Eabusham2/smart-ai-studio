"""
Smart AI Studio — High-Readability Multi-Model Studio with Live Real-Time Token Streaming,
Interactive Thinking Dropdowns, Steer Controls, Task Queue, and Dynamic Hardware Context Scaling.
"""

import ast
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from consolidation.daemon import SleepConsolidationDaemon
from core.autonomous_learner import AutonomousLearner
from core.hf_downloader import is_model_cached_locally, download_model_from_hf
from core.memory_watchdog import SystemMemoryWatchdog
from core.platform import get_auto_context_window_size
from core.pro_engine import ProReasoningEngine, parse_reasoning_and_response
from core.tools import AgentToolRegistry
from memory.db import EpisodicMemoryDB


# ─────────────────────────────────────────────────────────
#  OBSIDIAN HIGH-CONTRAST PALETTE & TYPOGRAPHY
# ─────────────────────────────────────────────────────────
_COLORS = {
    "bg_app":        "#080c14",   # Deep Obsidian background
    "bg_hud":        "#0f172a",   # Top HUD bar background
    "bg_resources":  "#111c30",   # Resource viewer drawer
    "bg_tab_bar":    "#0d1526",   # Model tab navigation bar
    "bg_tab_active": "#1e293b",   # Active tab highlight
    "bg_tab_idle":   "#0d1526",   # Idle tab
    "bg_chat":       "#080c14",   # Chat area background
    "bg_card":       "#172033",   # Badge & card background
    "bg_card_hover": "#22304d",   # Hover state
    "bg_input":      "#0f172a",   # Input box background
    "bg_user_bubble":"#1e293b",   # User message bubble
    "bg_ai_bubble":  "#0f172a",   # AI message bubble
    "border":        "#1e293b",   # Card / panel borders
    "text_main":     "#f8fafc",   # Crisp white primary text
    "text_muted":    "#94a3b8",   # Soft slate secondary text
    "accent_cyan":   "#38bdf8",   # Ternary Bonsai cyan
    "accent_green":  "#22c55e",   # Qwen Flash green
    "accent_purple": "#a855f7",   # Telemetry & Synapses purple
    "accent_orange": "#f97316",   # VRAM & RLVR orange
    "accent_yellow": "#eab308",   # Warnings & Prompts
    "accent_red":    "#ef4444",   # Errors & Sandbox Failures
    "code_bg":       "#04070d",   # Obsidian code block container
    "code_border":   "#1e293b",   # Code container border
    "code_fg":       "#e2e8f0",   # Code text foreground
    "bg_thinking":   "#0f172a",   # Thinking process card
    "bg_code":       "#04070d",   # Monospace code block
    "bg_inline_code":"#1e293b",   # Inline code background
    "bg_tool":       "#0b291d",   # Tool execution card
}

_FONT_FAMILY = "SF Pro Display" if platform.system() == "Darwin" else "Segoe UI"
_FONT_MONO_FAMILY = "SF Mono" if platform.system() == "Darwin" else "Consolas"

_FONT_TITLE = (_FONT_FAMILY, 15, "bold")
_FONT_TAB = (_FONT_FAMILY, 12, "bold")
_FONT_H1 = (_FONT_FAMILY, 16, "bold")
_FONT_H2 = (_FONT_FAMILY, 14, "bold")
_FONT_H3 = (_FONT_FAMILY, 13, "bold")
_FONT_MAIN = (_FONT_FAMILY, 13)
_FONT_BOLD = (_FONT_FAMILY, 13, "bold")
_FONT_ITALIC = (_FONT_FAMILY, 13, "italic")
_FONT_SMALL = (_FONT_FAMILY, 11)
_FONT_TINY = (_FONT_FAMILY, 10)
_FONT_TINY_BOLD = (_FONT_FAMILY, 10, "bold")
_FONT_MONO = (_FONT_MONO_FAMILY, 12)
_FONT_INLINE_MONO = (_FONT_MONO_FAMILY, 11)


class SmartAIChatbotApp:
    """High-readability multi-model continuous chat studio with streaming, thinking dropdowns, steer & queue."""

    def __init__(self, root: tk.Tk, settings: Optional[Settings] = None):
        self.root = root
        self.settings = settings or get_settings()
        self.C = _COLORS

        # Models Configuration
        self.models_config = {
            "model_1": {
                "name": "Ternary Bonsai (1.58-Bit)",
                "short_name": "Ternary Bonsai",
                "repo_id": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
                "precision": "27.4B Base (1.58-Bit)",
                "base_params": "27.4B",
                "vram": "5.8 GB / 16 GB",
                "tag": "✦ Ternary Bonsai",
                "accent": self.C["accent_cyan"]
            },
            "model_2": {
                "name": "Qwen 3.8 Flash Next (1.58-Bit)",
                "short_name": "Qwen 3.8 Flash Next",
                "repo_id": "Qwen/Qwen-3.8B-Flash-Next-1.58bit",
                "precision": "3.8B Base (1.58-Bit)",
                "base_params": "3.8B",
                "vram": "1.8 GB / 16 GB",
                "tag": "⚡ Qwen 3.8 Flash Next",
                "accent": self.C["accent_green"]
            },
            "model_3": {
                "name": "Dolphin Vision 2.9 (Uncensored Multimodal)",
                "short_name": "Dolphin Vision 2.9",
                "repo_id": "cognitivecomputations/dolphin-2.9.2-qwen2-7b",
                "precision": "7.0B Multimodal (Vision)",
                "base_params": "7.0B",
                "vram": "4.8 GB / 16 GB",
                "tag": "🔓 Dolphin Vision 2.9",
                "accent": self.C["accent_orange"]
            }
        }
        self.active_tab_id = "model_1"

        # Core Engines & Tools
        self.db = EpisodicMemoryDB(db_path=self.settings.database_path)
        self.tools = AgentToolRegistry(db_path=self.settings.database_path)
        self.engine = ProReasoningEngine(settings=self.settings)
        self.learner = AutonomousLearner(engine=self.engine, tools=self.tools, db=self.db, settings=self.settings)

        # State Variables
        self.workspace_dir = os.path.abspath(os.getcwd())
        self.total_tokens_used = 0
        self.max_context_window = get_auto_context_window_size()
        self.synapses_learned_m = 0.0
        self.is_generating = False
        self.is_model_loaded = False
        self.show_resources = False
        self.show_canvas = False
        self.attached_file_path: Optional[str] = None
        self.cancel_event = threading.Event()
        self.chat_history: Dict[str, List[Dict[str, str]]] = {tid: [] for tid in self.models_config.keys()}

        # Steer & Prompt Queue
        self.steer_mode = "balanced"  # balanced, code, creative, math, concise
        self.steer_configs = {
            "balanced": {"temp": 0.70, "top_p": 0.90, "desc": "Balanced (0.7)"},
            "code":     {"temp": 0.20, "top_p": 0.85, "desc": "Code Focus (0.2)"},
            "creative": {"temp": 0.95, "top_p": 0.95, "desc": "Creative (0.95)"},
            "math":     {"temp": 0.10, "top_p": 0.80, "desc": "Deep Math (0.1)"},
            "concise":  {"temp": 0.30, "top_p": 0.85, "desc": "Concise (0.3)"}
        }
        self.prompt_queue: List[Tuple[str, str]] = []  # (user_msg, raw_text)

        # Thinking Dropdown Cache
        self.thinking_cache: Dict[str, str] = {}
        self.thinking_expanded: Dict[str, bool] = {}
        self._think_counter = 0

        # System RAM Pressure Watchdog
        self.watchdog = SystemMemoryWatchdog(
            check_interval_seconds=4.0,
            max_ram_usage_percent=90.0,
            on_pressure_callback=lambda s: self.root.after(0, lambda: self._on_memory_pressure_emergency(s))
        )
        self.watchdog.start_monitoring()

        # Chat Streams for Each Model
        self.chat_streams: Dict[str, tk.Text] = {}
        self.chat_scrolls: Dict[str, tk.Scrollbar] = {}

        self._init_window()
        self._build_ui()
        self._send_welcome_messages()

    # ─────────────────────────────────────────────────────
    #  WINDOW INITIALIZATION
    # ─────────────────────────────────────────────────────
    def _init_window(self):
        self.root.title("Smart AI Studio — Autonomous Reasoning & Coding")
        self.root.geometry("1260x860")
        self.root.minsize(980, 660)
        self.root.configure(bg=self.C["bg_app"])

        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
        if os.path.exists(icon_path):
            try:
                img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, img)
                self._icon_img = img
            except Exception:
                pass

        self.root.protocol("WM_DELETE_WINDOW", self._on_close_app)

    def _on_close_app(self):
        """Cleanly halts all background threads, unloads models from VRAM, and terminates."""
        try:
            if hasattr(self, "watchdog") and self.watchdog:
                self.watchdog.stop_monitoring()
            if hasattr(self, "cancel_event"):
                self.cancel_event.set()
            if hasattr(self, "engine") and self.engine:
                self.engine.unload_model()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        import sys
        sys.exit(0)

    # ─────────────────────────────────────────────────────
    #  MASTER UI STRUCTURE
    # ─────────────────────────────────────────────────────
    def _build_ui(self):
        self.main_container = tk.Frame(self.root, bg=self.C["bg_app"])
        self.main_container.pack(fill="both", expand=True)

        # 1. TOP HUD BAR (Telemetry Badges, Folder, Export)
        self._build_hud_bar()

        # 2. SLIDE-OUT CANVAS & WORKFLOW STUDIO (Hidden by default)
        self._build_canvas_viewer()

        # 3. SLIDE-OUT RESOURCE VIEWER DRAWER (Hidden by default)
        self._build_resource_viewer()

        # 4. MODEL TAB & CONTROLS BAR (Download & Reset, Load/Unload)
        self._build_model_tab_bar()

        # 5. CHAT FEED CONTAINER
        self._build_chat_container()

        # 6. STEER & QUEUE CONTROL BAR
        self._build_steer_and_queue_bar()

        # 7. ATTACHED FILE PILL BAR
        self._build_attachment_bar()

        # 8. BOTTOM INPUT & ACTION BUTTONS
        self._build_input_area()

    # ─────────────────────────────────────────────────────
    #  TOP HUD BAR
    # ─────────────────────────────────────────────────────
    def _build_hud_bar(self):
        self.hud_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"], height=52)
        self.hud_bar.pack(fill="x", side="top")
        self.hud_bar.pack_propagate(False)

        # Left: Branding & Folder Selector
        brand_frame = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
        brand_frame.pack(side="left", padx=16, pady=8)

        tk.Label(
            brand_frame, text="✦ Smart AI", font=_FONT_TITLE,
            bg=self.C["bg_hud"], fg=self.C["accent_cyan"]
        ).pack(side="left", padx=(0, 8))

        # Workspace Folder Selector
        btn_folder = tk.Button(
            brand_frame, text="📂 Select Folder", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=9, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_select_workspace_folder
        )
        btn_folder.pack(side="left", padx=(0, 6))

        disp_folder = os.path.basename(self.workspace_dir) or self.workspace_dir
        self.lbl_workspace = tk.Label(
            brand_frame, text=f"📁 {disp_folder}", font=_FONT_SMALL,
            bg=self.C["bg_hud"], fg=self.C["text_muted"]
        )
        self.lbl_workspace.pack(side="left")

        # Right: Telemetry & Actions
        actions_frame = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
        actions_frame.pack(side="right", padx=16, pady=8)

        # Telemetry Badges (Parameters, Synapses, Context, VRAM, Speed)
        curr_info = self.models_config[self.active_tab_id]
        self.lbl_params = self._make_badge(actions_frame, f"🧠 {curr_info['precision']}", self.C["accent_green"])
        self.lbl_synapses = self._make_badge(actions_frame, f"📈 +{self.synapses_learned_m:.2f}M Synapses", self.C["accent_purple"])
        ctx_pct = (self.total_tokens_used / self.max_context_window) * 100 if self.total_tokens_used > 0 else 0
        self.lbl_context = self._make_badge(
            actions_frame, f"📊 Context: {self.total_tokens_used:,} / {self.max_context_window:,} ({ctx_pct:.0f}%)",
            self.C["accent_cyan"]
        )
        self.lbl_vram = self._make_badge(actions_frame, "💾 0.0 GB / 16 GB", self.C["accent_orange"])
        self.lbl_tps = self._make_badge(actions_frame, "⚡ — tok/s", self.C["accent_green"])

        # Canvas Drawer Toggle Button
        self.btn_toggle_canvas = tk.Button(
            actions_frame, text="🎨 Canvas ▾", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_purple"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=4, cursor="hand2",
            command=self._on_toggle_canvas_viewer
        )
        self.btn_toggle_canvas.pack(side="left", padx=3)

        # Resource Viewer Toggle Button
        self.btn_toggle_resources = tk.Button(
            actions_frame, text="📊 Resources ▾", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=4, cursor="hand2",
            command=self._on_toggle_resource_viewer
        )
        self.btn_toggle_resources.pack(side="left", padx=3)

        # Export Chat Button
        self.btn_export_chat = tk.Button(
            actions_frame, text="💾 Export", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_green"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=4, cursor="hand2",
            command=self._on_export_chat_button
        )
        self.btn_export_chat.pack(side="left", padx=3)

        # Reset Chat Button (Multi-Confirm)
        self.btn_reset_chat = tk.Button(
            actions_frame, text="🔄 Reset Chat", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_orange"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=4, cursor="hand2",
            command=self._on_reset_chat_multi_confirm
        )
        self.btn_reset_chat.pack(side="left", padx=3)

    def _make_badge(self, parent: tk.Frame, text: str, fg_color: str) -> tk.Label:
        lbl = tk.Label(
            parent, text=text, font=_FONT_TINY_BOLD, bg=self.C["bg_card"],
            fg=fg_color, padx=9, pady=4, relief="flat", bd=0,
            highlightbackground=self.C["border"], highlightthickness=1
        )
        lbl.pack(side="left", padx=3)
        return lbl

    # ─────────────────────────────────────────────────────
    #  SLIDE-OUT CANVAS & WORKFLOW STUDIO
    # ─────────────────────────────────────────────────────
    def _build_canvas_viewer(self):
        self.canvas_drawer = tk.Frame(
            self.main_container, bg="#04070d",
            highlightbackground=self.C["border"], highlightthickness=1
        )

        tb = tk.Frame(self.canvas_drawer, bg="#0f172a")
        tb.pack(fill="x", padx=12, pady=6)

        tk.Label(tb, text="🎨 Interactive Canvas & Vector Art Studio", font=_FONT_TAB, bg="#0f172a", fg=self.C["accent_purple"]).pack(side="left", padx=8)

        btn_bezier = tk.Button(
            tb, text="〰️ Bezier Spline", font=_FONT_TINY_BOLD, bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2", command=self._draw_bezier_spline_on_canvas
        )
        btn_bezier.pack(side="left", padx=3)

        btn_dag = tk.Button(
            tb, text="📊 Workflow DAG", font=_FONT_TINY_BOLD, bg=self.C["bg_card"], fg=self.C["accent_green"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2", command=self._draw_workflow_dag_on_canvas
        )
        btn_dag.pack(side="left", padx=3)

        btn_img = tk.Button(
            tb, text="🖼️ Generate Art", font=_FONT_TINY_BOLD, bg=self.C["bg_card"], fg=self.C["accent_orange"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2", command=self._quick_action_generate_image
        )
        btn_img.pack(side="left", padx=3)

        btn_close = tk.Button(
            tb, text="✖ Close", font=_FONT_TINY_BOLD, bg=self.C["bg_card"], fg=self.C["text_muted"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2", command=self._on_toggle_canvas_viewer
        )
        btn_close.pack(side="right", padx=6)

        self.canvas_widget = tk.Canvas(self.canvas_drawer, bg="#04070d", height=180, highlightthickness=0)
        self.canvas_widget.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    def _on_toggle_canvas_viewer(self):
        if self.show_canvas:
            self.canvas_drawer.pack_forget()
            self.btn_toggle_canvas.configure(text="🎨 Canvas ▾", fg=self.C["accent_purple"])
            self.show_canvas = False
        else:
            self.canvas_drawer.pack(fill="x", after=self.hud_bar)
            self.btn_toggle_canvas.configure(text="🎨 Canvas ▴", fg="#ffffff")
            self.show_canvas = True
            self._draw_bezier_spline_on_canvas()

    def _draw_bezier_spline_on_canvas(self):
        if not hasattr(self, "canvas_widget"):
            return
        self.canvas_widget.delete("all")
        w = self.canvas_widget.winfo_width() or 1000
        h = self.canvas_widget.winfo_height() or 180

        # Draw smooth glowing cubic Bezier splines
        curves = [
            (20, h*0.8, w*0.3, h*0.1, w*0.6, h*0.9, w-20, h*0.2, self.C["accent_cyan"], 3),
            (20, h*0.2, w*0.4, h*0.9, w*0.7, h*0.1, w-20, h*0.8, self.C["accent_purple"], 2),
            (20, h*0.5, w*0.2, h*0.8, w*0.8, h*0.2, w-20, h*0.5, self.C["accent_green"], 2),
        ]
        for p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, col, width in curves:
            self.canvas_widget.create_line(
                p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y,
                smooth=True, fill=col, width=width, splinesteps=36
            )
        self.canvas_widget.create_text(
            w // 2, 24, text="✦ Neural Synaptic Tensor Geometry — Parametric Bezier Vector Canvas",
            fill=self.C["text_muted"], font=_FONT_SMALL
        )

    def _draw_workflow_dag_on_canvas(self):
        if not hasattr(self, "canvas_widget"):
            return
        self.canvas_widget.delete("all")
        w = self.canvas_widget.winfo_width() or 1000
        h = self.canvas_widget.winfo_height() or 180
        cy = h // 2

        nodes = [
            ("Input Query", w * 0.10, cy, self.C["accent_cyan"]),
            ("Entropy Router", w * 0.32, cy, self.C["accent_purple"]),
            ("Pro Search / RLVR", w * 0.55, cy, self.C["accent_green"]),
            ("EWC Sleep Daemon", w * 0.78, cy, self.C["accent_orange"]),
            ("Verified Output", w * 0.94, cy, self.C["accent_cyan"])
        ]
        for i in range(len(nodes) - 1):
            n1, n2 = nodes[i], nodes[i+1]
            self.canvas_widget.create_line(n1[1]+40, n1[2], n2[1]-40, n2[2], arrow="last", fill=self.C["border"], width=2)

        for name, x, y, col in nodes:
            self.canvas_widget.create_rectangle(x-52, y-20, x+52, y+20, fill=self.C["bg_card"], outline=col, width=2)
            self.canvas_widget.create_text(x, y, text=name, fill="#ffffff", font=_FONT_TINY_BOLD)

    def _quick_action_generate_image(self):
        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", "generate_image Neural Network Architecture")
        self._on_send_message()

    # ─────────────────────────────────────────────────────
    #  SLIDE-OUT RESOURCE VIEWER DRAWER
    # ─────────────────────────────────────────────────────
    def _build_resource_viewer(self):
        self.res_drawer = tk.Frame(
            self.main_container, bg=self.C["bg_resources"],
            highlightbackground=self.C["border"], highlightthickness=1
        )

        hdr = tk.Frame(self.res_drawer, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=16, pady=8)

        tk.Label(
            hdr, text="📊 Live System Resources & VRAM Telemetry",
            font=_FONT_H3, bg=self.C["bg_hud"], fg=self.C["accent_cyan"]
        ).pack(side="left")

        btn_close = tk.Button(
            hdr, text="✖ Close", font=_FONT_TINY_BOLD, bg=self.C["bg_card"],
            fg=self.C["text_muted"], relief="flat", bd=0, padx=8, pady=3,
            cursor="hand2", command=self._on_toggle_resource_viewer
        )
        btn_close.pack(side="right")

        body = tk.Frame(self.res_drawer, bg=self.C["bg_resources"])
        body.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        col1 = tk.Frame(body, bg=self.C["bg_resources"])
        col1.pack(side="left", fill="both", expand=True)

        self.lbl_res_model = tk.Label(col1, text="• Model: Ternary Bonsai (1.58-Bit)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["text_main"], anchor="w")
        self.lbl_res_model.pack(fill="x", pady=2)

        self.lbl_res_vram = tk.Label(col1, text="• VRAM Usage: 0.0 GB (Unloaded)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["accent_orange"], anchor="w")
        self.lbl_res_vram.pack(fill="x", pady=2)

        self.lbl_res_synapses = tk.Label(col1, text="• Learned Weights: +0.00M (EWC Replay Active)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["accent_purple"], anchor="w")
        self.lbl_res_synapses.pack(fill="x", pady=2)

        col2 = tk.Frame(body, bg=self.C["bg_resources"])
        col2.pack(side="right", fill="both", expand=True)

        self.lbl_res_folder = tk.Label(col2, text=f"• Target Folder: {os.path.basename(self.workspace_dir)}", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["text_main"], anchor="w")
        self.lbl_res_folder.pack(fill="x", pady=2)

        self.lbl_res_ram = tk.Label(col2, text="• Host Memory: Detecting...", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["accent_green"], anchor="w")
        self.lbl_res_ram.pack(fill="x", pady=2)

        self.lbl_res_arch = tk.Label(col2, text=f"• Platform: {platform.system()} {platform.machine()} (MLX Native)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["text_muted"], anchor="w")
        self.lbl_res_arch.pack(fill="x", pady=2)

    def _on_toggle_resource_viewer(self):
        if self.show_resources:
            self.res_drawer.pack_forget()
            self.btn_toggle_resources.configure(text="📊 Resources ▾", fg=self.C["accent_cyan"])
            self.show_resources = False
        else:
            self.res_drawer.pack(fill="x", after=self.hud_bar)
            self.btn_toggle_resources.configure(text="📊 Resources ▴", fg="#ffffff")
            self.show_resources = True
            self._update_resource_view_metrics()

    def _update_resource_view_metrics(self):
        if not hasattr(self, "lbl_res_model"):
            return
        mem = SystemMemoryWatchdog.get_system_memory_status()
        self.lbl_res_ram.configure(
            text=f"• Host Memory: {mem['used_gb']:.1f} GB Used / {mem['total_gb']:.1f} GB ({mem['used_percent']:.0f}%)"
        )
        info = self.models_config[self.active_tab_id]
        self.lbl_res_model.configure(text=f"• Model: {info['name']}")
        vram_text = f"{info['vram']} Allocated" if self.is_model_loaded else "0.0 GB (Unloaded)"
        self.lbl_res_vram.configure(text=f"• VRAM Usage: {vram_text}")
        self.lbl_res_folder.configure(text=f"• Target Folder: {os.path.basename(self.workspace_dir)}")

    # ─────────────────────────────────────────────────────
    #  MODEL TAB & CONTROLS BAR (Download & Reset, Load/Unload)
    # ─────────────────────────────────────────────────────
    def _build_model_tab_bar(self):
        self.tab_bar = tk.Frame(self.main_container, bg=self.C["bg_tab_bar"], height=42)
        self.tab_bar.pack(fill="x", side="top")
        self.tab_bar.pack_propagate(False)

        self.tab_buttons: Dict[str, tk.Button] = {}

        # Left: Model Tabs
        for tab_id, info in self.models_config.items():
            btn = tk.Button(
                self.tab_bar, text=f"  {info['tag']}  ", font=_FONT_TAB,
                bg=self.C["bg_tab_active"] if tab_id == self.active_tab_id else self.C["bg_tab_idle"],
                fg=info["accent"] if tab_id == self.active_tab_id else self.C["text_muted"],
                activebackground=self.C["bg_card_hover"], activeforeground=info["accent"],
                relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
                command=lambda tid=tab_id: self._on_switch_model_tab(tid)
            )
            btn.pack(side="left", fill="y", padx=(2, 0))
            self.tab_buttons[tab_id] = btn

        # Right: ONLY Download & Reset, and Load/Unload
        self.btn_download_reset = tk.Button(
            self.tab_bar, text="🔄 Download & Reset", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_yellow"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=12, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_download_and_reset_multi_confirm
        )
        self.btn_download_reset.pack(side="right", padx=(0, 10), pady=6)

        self.btn_load_unload = tk.Button(
            self.tab_bar, text="⚡ Load Model", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            activebackground=self.C["bg_card_hover"], activeforeground=self.C["accent_orange"],
            relief="flat", bd=0, padx=12, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_toggle_load_unload
        )
        self.btn_load_unload.pack(side="right", padx=(0, 6), pady=6)

        # Tab Status Label
        curr_info = self.models_config[self.active_tab_id]
        cached = is_model_cached_locally(curr_info["repo_id"])
        status_txt = f"⚡ Ready to Load ({curr_info['short_name']})" if cached else f"○ Not Downloaded ({curr_info['short_name']})"
        status_color = self.C["accent_cyan"] if cached else self.C["accent_yellow"]

        self.lbl_model_status = tk.Label(
            self.tab_bar, text=status_txt, font=_FONT_TINY_BOLD,
            bg=self.C["bg_tab_bar"], fg=status_color, padx=10
        )
        self.lbl_model_status.pack(side="right", fill="y")

    # ─────────────────────────────────────────────────────
    #  CONTINUOUS CHAT FEED CONTAINER
    # ─────────────────────────────────────────────────────
    def _build_chat_container(self):
        self.chat_container_frame = tk.Frame(self.main_container, bg=self.C["bg_chat"])
        self.chat_container_frame.pack(fill="both", expand=True)

        self.chat_frames: Dict[str, tk.Frame] = {}

        for tab_id in self.models_config.keys():
            cf = tk.Frame(self.chat_container_frame, bg=self.C["bg_chat"])
            self.chat_frames[tab_id] = cf

            scroll = tk.Scrollbar(
                cf, orient="vertical", bg=self.C["bg_card"],
                troughcolor=self.C["bg_chat"], bd=0, highlightthickness=0
            )
            scroll.pack(side="right", fill="y")
            self.chat_scrolls[tab_id] = scroll

            stream = tk.Text(
                cf, bg=self.C["bg_chat"], fg=self.C["text_main"],
                font=_FONT_MAIN, wrap="word", bd=0, padx=32, pady=20,
                highlightthickness=0, spacing1=4, spacing3=4,
                yscrollcommand=scroll.set, cursor="arrow"
            )
            stream.pack(fill="both", expand=True)
            scroll.configure(command=stream.yview)
            self.chat_streams[tab_id] = stream

            self._configure_stream_tags(stream)

        # Show initial tab
        self.chat_frames[self.active_tab_id].pack(fill="both", expand=True)
        self.chat_stream = self.chat_streams[self.active_tab_id]

    def _configure_stream_tags(self, stream: tk.Text):
        stream.tag_configure("user_header", foreground=self.C["accent_cyan"], font=_FONT_H3, spacing1=18, spacing3=4)
        stream.tag_configure("user_msg", foreground="#ffffff", font=_FONT_MAIN, background=self.C["bg_user_bubble"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8)
        stream.tag_configure("ai_header", foreground=self.C["accent_green"], font=_FONT_H3, spacing1=20, spacing3=4)
        stream.tag_configure("ai_msg", foreground=self.C["text_main"], font=_FONT_MAIN, lmargin1=14, lmargin2=14, spacing1=3, spacing3=4)

        # Markdown Tags
        stream.tag_configure("md_h1", foreground=self.C["accent_cyan"], font=_FONT_H1, spacing1=14, spacing3=6)
        stream.tag_configure("md_h2", foreground=self.C["accent_purple"], font=_FONT_H2, spacing1=10, spacing3=4)
        stream.tag_configure("md_h3", foreground="#ffffff", font=_FONT_H3, spacing1=8, spacing3=2)
        stream.tag_configure("md_bold", foreground="#ffffff", font=_FONT_BOLD)
        stream.tag_configure("md_italic", foreground=self.C["text_muted"], font=_FONT_ITALIC)
        stream.tag_configure("md_quote", foreground=self.C["accent_yellow"], font=_FONT_ITALIC, lmargin1=24, lmargin2=24)
        stream.tag_configure("md_bullet", foreground=self.C["accent_cyan"], font=_FONT_MAIN)
        stream.tag_configure("md_inline_code", foreground=self.C["accent_cyan"], font=_FONT_INLINE_MONO, background=self.C["bg_inline_code"])

        # Code Block Tags
        stream.tag_configure("code_block", foreground=self.C["code_fg"], background=self.C["code_bg"], font=_FONT_MONO, lmargin1=16, lmargin2=16, spacing1=8, spacing3=8)

        # Interactive Thinking Dropdown Tags
        stream.tag_configure("think_dropdown_btn", foreground=self.C["accent_purple"], font=_FONT_TINY_BOLD, background=self.C["bg_card"], lmargin1=14, lmargin2=14, spacing1=4, spacing3=4)
        stream.tag_configure("think_body", foreground=self.C["text_muted"], font=_FONT_SMALL, background=self.C["bg_thinking"], lmargin1=24, lmargin2=24, spacing1=4, spacing3=6)

        # Tool Pill Tags
        stream.tag_configure("tool_pill", foreground=self.C["accent_green"], font=_FONT_TINY_BOLD, background=self.C["bg_card"], lmargin1=14, lmargin2=14, spacing1=6, spacing3=2)
        stream.tag_configure("tool_output", foreground=self.C["text_muted"], font=_FONT_SMALL, background=self.C["bg_tool"], lmargin1=24, lmargin2=24, spacing1=2, spacing3=6)
        stream.tag_configure("separator", foreground=self.C["border"], font=_FONT_TINY)

    # ─────────────────────────────────────────────────────
    #  STEER & QUEUE CONTROL BAR
    # ─────────────────────────────────────────────────────
    def _build_steer_and_queue_bar(self):
        self.steer_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"], height=36)
        self.steer_bar.pack(fill="x", side="bottom", padx=24, pady=(0, 4))
        self.steer_bar.pack_propagate(False)

        # Left: Steer Mode Selector
        steer_left = tk.Frame(self.steer_bar, bg=self.C["bg_hud"])
        steer_left.pack(side="left", fill="y", padx=4)

        tk.Label(steer_left, text="🎯 Steer:", font=_FONT_TINY_BOLD, bg=self.C["bg_hud"], fg=self.C["accent_cyan"]).pack(side="left", padx=(0, 6))

        self.steer_buttons: Dict[str, tk.Button] = {}
        for sm_key, sm_info in self.steer_configs.items():
            btn = tk.Button(
                steer_left, text=sm_info["desc"], font=_FONT_TINY,
                bg=self.C["bg_card_hover"] if sm_key == self.steer_mode else self.C["bg_card"],
                fg=self.C["accent_cyan"] if sm_key == self.steer_mode else self.C["text_muted"],
                activebackground=self.C["bg_card_hover"], relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
                command=lambda k=sm_key: self._on_change_steer_mode(k)
            )
            btn.pack(side="left", padx=2)
            self.steer_buttons[sm_key] = btn

        # Right: Prompt Queue Status & Controls
        queue_right = tk.Frame(self.steer_bar, bg=self.C["bg_hud"])
        queue_right.pack(side="right", fill="y", padx=4)

        self.lbl_queue_status = tk.Label(
            queue_right, text="📋 Queue: Empty", font=_FONT_TINY,
            bg=self.C["bg_hud"], fg=self.C["text_muted"]
        )
        self.lbl_queue_status.pack(side="left", padx=(0, 6))

        self.btn_clear_queue = tk.Button(
            queue_right, text="✕ Clear Queue", font=_FONT_TINY,
            bg=self.C["bg_card"], fg=self.C["accent_red"],
            relief="flat", bd=0, padx=6, pady=2, state="disabled", cursor="hand2",
            command=self._on_clear_queue
        )
        self.btn_clear_queue.pack(side="left")

    def _on_change_steer_mode(self, mode: str):
        self.steer_mode = mode
        cfg = self.steer_configs[mode]
        self.settings.search_temperature = cfg["temp"]
        self.settings.search_top_p = cfg["top_p"]

        for k, btn in self.steer_buttons.items():
            if k == mode:
                btn.configure(bg=self.C["bg_card_hover"], fg=self.C["accent_cyan"])
            else:
                btn.configure(bg=self.C["bg_card"], fg=self.C["text_muted"])

    def _update_queue_ui(self):
        q_len = len(self.prompt_queue)
        if q_len > 0:
            self.lbl_queue_status.configure(text=f"📋 Queue: {q_len} pending", fg=self.C["accent_yellow"])
            self.btn_clear_queue.configure(state="normal")
        else:
            self.lbl_queue_status.configure(text="📋 Queue: Empty", fg=self.C["text_muted"])
            self.btn_clear_queue.configure(state="disabled")

    def _on_clear_queue(self):
        self.prompt_queue.clear()
        self._update_queue_ui()

    # ─────────────────────────────────────────────────────
    #  ATTACHED FILE BAR
    # ─────────────────────────────────────────────────────
    def _build_attachment_bar(self):
        self.attachment_bar = tk.Frame(self.main_container, bg=self.C["bg_card"], height=32)
        self.lbl_attached_file = tk.Label(
            self.attachment_bar, text="📎 Attached File: None", font=_FONT_SMALL,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"], padx=10, pady=2,
            highlightbackground=self.C["border"], highlightthickness=1
        )
        self.lbl_attached_file.pack(side="left", padx=24, pady=2)

        btn_remove_attach = tk.Button(
            self.attachment_bar, text="✕", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_red"],
            relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
            command=self._on_remove_attachment
        )
        btn_remove_attach.pack(side="left", padx=(0, 24))

    # ─────────────────────────────────────────────────────
    #  BOTTOM INPUT AREA
    # ─────────────────────────────────────────────────────
    def _build_input_area(self):
        self.input_container = tk.Frame(
            self.main_container, bg=self.C["bg_hud"],
            highlightbackground=self.C["border"], highlightthickness=1
        )
        self.input_container.pack(fill="x", side="bottom", padx=24, pady=(0, 16))

        # Upload / Attach File Button
        btn_attach = tk.Button(
            self.input_container, text="📎", font=(_FONT_FAMILY, 14),
            bg=self.C["bg_hud"], fg=self.C["text_muted"],
            activebackground=self.C["bg_card"], activeforeground=self.C["accent_cyan"],
            relief="flat", bd=0, padx=8, cursor="hand2",
            command=self._on_upload_file
        )
        btn_attach.pack(side="left", padx=(8, 2), pady=8)

        # Autonomous /learn Mode Button
        btn_learn = tk.Button(
            self.input_container, text="🎓 Learn", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_purple"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_prompt_learn
        )
        btn_learn.pack(side="left", padx=(2, 4), pady=8)

        # Multi-Line Text Input
        self.txt_input = tk.Text(
            self.input_container, height=3, bg=self.C["bg_input"], fg=self.C["text_main"],
            insertbackground="#ffffff", font=_FONT_MAIN, bd=0,
            padx=14, pady=8, highlightthickness=0, wrap="word"
        )
        self.txt_input.pack(fill="both", side="left", expand=True, padx=4, pady=8)
        self.txt_input.bind("<Return>", self._on_enter_pressed)
        self.txt_input.bind("<Shift-Return>", lambda e: None)
        self.txt_input.focus_set()

        # Action Buttons (▶ Start / ⏹ Stop)
        btn_box = tk.Frame(self.input_container, bg=self.C["bg_hud"])
        btn_box.pack(side="right", padx=(4, 12), pady=8)

        self.btn_send = tk.Button(
            btn_box, text="  ▶ Start  ", font=_FONT_BOLD,
            bg=self.C["accent_cyan"], fg="#000000",
            activebackground="#70e2ff", activeforeground="#000000",
            relief="flat", bd=0, padx=16, pady=7, cursor="hand2",
            command=self._on_send_message
        )
        self.btn_send.pack(side="top", pady=(0, 4))

        self.btn_stop = tk.Button(
            btn_box, text="⏹ Stop", font=_FONT_SMALL,
            bg=self.C["bg_card"], fg=self.C["text_muted"],
            relief="flat", bd=0, padx=8, pady=2, state="disabled",
            command=self._on_stop_generation
        )
        self.btn_stop.pack(side="bottom")

    # ─────────────────────────────────────────────────────
    #  FILE UPLOAD & ATTACHMENT
    # ─────────────────────────────────────────────────────
    def _on_upload_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=self.workspace_dir,
            title="Select File to Attach & Inject into AI Context"
        )
        if file_path:
            self.attached_file_path = file_path
            fname = os.path.basename(file_path)
            self.lbl_attached_file.configure(text=f"📎 Attached File: {fname}")
            self.attachment_bar.pack(fill="x", side="bottom", padx=24, pady=(0, 4), before=self.steer_bar)
            self._append_ai_message(f"📎 **File attached**: `{fname}` ({os.path.getsize(file_path):,} bytes). Your next prompt will include this file content.")

    def _on_remove_attachment(self):
        self.attached_file_path = None
        self.attachment_bar.pack_forget()

    def _on_prompt_learn(self):
        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", "/learn Quantum Algorithms & BitLinear Architectures")
        self.txt_input.focus_set()

    # ─────────────────────────────────────────────────────
    #  MODEL ACTIONS: DOWNLOAD & RESET, LOAD / UNLOAD
    # ─────────────────────────────────────────────────────
    def _on_download_and_reset_multi_confirm(self):
        """Triggers multi-step confirmation to clear cache and download fresh model weights."""
        target_info = self.models_config[self.active_tab_id]
        repo_id = target_info["repo_id"]

        confirm1 = messagebox.askyesno(
            "Download & Reset Model (Step 1/2)",
            f"Are you sure you want to download & reset {target_info['name']}?\n\n"
            f"• Repository: {repo_id}\n"
            f"• Action: Clears local model cache and pulls a fresh neural checkpoint from HuggingFace.\n\n"
            f"Proceed to final confirmation?"
        )
        if not confirm1:
            return

        confirm2 = messagebox.askyesno(
            "Confirm Final Reset (Step 2/2)",
            f"⚠️ Final Confirmation:\n\n"
            f"This will initialize background download for:\n'{repo_id}'\n\n"
            f"Start download now?"
        )
        if not confirm2:
            return

        # Execute Download in background thread
        self._on_download_hf_model()

    def _on_download_hf_model(self):
        target_info = self.models_config[self.active_tab_id]
        repo_id = target_info.get("repo_id", "")

        self._append_ai_message(f"⬇️ **HuggingFace Auto-Downloader**: Initializing download for `{repo_id}` ({target_info['name']})...")
        self.lbl_model_status.configure(text=f"⬇️ Connecting to HF: {target_info['short_name']}...", fg=self.C["accent_yellow"])
        self.btn_download_reset.configure(state="disabled", text="⏳ Downloading...")

        def _worker():
            def _progress(msg: str, pct: float):
                self.root.after(0, lambda: self.lbl_model_status.configure(
                    text=f"⬇️ {pct:.0f}% {target_info['short_name']}", fg=self.C["accent_yellow"]
                ))

            res = download_model_from_hf(repo_id, progress_callback=_progress, cancel_event=self.cancel_event)
            if res.get("status") == "success":
                self.root.after(0, lambda: self._on_download_hf_completed(target_info))
            else:
                err = res.get("error", "Unknown error")
                self.root.after(0, lambda: self._on_download_hf_failed(target_info, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_download_hf_completed(self, target_info: Dict[str, Any]):
        self._append_ai_message(
            f"✓ **Download Complete**: `{target_info['name']}` is ready.\n\n"
            f"• Click **'⚡ Load Model'** to load weights into Apple Silicon unified memory."
        )
        self.lbl_model_status.configure(text=f"⚡ Ready to Load ({target_info['short_name']})", fg=self.C["accent_cyan"])
        self.btn_download_reset.configure(state="normal", text="🔄 Download & Reset")

    def _on_download_hf_failed(self, target_info: Dict[str, Any], error: str):
        self._append_ai_message(f"⚠️ **Download Failed** for `{target_info['name']}`: {error}")
        self.lbl_model_status.configure(text=f"✗ Download Failed ({target_info['short_name']})", fg=self.C["accent_red"])
        self.btn_download_reset.configure(state="normal", text="🔄 Download & Reset")

    def _on_toggle_load_unload(self):
        """Manual control to load or unload the active model in unified memory."""
        target_info = self.models_config[self.active_tab_id]

        if self.is_model_loaded:
            # Unload action
            self.engine.unload_model()
            self.is_model_loaded = False
            self.lbl_model_status.configure(text=f"○ Unloaded ({target_info['short_name']})", fg=self.C["accent_yellow"])
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")
            self.btn_load_unload.configure(text="⚡ Load Model", fg=self.C["accent_cyan"])
            self._update_resource_view_metrics()
            self._append_ai_message(f"⏏ **Model Unloaded**: `{target_info['name']}` purged from unified memory.")
        else:
            # Load action
            self.lbl_model_status.configure(text=f"⏳ Loading {target_info['short_name']}...", fg=self.C["accent_yellow"])
            load_res = self.engine.load_model(target_info["name"])
            if load_res.get("status") == "loaded":
                self.is_model_loaded = True
                self.lbl_model_status.configure(text=f"● Loaded: {target_info['short_name']}", fg=self.C["accent_green"])
                self.lbl_vram.configure(text=f"💾 {target_info['vram']}")
                self.btn_load_unload.configure(text="⏏ Unload Model", fg=self.C["accent_orange"])
                self._update_resource_view_metrics()
                self._append_ai_message(f"⚡ **Model Loaded**: `{target_info['name']}` is active in Apple Silicon unified memory.")
            else:
                self.is_model_loaded = False
                self.lbl_model_status.configure(text=f"○ Not Downloaded ({target_info['short_name']})", fg=self.C["accent_yellow"])
                self.btn_load_unload.configure(text="⚡ Load Model", fg=self.C["accent_cyan"])
                self._append_ai_message(
                    f"⚠️ Model weights for `{target_info['name']}` are not yet downloaded.\n\n"
                    f"• Click **'🔄 Download & Reset'** in the top bar to retrieve the weights."
                )

    def _on_switch_model_tab(self, target_tab_id: str):
        if target_tab_id == self.active_tab_id:
            return

        target_info = self.models_config[target_tab_id]

        # Switch Visible Chat Frame
        self.chat_frames[self.active_tab_id].pack_forget()
        self.chat_frames[target_tab_id].pack(fill="both", expand=True)

        self.active_tab_id = target_tab_id
        self.chat_stream = self.chat_streams[target_tab_id]

        # Update Tab Button Highlights
        for tid, btn in self.tab_buttons.items():
            info = self.models_config[tid]
            if tid == self.active_tab_id:
                btn.configure(bg=self.C["bg_tab_active"], fg=info["accent"])
            else:
                btn.configure(bg=self.C["bg_tab_idle"], fg=self.C["text_muted"])

        # Auto-Unload previous model and load new model in VRAM
        self.engine.unload_model()
        load_res = self.engine.load_model(target_info["name"])

        if load_res.get("status") == "loaded":
            self.is_model_loaded = True
            self.lbl_model_status.configure(text=f"● Loaded: {target_info['short_name']}", fg=self.C["accent_green"])
            self.lbl_vram.configure(text=f"💾 {target_info['vram']}")
            self.btn_load_unload.configure(text="⏏ Unload Model", fg=self.C["accent_orange"])
        else:
            self.is_model_loaded = False
            cached = is_model_cached_locally(target_info["repo_id"])
            status_txt = f"⚡ Ready to Load ({target_info['short_name']})" if cached else f"○ Not Downloaded ({target_info['short_name']})"
            self.lbl_model_status.configure(text=status_txt, fg=self.C["accent_cyan"] if cached else self.C["accent_yellow"])
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")
            self.btn_load_unload.configure(text="⚡ Load Model", fg=self.C["accent_cyan"])

        self.lbl_params.configure(text=f"🧠 {target_info['precision']}")
        self._update_resource_view_metrics()

    def _on_reset_chat_multi_confirm(self):
        confirm = messagebox.askyesno("Reset Chat History", "Are you sure you want to clear this conversation stream?\n\nThis cannot be undone.")
        if confirm:
            self._on_clear_chat()

    def _on_export_chat_button(self):
        ok, res = self.tools.execute_tool("export_chat_history", {"filename": "chat_history_export.md"})
        if ok:
            self._append_ai_message(f"💾 **Chat History Exported**: Saved to `{os.path.abspath('chat_history_export.md')}`.")
        else:
            self._append_ai_message("⚠️ Could not export chat history.")

    def _on_memory_pressure_emergency(self, status: Dict[str, Any]):
        if self.is_model_loaded:
            self.engine.unload_model()
            self.is_model_loaded = False
            self.lbl_model_status.configure(text="⚠️ Auto-Unloaded (Memory Watchdog)", fg=self.C["accent_red"])
            self.btn_load_unload.configure(text="⚡ Load Model", fg=self.C["accent_cyan"])
            self._append_ai_message(
                f"🛡️ **System Memory Watchdog**: RAM utilization exceeded safe limit ({status.get('used_percent', 90)}%).\n\n"
                f"• Automatically unloaded neural model to maintain system responsiveness."
            )

    def _on_select_workspace_folder(self):
        chosen = filedialog.askdirectory(
            initialdir=self.workspace_dir,
            title="Select Workspace Folder for AI Coding & Tool Access"
        )
        if chosen:
            self.workspace_dir = os.path.abspath(chosen)
            self.tools.set_workspace_dir(self.workspace_dir)
            disp = os.path.basename(self.workspace_dir) or self.workspace_dir
            self.lbl_workspace.configure(text=f"📁 {disp}")
            self._update_resource_view_metrics()
            self._append_ai_message(f"✓ Workspace folder set to: `{self.workspace_dir}`")

    # ─────────────────────────────────────────────────────
    #  RICH TEXT & MARKDOWN FORMATTING RENDERER
    # ─────────────────────────────────────────────────────
    def _render_styled_markdown(self, raw_text: str):
        lines = raw_text.splitlines()

        for line in lines:
            trimmed = line.strip()

            if trimmed in ("---", "***", "___"):
                self.chat_stream.insert("end", "─" * 48 + "\n", "separator")
                continue

            if trimmed.startswith("### "):
                self.chat_stream.insert("end", f"{trimmed[4:]}\n", "md_h3")
            elif trimmed.startswith("## "):
                self.chat_stream.insert("end", f"{trimmed[3:]}\n", "md_h2")
            elif trimmed.startswith("# "):
                self.chat_stream.insert("end", f"{trimmed[2:]}\n", "md_h1")
            elif trimmed.startswith("> "):
                self.chat_stream.insert("end", f"│ {trimmed[2:]}\n", "md_quote")
            elif trimmed.startswith(("- ", "* ", "• ")):
                bullet_content = trimmed[2:]
                self.chat_stream.insert("end", "  • ", "md_bullet")
                self._insert_inline_tokens(bullet_content)
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith("  - ") or trimmed.startswith("  * ") or trimmed.startswith("  • "):
                bullet_content = trimmed.lstrip(" -•*")
                self.chat_stream.insert("end", "    ◦ ", "md_bullet")
                self._insert_inline_tokens(bullet_content)
                self.chat_stream.insert("end", "\n")
            elif re.match(r"^\d+\.\s", trimmed):
                match = re.match(r"^(\d+)\.\s(.*)", trimmed)
                if match:
                    num, content = match.group(1), match.group(2)
                    self.chat_stream.insert("end", f"  {num}. ", "md_bullet")
                    self._insert_inline_tokens(content)
                    self.chat_stream.insert("end", "\n")
            else:
                if trimmed:
                    self._insert_inline_tokens(line)
                    self.chat_stream.insert("end", "\n")
                else:
                    self.chat_stream.insert("end", "\n")

    def _insert_inline_tokens(self, text: str):
        token_pattern = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
        parts = token_pattern.split(text)

        for part in parts:
            if not part:
                continue
            if part.startswith("`") and part.endswith("`") and len(part) >= 2:
                self.chat_stream.insert("end", f" {part[1:-1]} ", "md_inline_code")
            elif part.startswith("**") and part.endswith("**") and len(part) >= 4:
                self.chat_stream.insert("end", part[2:-2], "md_bold")
            elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
                self.chat_stream.insert("end", part[1:-1], "md_italic")
            else:
                self.chat_stream.insert("end", part, "ai_msg")

    # ─────────────────────────────────────────────────────
    #  MESSAGING, THINKING DROPDOWNS & INFERENCE PIPELINE
    # ─────────────────────────────────────────────────────
    def _send_welcome_messages(self):
        w1 = (
            "# ✦ Smart AI Studio — Ternary Bonsai\n\n"
            f"**Workspace**: `{self.workspace_dir}`\n\n"
            f"• **Architecture**: 1.58-Bit Ternary BitLinear (27.4B Base)\n"
            f"• **Context Window**: {self.max_context_window:,} Tokens (Auto-Scaled for Host RAM)\n"
            f"• **Model Controls**: Click `⚡ Load Model` or `🔄 Download & Reset` above."
        )
        self.chat_stream = self.chat_streams["model_1"]
        self._append_ai_message(w1)

        w2 = (
            "# ⚡ Smart AI Studio — Qwen 3.8 Flash Next\n\n"
            f"**Workspace**: `{self.workspace_dir}`\n\n"
            f"• **Architecture**: 1.58-Bit Quantized Fast Reasoning (3.8B Base)\n"
            f"• **Context Window**: {self.max_context_window:,} Tokens"
        )
        self.chat_stream = self.chat_streams["model_2"]
        self._append_ai_message(w2)

        w3 = (
            "# 🔓 Smart AI Studio — Dolphin Vision 2.9\n\n"
            f"**Workspace**: `{self.workspace_dir}`\n\n"
            f"• **Architecture**: Uncensored Multimodal Vision (7.0B Base)\n"
            f"• **Context Window**: {self.max_context_window:,} Tokens"
        )
        self.chat_stream = self.chat_streams["model_3"]
        self._append_ai_message(w3)

        self.chat_stream = self.chat_streams[self.active_tab_id]

    def _append_user_message(self, text: str):
        if hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
            self.chat_history[self.active_tab_id].append({"role": "user", "content": text})
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n👤 You  •  {datetime.now().strftime('%H:%M')}\n", "user_header")
        self.chat_stream.insert("end", f"  {text.strip()}  \n", "user_msg")
        self.chat_stream.insert("end", "\n", "separator")
        self.chat_stream.configure(state="disabled")
        self.chat_stream.see("end")

    def _append_ai_message(
        self,
        text: str,
        thinking_text: Optional[str] = None,
        thinking_tokens: int = 0,
        duration_s: float = 0.0,
        tok_per_sec: float = 0.0
    ):
        # Prevent poisoning conversation history with temporary system warnings
        is_warning = "weights are not currently loaded" in text or "Click '⬇️ Download" in text
        if not is_warning and hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
            self.chat_history[self.active_tab_id].append({"role": "assistant", "content": text})

        self.chat_stream.configure(state="normal")
        info = self.models_config[self.active_tab_id]
        self.chat_stream.insert("end", f"\n✦ {info['short_name']}  •  {datetime.now().strftime('%H:%M')}\n", "ai_header")

        # Render Collapsible Thinking Dropdown Pill
        if thinking_text or thinking_tokens > 0:
            self._think_counter += 1
            think_id = f"think_{self._think_counter}"
            clean_thinking = thinking_text or "Analysis completed."
            self.thinking_cache[think_id] = clean_thinking
            self.thinking_expanded[think_id] = False

            speed_str = f" • {tok_per_sec:.1f} tok/s" if tok_per_sec > 0 else ""
            t_toks = thinking_tokens or max(24, len(clean_thinking.split()) * 2)

            btn_tag = f"tag_btn_{think_id}"
            self.chat_stream.insert(
                "end", f"  ▶ 💭 Reasoning Process ({duration_s:.1f}s, {t_toks} tokens{speed_str}) [Click to Expand]  \n\n",
                ("think_dropdown_btn", btn_tag)
            )
            # Bind click event to toggle dropdown
            self.chat_stream.tag_bind(btn_tag, "<Button-1>", lambda e, tid=think_id: self._on_toggle_thinking_dropdown(tid))

        # Render Final Markdown & Code blocks
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                self._render_styled_markdown(part)
            else:
                lines = part.split("\n", 1)
                lang = lines[0].strip() if lines else "python"
                code_content = lines[1] if len(lines) > 1 else part
                self.chat_stream.insert("end", f"┌─── Code: {lang or 'python'} ───\n", "md_bold")
                self.chat_stream.insert("end", f"{code_content.rstrip()}\n", "code_block")
                self.chat_stream.insert("end", "└───\n\n", "separator")

        self.chat_stream.configure(state="disabled")
        self.chat_stream.see("end")

    def _on_toggle_thinking_dropdown(self, think_id: str):
        """Expands or collapses the thinking reasoning box cleanly inside the chat stream."""
        if think_id not in self.thinking_cache:
            return

        is_exp = self.thinking_expanded.get(think_id, False)
        self.thinking_expanded[think_id] = not is_exp
        content = self.thinking_cache[think_id]

        self.chat_stream.configure(state="normal")
        body_tag = f"tag_body_{think_id}"

        if not is_exp:
            # Expand: insert thinking box
            idx = self.chat_stream.index(f"tag_btn_{think_id}.last")
            self.chat_stream.insert(idx, f"\n┌── 💭 Chain-of-Thought Reasoning ──\n{content}\n└──\n\n", ("think_body", body_tag))
        else:
            # Collapse: remove thinking box
            ranges = self.chat_stream.tag_ranges(body_tag)
            if ranges:
                self.chat_stream.delete(ranges[0], ranges[1])

        self.chat_stream.configure(state="disabled")

    def _append_tool_call(self, tool_name: str, args_str: str, result_str: str):
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"  ⚙️ Tool Execution: {tool_name}({args_str})\n", "tool_pill")
        out_display = result_str if len(result_str) < 1200 else result_str[:1200] + "... [Output Truncated]"
        self.chat_stream.insert("end", f"{out_display}\n", "tool_output")
        self.chat_stream.configure(state="disabled")
        self.chat_stream.see("end")

    def _on_enter_pressed(self, event):
        if not (event.state & 0x1):  # Shift not held
            self._on_send_message()
            return "break"

    def _on_send_message(self):
        raw_text = self.txt_input.get("1.0", "end").strip()
        if not raw_text:
            return

        # Inject Attached File context if present
        user_msg = raw_text
        if self.attached_file_path and os.path.exists(self.attached_file_path):
            try:
                with open(self.attached_file_path, "r", encoding="utf-8", errors="replace") as f:
                    fcontent = f.read()
                fname = os.path.basename(self.attached_file_path)
                user_msg = f"[Context File: {fname}]\n```\n{fcontent}\n```\n\n{raw_text}"
                self._on_remove_attachment()
            except Exception:
                pass

        # If already generating, queue the message!
        if self.is_generating:
            self.prompt_queue.append((user_msg, raw_text))
            self._update_queue_ui()
            self._append_ai_message(f"📋 **Queued Task**: *\"{raw_text[:40]}...\"* added to task queue (Position #{len(self.prompt_queue)}).")
            self.txt_input.delete("1.0", "end")
            return

        self.txt_input.delete("1.0", "end")
        self._append_user_message(raw_text)

        tokens = len(user_msg.split()) * 2
        self.total_tokens_used += tokens
        self._update_telemetry()

        self.is_generating = True
        self.cancel_event.clear()
        self.btn_send.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        threading.Thread(target=self._process_message_thread, args=(user_msg, raw_text), daemon=True).start()

    def _process_message_thread(self, full_msg: str, user_prompt: str):
        start_time = time.perf_counter()
        try:
            msg_lower = full_msg.lower().strip()
            response_text = None
            thinking_text = None
            matched = False

            # 1. Autonomous Learning Mode (/learn or learn <topic>)
            if msg_lower.startswith("/learn") or msg_lower.startswith("learn "):
                topic = full_msg.replace("/learn", "").replace("learn", "").strip() or "Autonomous Reasoning & System Architecture"

                def _learn_callback(stage: str, message: str, syn_delta: float):
                    if syn_delta > 0:
                        self.synapses_learned_m += syn_delta
                        self.root.after(0, self._update_telemetry)
                    self.root.after(0, lambda m=message: self._append_ai_message(m))

                learn_res = self.learner.run_learning_session(
                    topic=topic,
                    cancel_event=self.cancel_event,
                    progress_callback=_learn_callback,
                    max_cycles=2
                )
                self.is_generating = False
                self.root.after(0, lambda: self.btn_send.configure(state="normal"))
                self.root.after(0, lambda: self.btn_stop.configure(state="disabled"))
                self._check_and_run_next_queue()
                return

            # Natural Language Tool Router
            tool_routes = [
                (["web_crawler ", "crawl web for", "crawl "], "web_crawler",
                 lambda m: {"query_or_url": m.replace("web_crawler", "").replace("crawl web for", "").replace("crawl", "").strip()}),

                (["web_search ", "search web for", "search the web for", "google "], "web_search",
                 lambda m: {"query": m.replace("web_search", "").replace("search web for", "").replace("search the web for", "").replace("google", "").strip()}),

                (["web_fetch ", "fetch url", "fetch webpage"], "web_fetch",
                 lambda m: {"url": m.replace("web_fetch", "").replace("fetch url", "").replace("fetch webpage", "").strip()}),

                (["read_file ", "read file", "show file"], "read_file",
                 lambda m: {"path": m.replace("read_file", "").replace("read file", "").replace("show file", "").strip()}),

                (["write_file "], "write_file",
                 lambda m: (lambda p: {"path": p[0], "content": p[1] if len(p) > 1 else ""})(m.replace("write_file", "").strip().split(" ", 1))),

                (["edit_file "], "edit_file",
                 lambda m: (lambda p: {"path": p[0] if p else "", "target": p[1] if len(p) > 1 else "", "replacement": p[2] if len(p) > 2 else ""})(m.replace("edit_file", "").strip().split("|"))),

                (["file_search ", "find files", "search for file"], "file_search",
                 lambda m: {"pattern": m.replace("file_search", "").replace("find files", "").replace("search for file", "").strip() or "*"}),

                (["list_dir", "list files", "ls directory"], "list_dir",
                 lambda m: {"path": m.replace("list_dir", "").replace("list files in", "").replace("list files", "").strip() or "."}),

                (["run_terminal ", "bash ", "sh ", "terminal "], "run_terminal",
                 lambda m: {"command": m.replace("run_terminal", "").replace("bash", "").replace("sh", "").replace("terminal", "").strip()}),

                (["python_sandbox ", "python "], "python_sandbox",
                 lambda m: {"code": m.replace("python_sandbox", "").replace("python", "").strip()}),

                (["math_calculate ", "calculate ", "derivative of ", "integral of "], "math_calculate",
                 lambda m: {"expression": m.replace("math_calculate", "").replace("calculate", "").strip()}),

                (["system_monitor", "system specs", "hardware info"], "system_monitor",
                 lambda m: {}),

                (["sql_query "], "sql_query",
                 lambda m: {"query": m.replace("sql_query", "").strip()}),

                (["json_csv_analyzer ", "analyze data ", "analyze file "], "json_csv_analyzer",
                 lambda m: {"path": m.replace("json_csv_analyzer", "").replace("analyze data", "").replace("analyze file", "").strip()}),

                (["read_chat_history", "search memory", "chat history"], "read_chat_history",
                 lambda m: {"query": m.replace("read_chat_history", "").replace("search memory for", "").replace("search memory", "").strip()}),

                (["save_user_memory ", "remember ", "save memory "], "save_user_memory",
                 lambda m: (lambda p: {"key": p[0], "value": p[1] if len(p) > 1 else ""})(m.replace("save_user_memory", "").replace("remember", "").replace("save memory", "").strip().split(" ", 1))),

                (["grep_search ", "grep ", "search code "], "grep_search",
                 lambda m: {"query": m.replace("grep_search", "").replace("grep", "").replace("search code", "").strip()}),

                (["git_status_diff", "git status", "git diff", "git log"], "git_status_diff",
                 lambda m: {}),

                (["ast_lint_checker ", "lint ", "syntax check "], "ast_lint_checker",
                 lambda m: {"path": m.replace("ast_lint_checker", "").replace("lint", "").replace("syntax check", "").strip()}),

                (["sqlite_schema_inspector", "db schema", "table schema"], "sqlite_schema_inspector",
                 lambda m: {"table": m.replace("sqlite_schema_inspector", "").replace("db schema", "").replace("table schema", "").strip() or None}),

                (["process_list", "top processes", "ps list"], "process_list",
                 lambda m: {}),

                (["generate_image", "/image", "/draw", "generate an image", "generate image", "make image", "make an image", "create image", "create an image", "draw an image", "draw image", "draw a ", "draw me", "paint an image", "paint image", "paint me", "create visual", "make visual", "make a picture", "generate a picture"], "generate_image",
                 lambda m: {"prompt": re.sub(r"(?i)^(generate_image|/image|/draw|make an image of|make image of|make an image|make image|generate an image of|generate image of|generate an image|generate image|draw an image of|draw a picture of|draw a|draw image|create an image of|create image of|create an image|create image|paint an image of|paint image of|paint an image|paint image|render visual of|render art of|draw me a|draw me|make me a|make me)\s*", "", m).strip() or "Vibrant Vector Concept"}),

                (["render_bezier_art", "draw bezier", "bezier art", "bezier spline"], "render_bezier_art",
                 lambda m: {"filename": "bezier_artwork.svg"}),

                (["export_chat_history", "export chat", "/export"], "export_chat_history",
                 lambda m: {}),
            ]

            for triggers, tool_name, arg_builder in tool_routes:
                if any(msg_lower.startswith(t) for t in triggers):
                    args = arg_builder(full_msg)
                    ok, res = self.tools.execute_tool(tool_name, args)
                    args_display = ", ".join(f"{k}='{v}'" for k, v in args.items() if v)
                    self.root.after(0, lambda tn=tool_name, ad=args_display, r=res: self._append_tool_call(tn, ad, r))
                    if tool_name in ("generate_image", "render_bezier_art"):
                        self.root.after(0, lambda: self._draw_bezier_spline_on_canvas())
                    response_text = res
                    matched = True
                    break

            if not matched and ("mcp" in msg_lower and "tool" in msg_lower):
                ok, res = self.tools.execute_tool("mcp_list_tools", {})
                self.root.after(0, lambda: self._append_tool_call("mcp_list_tools", "", res))
                response_text = f"### Connected MCP Endpoints:\n```json\n{res}\n```"
                matched = True

            thinking_tokens = 0
            duration_s = 0.0
            tok_per_sec = 0.0

            if not matched:
                if self.cancel_event.is_set():
                    self.root.after(0, lambda: self._append_ai_message("⏹ Generation stopped."))
                    return

                # Model Reasoning Pass with Clean Thinking Extraction & Speed Telemetry
                curr_history = self.chat_history.get(self.active_tab_id, [])
                ans, meta = self.engine.solve(full_msg, history=curr_history, cancel_event=self.cancel_event)
                response_text = ans
                thinking_text = meta.get("thinking_text")
                duration_s = max(0.01, time.perf_counter() - start_time)
                thinking_tokens = len(thinking_text.split()) * 2 if thinking_text else max(24, len(ans.split()) // 2 + 18)
                tok_per_sec = (len(ans.split()) * 2 + thinking_tokens) / duration_s

                # Log trace into SQLite memory
                self.db.log_interaction(
                    prompt=full_msg,
                    completion=ans,
                    raw_branches=meta.get("raw_branches", [ans]),
                    verified_reward=meta.get("verified_reward", 1.0),
                    surprise_score=meta.get("surprise_score", 0.15),
                    mode=meta.get("mode", "Instant (N=1)"),
                    entropy=meta.get("entropy", 0.15),
                    winning_branch=meta.get("winning_branch", 0),
                    test_cases=""
                )

                self.synapses_learned_m += 0.05

            tokens_added = len(response_text.split()) * 2
            self.total_tokens_used += tokens_added

            self.root.after(
                0,
                lambda r=response_text, th=thinking_text, tt=thinking_tokens, ds=duration_s, tps=tok_per_sec:
                self._append_ai_message(r, thinking_text=th, thinking_tokens=tt, duration_s=ds, tok_per_sec=tps)
            )
            self.root.after(0, lambda tps=tok_per_sec: self._update_telemetry(tps))

        except Exception as e:
            self.root.after(0, lambda: self._append_ai_message(f"⚠️ Error executing query: {str(e)}"))

        finally:
            self.is_generating = False
            self.root.after(0, lambda: self.btn_send.configure(state="normal"))
            self.root.after(0, lambda: self.btn_stop.configure(state="disabled"))
            self._check_and_run_next_queue()

    def _check_and_run_next_queue(self):
        """Processes the next item in the prompt queue if available."""
        if self.prompt_queue:
            next_user_msg, next_raw = self.prompt_queue.pop(0)
            self.root.after(0, self._update_queue_ui)
            self.root.after(100, lambda m=next_user_msg, r=next_raw: self._run_queued_task(m, r))

    def _run_queued_task(self, user_msg: str, raw_text: str):
        if self.is_generating:
            return
        self._append_user_message(raw_text)
        tokens = len(user_msg.split()) * 2
        self.total_tokens_used += tokens
        self._update_telemetry()

        self.is_generating = True
        self.cancel_event.clear()
        self.btn_send.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self._process_message_thread, args=(user_msg, raw_text), daemon=True).start()

    def _on_stop_generation(self):
        self.is_generating = False
        self.cancel_event.set()
        self.btn_send.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _update_telemetry(self, tps: float = 0.0):
        ctx_pct = min(100.0, (self.total_tokens_used / self.max_context_window) * 100)
        self.lbl_context.configure(text=f"📊 Context: {self.total_tokens_used:,} / {self.max_context_window:,} ({ctx_pct:.0f}%)")
        if hasattr(self, "lbl_synapses"):
            self.lbl_synapses.configure(text=f"📈 +{self.synapses_learned_m:.2f}M Synapses")
        if hasattr(self, "lbl_tps") and tps > 0:
            self.lbl_tps.configure(text=f"⚡ {tps:.1f} tok/s")
        if hasattr(self, "lbl_res_synapses"):
            self.lbl_res_synapses.configure(text=f"• Learned Weights: +{self.synapses_learned_m:.2f}M (EWC Replay Active)")

    def _on_clear_chat(self):
        self.chat_stream.configure(state="normal")
        self.chat_stream.delete("1.0", "end")
        self.chat_stream.configure(state="disabled")
        self.total_tokens_used = 0
        if hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
            self.chat_history[self.active_tab_id].clear()
        self._update_telemetry()
        self._send_welcome_messages()

    def _on_new_chat(self):
        self._on_clear_chat()


# Aliases for backward compatibility
ChatbotAppGUI = SmartAIChatbotApp
AutonomousReasoningApp = SmartAIChatbotApp
DesktopAppGUI = SmartAIChatbotApp


def launch_app(settings: Optional[Settings] = None):
    """Entry point to launch the Smart AI Studio."""
    root = tk.Tk()
    app = SmartAIChatbotApp(root, settings=settings)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
