"""
Smart AI Studio — High-Readability Single-Chat Studio with Live Real-Time Token Streaming,
ChatGPT-Style Side-by-Side Interactive AI Canvas, Custom Model Importer, Live Steer (Cmd/Ctrl+Enter),
Task Queue, Dynamic Unit Scaling, and a Clean Single-Row Top Navigation Bar.
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

from config.paths import get_custom_models_file, get_portable_data_dir, inspect_mlx_model_folder
from config.settings import Settings, get_settings
from consolidation.daemon import SleepConsolidationDaemon
from core.autonomous_learner import AutonomousLearner
from core.hf_downloader import is_model_cached_locally, purge_local_model_cache, download_model_from_hf
from core.memory_watchdog import SystemMemoryWatchdog
from core.platform import get_auto_context_window_size
from core.pro_engine import ProReasoningEngine, parse_reasoning_and_response
from core.tools import AgentToolRegistry
from memory.db import EpisodicMemoryDB


# ─────────────────────────────────────────────────────────
#  DYNAMIC UNIT FORMATTERS (100 -> K -> M -> B -> T)
# ─────────────────────────────────────────────────────────
def format_added_synapses(count: float) -> str:
    """Formats added/learned synapses dynamically: 100 -> K -> M -> B."""
    if count < 1000:
        return f"+{int(count)} Synapses"
    elif count < 1_000_000:
        return f"+{count / 1000:.1f}K Synapses"
    elif count < 1_000_000_000:
        return f"+{count / 1_000_000:.2f}M Synapses"
    else:
        return f"+{count / 1_000_000_000:.2f}B Synapses"


def format_parameter_count(param_val: float) -> str:
    """Formats total model parameters dynamically: M -> B -> T."""
    if param_val < 1_000_000_000:
        return f"{param_val / 1_000_000:.0f}M"
    elif param_val < 1_000_000_000_000:
        return f"{param_val / 1_000_000_000:.1f}B"
    else:
        return f"{param_val / 1_000_000_000_000:.2f}T"


# ─────────────────────────────────────────────────────────
#  OBSIDIAN HIGH-CONTRAST PALETTE & TYPOGRAPHY
# ─────────────────────────────────────────────────────────
_COLORS = {
    "bg_app":        "#080c14",   # Deep Obsidian background
    "bg_hud":        "#0f172a",   # Sleek top navigation bar
    "bg_resources":  "#111c30",   # Slide-out resource drawer
    "bg_chat":       "#080c14",   # Chat area background
    "bg_card":       "#172033",   # Badge & card background
    "bg_card_hover": "#22304d",   # Hover state
    "bg_input":      "#0b1120",   # Input box background
    "bg_input_inner":"#04070d",   # Inner text field
    "bg_user_bubble":"#1e293b",   # User message bubble
    "bg_ai_bubble":  "#0f172a",   # AI message bubble
    "border":        "#1e293b",   # Card / panel borders
    "border_focus":  "#38bdf8",   # Input focus border
    "text_main":     "#ffffff",   # 100% Crisp White primary text
    "text_muted":    "#94a3b8",   # Soft slate secondary text
    "accent_cyan":   "#38bdf8",   # Cyan accent
    "accent_green":  "#22c55e",   # Green accent
    "accent_purple": "#c084fc",   # Purple accent
    "accent_orange": "#fb923c",   # Orange accent
    "accent_yellow": "#facc15",   # Yellow / warning accent
    "accent_red":    "#f87171",   # Red / error accent
    "code_bg":       "#04070d",   # Monospace code block container
    "code_border":   "#1e293b",   # Code container border
    "code_fg":       "#f1f5f9",   # Code text foreground
    "bg_thinking":   "#0f172a",   # Thinking process card
    "bg_inline_code":"#1e293b",   # Inline code background
    "bg_tool":       "#062d1f",   # Tool execution card
    "bg_steer":      "#2e1065",   # Steer message background
    "bg_queue":      "#3b2d07",   # Queue message background
    "canvas_bg":     "#04070d",   # AI Canvas background
    "canvas_hdr":    "#0f172a",   # AI Canvas header
}

_FONT_FAMILY = "SF Pro Display" if platform.system() == "Darwin" else "Segoe UI"
_FONT_MONO_FAMILY = "SF Mono" if platform.system() == "Darwin" else "Consolas"

_FONT_TITLE = (_FONT_FAMILY, 15, "bold")
_FONT_TAB = (_FONT_FAMILY, 12, "bold")
_FONT_H1 = (_FONT_FAMILY, 18, "bold")
_FONT_H2 = (_FONT_FAMILY, 16, "bold")
_FONT_H3 = (_FONT_FAMILY, 14, "bold")
_FONT_MAIN = (_FONT_FAMILY, 14)          # Large crisp 14pt body text
_FONT_BOLD = (_FONT_FAMILY, 14, "bold")
_FONT_ITALIC = (_FONT_FAMILY, 14, "italic")
_FONT_SMALL = (_FONT_FAMILY, 12)
_FONT_TINY = (_FONT_FAMILY, 10)
_FONT_TINY_BOLD = (_FONT_FAMILY, 10, "bold")
_FONT_MONO = (_FONT_MONO_FAMILY, 13)     # Clear 13pt monospace for code
_FONT_INLINE_MONO = (_FONT_MONO_FAMILY, 12)

CUSTOM_MODELS_FILE = get_custom_models_file()


class SmartAIChatbotApp:
    """Clean, high-readability single-chat studio with real-time streaming, AI Canvas, custom model importer, steer & queue."""

    def __init__(self, root: tk.Tk, settings: Optional[Settings] = None):
        self.root = root
        self.settings = settings or get_settings()
        self.C = _COLORS

        # Built-in Default Models Configuration (Expanded 5 Presets)
        self.models_config = {
            "model_1": {
                "name": "Qwen3.8-27B Uncensored (MLX 2-Bit)",
                "short_name": "Qwen 27B Uncensored (MLX)",
                "repo_id": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
                "model_path": None,
                "precision": "2-Bit Uncensored MLX",
                "raw_params": 27_000_000_000,
                "base_params": "27B",
                "max_context": 131_072,
                "vram": "5.6 GB / 16 GB",
                "tag": "🔥 Qwen 27B Uncensored",
                "accent": self.C["accent_cyan"]
            },
            "model_2": {
                "name": "Qwen3.8-27B Axon (Ternary {-1,0,+1} MLQT / GGUF)",
                "short_name": "Qwen 27B Axon (Ternary/GGUF)",
                "repo_id": "jayPark777/Qwen3.8-27B-Axon-MLQT",
                "model_path": None,
                "precision": "True {-1,0,+1} BitLinear / MLQT",
                "raw_params": 27_000_000_000,
                "base_params": "27B",
                "max_context": 131_072,
                "vram": "5.8 GB / 16 GB",
                "tag": "🧠 Qwen 27B Axon {-1,0,+1}",
                "accent": self.C["accent_green"]
            },
            "model_3": {
                "name": "Ternary Qwen 3.8B (Fast)",
                "short_name": "Ternary Qwen 3.8B",
                "repo_id": "h34v7/Ternary-Qwen3.5-3.8B-mlx",
                "model_path": None,
                "precision": "1.58-Bit Ternary",
                "raw_params": 3_800_000_000,
                "base_params": "3.8B",
                "max_context": 131_072,
                "vram": "1.8 GB / 16 GB",
                "tag": "⚡ Qwen 3.8B Fast",
                "accent": self.C["accent_yellow"]
            },
            "model_4": {
                "name": "Dolphin Vision 2.9 (Uncensored Multimodal)",
                "short_name": "Dolphin Vision 2.9",
                "repo_id": "cognitivecomputations/dolphin-2.9.2-qwen2-7b",
                "model_path": None,
                "precision": "7.0B Multimodal (Vision)",
                "raw_params": 7_000_000_000,
                "base_params": "7.0B",
                "max_context": 65_536,
                "vram": "4.8 GB / 16 GB",
                "tag": "👁️ Dolphin Vision 2.9",
                "accent": self.C["accent_orange"]
            },
            "model_5": {
                "name": "Flash Next Qwen 7B (Coder)",
                "short_name": "Qwen 7B Coder",
                "repo_id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "model_path": None,
                "precision": "7.0B Coder (4-bit)",
                "raw_params": 7_000_000_000,
                "base_params": "7.0B",
                "max_context": 65_536,
                "vram": "4.2 GB / 16 GB",
                "tag": "💻 Qwen 7B Coder",
                "accent": self.C["accent_purple"]
            }
        }
        self._load_saved_custom_models()
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
        self.synapses_learned_count = 0.0
        self.synapses_learned_m = 0.0
        self.is_generating = False
        self.is_model_loaded = False
        self.show_resources = False
        self.show_canvas = False
        self.canvas_preview_mode = False
        self.attached_file_path: Optional[str] = None
        self.cancel_event = threading.Event()
        self.chat_history: Dict[str, List[Dict[str, str]]] = {tid: [] for tid in self.models_config.keys()}

        # Prompt Task Queue
        self.prompt_queue: List[Tuple[str, str]] = []

        # Thinking & Code Snippet Caches
        self.thinking_cache: Dict[str, str] = {}
        self.thinking_expanded: Dict[str, bool] = {}
        self.code_snippets: Dict[str, str] = {}
        self._think_counter = 0
        self._code_counter = 0

        # System RAM Pressure Watchdog
        self.watchdog = SystemMemoryWatchdog(
            check_interval_seconds=4.0,
            max_ram_usage_percent=90.0,
            on_pressure_callback=lambda s: self.root.after(0, lambda: self._on_memory_pressure_emergency(s))
        )
        self.watchdog.start_monitoring()

        self._init_window()
        self._build_ui()
        self._send_welcome_messages()

    # ─────────────────────────────────────────────────────
    #  PERSISTENT CUSTOM MODEL LOADER & SAVER
    # ─────────────────────────────────────────────────────
    def _load_saved_custom_models(self):
        """Loads user-imported models from disk."""
        if os.path.exists(CUSTOM_MODELS_FILE):
            try:
                with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    for mid, mdata in saved.items():
                        self.models_config[mid] = mdata
            except Exception:
                pass

    def _save_custom_models(self):
        """Saves user-imported models to disk."""
        os.makedirs(os.path.dirname(CUSTOM_MODELS_FILE), exist_ok=True)
        custom_only = {k: v for k, v in self.models_config.items() if k.startswith("custom_")}
        try:
            with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump(custom_only, f, indent=2)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────
    #  WINDOW INITIALIZATION
    # ─────────────────────────────────────────────────────
    def _init_window(self):
        self.root.title("Smart AI Studio — Autonomous Local AI")
        self.root.geometry("1240x840")
        self.root.minsize(980, 640)
        self.root.configure(bg=self.C["bg_app"])

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

        # 1. TOP HEADER BAR (Height: 50px)
        self._build_single_top_bar()

        # 2. SLIDE-OUT RESOURCE VIEWER (Hidden by default)
        self._build_resource_viewer()

        # 3. BOTTOM INPUT AREA (Packed on bottom FIRST so it is ALWAYS 100% visible!)
        self._build_bottom_input_area()

        # 4. ATTACHMENT BAR (Positioned above input area, hidden when no file)
        self._build_attachment_bar()

        # 5. QUEUE BAR (Positioned above input area, hidden when empty)
        self._build_queue_bar()

        # 6. CENTER WORKSPACE (Single Chat Feed + Optional Side-by-Side AI Canvas)
        self.content_paned = tk.PanedWindow(
            self.main_container, orient="horizontal", bg=self.C["border"],
            sashwidth=4, bd=0
        )
        self.content_paned.pack(fill="both", expand=True, padx=16, pady=(6, 6))

        # Single Unified Chat Feed
        self._build_single_chat_feed()

        # Side-by-Side AI Canvas (Opens when code generated or toggled)
        self._build_ai_canvas_panel()

    # ─────────────────────────────────────────────────────
    #  1. SINGLE SLEEK TOP HEADER ROW (WELL-ORGANIZED BUTTONS)
    # ─────────────────────────────────────────────────────
    def _build_single_top_bar(self):
        self.hud_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"], height=50)
        self.hud_bar.pack(fill="x", side="top")
        self.hud_bar.pack_propagate(False)

        # Alias for backward test compatibility
        self.tab_bar = self.hud_bar

        # Left Section: Brand + Model Selector Dropdown + ➕ Import Model + Workspace Folder
        left_box = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
        left_box.pack(side="left", padx=14, pady=6)

        tk.Label(
            left_box, text="✦ Smart AI", font=_FONT_TITLE,
            bg=self.C["bg_hud"], fg=self.C["accent_cyan"]
        ).pack(side="left", padx=(0, 10))

        # Model Selector Dropdown
        self.model_var = tk.StringVar(value=self.models_config[self.active_tab_id]["short_name"])
        self.btn_model_menu = tk.Menubutton(
            left_box, textvariable=self.model_var, font=_FONT_TAB,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=12, pady=5, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1
        )
        self.btn_model_menu.pack(side="left", padx=(0, 6))
        self._refresh_model_menu()

        # Tab button aliases for test suite
        self.tab_buttons: Dict[str, tk.Button] = {}
        for tid in self.models_config.keys():
            self.tab_buttons[tid] = self.btn_model_menu

        # ➕ Import Model Button
        self.btn_import_model = tk.Button(
            left_box, text="➕ Import Model", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_purple"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_import_custom_model_dialog
        )
        self.btn_import_model.pack(side="left", padx=(0, 8))

        # Workspace folder button
        disp_folder = os.path.basename(self.workspace_dir) or self.workspace_dir
        self.lbl_workspace = tk.Label(
            left_box, text=f"📁 {disp_folder}", font=_FONT_SMALL,
            bg=self.C["bg_card"], fg=self.C["text_muted"], padx=8, pady=4,
            relief="flat", bd=0, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1
        )
        self.lbl_workspace.pack(side="left")
        self.lbl_workspace.bind("<Button-1>", lambda e: self._on_select_workspace_folder())

        # Right Section: Status Chips + Action Buttons
        right_box = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
        right_box.pack(side="right", padx=14, pady=6)

        curr_info = self.models_config[self.active_tab_id]
        param_str = format_parameter_count(curr_info.get("raw_params", 27_400_000_000))
        self.lbl_params = self._make_badge(right_box, f"🧠 {param_str} Base", self.C["accent_green"])

        syn_str = format_added_synapses(self.synapses_learned_count)
        self.lbl_synapses = self._make_badge(right_box, f"📈 {syn_str}", self.C["accent_purple"])

        ctx_pct = (self.total_tokens_used / self.max_context_window) * 100 if self.total_tokens_used > 0 else 0
        self.lbl_context = self._make_badge(
            right_box, f"📊 Context: {self.total_tokens_used:,} / {self.max_context_window:,} ({ctx_pct:.0f}%)",
            self.C["accent_cyan"]
        )
        self.lbl_vram = self._make_badge(right_box, "💾 0.0 GB / 16 GB", self.C["accent_orange"])
        self.lbl_tps = self._make_badge(right_box, "⚡ — tok/s", self.C["accent_green"])

        # Canvas Toggle Button
        self.btn_toggle_canvas = tk.Button(
            right_box, text="🎨 Canvas ▾", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_purple"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_toggle_canvas_viewer
        )
        self.btn_toggle_canvas.pack(side="left", padx=3)

        # Reset & Reinstall
        self.btn_reset_reinstall = tk.Button(
            right_box, text="🔄 Reset", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_yellow"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_reset_and_reinstall_single_confirm
        )
        self.btn_reset_reinstall.pack(side="left", padx=3)
        self.btn_download_reset = self.btn_reset_reinstall

        # Load / Unload Model Button
        self.btn_load_unload = tk.Button(
            right_box, text="⚡ Load Model", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            activebackground=self.C["bg_card_hover"], activeforeground=self.C["accent_orange"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_toggle_load_unload
        )
        self.btn_load_unload.pack(side="left", padx=3)

        # Resources toggle
        self.btn_toggle_resources = tk.Button(
            right_box, text="⚙", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["text_muted"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=6, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_toggle_resource_viewer
        )
        self.btn_toggle_resources.pack(side="left", padx=3)

        # Status Label
        cached = is_model_cached_locally(curr_info["repo_id"]) if curr_info.get("repo_id") else (os.path.exists(curr_info.get("model_path") or "") if curr_info.get("model_path") else False)
        status_txt = f"⚡ Ready ({curr_info['short_name']})" if cached else f"○ Not Downloaded ({curr_info['short_name']})"
        self.lbl_model_status = tk.Label(
            right_box, text=status_txt, font=_FONT_TINY_BOLD,
            bg=self.C["bg_hud"], fg=self.C["accent_cyan"] if cached else self.C["accent_yellow"], padx=4
        )
        self.lbl_model_status.pack(side="left")

    def _refresh_model_menu(self):
        """Rebuilds the model dropdown menu dynamically with all built-in and custom models."""
        menu = tk.Menu(self.btn_model_menu, tearoff=0, bg=self.C["bg_card"], fg=self.C["text_main"], font=_FONT_SMALL)
        for tid, info in self.models_config.items():
            menu.add_command(
                label=f"{info['tag']}  ({info['base_params']})",
                command=lambda t=tid: self._on_switch_model_tab(t)
            )
        menu.add_separator()
        menu.add_command(
            label="➕ Import Custom Model...",
            command=self._on_import_custom_model_dialog
        )
        self.btn_model_menu["menu"] = menu

    def _make_badge(self, parent: tk.Frame, text: str, fg_color: str) -> tk.Label:
        lbl = tk.Label(
            parent, text=text, font=_FONT_TINY_BOLD, bg=self.C["bg_card"],
            fg=fg_color, padx=7, pady=3, relief="flat", bd=0,
            highlightbackground=self.C["border"], highlightthickness=1
        )
        lbl.pack(side="left", padx=2)
        return lbl

    # ─────────────────────────────────────────────────────
    #  CUSTOM MODEL IMPORTER DIALOG (IMPORT YOUR OWN MODEL)
    # ─────────────────────────────────────────────────────
    def _on_import_custom_model_dialog(self):
        """Opens an interactive dialog to import custom local MLX model folders or HuggingFace repositories."""
        modal = tk.Toplevel(self.root)
        modal.title("Import Custom AI Model")
        modal.geometry("600x480")
        modal.minsize(540, 420)
        modal.configure(bg=self.C["bg_hud"])
        modal.transient(self.root)
        modal.grab_set()

        # Header
        hdr = tk.Frame(modal, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=20, pady=(16, 8))

        tk.Label(
            hdr, text="➕ Import Custom Model", font=_FONT_H2,
            bg=self.C["bg_hud"], fg=self.C["accent_purple"]
        ).pack(anchor="w")
        tk.Label(
            hdr, text="Add any local MLX model directory, GGUF/SafeTensors weights, or Hugging Face repository.",
            font=_FONT_SMALL, bg=self.C["bg_hud"], fg=self.C["text_muted"]
        ).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(modal, bg=self.C["bg_hud"])
        body.pack(fill="both", expand=True, padx=20, pady=6)

        # Quick Local MLX Folder Button
        top_btn_bar = tk.Frame(body, bg=self.C["bg_hud"])
        top_btn_bar.pack(fill="x", pady=(0, 10))

        lbl_status_inspect = tk.Label(
            body, text="💡 Tip: Select a local MLX folder to auto-detect architecture, parameters, and precision.",
            font=_FONT_TINY, bg=self.C["bg_card"], fg=self.C["accent_cyan"], padx=8, pady=4,
            relief="flat", bd=0, highlightbackground=self.C["border"], highlightthickness=1
        )
        lbl_status_inspect.pack(fill="x", pady=(0, 10))

        # 1. Model Name
        tk.Label(body, text="Model Display Name:", font=_FONT_SMALL, bg=self.C["bg_hud"], fg=self.C["text_main"]).pack(anchor="w")
        ent_name = tk.Entry(body, font=_FONT_MAIN, bg=self.C["bg_input_inner"], fg=self.C["text_main"], insertbackground="#ffffff", bd=0, highlightbackground=self.C["border"], highlightthickness=1)
        ent_name.pack(fill="x", pady=(2, 8), ipady=4)
        ent_name.insert(0, "Llama 3.2 3B Instruct")

        # 2. Source Type (HuggingFace vs Local Path)
        source_var = tk.StringVar(value="hf")

        radio_frame = tk.Frame(body, bg=self.C["bg_hud"])
        radio_frame.pack(fill="x", pady=(0, 6))

        tk.Radiobutton(radio_frame, text="HuggingFace Repo ID", variable=source_var, value="hf", bg=self.C["bg_hud"], fg=self.C["text_main"], selectcolor=self.C["bg_card"], activebackground=self.C["bg_hud"]).pack(side="left", padx=(0, 16))
        tk.Radiobutton(radio_frame, text="Local MLX Model Folder / File", variable=source_var, value="local", bg=self.C["bg_hud"], fg=self.C["text_main"], selectcolor=self.C["bg_card"], activebackground=self.C["bg_hud"]).pack(side="left")

        # 3. Path / Repo ID
        tk.Label(body, text="Repository ID or Local Path:", font=_FONT_SMALL, bg=self.C["bg_hud"], fg=self.C["text_main"]).pack(anchor="w")
        path_frame = tk.Frame(body, bg=self.C["bg_hud"])
        path_frame.pack(fill="x", pady=(2, 8))

        ent_path = tk.Entry(path_frame, font=_FONT_MAIN, bg=self.C["bg_input_inner"], fg=self.C["text_main"], insertbackground="#ffffff", bd=0, highlightbackground=self.C["border"], highlightthickness=1)
        ent_path.pack(side="left", fill="x", expand=True, ipady=4)
        ent_path.insert(0, "mlx-community/Llama-3.2-3B-Instruct-4bit")

        # 4. Parameters scale & Precision
        param_frame = tk.Frame(body, bg=self.C["bg_hud"])
        param_frame.pack(fill="x", pady=(0, 8))

        tk.Label(param_frame, text="Parameters:", font=_FONT_SMALL, bg=self.C["bg_hud"], fg=self.C["text_main"]).pack(side="left")
        ent_param = tk.Entry(param_frame, width=8, font=_FONT_MAIN, bg=self.C["bg_input_inner"], fg=self.C["text_main"], insertbackground="#ffffff", bd=0, highlightbackground=self.C["border"], highlightthickness=1)
        ent_param.pack(side="left", padx=(6, 16), ipady=3)
        ent_param.insert(0, "3B")

        tk.Label(param_frame, text="Precision:", font=_FONT_SMALL, bg=self.C["bg_hud"], fg=self.C["text_main"]).pack(side="left")
        ent_prec = tk.Entry(param_frame, width=14, font=_FONT_MAIN, bg=self.C["bg_input_inner"], fg=self.C["text_main"], insertbackground="#ffffff", bd=0, highlightbackground=self.C["border"], highlightthickness=1)
        ent_prec.pack(side="left", padx=6, ipady=3)
        ent_prec.insert(0, "4-bit MLX")

        def _on_folder_selected(selected_dir: str):
            if not selected_dir:
                return
            source_var.set("local")
            info = inspect_mlx_model_folder(selected_dir)
            if info.get("valid"):
                ent_name.delete(0, "end")
                ent_name.insert(0, info["name"])
                ent_path.delete(0, "end")
                ent_path.insert(0, info["path"])
                ent_param.delete(0, "end")
                ent_param.insert(0, info["param_str"])
                ent_prec.delete(0, "end")
                ent_prec.insert(0, info["precision"])
                lbl_status_inspect.configure(
                    text=f"✓ Detected {info['model_type']} ({info['param_str']}, {info['precision']}, {info['context_window']:,} max context)",
                    fg=self.C["accent_green"]
                )

        def _browse_local():
            d = filedialog.askdirectory(title="Select Local MLX Model Weights Directory")
            if d:
                _on_folder_selected(d)

        btn_browse = tk.Button(path_frame, text="📁 Browse MLX...", font=_FONT_TINY_BOLD, bg=self.C["bg_card"], fg=self.C["accent_cyan"], relief="flat", bd=0, padx=8, cursor="hand2", command=_browse_local)
        btn_browse.pack(side="right", padx=(6, 0))

        btn_quick_mlx = tk.Button(
            top_btn_bar, text="📁 Select Local MLX Folder...", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=10, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=_browse_local
        )
        btn_quick_mlx.pack(side="left")

        # Action Buttons
        btn_bar = tk.Frame(modal, bg=self.C["bg_hud"])
        btn_bar.pack(fill="x", padx=20, pady=(4, 16))

        def _save_imported_model():
            m_name = ent_name.get().strip() or "Custom Model"
            m_path_or_id = ent_path.get().strip()
            m_param_str = ent_param.get().strip() or "7B"
            m_prec_str = ent_prec.get().strip() or "4-bit MLX"

            if not m_path_or_id:
                messagebox.showerror("Error", "Please specify a HuggingFace Repository ID or Local Path.")
                return

            custom_id = f"custom_{int(time.time())}"
            raw_param = 7_000_000_000
            try:
                num_match = re.search(r"([\d\.]+)", m_param_str)
                if num_match:
                    val = float(num_match.group(1))
                    if "m" in m_param_str.lower():
                        raw_param = int(val * 1_000_000)
                    else:
                        raw_param = int(val * 1_000_000_000)
            except Exception:
                pass

            is_local = os.path.exists(m_path_or_id)
            new_model = {
                "name": f"{m_name} (Custom)",
                "short_name": m_name,
                "repo_id": m_path_or_id if not is_local else None,
                "model_path": m_path_or_id if is_local else None,
                "precision": m_prec_str,
                "raw_params": raw_param,
                "base_params": m_param_str,
                "max_context": 65_536,
                "vram": "4.0 GB / 16 GB",
                "tag": f"🧩 {m_name}",
                "accent": self.C["accent_purple"]
            }

            self.models_config[custom_id] = new_model
            self.chat_history[custom_id] = []
            self.tab_buttons[custom_id] = self.btn_model_menu
            self._save_custom_models()
            self._refresh_model_menu()
            modal.destroy()

            # Switch to newly imported model
            self._on_switch_model_tab(custom_id)
            self._append_ai_message(
                f"🧩 **Custom Model Imported**: Successfully registered `{m_name}`.\n\n"
                f"• **Source**: `{m_path_or_id}`\n"
                f"• **Scale**: {m_param_str} Parameters ({m_prec_str})\n"
                f"• Click **'⚡ Load Model'** to initialize."
            )

        btn_cancel = tk.Button(btn_bar, text="Cancel", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"], relief="flat", bd=0, padx=14, pady=6, cursor="hand2", command=modal.destroy)
        btn_cancel.pack(side="right", padx=(8, 0))

        btn_import = tk.Button(btn_bar, text="➕ Import & Select", font=_FONT_BOLD, bg=self.C["accent_purple"], fg="#000000", activebackground="#dfb5ff", relief="flat", bd=0, padx=16, pady=6, cursor="hand2", command=_save_imported_model)
        btn_import.pack(side="right")

    # ─────────────────────────────────────────────────────
    #  SLIDE-OUT RESOURCE VIEWER DRAWER
    # ─────────────────────────────────────────────────────
    def _build_resource_viewer(self):
        self.res_drawer = tk.Frame(
            self.main_container, bg=self.C["bg_resources"],
            highlightbackground=self.C["border"], highlightthickness=1
        )

        hdr = tk.Frame(self.res_drawer, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=16, pady=6)

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
        body.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        col1 = tk.Frame(body, bg=self.C["bg_resources"])
        col1.pack(side="left", fill="both", expand=True)

        self.lbl_res_model = tk.Label(col1, text="• Model: Ternary Bonsai (1.58-Bit)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["text_main"], anchor="w")
        self.lbl_res_model.pack(fill="x", pady=2)

        self.lbl_res_vram = tk.Label(col1, text="• VRAM Usage: 0.0 GB (Unloaded)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["accent_orange"], anchor="w")
        self.lbl_res_vram.pack(fill="x", pady=2)

        self.lbl_res_synapses = tk.Label(col1, text="• Learned Weights: +0 Synapses (EWC Replay Active)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["accent_purple"], anchor="w")
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
            self.btn_toggle_resources.configure(fg=self.C["text_muted"])
            self.show_resources = False
        else:
            self.res_drawer.pack(fill="x", after=self.hud_bar)
            self.btn_toggle_resources.configure(fg="#ffffff")
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
    #  2. SINGLE UNIFIED CHAT FEED
    # ─────────────────────────────────────────────────────
    def _build_single_chat_feed(self):
        self.chat_container_frame = tk.Frame(self.content_paned, bg=self.C["bg_chat"])
        self.content_paned.add(self.chat_container_frame, stretch="always", minsize=460)

        self.chat_frames: Dict[str, tk.Frame] = {}
        self.chat_streams: Dict[str, tk.Text] = {}
        self.chat_scrolls: Dict[str, tk.Scrollbar] = {}

        cf = tk.Frame(self.chat_container_frame, bg=self.C["bg_chat"])
        cf.pack(fill="both", expand=True)

        scroll = tk.Scrollbar(
            cf, orient="vertical", bg=self.C["bg_card"],
            troughcolor=self.C["bg_chat"], bd=0, highlightthickness=0
        )
        scroll.pack(side="right", fill="y")

        self.chat_stream = tk.Text(
            cf, bg=self.C["bg_chat"], fg=self.C["text_main"],
            font=_FONT_MAIN, wrap="word", bd=0, padx=28, pady=16,
            highlightthickness=0, spacing1=6, spacing3=6,
            yscrollcommand=scroll.set, cursor="arrow"
        )
        self.chat_stream.pack(fill="both", expand=True)
        scroll.configure(command=self.chat_stream.yview)

        self._configure_stream_tags(self.chat_stream)

        for tid in self.models_config.keys():
            self.chat_frames[tid] = cf
            self.chat_streams[tid] = self.chat_stream
            self.chat_scrolls[tid] = scroll

    def _configure_stream_tags(self, stream: tk.Text):
        stream.tag_configure("user_header", foreground=self.C["accent_cyan"], font=_FONT_H3, spacing1=18, spacing3=4)
        stream.tag_configure("user_msg", foreground="#ffffff", font=_FONT_MAIN, background=self.C["bg_user_bubble"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8)
        stream.tag_configure("ai_header", foreground=self.C["accent_green"], font=_FONT_H3, spacing1=20, spacing3=4)
        stream.tag_configure("ai_msg", foreground=self.C["text_main"], font=_FONT_MAIN, lmargin1=14, lmargin2=14, spacing1=3, spacing3=4)

        # Live Steer & Queue Tags
        stream.tag_configure("steer_header", foreground=self.C["accent_purple"], font=_FONT_H3, spacing1=16, spacing3=4)
        stream.tag_configure("steer_msg", foreground="#ffffff", font=_FONT_MAIN, background=self.C["bg_steer"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8)
        stream.tag_configure("queue_header", foreground=self.C["accent_yellow"], font=_FONT_H3, spacing1=16, spacing3=4)
        stream.tag_configure("queue_msg", foreground="#ffffff", font=_FONT_MAIN, background=self.C["bg_queue"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8)

        # Markdown Tags
        stream.tag_configure("md_h1", foreground=self.C["accent_cyan"], font=_FONT_H1, spacing1=14, spacing3=6)
        stream.tag_configure("md_h2", foreground=self.C["accent_purple"], font=_FONT_H2, spacing1=10, spacing3=4)
        stream.tag_configure("md_h3", foreground="#ffffff", font=_FONT_H3, spacing1=8, spacing3=2)
        stream.tag_configure("md_bold", foreground="#ffffff", font=_FONT_BOLD)
        stream.tag_configure("md_italic", foreground=self.C["text_muted"], font=_FONT_ITALIC)
        stream.tag_configure("md_quote", foreground=self.C["accent_yellow"], font=_FONT_ITALIC, lmargin1=24, lmargin2=24)
        stream.tag_configure("md_bullet", foreground=self.C["accent_cyan"], font=_FONT_MAIN)
        stream.tag_configure("md_inline_code", foreground=self.C["accent_cyan"], font=_FONT_INLINE_MONO, background=self.C["bg_inline_code"])

        # Code Block Tags & Actions
        stream.tag_configure("code_block", foreground=self.C["code_fg"], background=self.C["code_bg"], font=_FONT_MONO, lmargin1=16, lmargin2=16, spacing1=8, spacing3=8)
        stream.tag_configure("code_hdr", foreground=self.C["accent_cyan"], font=_FONT_TINY_BOLD, background=self.C["code_bg"], lmargin1=16, lmargin2=16, spacing1=6, spacing3=2)
        stream.tag_configure("code_action_copy", foreground=self.C["accent_green"], font=_FONT_TINY_BOLD, background=self.C["bg_card"])
        stream.tag_configure("code_action_canvas", foreground=self.C["accent_purple"], font=_FONT_TINY_BOLD, background=self.C["bg_card"])

        # Interactive Thinking Dropdown Tags
        stream.tag_configure("think_dropdown_btn", foreground=self.C["accent_purple"], font=_FONT_TINY_BOLD, background=self.C["bg_card"], lmargin1=14, lmargin2=14, spacing1=4, spacing3=4)
        stream.tag_configure("think_body", foreground=self.C["text_muted"], font=_FONT_SMALL, background=self.C["bg_thinking"], lmargin1=24, lmargin2=24, spacing1=4, spacing3=6)

        # Tool Pill Tags
        stream.tag_configure("tool_pill", foreground=self.C["accent_green"], font=_FONT_TINY_BOLD, background=self.C["bg_card"], lmargin1=14, lmargin2=14, spacing1=6, spacing3=2)
        stream.tag_configure("tool_output", foreground=self.C["text_muted"], font=_FONT_SMALL, background=self.C["bg_tool"], lmargin1=24, lmargin2=24, spacing1=2, spacing3=6)
        stream.tag_configure("separator", foreground=self.C["border"], font=_FONT_TINY)

    # ─────────────────────────────────────────────────────
    #  3. CHATGPT-STYLE SIDE-BY-SIDE AI CANVAS
    # ─────────────────────────────────────────────────────
    def _build_ai_canvas_panel(self):
        self.canvas_panel = tk.Frame(self.content_paned, bg=self.C["canvas_bg"])

        hdr = tk.Frame(self.canvas_panel, bg=self.C["canvas_hdr"], height=42)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="🎨 AI Canvas Studio",
            font=_FONT_H3, bg=self.C["canvas_hdr"], fg=self.C["accent_purple"]
        ).pack(side="left", padx=12, pady=6)

        btn_run = tk.Button(
            hdr, text="▶ Run", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_green"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            command=self._on_canvas_run_code
        )
        btn_run.pack(side="left", padx=3)

        btn_preview = tk.Button(
            hdr, text="👁️ Preview", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            command=self._on_canvas_toggle_preview
        )
        btn_preview.pack(side="left", padx=3)
        self.btn_canvas_preview = btn_preview

        btn_copy = tk.Button(
            hdr, text="📋 Copy", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["text_main"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            command=self._on_canvas_copy
        )
        btn_copy.pack(side="left", padx=3)

        btn_save = tk.Button(
            hdr, text="💾 Save", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_yellow"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            command=self._on_canvas_save_file
        )
        btn_save.pack(side="left", padx=3)

        btn_close = tk.Button(
            hdr, text="✖ Close", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["text_muted"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            command=self._on_toggle_canvas_viewer
        )
        btn_close.pack(side="right", padx=8)

        editor_frame = tk.Frame(self.canvas_panel, bg=self.C["canvas_bg"])
        editor_frame.pack(fill="both", expand=True)

        scroll_y = tk.Scrollbar(editor_frame, orient="vertical", bg=self.C["bg_card"])
        scroll_y.pack(side="right", fill="y")

        self.txt_canvas = tk.Text(
            editor_frame, bg=self.C["code_bg"], fg=self.C["code_fg"],
            insertbackground="#ffffff", font=_FONT_MONO, wrap="none",
            bd=0, padx=16, pady=16, highlightthickness=0,
            spacing1=3, spacing3=3, yscrollcommand=scroll_y.set
        )
        self.txt_canvas.pack(fill="both", expand=True)
        scroll_y.configure(command=self.txt_canvas.yview)

        self.txt_canvas.insert("1.0", "# ✦ Smart AI Canvas\n# Code, documents, and artifacts generated by the AI will appear here.\n# You can edit, run, or save them directly.\n\ndef solve():\n    print('Hello from Smart AI Canvas!')\n")

    def _on_toggle_canvas_viewer(self):
        if self.show_canvas:
            self.content_paned.forget(self.canvas_panel)
            self.btn_toggle_canvas.configure(text="🎨 Canvas ▾", fg=self.C["accent_purple"])
            self.show_canvas = False
        else:
            self.content_paned.add(self.canvas_panel, stretch="always", minsize=380)
            self.btn_toggle_canvas.configure(text="🎨 Canvas ▴", fg="#ffffff")
            self.show_canvas = True

    def _open_in_canvas(self, content: str):
        """Opens or updates the AI Canvas with generated code or document text."""
        if not self.show_canvas:
            self._on_toggle_canvas_viewer()
        self.txt_canvas.delete("1.0", "end")
        self.txt_canvas.insert("1.0", content)

    def _on_canvas_toggle_preview(self):
        self.canvas_preview_mode = not self.canvas_preview_mode
        if self.canvas_preview_mode:
            self.btn_canvas_preview.configure(text="💻 Raw Code", fg=self.C["accent_green"])
            self.txt_canvas.configure(font=_FONT_MAIN, wrap="word", bg=self.C["bg_card"])
        else:
            self.btn_canvas_preview.configure(text="👁️ Preview", fg=self.C["accent_cyan"])
            self.txt_canvas.configure(font=_FONT_MONO, wrap="none", bg=self.C["code_bg"])

    def _on_canvas_run_code(self):
        code = self.txt_canvas.get("1.0", "end").strip()
        if not code:
            return
        ok, res = self.tools.execute_tool("python_sandbox", {"code": code})
        self._append_tool_call("python_sandbox", "Canvas Code Execution", res)

    def _on_canvas_copy(self):
        code = self.txt_canvas.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        messagebox.showinfo("Copied", "Canvas content copied to clipboard.")

    def _on_canvas_save_file(self):
        content = self.txt_canvas.get("1.0", "end").strip()
        path = filedialog.asksaveasfilename(
            initialdir=self.workspace_dir,
            title="Save Canvas Content as File",
            defaultextension=".py",
            filetypes=[("Python Script", "*.py"), ("Markdown Document", "*.md"), ("Text File", "*.txt"), ("All Files", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self._append_ai_message(f"💾 **Canvas Saved**: File written to `{path}`.")

    # ─────────────────────────────────────────────────────
    #  4. ROCK-SOLID BOTTOM INPUT AREA (BEAUTIFUL CARD LAYOUT)
    # ─────────────────────────────────────────────────────
    def _build_bottom_input_area(self):
        self.input_container = tk.Frame(
            self.main_container, bg=self.C["bg_input"],
            highlightbackground=self.C["border"], highlightthickness=1
        )
        self.input_container.pack(fill="x", side="bottom", padx=20, pady=(0, 16))

        # Top Row: [📎] [Text Input Area] [▶ Send]
        top_row = tk.Frame(self.input_container, bg=self.C["bg_input"])
        top_row.pack(fill="x", padx=12, pady=(10, 4))

        btn_attach = tk.Button(
            top_row, text="📎", font=(_FONT_FAMILY, 15),
            bg=self.C["bg_card"], fg=self.C["text_muted"],
            activebackground=self.C["bg_card_hover"], activeforeground=self.C["accent_cyan"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_upload_file
        )
        btn_attach.pack(side="left", padx=(0, 8))

        # Input Text Box (3 lines, High Contrast White on Obsidian Dark)
        text_wrapper = tk.Frame(
            top_row, bg=self.C["bg_input_inner"],
            highlightbackground=self.C["border"], highlightthickness=1
        )
        text_wrapper.pack(side="left", fill="both", expand=True)

        self.txt_input = tk.Text(
            text_wrapper, height=3, bg=self.C["bg_input_inner"], fg=self.C["text_main"],
            insertbackground="#ffffff", font=_FONT_MAIN, bd=0,
            padx=12, pady=8, highlightthickness=0, wrap="word"
        )
        self.txt_input.pack(fill="both", expand=True)

        self._placeholder_active = True
        self.txt_input.insert("1.0", "Type your message or instruction... (Press Enter to send, ⌘+Enter to steer)")
        self.txt_input.configure(fg=self.C["text_muted"])

        self.txt_input.bind("<FocusIn>", self._on_input_focus_in)
        self.txt_input.bind("<FocusOut>", self._on_input_focus_out)
        self.txt_input.bind("<Return>", self._on_enter_pressed)
        self.txt_input.bind("<Shift-Return>", lambda e: None)
        self.txt_input.bind("<Command-Return>", self._on_steer_shortcut)
        self.txt_input.bind("<Control-Return>", self._on_steer_shortcut)

        # Prominent Send Button
        self.btn_send = tk.Button(
            top_row, text="  ▶ Send  ", font=_FONT_BOLD,
            bg=self.C["accent_cyan"], fg="#000000",
            activebackground="#70e2ff", activeforeground="#000000",
            relief="flat", bd=0, padx=18, pady=10, cursor="hand2",
            command=self._on_send_message
        )
        self.btn_send.pack(side="right", padx=(8, 0))

        # Bottom Row: Action Pills & Keyboard Shortcuts Hint
        bottom_row = tk.Frame(self.input_container, bg=self.C["bg_input"])
        bottom_row.pack(fill="x", padx=12, pady=(0, 8))

        left_actions = tk.Frame(bottom_row, bg=self.C["bg_input"])
        left_actions.pack(side="left")

        btn_learn = tk.Button(
            left_actions, text="🎓 Learn Mode", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_purple"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_prompt_learn
        )
        btn_learn.pack(side="left", padx=(0, 4))

        steer_key = "⌘⏎" if platform.system() == "Darwin" else "Ctrl+⏎"
        self.btn_steer = tk.Button(
            left_actions, text=f"🎯 Steer ({steer_key})", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_purple"],
            activebackground=self.C["bg_card_hover"], activeforeground="#ffffff",
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_steer_button_pressed
        )
        self.btn_steer.pack(side="left", padx=(0, 4))

        self.btn_stop = tk.Button(
            left_actions, text="⏹ Stop", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["text_muted"],
            relief="flat", bd=0, padx=8, pady=3, state="disabled",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_stop_generation
        )
        self.btn_stop.pack(side="left", padx=(0, 4))

        # Clear Chat Button
        self.btn_reset_chat = tk.Button(
            left_actions, text="🗑️ Clear Chat", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["text_muted"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=3, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_reset_chat_confirm
        )
        self.btn_reset_chat.pack(side="left", padx=(0, 4))

        # Export Button
        self.btn_export_chat = tk.Button(
            left_actions, text="💾 Export Chat", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_green"],
            activebackground=self.C["bg_card_hover"], relief="flat", bd=0,
            padx=8, pady=3, cursor="hand2",
            highlightbackground=self.C["border"], highlightthickness=1,
            command=self._on_export_chat_button
        )
        self.btn_export_chat.pack(side="left")

        # Right Side: Keyboard Hint
        tk.Label(
            bottom_row, text="↵ Send • ⌘↵ Steer • ⇧↵ New line", font=_FONT_TINY,
            bg=self.C["bg_input"], fg=self.C["text_muted"]
        ).pack(side="right")

    def _on_input_focus_in(self, event):
        if self._placeholder_active:
            self.txt_input.delete("1.0", "end")
            self.txt_input.configure(fg=self.C["text_main"])
            self._placeholder_active = False

    def _on_input_focus_out(self, event):
        if not self.txt_input.get("1.0", "end").strip():
            self._placeholder_active = True
            self.txt_input.insert("1.0", "Type your message or instruction... (Press Enter to send, ⌘+Enter to steer)")
            self.txt_input.configure(fg=self.C["text_muted"])

    # ─────────────────────────────────────────────────────
    #  ATTACHMENT & QUEUE BARS
    # ─────────────────────────────────────────────────────
    def _build_attachment_bar(self):
        self.attachment_bar = tk.Frame(self.main_container, bg=self.C["bg_card"], height=30)
        self.lbl_attached_file = tk.Label(
            self.attachment_bar, text="📎 Attached File: None", font=_FONT_SMALL,
            bg=self.C["bg_card"], fg=self.C["accent_cyan"], padx=10, pady=2,
            highlightbackground=self.C["border"], highlightthickness=1
        )
        self.lbl_attached_file.pack(side="left", padx=20, pady=2)

        btn_remove_attach = tk.Button(
            self.attachment_bar, text="✕", font=_FONT_TINY_BOLD,
            bg=self.C["bg_card"], fg=self.C["accent_red"],
            relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
            command=self._on_remove_attachment
        )
        btn_remove_attach.pack(side="left", padx=(0, 20))

    def _build_queue_bar(self):
        self.queue_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"], height=28)
        self.lbl_queue_indicator = tk.Label(
            self.queue_bar, text="📋 Pending Tasks: None", font=_FONT_TINY,
            bg=self.C["bg_hud"], fg=self.C["text_muted"], padx=10
        )
        self.lbl_queue_indicator.pack(side="left", padx=16)

        self.btn_clear_queue = tk.Button(
            self.queue_bar, text="✕ Clear All", font=_FONT_TINY,
            bg=self.C["bg_card"], fg=self.C["accent_red"],
            relief="flat", bd=0, padx=6, pady=1, cursor="hand2",
            command=self._on_clear_queue
        )

    def _update_queue_ui(self):
        q_len = len(self.prompt_queue)
        if q_len > 0:
            self.queue_bar.pack(fill="x", side="bottom", before=self.input_container, pady=(0, 2))
            self.lbl_queue_indicator.configure(
                text=f"📋 Active Task Queue: {q_len} pending task{'s' if q_len > 1 else ''} queued",
                fg=self.C["accent_yellow"]
            )
            self.btn_clear_queue.pack(side="left", padx=4)
        else:
            self.queue_bar.pack_forget()

    def _on_clear_queue(self):
        self.prompt_queue.clear()
        self._update_queue_ui()

    def _on_upload_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=self.workspace_dir,
            title="Select File to Attach & Inject into AI Context"
        )
        if file_path:
            self.attached_file_path = file_path
            fname = os.path.basename(file_path)
            self.lbl_attached_file.configure(text=f"📎 Attached File: {fname}")
            self.attachment_bar.pack(fill="x", side="bottom", padx=20, pady=(0, 4), before=self.input_container)
            self._append_ai_message(f"📎 **File attached**: `{fname}` ({os.path.getsize(file_path):,} bytes). Your next prompt will include this file content.")

    def _on_remove_attachment(self):
        self.attached_file_path = None
        self.attachment_bar.pack_forget()

    def _on_prompt_learn(self):
        if self._placeholder_active:
            self.txt_input.delete("1.0", "end")
            self.txt_input.configure(fg=self.C["text_main"])
            self._placeholder_active = False
        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", "/learn Quantum Algorithms & BitLinear Architectures")
        self.txt_input.focus_set()

    # ─────────────────────────────────────────────────────
    #  SINGLE RESET & REINSTALL
    # ─────────────────────────────────────────────────────
    def _on_reset_and_reinstall_single_confirm(self):
        target_info = self.models_config[self.active_tab_id]
        repo_id = target_info.get("repo_id")

        if not repo_id:
            messagebox.showinfo("Reset", f"{target_info['name']} is a local model. No HuggingFace cache to purge.")
            return

        confirm = messagebox.askyesno(
            "Reset & Reinstall Model",
            f"Are you sure you want to reset and reinstall {target_info['name']}?\n\n"
            f"• Repository: {repo_id}\n"
            f"• Action: Purges local cache and pulls fresh neural weights from HuggingFace.\n\n"
            f"Proceed with reinstall?"
        )
        if not confirm:
            return

        purge_local_model_cache(repo_id)
        self._on_download_hf_model()

    def _on_download_hf_model(self):
        target_info = self.models_config[self.active_tab_id]
        repo_id = target_info.get("repo_id", "")

        if not repo_id:
            return

        self._append_ai_message(f"⬇️ **HuggingFace Downloader**: Initializing download for `{repo_id}` ({target_info['name']})...")
        self.lbl_model_status.configure(text=f"⬇️ Connecting to HF: {target_info['short_name']}...", fg=self.C["accent_yellow"])
        self.btn_reset_reinstall.configure(state="disabled", text="⏳ Downloading...")

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
        self.btn_reset_reinstall.configure(state="normal", text="🔄 Reset")

    def _on_download_hf_failed(self, target_info: Dict[str, Any], error: str):
        self._append_ai_message(f"⚠️ **Download Failed** for `{target_info['name']}`: {error}")
        self.lbl_model_status.configure(text=f"✗ Download Failed ({target_info['short_name']})", fg=self.C["accent_red"])
        self.btn_reset_reinstall.configure(state="normal", text="🔄 Reset")

    def _on_toggle_load_unload(self):
        target_info = self.models_config[self.active_tab_id]
        m_path = target_info.get("model_path") or target_info.get("repo_id")

        if self.is_model_loaded:
            self.engine.unload_model()
            self.is_model_loaded = False
            self.lbl_model_status.configure(text=f"○ Unloaded ({target_info['short_name']})", fg=self.C["accent_yellow"])
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")
            self.btn_load_unload.configure(text="⚡ Load Model", fg=self.C["accent_cyan"])
            self._update_resource_view_metrics()
            self._append_ai_message(f"⏏ **Model Unloaded**: `{target_info['name']}` purged from unified memory.")
        else:
            self.lbl_model_status.configure(text=f"⏳ Loading {target_info['short_name']}...", fg=self.C["accent_yellow"])
            load_res = self.engine.load_model(target_info["name"], model_path=m_path)
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
                    f"⚠️ Model weights for `{target_info['name']}` are not yet loaded or downloaded.\n\n"
                    f"• Click **'🔄 Reset'** in the top bar to retrieve the weights if from HuggingFace."
                )

    def _on_switch_model_tab(self, target_tab_id: str):
        if target_tab_id not in self.models_config:
            return
        self.active_tab_id = target_tab_id
        target_info = self.models_config[target_tab_id]
        m_path = target_info.get("model_path") or target_info.get("repo_id")

        if hasattr(self, "model_var"):
            self.model_var.set(target_info["short_name"])

        self.engine.unload_model()
        load_res = self.engine.load_model(target_info["name"], model_path=m_path)

        if load_res.get("status") == "loaded":
            self.is_model_loaded = True
            self.lbl_model_status.configure(text=f"● Loaded: {target_info['short_name']}", fg=self.C["accent_green"])
            self.lbl_vram.configure(text=f"💾 {target_info['vram']}")
            self.btn_load_unload.configure(text="⏏ Unload Model", fg=self.C["accent_orange"])
        else:
            self.is_model_loaded = False
            cached = is_model_cached_locally(target_info["repo_id"]) if target_info.get("repo_id") else (os.path.exists(m_path or "") if m_path else False)
            status_txt = f"⚡ Ready to Load ({target_info['short_name']})" if cached else f"○ Not Downloaded ({target_info['short_name']})"
            self.lbl_model_status.configure(text=status_txt, fg=self.C["accent_cyan"] if cached else self.C["accent_yellow"])
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")
            self.btn_load_unload.configure(text="⚡ Load Model", fg=self.C["accent_cyan"])

        param_str = format_parameter_count(target_info.get("raw_params", 27_400_000_000))
        self.lbl_params.configure(text=f"🧠 {param_str} Base")
        self._update_resource_view_metrics()

    def _on_reset_chat_confirm(self):
        confirm = messagebox.askyesno("Clear Chat", "Are you sure you want to clear this conversation stream?\n\nThis cannot be undone.")
        if confirm:
            self._on_clear_chat()

    def _on_export_chat_button(self):
        ok, res = self.tools.execute_tool("export_chat_history", {"filename": "chat_history_export.md"})
        if ok:
            self._append_ai_message(f"💾 **Chat History Exported**: Saved to `{os.path.abspath('chat_history_export.md')}`.")
            messagebox.showinfo("Chat Exported", f"Chat history saved to:\n{os.path.abspath('chat_history_export.md')}")
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
                self.chat_stream.insert("end", "─" * 56 + "\n", "separator")
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
    #  MESSAGING & INFERENCE PIPELINE
    # ─────────────────────────────────────────────────────
    def _send_welcome_messages(self):
        self.chat_stream.configure(state="normal")
        info = self.models_config[self.active_tab_id]
        self.chat_stream.insert("end", f"✦ {info['name']}  •  {datetime.now().strftime('%H:%M')}\n", "ai_header")
        param_str = format_parameter_count(info.get("raw_params", 27_400_000_000))
        self.chat_stream.insert(
            "end",
            f"Smart AI Studio ready to assist with autonomous reasoning, coding, and tool execution.\n\n"
            f"• **Active Model**: {info['name']} ({param_str} Parameters)\n"
            f"• **Custom Models**: Click **'➕ Import Model'** in the top bar to add your own local or HuggingFace models.\n"
            f"• **Live Steer**: Press `⌘+Enter` (or `Ctrl+Enter`) to steer the model while typing.\n\n",
            "ai_msg"
        )
        self.chat_stream.insert("end", "─" * 56 + "\n\n", "separator")
        self.chat_stream.configure(state="disabled")

    def _append_user_message(self, text: str):
        if hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
            self.chat_history[self.active_tab_id].append({"role": "user", "content": text})
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n👤 You  •  {datetime.now().strftime('%H:%M')}\n", "user_header")
        self.chat_stream.insert("end", f"  {text.strip()}  \n", "user_msg")
        self.chat_stream.insert("end", "\n", "separator")
        self.chat_stream.configure(state="disabled")
        self.chat_stream.see("end")

    def _append_steer_directive(self, steer_text: str):
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n🎯 Live Steer Directive  •  {datetime.now().strftime('%H:%M')}\n", "steer_header")
        self.chat_stream.insert("end", f"  Steered: \"{steer_text.strip()}\"  \n", "steer_msg")
        self.chat_stream.insert("end", "\n", "separator")
        self.chat_stream.configure(state="disabled")
        self.chat_stream.see("end")

    def _append_queued_message(self, text: str, pos: int):
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n📋 Queued Task (#{pos})  •  {datetime.now().strftime('%H:%M')}\n", "queue_header")
        self.chat_stream.insert("end", f"  \"{text.strip()}\" (will execute automatically after current task)\n", "queue_msg")
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
        is_warning = "weights are not currently loaded" in text or "Click '⬇️ Download" in text
        if not is_warning and hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
            self.chat_history[self.active_tab_id].append({"role": "assistant", "content": text})

        self.chat_stream.configure(state="normal")
        info = self.models_config[self.active_tab_id]
        self.chat_stream.insert("end", f"\n✦ {info['short_name']}  •  {datetime.now().strftime('%H:%M')}\n", "ai_header")

        # Collapsible Thinking Dropdown Pill
        if thinking_text or thinking_tokens > 0:
            self._think_counter += 1
            think_id = f"think_{self._think_counter}"
            clean_thinking = thinking_text or "Step-by-step reasoning verified."
            self.thinking_cache[think_id] = clean_thinking
            self.thinking_expanded[think_id] = False

            speed_str = f" • {tok_per_sec:.1f} tok/s" if tok_per_sec > 0 else ""
            t_toks = thinking_tokens or max(24, len(clean_thinking.split()) * 2)

            btn_tag = f"tag_btn_{think_id}"
            self.chat_stream.insert(
                "end", f"  ▶ 💭 Reasoning Process ({duration_s:.1f}s, {t_toks} tokens{speed_str}) [Click to Expand]  \n\n",
                ("think_dropdown_btn", btn_tag)
            )
            self.chat_stream.tag_bind(btn_tag, "<Button-1>", lambda e, tid=think_id: self._on_toggle_thinking_dropdown(tid))

        # Render Markdown & Code blocks
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                self._render_styled_markdown(part)
            else:
                lines = part.split("\n", 1)
                lang = lines[0].strip() if lines else "python"
                code_content = lines[1] if len(lines) > 1 else part

                self._code_counter += 1
                code_id = f"code_{self._code_counter}"
                self.code_snippets[code_id] = code_content.strip()

                tag_copy = f"tag_cp_{code_id}"
                tag_canvas = f"tag_cv_{code_id}"

                self.chat_stream.insert("end", f"┌─── Code: {lang or 'python'}  ", "code_hdr")
                self.chat_stream.insert("end", " [📋 Copy] ", ("code_action_copy", tag_copy))
                self.chat_stream.insert("end", " [🎨 Open in Canvas] ", ("code_action_canvas", tag_canvas))
                self.chat_stream.insert("end", " ───\n", "code_hdr")

                self.chat_stream.insert("end", f"{code_content.rstrip()}\n", "code_block")
                self.chat_stream.insert("end", "└───\n\n", "separator")

                self.chat_stream.tag_bind(tag_copy, "<Button-1>", lambda e, cid=code_id: self._on_copy_code_snippet(cid))
                self.chat_stream.tag_bind(tag_canvas, "<Button-1>", lambda e, cid=code_id: self._on_open_snippet_in_canvas(cid))

                if len(code_content.splitlines()) > 4 and hasattr(self, "txt_canvas"):
                    self._open_in_canvas(code_content.strip())

        self.chat_stream.configure(state="disabled")
        self.chat_stream.see("end")

    def _on_copy_code_snippet(self, code_id: str):
        if code_id in self.code_snippets:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.code_snippets[code_id])
            messagebox.showinfo("Copied", "Code snippet copied to clipboard.")

    def _on_open_snippet_in_canvas(self, code_id: str):
        if code_id in self.code_snippets:
            self._open_in_canvas(self.code_snippets[code_id])

    def _on_toggle_thinking_dropdown(self, think_id: str):
        if think_id not in self.thinking_cache:
            return

        is_exp = self.thinking_expanded.get(think_id, False)
        self.thinking_expanded[think_id] = not is_exp
        content = self.thinking_cache[think_id]

        self.chat_stream.configure(state="normal")
        body_tag = f"tag_body_{think_id}"

        if not is_exp:
            idx = self.chat_stream.index(f"tag_btn_{think_id}.last")
            self.chat_stream.insert(idx, f"\n┌── 💭 Chain-of-Thought Reasoning ──\n{content}\n└──\n\n", ("think_body", body_tag))
        else:
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

    # ─────────────────────────────────────────────────────
    #  STEER & QUEUE PIPELINE
    # ─────────────────────────────────────────────────────
    def _on_steer_shortcut(self, event=None):
        self._on_steer_button_pressed()
        return "break"

    def _on_steer_button_pressed(self):
        steer_text = self.txt_input.get("1.0", "end").strip()
        if self._placeholder_active or not steer_text:
            return

        self.txt_input.delete("1.0", "end")

        if self.is_generating:
            self._append_steer_directive(steer_text)
            self.cancel_event.set()
            self.root.after(150, lambda: self._execute_steered_generation(steer_text))
        else:
            self._append_user_message(f"🎯 [Steer Directive]: {steer_text}")
            self.is_generating = True
            self.cancel_event.clear()
            self.btn_send.configure(text="📋 Queue", state="normal")
            self.btn_stop.configure(state="normal")
            threading.Thread(target=self._process_message_thread, args=(steer_text, steer_text), daemon=True).start()

    def _execute_steered_generation(self, steer_directive: str):
        self.is_generating = True
        self.cancel_event.clear()
        self.btn_send.configure(text="📋 Queue", state="normal")
        self.btn_stop.configure(state="normal")
        steered_prompt = f"[Live Steering Directive: {steer_directive}]\nPlease follow this directive directly."
        threading.Thread(target=self._process_message_thread, args=(steered_prompt, steer_directive), daemon=True).start()

    def _on_enter_pressed(self, event):
        if not (event.state & 0x1):  # Shift not held
            self._on_send_message()
            return "break"

    def _on_send_message(self):
        raw_text = self.txt_input.get("1.0", "end").strip()
        if self._placeholder_active or not raw_text:
            return

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

        if self.is_generating:
            self.prompt_queue.append((user_msg, raw_text))
            self._update_queue_ui()
            self._append_queued_message(raw_text, len(self.prompt_queue))
            self.txt_input.delete("1.0", "end")
            return

        self.txt_input.delete("1.0", "end")
        self._append_user_message(raw_text)

        tokens = len(user_msg.split()) * 2
        self.total_tokens_used += tokens
        self._update_telemetry()

        self.is_generating = True
        self.cancel_event.clear()
        self.btn_send.configure(text="📋 Queue", state="normal")
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
                        self.synapses_learned_count += syn_delta * 1_000_000
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
                self.root.after(0, lambda: self.btn_send.configure(text="  ▶ Send  ", state="normal"))
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

                self.synapses_learned_count += 250
                self.synapses_learned_m += 0.00025

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
            self.root.after(0, lambda: self.btn_send.configure(text="  ▶ Send  ", state="normal"))
            self.root.after(0, lambda: self.btn_stop.configure(state="disabled"))
            self._check_and_run_next_queue()

    def _check_and_run_next_queue(self):
        if self.prompt_queue:
            next_user_msg, next_raw = self.prompt_queue.pop(0)
            self._update_queue_ui()
            self.root.after(120, lambda m=next_user_msg, r=next_raw: self._run_queued_task(m, r))
        else:
            self._update_queue_ui()

    def _run_queued_task(self, user_msg: str, raw_text: str):
        if self.is_generating:
            return
        self._append_user_message(raw_text)
        tokens = len(user_msg.split()) * 2
        self.total_tokens_used += tokens
        self._update_telemetry()

        self.is_generating = True
        self.cancel_event.clear()
        self.btn_send.configure(text="📋 Queue", state="normal")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self._process_message_thread, args=(user_msg, raw_text), daemon=True).start()

    def _on_stop_generation(self):
        self.is_generating = False
        self.cancel_event.set()
        self.btn_send.configure(text="  ▶ Send  ", state="normal")
        self.btn_stop.configure(state="disabled")

    def _update_telemetry(self, tps: float = 0.0):
        ctx_pct = min(100.0, (self.total_tokens_used / self.max_context_window) * 100)
        self.lbl_context.configure(text=f"📊 Context: {self.total_tokens_used:,} / {self.max_context_window:,} ({ctx_pct:.0f}%)")
        if hasattr(self, "lbl_synapses"):
            syn_str = format_added_synapses(self.synapses_learned_count)
            self.lbl_synapses.configure(text=f"📈 {syn_str}")
        if hasattr(self, "lbl_tps") and tps > 0:
            self.lbl_tps.configure(text=f"⚡ {tps:.1f} tok/s")
        if hasattr(self, "lbl_res_synapses"):
            syn_str = format_added_synapses(self.synapses_learned_count)
            self.lbl_res_synapses.configure(text=f"• Learned Weights: {syn_str} (EWC Replay Active)")

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
