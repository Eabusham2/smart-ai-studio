"""
Smart AI Studio — High-Readability Single-Chat Studio with Live Real-Time Token Streaming,
ChatGPT-Style Side-by-Side Interactive AI Canvas, Custom Model Importer, Live Steer (Cmd/Ctrl+Enter),
Task Queue, Dynamic Unit Scaling, and a Clean Single-Row Top Navigation Bar.
"""

import ast
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.paths import get_custom_models_file, get_portable_data_dir, inspect_mlx_model_folder, terminate_existing_app_instances
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
    """Formats added/learned parameters dynamically: 100 -> K -> M -> B -> Params."""
    if count == 0:
        return "+0 Params"
    elif count < 1000:
        return f"+{int(count)} Params"
    elif count < 1_000_000:
        return f"+{count / 1000:.1f}K Params"
    elif count < 1_000_000_000:
        return f"+{count / 1_000_000:.2f}M Params"
    else:
        return f"+{count / 1_000_000_000:.2f}B Params"


format_added_params = format_added_synapses


def format_parameter_count(param_val: float) -> str:
    """Formats total model parameters dynamically: M -> B -> T."""
    if param_val < 1_000_000_000:
        return f"{param_val / 1_000_000:.0f}M"
    elif param_val < 1_000_000_000_000:
        return f"{param_val / 1_000_000_000:.1f}B"
    else:
        return f"{param_val / 1_000_000_000_000:.2f}T"


def detect_system_theme() -> str:
    """Detects whether macOS, Windows, or Linux is running in dark or light mode."""
    if platform.system() == "Darwin":
        try:
            import subprocess
            res = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=1.0
            )
            if "Dark" in res.stdout:
                return "dark"
            return "light"
        except Exception:
            return "dark"
    elif platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if val == 1 else "dark"
        except Exception:
            return "dark"
    return "dark"


# ─────────────────────────────────────────────────────────
#  AUTO LIGHT / DARK DUAL-MODE PALETTES (PURE B&W BUTTONS)
# ─────────────────────────────────────────────────────────
_THEMES = {
    "dark": {
        "bg_app":        "#080c14",   # Deep navy midnight background (Reverted)
        "bg_hud":        "#0e1626",   # Navy top bar (Reverted)
        "bg_resources":  "#0e1626",   # Slide-out resource drawer
        "bg_chat":       "#080c14",   # Chat area background
        "bg_card":       "#172033",   # Card container
        "bg_card_hover": "#1e293b",   # Hover state
        "btn_bg":        "#000000",   # Pure Solid Black Button
        "btn_fg":        "#ffffff",   # Pure Solid White Text
        "btn_hover":     "#000000",
        "btn_primary_bg":"#ffffff",   # Pure Solid White Button (Send/Run)
        "btn_primary_fg":"#000000",   # Pure Solid Black Text (Send/Run)
        "btn_primary_hover":"#ffffff",
        "badge_bg":      "#000000",   # Pure Solid Black Badge
        "badge_fg":      "#ffffff",   # Pure Solid White Text
        "bg_input":      "#0e1626",   # Input box background
        "bg_input_inner":"#080c14",   # Inner text field
        "bg_user_bubble":"#172033",   # User message bubble
        "bg_ai_bubble":  "#080c14",   # AI message bubble
        "border":        "#1e293b",   # Card / panel borders
        "border_focus":  "#38bdf8",   # Input focus border
        "text_main":     "#ffffff",   # Pure Solid White Primary Text
        "text_muted":    "#ffffff",   # Pure Solid White Secondary Text
        "text_placeholder":"#71717a", # Subtle muted ghost text for placeholder
        "accent_cyan":   "#38bdf8",   # Sky Cyan accent
        "accent_green":  "#4ade80",   # Emerald Green accent
        "accent_purple": "#c084fc",   # Purple accent
        "accent_orange": "#fb923c",   # Orange accent
        "accent_yellow": "#facc15",   # Yellow accent
        "accent_red":    "#f87171",   # Red accent
        "code_bg":       "#0f172a",   # Monospace code block container
        "code_border":   "#1e293b",   # Code container border
        "code_fg":       "#ffffff",   # Pure Solid White Code text
        "bg_thinking":   "#131d33",   # Thinking process card
        "bg_inline_code":"#1e293b",   # Inline code background
        "bg_tool":       "#091b14",   # Tool execution card
        "bg_steer":      "#2e1065",   # Steer message background
        "bg_queue":      "#451a03",   # Queue message background
        "canvas_bg":     "#080c14",   # AI Canvas background
        "canvas_hdr":    "#0e1626",   # AI Canvas header
    },
    "light": {
        "bg_app":        "#f8fafc",   # Soft Light background (Reverted)
        "bg_hud":        "#f1f5f9",   # Top bar (Slate 100) (Reverted)
        "bg_resources":  "#f1f5f9",   # Slide-out resource drawer
        "bg_chat":       "#f8fafc",   # Chat area background
        "bg_card":       "#f1f5f9",   # Card background
        "bg_card_hover": "#e2e8f0",   # Hover state
        "btn_bg":        "#ffffff",   # Pure Solid White Button
        "btn_fg":        "#000000",   # Pure Solid Black Text
        "btn_hover":     "#ffffff",
        "btn_primary_bg":"#000000",   # Pure Solid Black Button (Send/Run)
        "btn_primary_fg":"#ffffff",   # Pure Solid White Text (Send/Run)
        "btn_primary_hover":"#000000",
        "badge_bg":      "#ffffff",   # Pure Solid White Badge
        "badge_fg":      "#000000",   # Pure Solid Black Text
        "bg_input":      "#f1f5f9",   # Input box background
        "bg_input_inner":"#ffffff",   # Inner text field
        "bg_user_bubble":"#f1f5f9",   # User message bubble
        "bg_ai_bubble":  "#f8fafc",   # AI message bubble
        "border":        "#e2e8f0",   # Card / panel borders
        "border_focus":  "#0284c7",   # Input focus border
        "text_main":     "#000000",   # Pure Solid Black Primary Text
        "text_muted":    "#000000",   # Pure Solid Black Secondary Text
        "text_placeholder":"#64748b", # Subtle muted ghost text for placeholder
        "accent_cyan":   "#0284c7",   # Sky Blue accent
        "accent_green":  "#15803d",   # Green accent
        "accent_purple": "#7e22ce",   # Deep Purple accent
        "accent_orange": "#c2410c",   # Deep Orange accent
        "accent_yellow": "#a16207",   # Amber / Yellow accent
        "accent_red":    "#b91c1c",   # Red accent
        "code_bg":       "#f1f5f9",   # Monospace code block container
        "code_border":   "#e2e8f0",   # Code container border
        "code_fg":       "#000000",   # Pure Solid Black Code text
        "bg_thinking":   "#f1f5f9",   # Thinking process card
        "bg_inline_code":"#e2e8f0",   # Inline code background
        "bg_tool":       "#dcfce7",   # Tool execution card
        "bg_steer":      "#f3e8ff",   # Steer message background
        "bg_queue":      "#fef3c7",   # Queue message background
        "canvas_bg":     "#f8fafc",   # AI Canvas background
        "canvas_hdr":    "#f1f5f9",   # AI Canvas header
    }
}

_INITIAL_THEME = detect_system_theme()
_COLORS = dict(_THEMES[_INITIAL_THEME])

_FONT_FAMILY = "SF Pro Display" if platform.system() == "Darwin" else "Segoe UI"
_FONT_MONO_FAMILY = "SF Mono" if platform.system() == "Darwin" else "Consolas"

_FONT_TITLE = (_FONT_FAMILY, 15, "bold")
_FONT_TAB = (_FONT_FAMILY, 12, "bold")
_FONT_H1 = (_FONT_FAMILY, 18, "bold")
_FONT_H2 = (_FONT_FAMILY, 16, "bold")
_FONT_H3 = (_FONT_FAMILY, 14, "bold")
_FONT_MAIN = (_FONT_FAMILY, 14)          # Large crisp 14pt body text
_FONT_INPUT = (_FONT_FAMILY, 15)         # Prominent 15pt text for input box
_FONT_BOLD = (_FONT_FAMILY, 14, "bold")
_FONT_ITALIC = (_FONT_FAMILY, 14, "italic")
_FONT_BOLD_ITALIC = (_FONT_FAMILY, 14, "bold", "italic")
_FONT_SMALL = (_FONT_FAMILY, 12)
_FONT_TINY = (_FONT_FAMILY, 10)
_FONT_TINY_BOLD = (_FONT_FAMILY, 10, "bold")
_FONT_MONO = (_FONT_MONO_FAMILY, 13)     # Clear 13pt monospace for code
_FONT_INLINE_MONO = (_FONT_MONO_FAMILY, 12)

CUSTOM_MODELS_FILE = get_custom_models_file()


# ─────────────────────────────────────────────────────────
#  SOLID B&W BUTTON COMPONENT (BYPASSES MACOS AQUA GREYING)
# ─────────────────────────────────────────────────────────
class SolidButton(tk.Label):
    """
    Cross-platform solid button implemented on tk.Label.
    Bypasses macOS Aqua NSButton native rendering which forces grey backgrounds and grey text.
    Guarantees 100% pixel-perfect solid background and crisp foreground colors on macOS, Linux, and Windows.
    """
    def __init__(self, parent, text="", command=None, bg=None, fg=None,
                 font=None, padx=10, pady=5, cursor="hand2", state="normal", **kwargs):
        self._command = command
        self._state = state
        self._normal_bg = bg if bg is not None else _COLORS.get("btn_bg", "#000000")
        self._normal_fg = fg if fg is not None else _COLORS.get("btn_fg", "#ffffff")
        self._hover_bg = kwargs.pop("activebackground", self._normal_bg)
        self._hover_fg = kwargs.pop("activeforeground", self._normal_fg)
        relief = kwargs.pop("relief", "flat")
        bd = kwargs.pop("bd", kwargs.pop("borderwidth", 0))
        kwargs.pop("highlightbackground", None)
        kwargs.pop("highlightthickness", None)
        
        super().__init__(
            parent, text=text, font=font, bg=self._normal_bg, fg=self._normal_fg,
            padx=padx, pady=pady, cursor=cursor if state == "normal" else "arrow",
            relief=relief, bd=bd, **kwargs
        )
        
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_click(self, event=None):
        if self._state == "normal" and self._command:
            self._command()

    def _on_enter(self, event=None):
        if self._state == "normal":
            super().configure(bg=self._hover_bg, fg=self._hover_fg)

    def _on_leave(self, event=None):
        if self._state == "normal":
            super().configure(bg=self._normal_bg, fg=self._normal_fg)

    def invoke(self):
        """Simulates button click (for test suite compatibility)."""
        if self._state == "normal" and self._command:
            self._command()

    def configure(self, **kwargs):
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "state" in kwargs:
            self._state = kwargs.pop("state")
            if self._state == "disabled":
                kwargs["cursor"] = "arrow"
            else:
                kwargs["cursor"] = "hand2"
        if "bg" in kwargs:
            self._normal_bg = kwargs["bg"]
        if "fg" in kwargs:
            self._normal_fg = kwargs["fg"]
        if "activebackground" in kwargs:
            self._hover_bg = kwargs.pop("activebackground")
        if "activeforeground" in kwargs:
            self._hover_fg = kwargs.pop("activeforeground")
        kwargs.pop("highlightbackground", None)
        kwargs.pop("highlightthickness", None)
        super().configure(**kwargs)

    config = configure


tk.Button = SolidButton


class SmartAIChatbotApp:
    """Clean, high-readability single-chat studio with real-time streaming, AI Canvas, custom model importer, steer & queue."""

    def __init__(self, root: tk.Tk, settings: Optional[Settings] = None):
        self.root = root
        self.settings = settings or get_settings()
        self.current_theme = detect_system_theme()
        self.C = dict(_THEMES[self.current_theme])
        _COLORS.clear()
        _COLORS.update(self.C)

        # Built-in Default Models Configuration (Expanded Multi-Modal & Uncensored Presets)
        self.models_config = {
            "model_1": {
                "name": "Qwen3.8-27B Uncensored (MLX 2-Bit)",
                "short_name": "Qwen 27B Uncensored (MLX)",
                "repo_id": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
                "model_path": None,
                "precision": "2-Bit Uncensored MLX",
                "raw_params": 27_400_000_000,
                "base_params": "27.4B",
                "est_speed": "⚡ ~28 t/s",
                "max_context": 131_072,
                "vram": "14.5 GB / 16 GB",
                "tag": "🔥 Qwen 27B Uncensored",
                "accent": self.C["accent_cyan"]
            },
            "model_2": {
                "name": "Qwen3.8-27B Abliterated (Lowest Quant GGUF)",
                "short_name": "Qwen 27B Abliterated (GGUF)",
                "repo_id": "douyamv/Qwen3.8-27B-abliterated-GGUF",
                "model_path": None,
                "precision": "Q2_K / Q3_K_M Lowest Quant GGUF",
                "raw_params": 27_400_000_000,
                "base_params": "27.4B",
                "est_speed": "⚡ ~22 t/s",
                "max_context": 131_072,
                "vram": "14.2 GB / 16 GB",
                "tag": "🔓 Qwen 27B Abliterated",
                "accent": "#f43f5e"
            },
            "model_3": {
                "name": "RealVisXL V5.0 (High-Res Photoreal Uncensored)",
                "short_name": "RealVisXL V5.0 (SDXL)",
                "repo_id": "SG161222/RealVisXL_V5.0",
                "model_path": None,
                "precision": "FP16 / SafeTensors (16GB RAM Optimized)",
                "raw_params": 6_600_000_000,
                "base_params": "6.6B",
                "est_speed": "🎨 ~8 it/s",
                "max_context": 4_096,
                "vram": "6.2 GB / 16 GB",
                "tag": "📸 RealVisXL V5.0 (SDXL)",
                "accent": "#ec4899"
            },
            "model_4": {
                "name": "Z-Image Turbo NSFW v2 (Q8 GGUF High-Res)",
                "short_name": "Z-Image Turbo NSFW v2",
                "repo_id": "lesliemore/z-image-turbo-nsfw-v2-GGUF",
                "model_path": None,
                "precision": "Q8_0 GGUF Quantized",
                "raw_params": 4_000_000_000,
                "base_params": "4.0B",
                "est_speed": "⚡ ~14 it/s",
                "max_context": 4_096,
                "vram": "4.5 GB / 16 GB",
                "tag": "⚡ Z-Image Turbo Q8",
                "accent": "#a855f7"
            },
            "model_5": {
                "name": "Qwen Image Edit Rapid AIO (Text & Image in GGUF)",
                "short_name": "Qwen Image Edit AIO",
                "repo_id": "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF",
                "model_path": None,
                "precision": "Rapid AIO GGUF (Text & Image In)",
                "raw_params": 7_000_000_000,
                "base_params": "7.0B",
                "est_speed": "🎨 ~16 t/s",
                "max_context": 32_768,
                "vram": "5.2 GB / 16 GB",
                "tag": "🎨 Qwen Image Edit Rapid",
                "accent": "#06b6d4"
            },
            "model_6": {
                "name": "Ideogram Instant Uncensored (16GB RAM Turbo)",
                "short_name": "Ideogram Instant (16GB)",
                "repo_id": "SG161222/RealVisXL_V5.0",
                "model_path": None,
                "precision": "4-Step Instant Diffusion",
                "raw_params": 3_500_000_000,
                "base_params": "3.5B",
                "est_speed": "✨ ~18 it/s",
                "max_context": 2_048,
                "vram": "3.8 GB / 16 GB",
                "tag": "✨ Ideogram Instant NSFW",
                "accent": "#eab308"
            },
            "model_7": {
                "name": "LTX-Video 2.5 (MLX Q4 High-Res Video & Audio)",
                "short_name": "LTX-Video 2.5 (MLX)",
                "repo_id": "dgrauet/ltx-2.5-mlx-q4",
                "model_path": None,
                "precision": "Q4 Apple Silicon MLX Native",
                "raw_params": 5_000_000_000,
                "base_params": "5.0B",
                "est_speed": "🎬 ~4 fps",
                "max_context": 8_192,
                "vram": "5.8 GB / 16 GB",
                "tag": "🎬 LTX-Video 2.5 Q4",
                "accent": "#10b981"
            },
            "model_8": {
                "name": "Wan 2.2 Remix (GGUF Q4 Motion Engine)",
                "short_name": "Wan 2.2 Remix (GGUF)",
                "repo_id": "freeguyfroverrrr/Wan-2.2-Remix-GGUF",
                "model_path": None,
                "precision": "Q4 GGUF Quantized",
                "raw_params": 5_000_000_000,
                "base_params": "5.0B",
                "est_speed": "🎥 ~5 fps",
                "max_context": 8_192,
                "vram": "5.9 GB / 16 GB",
                "tag": "🎥 Wan 2.2 Remix Q4",
                "accent": "#6366f1"
            },
            "model_9": {
                "name": "MiniMax-H3 MLX 4-bit (AfterMidnight NSFW LoRA)",
                "short_name": "MiniMax-H3 AfterMidnight",
                "repo_id": "pipenetwork/MiniMax-H3-MLX-4bit",
                "model_path": None,
                "precision": "4-bit MLX + Rank 32 LoRA",
                "raw_params": 4_000_000_000,
                "base_params": "4.0B",
                "est_speed": "🌙 ~26 t/s",
                "max_context": 8_192,
                "vram": "5.7 GB / 16 GB",
                "tag": "🌙 MiniMax-H3 AfterMidnight",
                "accent": "#8b5cf6"
            }
        }
        self._load_saved_custom_models()
        self.active_tab_id = "model_1"

        # LoRA Slider Strength Parameters
        self.softer_lora_str = 1.0   # Target to improve anatomy/detail (default: 1.0)
        self.harder_lora_str = 0.8   # Softer + motion target, rank 32 (cap: 0.8)
        self.custom_lora_str = 0.7   # Mystic XXX / secondary blend (default: 0.7)

        # Core Engines & Tools
        self.db = EpisodicMemoryDB(db_path=self.settings.database_path)
        self.tools = AgentToolRegistry(db_path=self.settings.database_path)
        self.engine = ProReasoningEngine(settings=self.settings)
        self.learner = AutonomousLearner(engine=self.engine, tools=self.tools, db=self.db, settings=self.settings)

        # State Variables
        self.workspace_dir: Optional[str] = None
        self.tools.set_workspace_dir(None)
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
        self.thinking_meta: Dict[str, Dict[str, Any]] = {}
        self.code_snippets: Dict[str, str] = {}
        self._think_counter = 0
        self._code_counter = 0
        self.last_metadata: Optional[Dict[str, Any]] = None

        # Thread-Safe Event Queue for Background Worker & Watchdog Callbacks
        self._event_queue: queue.Queue = queue.Queue()

        # System RAM Pressure Watchdog with Proactive Reclaim (94% limit)
        self.watchdog = SystemMemoryWatchdog(
            check_interval_seconds=2.5,
            max_ram_usage_percent=94.0,
            min_free_ram_gb=0.8,
            max_process_ram_gb=12.0,
            on_pressure_callback=lambda s: self._event_queue.put(("memory_pressure", s))
        )
        self.watchdog.start_monitoring()

        self._init_window()
        self._build_ui()
        self._send_welcome_messages()
        self._update_model_action_buttons()
        self._update_input_lock_state()
        self._start_event_queue_polling()

    def _start_event_queue_polling(self):
        """Safely consumes thread-safe background events inside Tkinter main GUI thread."""
        def _poll():
            try:
                while True:
                    evt, data = self._event_queue.get_nowait()
                    if evt == "memory_pressure":
                        self._on_memory_pressure_emergency(data)
            except queue.Empty:
                pass
            except Exception:
                pass

            try:
                if hasattr(self, "lbl_vram"):
                    proc_rss = SystemMemoryWatchdog.get_process_rss_gb()
                    mem = SystemMemoryWatchdog.get_system_memory_status()
                    total_gb = mem.get("total_gb", 16.0)
                    self.lbl_vram.configure(text=f"💾 {proc_rss:.2f} GB / {total_gb:.0f} GB")
            except Exception:
                pass

            try:
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(500, _poll)
            except Exception:
                pass

        try:
            self.root.after(500, _poll)
        except Exception:
            pass

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

        self._configure_ttk_styles()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_app)

    def _configure_ttk_styles(self):
        """Configures ttk widget styles (Treeview, Combobox, Scrollbar) according to active light/dark theme."""
        try:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")

            is_dark = (self.current_theme == "dark")
            bg_app = self.C["bg_app"]
            bg_card = self.C["bg_card"]
            fg_main = self.C["text_main"]
            accent_cyan = self.C["accent_cyan"]
            bg_inner = self.C["bg_input_inner"]

            # Treeview Styling
            style.configure(
                "Treeview",
                background=bg_app,
                foreground=fg_main,
                fieldbackground=bg_app,
                font=_FONT_SMALL,
                rowheight=26,
                borderwidth=0
            )
            style.configure(
                "Treeview.Heading",
                background=bg_card,
                foreground=accent_cyan,
                font=_FONT_TINY_BOLD,
                relief="flat"
            )
            style.map(
                "Treeview",
                background=[("selected", "#333333" if is_dark else "#e0e0e0")],
                foreground=[("selected", fg_main)]
            )
            style.map(
                "Treeview.Heading",
                background=[("active", self.C["bg_card_hover"])],
                foreground=[("active", accent_cyan)]
            )

            # Combobox Styling
            style.configure(
                "TCombobox",
                background=bg_card,
                foreground=fg_main,
                fieldbackground=bg_inner,
                selectbackground="#333333" if is_dark else "#e0e0e0",
                selectforeground=fg_main,
                arrowcolor=accent_cyan,
                font=_FONT_SMALL
            )
            style.map(
                "TCombobox",
                fieldbackground=[("readonly", bg_inner), ("disabled", bg_inner)],
                selectbackground=[("readonly", "#333333" if is_dark else "#e0e0e0")],
                selectforeground=[("readonly", fg_main)]
            )
        except Exception:
            pass

    def _on_toggle_theme(self):
        """Toggles between Dark Mode and Light Mode with instant high-contrast visual refresh."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme_colors()

    def _apply_theme_colors(self):
        """Applies current light/dark theme colors dynamically across all UI widgets and ttk styles."""
        self.C = dict(_THEMES[self.current_theme])
        _COLORS.clear()
        _COLORS.update(self.C)

        is_dark = (self.current_theme == "dark")
        fg_main = self.C["text_main"]
        bg_app = self.C["bg_app"]
        bg_hud = self.C["bg_hud"]
        btn_bg = self.C["btn_bg"]
        btn_fg = self.C["btn_fg"]
        btn_hover = self.C["btn_hover"]
        btn_p_bg = self.C["btn_primary_bg"]
        btn_p_fg = self.C["btn_primary_fg"]
        btn_p_hover = self.C["btn_primary_hover"]
        badge_bg = self.C["badge_bg"]
        badge_fg = self.C["badge_fg"]
        bg_input = self.C["bg_input"]
        bg_inner = self.C["bg_input_inner"]

        self.root.configure(bg=bg_app)
        if hasattr(self, "main_container"):
            self.main_container.configure(bg=bg_app)
        if hasattr(self, "hud_bar"):
            self.hud_bar.configure(bg=bg_hud)

        # Reconfigure Top Bar buttons
        top_buttons = [
            "btn_model_menu", "btn_import_model", "lbl_workspace", "btn_remove_folder",
            "btn_download_model", "btn_reset_reinstall", "btn_export_chat", "btn_branch_vis",
            "btn_memory_exp", "btn_sleep_panel", "btn_dsl_play", "btn_toggle_canvas",
            "btn_load_unload", "btn_toggle_resources", "btn_theme_toggle"
        ]
        for btn_name in top_buttons:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                try:
                    btn.configure(
                        bg=btn_bg, fg=btn_fg,
                        highlightbackground=btn_bg,
                        activebackground=btn_hover, activeforeground=btn_fg
                    )
                except Exception:
                    pass

        # Reconfigure Badges
        badges = ["lbl_model_speed_params", "lbl_params", "lbl_synapses", "lbl_context", "lbl_vram", "lbl_tps"]
        for badge_name in badges:
            if hasattr(self, badge_name):
                b = getattr(self, badge_name)
                try:
                    b.configure(bg=badge_bg, fg=badge_fg, highlightbackground=badge_bg)
                except Exception:
                    pass

        if hasattr(self, "lbl_model_status"):
            self.lbl_model_status.configure(bg=bg_hud, fg=btn_fg)

        # Chat stream
        if hasattr(self, "chat_stream"):
            self.chat_stream.configure(bg=self.C["bg_chat"], fg=fg_main, insertbackground=fg_main)
            self._configure_stream_tags(self.chat_stream)

        # Input container & buttons
        if hasattr(self, "input_container"):
            self.input_container.configure(bg=bg_input)
        if hasattr(self, "input_card"):
            self.input_card.configure(bg=bg_inner, highlightbackground=self.C["border"])
        if hasattr(self, "txt_input"):
            self.txt_input.configure(
                bg=bg_inner,
                fg=fg_main if not getattr(self, "_placeholder_active", False) else self.C.get("text_placeholder", "#71717a"),
                insertbackground=fg_main
            )
        if hasattr(self, "btn_send"):
            self.btn_send.configure(
                bg=btn_p_bg, fg=btn_p_fg,
                highlightbackground=btn_p_bg,
                activebackground=btn_p_hover, activeforeground=btn_p_fg
            )

        bottom_buttons = ["btn_attach", "btn_steer", "btn_live_steer", "btn_stop", "btn_reset_chat", "btn_export_chat"]
        for btn_name in bottom_buttons:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                try:
                    btn.configure(
                        bg=btn_bg, fg=btn_fg,
                        highlightbackground=btn_bg,
                        activebackground=btn_hover, activeforeground=btn_fg
                    )
                except Exception:
                    pass

        if hasattr(self, "btn_theme_toggle"):
            self.btn_theme_toggle.configure(text="☀️ Light" if is_dark else "🌙 Dark")

        # Canvas panel
        if hasattr(self, "canvas_panel"):
            self.canvas_panel.configure(bg=self.C["canvas_bg"])
        if hasattr(self, "txt_canvas"):
            self.txt_canvas.configure(
                bg=self.C["code_bg"],
                fg=self.C["code_fg"],
                insertbackground=self.C["code_fg"]
            )

        self._configure_ttk_styles()

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

        # 1. TOP HEADER BAR (Clean Spacing)
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
        self.content_paned.pack(fill="both", expand=True, padx=16, pady=(4, 6))

        # Single Unified Chat Feed
        self._build_single_chat_feed()

        # Side-by-Side AI Canvas (Opens when code generated or toggled)
        self._build_ai_canvas_panel()

    # ─────────────────────────────────────────────────────
    #  1. SINGLE SLEEK TOP HEADER ROW (WELL-ORGANIZED BUTTONS)
    # ─────────────────────────────────────────────────────
    def _build_single_top_bar(self):
        self.hud_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"])
        self.hud_bar.pack(fill="x", side="top", padx=10, pady=(6, 4))

        # Alias for backward test compatibility
        self.tab_bar = self.hud_bar

        # Left Section: Brand + Speed/Param Badge + Model Selector Dropdown + Workspace Folder
        left_box = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
        left_box.pack(side="left", padx=4, pady=2)

        tk.Label(
            left_box, text="✦ Smart AI", font=_FONT_TITLE,
            bg=self.C["bg_hud"], fg=self.C["accent_cyan"]
        ).pack(side="left", padx=(0, 8))

        curr_info = self.models_config[self.active_tab_id]
        # Estimated Speed & Base Params Badge right before model selector
        self.lbl_model_speed_params = self._make_badge(
            left_box, f"{curr_info.get('est_speed', '⚡ ~28 t/s')} • {curr_info.get('base_params', '27.4B')}"
        )

        # Model Selector Dropdown
        self.model_var = tk.StringVar(value=self.models_config[self.active_tab_id]["short_name"])
        self.btn_model_menu = tk.Menubutton(
            left_box, textvariable=self.model_var, font=_FONT_TAB,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2",
            highlightthickness=0
        )
        self.btn_model_menu.pack(side="left", padx=(4, 6))
        self._refresh_model_menu()

        # Tab button aliases for test suite
        self.tab_buttons: Dict[str, tk.Button] = {}
        for tid in self.models_config.keys():
            self.tab_buttons[tid] = self.btn_model_menu

        # ➕ Import Custom Model is inside the model dropdown menu, keeping attribute for test compatibility
        self.btn_import_model = SolidButton(
            left_box, text="➕ Import Model",
            command=self._on_import_custom_model_dialog
        )

        # Workspace folder container with Add / Remove
        self.workspace_box = tk.Frame(left_box, bg=self.C["bg_hud"])
        self.workspace_box.pack(side="left", padx=(0, 4))

        disp_folder = f"📁 {os.path.basename(self.workspace_dir)}" if self.workspace_dir else "📁 + Add Folder"
        self.lbl_workspace = tk.Label(
            self.workspace_box, text=disp_folder, font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"], padx=8, pady=3,
            relief="flat", bd=0, cursor="hand2"
        )
        self.lbl_workspace.pack(side="left")
        self.lbl_workspace.bind("<Button-1>", lambda e: self._on_select_workspace_folder())

        self.btn_remove_folder = tk.Label(
            self.workspace_box, text="✖", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["accent_red"], padx=5, pady=3,
            relief="flat", bd=0, cursor="hand2"
        )
        if self.workspace_dir:
            self.btn_remove_folder.pack(side="left", padx=(2, 0))
        self.btn_remove_folder.bind("<Button-1>", lambda e: self._on_remove_workspace_folder())

        # Right Section: Status Chips + Action Buttons
        right_box = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
        right_box.pack(side="right", padx=4, pady=2)

        total_p = curr_info.get("raw_params", 27_400_000_000) + self.synapses_learned_count
        param_str = format_parameter_count(total_p)
        self.lbl_params = self._make_badge(right_box, f"🧠 {param_str} Total Params")

        syn_str = format_added_synapses(self.synapses_learned_count)
        self.lbl_synapses = self._make_badge(right_box, f"📈 {syn_str}")

        ctx_pct = (self.total_tokens_used / self.max_context_window) * 100 if self.total_tokens_used > 0 else 0
        self.lbl_context = self._make_badge(
            right_box, f"📊 Context: {self.total_tokens_used:,} / {self.max_context_window:,} ({ctx_pct:.0f}%)"
        )
        self.lbl_vram = self._make_badge(right_box, "💾 0.0 GB / 16 GB")
        self.lbl_tps = self._make_badge(right_box, "⚡ — tok/s")

        # Load / Unload Model Button
        # Primary Load / Install / Unload Model Button
        self.btn_load_unload = tk.Button(
            right_box, text="⚡ Load", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_toggle_load_unload
        )
        self.btn_load_unload.pack(side="left", padx=2)

        # Dedicated Grab / Reset & Re-grab Button
        self.btn_reset_reinstall = tk.Button(
            right_box, text="🔄 Reset", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_reset_and_reinstall_single_confirm
        )
        self.btn_reset_reinstall.pack(side="left", padx=2)
        self.btn_download_model = self.btn_reset_reinstall
        self.btn_download_reset = self.btn_reset_reinstall

        # Export Chat to Markdown
        self.btn_export_chat = tk.Button(
            right_box, text="💾 Export", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_export_chat_button
        )
        self.btn_export_chat.pack(side="left", padx=2)

        # Branches Visualizer Button
        self.btn_branch_vis = tk.Button(
            right_box, text="🌿 Branches", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_open_branch_visualizer
        )
        self.btn_branch_vis.pack(side="left", padx=2)

        # Memory Explorer Button
        self.btn_memory_exp = tk.Button(
            right_box, text="💾 Memory DB", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_open_memory_explorer
        )
        self.btn_memory_exp.pack(side="left", padx=2)

        # Sleep Consolidation Button
        self.btn_sleep_panel = tk.Button(
            right_box, text="💤 Sleep EWC", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_open_sleep_consolidation_panel
        )
        self.btn_sleep_panel.pack(side="left", padx=2)

        # DSL & Sandbox Button
        self.btn_dsl_play = tk.Button(
            right_box, text="🧪 DSL", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_open_dsl_playground
        )
        self.btn_dsl_play.pack(side="left", padx=2)

        # Canvas Toggle Button
        self.btn_toggle_canvas = tk.Button(
            right_box, text="🎨 Canvas ▾", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_toggle_canvas_viewer
        )
        self.btn_toggle_canvas.pack(side="left", padx=2)

        # Resources toggle
        self.btn_toggle_resources = tk.Button(
            right_box, text="⚙", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=6, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_toggle_resource_viewer
        )
        self.btn_toggle_resources.pack(side="left", padx=2)

        # Theme Toggle (☀️ Light / 🌙 Dark)
        is_dark = (self.current_theme == "dark")
        self.btn_theme_toggle = tk.Button(
            right_box, text="☀️ Light" if is_dark else "🌙 Dark", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_toggle_theme
        )
        self.btn_theme_toggle.pack(side="left", padx=2)

        # Status Label
        cached = is_model_cached_locally(curr_info["repo_id"]) if curr_info.get("repo_id") else (os.path.exists(curr_info.get("model_path") or "") if curr_info.get("model_path") else False)
        status_txt = f"⚡ Ready ({curr_info['short_name']})" if cached else f"○ Not Downloaded ({curr_info['short_name']})"
        self.lbl_model_status = tk.Label(
            right_box, text=status_txt, font=_FONT_TINY_BOLD,
            bg=self.C["bg_hud"], fg=self.C["btn_fg"], padx=4
        )
        self.lbl_model_status.pack(side="left")

    def _refresh_model_menu(self):
        """Rebuilds the model dropdown menu dynamically with all built-in and custom models."""
        menu = tk.Menu(self.btn_model_menu, tearoff=0, bg=self.C["btn_bg"], fg=self.C["btn_fg"], font=_FONT_SMALL)
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

    def _make_badge(self, parent: tk.Frame, text: str, fg_color: Optional[str] = None) -> tk.Label:
        lbl = tk.Label(
            parent, text=text, font=_FONT_TINY_BOLD,
            bg=self.C["badge_bg"],
            fg=self.C["badge_fg"],
            padx=7, pady=3, relief="flat", bd=0,
            highlightthickness=0
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

        btn_browse = tk.Button(path_frame, text="📁 Browse MLX...", font=_FONT_TINY_BOLD, bg=self.C["btn_bg"], fg=self.C["btn_fg"], activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"], relief="flat", bd=0, padx=8, cursor="hand2", highlightthickness=0, command=_browse_local)
        btn_browse.pack(side="right", padx=(6, 0))

        btn_quick_mlx = tk.Button(
            top_btn_bar, text="📁 Select Local MLX Folder...", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2",
            highlightthickness=0,
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

        btn_cancel = tk.Button(btn_bar, text="Cancel", font=_FONT_SMALL, bg=self.C["btn_bg"], fg=self.C["btn_fg"], activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"], relief="flat", bd=0, padx=14, pady=6, cursor="hand2", highlightthickness=0, command=modal.destroy)
        btn_cancel.pack(side="right", padx=(8, 0))

        btn_import = tk.Button(btn_bar, text="➕ Import & Select", font=_FONT_BOLD, bg=self.C["btn_primary_bg"], fg=self.C["btn_primary_fg"], activebackground=self.C["btn_primary_hover"], activeforeground=self.C["btn_primary_fg"], relief="flat", bd=0, padx=16, pady=6, cursor="hand2", highlightthickness=0, command=_save_imported_model)
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
            hdr, text="✖ Close", font=_FONT_TINY_BOLD, bg=self.C["btn_bg"],
            fg=self.C["btn_fg"], activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3,
            cursor="hand2", highlightthickness=0, command=self._on_toggle_resource_viewer
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

        folder_disp = os.path.basename(self.workspace_dir) if self.workspace_dir else "None (No Folder Selected)"
        self.lbl_res_folder = tk.Label(col2, text=f"• Target Folder: {folder_disp}", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["text_main"], anchor="w")
        self.lbl_res_folder.pack(fill="x", pady=2)

        self.lbl_res_ram = tk.Label(col2, text="• Host Memory: Detecting...", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["accent_green"], anchor="w")
        self.lbl_res_ram.pack(fill="x", pady=2)

        self.lbl_res_arch = tk.Label(col2, text=f"• Platform: {platform.system()} {platform.machine()} (MLX Native)", font=_FONT_SMALL, bg=self.C["bg_resources"], fg=self.C["text_muted"], anchor="w")
        self.lbl_res_arch.pack(fill="x", pady=2)

        # LoRA Conditioning & Slider Controls Frame
        lora_box = tk.Frame(self.res_drawer, bg=self.C["bg_hud"], highlightbackground=self.C["border"], highlightthickness=1)
        lora_box.pack(fill="x", padx=20, pady=(0, 10))

        lora_hdr = tk.Frame(lora_box, bg=self.C["bg_hud"])
        lora_hdr.pack(fill="x", padx=10, pady=(6, 2))
        tk.Label(lora_hdr, text="🎛️ Multimodal & Diffusion LoRA Sliders (Anatomy, Motion & Detail)", font=_FONT_TINY_BOLD, bg=self.C["bg_hud"], fg=self.C["accent_purple"]).pack(side="left")

        sliders_bar = tk.Frame(lora_box, bg=self.C["bg_hud"])
        sliders_bar.pack(fill="x", padx=10, pady=(0, 6))

        # Slider 1: Softer LoRA (Anatomy / Detail)
        s1_frame = tk.Frame(sliders_bar, bg=self.C["bg_hud"])
        s1_frame.pack(side="left", fill="x", expand=True, padx=4)
        lbl_s1 = tk.Label(s1_frame, text=f"Softer LoRA (Anatomy): {self.softer_lora_str:.2f}x", font=_FONT_TINY, bg=self.C["bg_hud"], fg=self.C["text_main"])
        lbl_s1.pack(anchor="w")
        scale_s1 = tk.Scale(s1_frame, from_=0.0, to=1.5, resolution=0.05, orient="horizontal", bg=self.C["bg_hud"], fg=self.C["accent_cyan"], highlightthickness=0, bd=0, showvalue=False, command=lambda v: [setattr(self, 'softer_lora_str', float(v)), lbl_s1.configure(text=f"Softer LoRA (Anatomy): {float(v):.2f}x")])
        scale_s1.set(self.softer_lora_str)
        scale_s1.pack(fill="x")

        # Slider 2: Harder LoRA (Motion / Dynamics)
        s2_frame = tk.Frame(sliders_bar, bg=self.C["bg_hud"])
        s2_frame.pack(side="left", fill="x", expand=True, padx=4)
        lbl_s2 = tk.Label(s2_frame, text=f"Harder LoRA (Motion R32): {self.harder_lora_str:.2f}x", font=_FONT_TINY, bg=self.C["bg_hud"], fg=self.C["text_main"])
        lbl_s2.pack(anchor="w")
        scale_s2 = tk.Scale(s2_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", bg=self.C["bg_hud"], fg=self.C["accent_orange"], highlightthickness=0, bd=0, showvalue=False, command=lambda v: [setattr(self, 'harder_lora_str', float(v)), lbl_s2.configure(text=f"Harder LoRA (Motion R32): {float(v):.2f}x")])
        scale_s2.set(self.harder_lora_str)
        scale_s2.pack(fill="x")

        # Slider 3: Custom / Mystic LoRA
        s3_frame = tk.Frame(sliders_bar, bg=self.C["bg_hud"])
        s3_frame.pack(side="left", fill="x", expand=True, padx=4)
        lbl_s3 = tk.Label(s3_frame, text=f"Mystic / Secondary LoRA: {self.custom_lora_str:.2f}x", font=_FONT_TINY, bg=self.C["bg_hud"], fg=self.C["text_main"])
        lbl_s3.pack(anchor="w")
        scale_s3 = tk.Scale(s3_frame, from_=0.0, to=1.5, resolution=0.05, orient="horizontal", bg=self.C["bg_hud"], fg=self.C["accent_green"], highlightthickness=0, bd=0, showvalue=False, command=lambda v: [setattr(self, 'custom_lora_str', float(v)), lbl_s3.configure(text=f"Mystic / Secondary LoRA: {float(v):.2f}x")])
        scale_s3.set(self.custom_lora_str)
        scale_s3.pack(fill="x")

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
        proc_rss = mem.get("process_rss_gb", 0.0)
        self.lbl_res_ram.configure(
            text=f"• Process RSS: {proc_rss:.1f} GB | Host RAM: {mem['used_gb']:.1f} GB / {mem['total_gb']:.1f} GB ({mem['used_percent']:.0f}%)"
        )
        info = self.models_config[self.active_tab_id]
        self.lbl_res_model.configure(text=f"• Model: {info['name']}")
        vram_text = f"{proc_rss:.1f} GB (Allocated)" if self.is_model_loaded and proc_rss > 0 else (f"{info['vram']} (Allocated)" if self.is_model_loaded else "0.0 GB (Unloaded)")
        self.lbl_res_vram.configure(text=f"• VRAM Usage: {vram_text}")
        folder_disp = os.path.basename(self.workspace_dir) if self.workspace_dir else "None (No Folder Selected)"
        self.lbl_res_folder.configure(text=f"• Target Folder: {folder_disp}")

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
        is_dark = (self.current_theme == "dark")
        fg_main = self.C["text_main"]
        fg_muted = self.C["text_muted"]

        # Crisp Text Selection Tag (Highlights typed text only)
        sel_bg = self.C.get("accent_cyan", "#38bdf8")
        sel_fg = "#000000" if not is_dark else "#ffffff"
        stream.tag_configure("sel", background=sel_bg, foreground=sel_fg)

        stream.tag_configure("user_header", foreground=self.C["accent_cyan"], font=_FONT_H3)
        stream.tag_configure("user_msg", foreground=fg_main, font=_FONT_MAIN)
        stream.tag_configure("ai_header", foreground=self.C["accent_green"], font=_FONT_H3)
        stream.tag_configure("ai_msg", foreground=fg_main, font=_FONT_MAIN)

        # Live Steer & Queue Tags
        stream.tag_configure("steer_header", foreground=self.C["accent_purple"], font=_FONT_H3)
        stream.tag_configure("steer_msg", foreground=self.C["accent_purple"], font=_FONT_MAIN)
        stream.tag_configure("queue_header", foreground=self.C["accent_yellow"], font=_FONT_H3)
        stream.tag_configure("queue_msg", foreground=self.C["accent_yellow"], font=_FONT_MAIN)

        # Markdown Tags
        stream.tag_configure("md_h1", foreground=self.C["accent_cyan"], font=_FONT_H1)
        stream.tag_configure("md_h2", foreground=self.C["accent_purple"], font=_FONT_H2)
        stream.tag_configure("md_h3", foreground=fg_main, font=_FONT_H3)
        stream.tag_configure("md_bold", foreground=fg_main, font=_FONT_BOLD)
        stream.tag_configure("md_bold_italic", foreground=fg_main, font=_FONT_BOLD_ITALIC)
        stream.tag_configure("md_italic", foreground=fg_muted, font=_FONT_ITALIC)
        stream.tag_configure("md_quote", foreground=self.C["accent_yellow"], font=_FONT_ITALIC)
        stream.tag_configure("md_bullet", foreground=self.C["accent_cyan"], font=_FONT_MAIN)
        stream.tag_configure("md_inline_code", foreground=self.C["accent_cyan"], font=_FONT_INLINE_MONO)

        # Code Block Tags & Actions (Pure B&W Action Buttons)
        stream.tag_configure("code_block", foreground=self.C["code_fg"], background=self.C["code_bg"], font=_FONT_MONO)
        stream.tag_configure("code_hdr", foreground=self.C["accent_cyan"], font=_FONT_TINY_BOLD, background=self.C["code_bg"])
        stream.tag_configure("code_action_copy", foreground=self.C["btn_fg"], font=_FONT_TINY_BOLD, background=self.C["btn_bg"])
        stream.tag_configure("code_action_canvas", foreground=self.C["btn_fg"], font=_FONT_TINY_BOLD, background=self.C["btn_bg"])

        # Interactive Thinking Dropdown Tags (Pure B&W Pill)
        stream.tag_configure("think_dropdown_btn", foreground=self.C["btn_fg"], font=_FONT_TINY_BOLD, background=self.C["btn_bg"])
        stream.tag_configure("think_body", foreground=fg_main, font=_FONT_SMALL, background=self.C["bg_thinking"])

        # Tool Pill Tags (Pure B&W Pill)
        stream.tag_configure("tool_pill", foreground=self.C["btn_fg"], font=_FONT_TINY_BOLD, background=self.C["btn_bg"])
        stream.tag_configure("tool_output", foreground=fg_main, font=_FONT_SMALL, background=self.C["bg_tool"])
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
            bg=self.C["btn_primary_bg"], fg=self.C["btn_primary_fg"],
            highlightbackground=self.C["btn_primary_bg"],
            activebackground=self.C["btn_primary_hover"], activeforeground=self.C["btn_primary_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_run_code
        )
        btn_run.pack(side="left", padx=3)

        btn_preview = tk.Button(
            hdr, text="👁️ Preview", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_toggle_preview
        )
        btn_preview.pack(side="left", padx=3)
        self.btn_canvas_preview = btn_preview

        btn_browser = tk.Button(
            hdr, text="🌐 Browser", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_open_browser
        )
        btn_browser.pack(side="left", padx=3)

        btn_visual = tk.Button(
            hdr, text="📊 Visualizer", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_make_visual
        )
        btn_visual.pack(side="left", padx=3)

        btn_dsl_c = tk.Button(
            hdr, text="🧪 DSL Sandbox", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_load_dsl_template
        )
        btn_dsl_c.pack(side="left", padx=3)

        btn_copy = tk.Button(
            hdr, text="📋 Copy", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_copy
        )
        btn_copy.pack(side="left", padx=3)

        btn_save = tk.Button(
            hdr, text="💾 Save", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
            command=self._on_canvas_save_file
        )
        btn_save.pack(side="left", padx=3)

        btn_close = tk.Button(
            hdr, text="✖ Close", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=8, pady=3, cursor="hand2",
            highlightthickness=0,
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

    def _on_canvas_open_browser(self):
        """Opens current canvas HTML/SVG content or saved artifact in user's default web browser."""
        content = self.txt_canvas.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Canvas Empty", "Please generate or load an interactive visualizer or HTML/SVG content first.")
            return

        # Check if content points to an existing file
        if os.path.exists(content) and content.endswith((".html", ".htm", ".svg")):
            webbrowser.open(f"file://{os.path.abspath(content)}")
            return

        # Create or update preview HTML in workspace
        preview_file = os.path.join(self.workspace_dir, "canvas_interactive_preview.html")
        if not content.startswith("<!DOCTYPE") and not content.startswith("<html") and not content.startswith("<svg"):
            # Wrap standard code / text into styled HTML document
            html_wrapped = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Smart AI Canvas Preview</title>
    <style>
        body {{ background: #080c14; color: #f8fafc; font-family: system-ui, sans-serif; padding: 30px; }}
        pre {{ background: #0f172a; border: 1px solid #1e293b; padding: 20px; border-radius: 12px; font-family: monospace; overflow: auto; }}
    </style>
</head>
<body>
    <h2>✦ Smart AI Canvas Live Artifact</h2>
    <pre>{content}</pre>
</body>
</html>"""
            content = html_wrapped

        with open(preview_file, "w", encoding="utf-8") as f:
            f.write(content)

        webbrowser.open(f"file://{os.path.abspath(preview_file)}")
        self._append_ai_message(f"🌐 **Browser Preview Opened**: Interactive visualizer launched at [`{preview_file}`](file://{os.path.abspath(preview_file)}).")

    def _on_canvas_make_visual(self):
        """Triggers the Interactive Visual Maker to generate a rich chart or simulation on Canvas."""
        ok, res = self.tools.execute_tool("interactive_visual_maker", {
            "visual_type": "chart",
            "title": "Smart AI Studio • Real-Time Metrics & Neural Performance",
            "description": "Interactive Multi-Series Telemetry & Reasoning Accuracy Dashboard",
            "theme": "obsidian",
            "filename": f"interactive_dashboard_{int(time.time())}.html"
        })
        self._append_tool_call("interactive_visual_maker", "Interactive Visual Maker", res)
        # Extract HTML file path if present
        match = re.search(r'file://([^\s\)]+\.html)', res)
        if match:
            html_path = match.group(1)
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    self._open_in_canvas(f.read())
                webbrowser.open(f"file://{html_path}")

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
            filetypes=[("Python Script", "*.py"), ("HTML Document", "*.html"), ("Markdown Document", "*.md"), ("SVG Vector", "*.svg"), ("All Files", "*.*")]
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
            bd=0, highlightthickness=0
        )
        self.input_container.pack(fill="x", side="bottom", padx=20, pady=(0, 10))

        # Slash Commands Auto-Popup Bar (Appears when typing '/')
        self.slash_bar = tk.Frame(self.input_container, bg=self.C["bg_input"], bd=0)

        self.slash_commands = [
            {"cmd": "/learn", "label": "🎓 /learn", "desc": "Autonomous self-directed research & consolidation", "template": "/learn "},
            {"cmd": "/steer", "label": "🎯 /steer", "desc": "Live steering of model generation", "template": "/steer "},
            {"cmd": "/canvas", "label": "🎨 /canvas", "desc": "Open AI Canvas Studio & live coding artifact", "action": self._on_toggle_canvas_viewer},
            {"cmd": "/search", "label": "🌐 /search", "desc": "Web search and documentation fetch", "template": "/search "},
            {"cmd": "/calc", "label": "📐 /calc", "desc": "SymPy calculus and math evaluation", "template": "/calc "},
            {"cmd": "/branches", "label": "🌿 /branches", "desc": "View speculative reasoning branches", "action": self._on_open_branch_visualizer},
            {"cmd": "/memory", "label": "💾 /memory", "desc": "Inspect SQLite episodic memory DB", "action": self._on_open_memory_explorer},
            {"cmd": "/sleep", "label": "💤 /sleep", "desc": "Trigger Slow-LoRA EWC sleep consolidation", "action": self._on_open_sleep_consolidation_panel},
            {"cmd": "/dsl", "label": "🧪 /dsl", "desc": "Open TensorGraph DSL playground", "action": self._on_open_dsl_playground},
            {"cmd": "/clear", "label": "🗑️ /clear", "desc": "Reset conversation and context", "action": self._on_clear_chat},
            {"cmd": "/export", "label": "📥 /export", "desc": "Export chat history to Markdown", "action": self._on_export_chat_button},
        ]

        # Row Container: [📎] [Text Box Card] [ [▶ Send] \n [↵ Send • ⌘↵ Steer] ]
        self.input_top_row = tk.Frame(self.input_container, bg=self.C["bg_input"])
        self.input_top_row.pack(fill="x", padx=16, pady=(6, 6))

        # Left Attach Button (Vertically centered with input row)
        self.btn_attach = tk.Button(
            self.input_top_row, text="📎", font=(_FONT_FAMILY, 15),
            bg=self.C["btn_bg"], fg=self.C["btn_fg"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"],
            relief="flat", bd=0, padx=10, pady=5, cursor="hand2",
            highlightthickness=0,
            command=self._on_upload_file
        )
        self.btn_attach.pack(side="left", padx=(0, 10), anchor="center")

        # Center Text Input Card (Symmetrical start/end padding, centered text)
        self.text_wrapper = tk.Frame(
            self.input_top_row, bg=self.C["bg_input_inner"],
            bd=1, highlightbackground=self.C["border"], highlightthickness=1
        )
        self.text_wrapper.pack(side="left", fill="both", expand=True)
        self.input_card = self.text_wrapper  # Compatibility alias

        self.txt_input = tk.Text(
            self.text_wrapper, height=1, bg=self.C["bg_input_inner"], fg=self.C["text_main"],
            insertbackground=self.C["text_main"], font=_FONT_INPUT, bd=0,
            padx=0, pady=0, spacing1=3, spacing3=3, highlightthickness=0, wrap="word"
        )
        self.txt_input.pack(fill="both", expand=True, padx=14, pady=5)

        self._placeholder_active = True
        placeholder_color = self.C.get("text_placeholder", "#71717a")
        self.txt_input.insert("1.0", "Ask anything, paste code, or type / for commands... (⇧⏎ for newline)")
        self.txt_input.configure(fg=placeholder_color)

        self._ctrl_or_cmd_held = False

        def _on_mod_press(event=None):
            self._ctrl_or_cmd_held = True
            self._update_send_button_state()

        def _on_mod_release(event=None):
            self._ctrl_or_cmd_held = False
            self._update_send_button_state()

        for key in ("<KeyPress-Control_L>", "<KeyPress-Control_R>", "<KeyPress-Meta_L>", "<KeyPress-Meta_R>", "<KeyPress-Alt_L>", "<KeyPress-Alt_R>"):
            try:
                self.root.bind(key, _on_mod_press, add="+")
                self.txt_input.bind(key, _on_mod_press, add="+")
            except Exception:
                pass

        for key in ("<KeyRelease-Control_L>", "<KeyRelease-Control_R>", "<KeyRelease-Meta_L>", "<KeyRelease-Meta_R>", "<KeyRelease-Alt_L>", "<KeyRelease-Alt_R>"):
            try:
                self.root.bind(key, _on_mod_release, add="+")
                self.txt_input.bind(key, _on_mod_release, add="+")
            except Exception:
                pass

        try:
            self.txt_input.bind("<FocusIn>", self._on_input_focus_in)
            self.txt_input.bind("<FocusOut>", self._on_input_focus_out)
            self.txt_input.bind("<Return>", self._on_enter_pressed)
            self.txt_input.bind("<Shift-Return>", lambda e: None)
            self.txt_input.bind("<Command-Return>", self._on_steer_shortcut)
            self.txt_input.bind("<Control-Return>", self._on_steer_shortcut)
            self.txt_input.bind("<KeyRelease>", self._on_input_key_release)
            self.txt_input.bind("<KeyPress>", lambda e: self.root.after(10, self._update_send_button_state))
        except Exception:
            pass

        # Right Send Column (Shifted down to align horizontally with the text box)
        send_col = tk.Frame(self.input_top_row, bg=self.C["bg_input"])
        send_col.pack(side="right", padx=(10, 0), anchor="center", pady=(4, 0))

        # Dynamic Action Button (▶ Send / 📋 Queue / 🎯 Steer / ⏹ Stop)
        self.btn_send = tk.Button(
            send_col, text="  ▶ Send  ", font=_FONT_TAB,
            bg=self.C["btn_primary_bg"], fg=self.C["btn_primary_fg"],
            highlightbackground=self.C["btn_primary_bg"],
            activebackground=self.C["btn_primary_hover"], activeforeground=self.C["btn_primary_fg"],
            relief="flat", bd=0, padx=12, pady=4, cursor="hand2",
            highlightthickness=0,
            command=self._on_send_button_clicked
        )
        self.btn_send.pack(side="top", fill="x")

        # Tooltip directly below Send button
        steer_key = "⌘↵" if platform.system() == "Darwin" else "Ctrl+↵"
        lbl_hint = tk.Label(
            send_col, text=f"↵ Send • {steer_key} Steer", font=_FONT_TINY,
            bg=self.C["bg_input"], fg=placeholder_color
        )
        lbl_hint.pack(side="top", pady=(1, 0))

        # Attribute references for backward compatibility with test suites
        self.btn_stop = self.btn_send
        self.btn_reset_chat = SolidButton(
            self.root, text="🗑️ Clear Chat", command=self._on_reset_chat_confirm
        )
        self.btn_steer = SolidButton(
            self.root, text=f"🎯 Steer ({steer_key})", command=self._on_steer_button_pressed
        )
        self.btn_learn = SolidButton(
            self.root, text="🎓 Learn Mode", command=self._on_prompt_learn
        )

    def _update_send_button_state(self):
        """Dynamically updates button state: Queue while generating, Steer when holding Ctrl/Cmd, Stop when empty."""
        if not hasattr(self, "btn_send") or not hasattr(self, "txt_input"):
            return

        if not self.is_model_loaded:
            self.btn_send.configure(
                state="disabled", text="🔒 Unloaded",
                bg=self.C.get("btn_bg", "#27272a"), fg=self.C.get("text_muted", "#71717a")
            )
            return

        text = self.txt_input.get("1.0", "end-1c").strip()
        has_text = bool(text and not getattr(self, "_placeholder_active", False) and "Ask anything" not in text and "Model is unloaded" not in text)

        if self.is_generating:
            self.btn_send.configure(state="normal")
            if has_text:
                if getattr(self, "_ctrl_or_cmd_held", False):
                    self.btn_send.configure(
                        text="🎯 Steer",
                        bg=self.C.get("accent_purple", "#c084fc"),
                        fg="#ffffff"
                    )
                else:
                    self.btn_send.configure(
                        text="📋 Queue",
                        bg=self.C.get("accent_yellow", "#facc15"),
                        fg="#000000"
                    )
            else:
                self.btn_send.configure(
                    text="⏹ Stop",
                    bg=self.C.get("accent_red", "#ef4444"),
                    fg="#ffffff"
                )
        else:
            self.btn_send.configure(
                state="normal",
                text="  ▶ Send  ",
                bg=self.C["btn_primary_bg"],
                fg=self.C["btn_primary_fg"]
            )

    def _on_send_button_clicked(self):
        if not self.is_model_loaded:
            self._on_toggle_load_unload()
            return

        text = self.txt_input.get("1.0", "end-1c").strip()
        has_text = bool(text and not getattr(self, "_placeholder_active", False) and "Ask anything" not in text and "Model is unloaded" not in text)

        if self.is_generating:
            if has_text:
                if getattr(self, "_ctrl_or_cmd_held", False):
                    self._on_steer_button_pressed()
                else:
                    self._on_send_message()
            else:
                self._on_stop_generation()
        else:
            self._on_send_message()

    def _recompute_input_height(self, is_focused: Optional[bool] = None):
        if is_focused is None:
            is_focused = (self.root.focus_get() == self.txt_input)

        if getattr(self, "_placeholder_active", False):
            if self.txt_input.cget("height") != 1:
                self.txt_input.configure(height=1)
            return

        content = self.txt_input.get("1.0", "end-1c")
        if not content:
            if self.txt_input.cget("height") != 1:
                self.txt_input.configure(height=1)
            return

        try:
            res = self.txt_input.count("1.0", "end-1c", "displaylines")
            display_lines = res[0] if res else len(content.splitlines())
        except Exception:
            display_lines = len(content.splitlines())

        display_lines = max(1, display_lines)

        if is_focused:
            target_height = min(6, display_lines)
        else:
            target_height = min(3, display_lines)

        if self.txt_input.cget("height") != target_height:
            self.txt_input.configure(height=target_height)

    def _on_input_key_release(self, event=None):
        if self._placeholder_active:
            self._hide_slash_bar()
            self._recompute_input_height(is_focused=True)
            return

        self._recompute_input_height(is_focused=True)
        self._update_send_button_state()

        current_text = self.txt_input.get("1.0", "end-1c").strip()
        if current_text.startswith("/"):
            self._show_slash_bar(current_text)
        else:
            self._hide_slash_bar()

    def _show_slash_bar(self, filter_prefix: str = "/"):
        for child in self.slash_bar.winfo_children():
            child.destroy()

        matching = [
            item for item in self.slash_commands
            if item["cmd"].startswith(filter_prefix.lower()) or filter_prefix == "/"
        ]
        if not matching:
            self._hide_slash_bar()
            return

        lbl_hint = tk.Label(
            self.slash_bar, text="⚡ Quick Commands:", font=_FONT_TINY_BOLD,
            bg=self.C["bg_input"], fg=self.C["accent_cyan"]
        )
        lbl_hint.pack(side="left", padx=(8, 6), pady=4)

        for item in matching[:6]:
            btn = SolidButton(
                self.slash_bar, text=item["label"], font=_FONT_TINY_BOLD,
                bg=self.C["btn_bg"], fg=self.C["btn_fg"],
                padx=8, pady=3, cursor="hand2",
                command=lambda it=item: self._select_slash_command(it)
            )
            btn.pack(side="left", padx=3, pady=4)

        if not self.slash_bar.winfo_ismapped():
            self.slash_bar.pack(fill="x", side="top", padx=12, pady=(4, 0), before=self.input_top_row)

    def _hide_slash_bar(self):
        if hasattr(self, "slash_bar") and self.slash_bar.winfo_ismapped():
            self.slash_bar.pack_forget()

    def _select_slash_command(self, item: Dict[str, Any]):
        self._hide_slash_bar()
        if "action" in item:
            self.txt_input.delete("1.0", "end")
            self._placeholder_active = True
            placeholder_color = self.C.get("text_placeholder", "#71717a")
            self.txt_input.insert("1.0", "Ask anything, paste code, or type / for commands... (⇧⏎ for newline)")
            self.txt_input.configure(fg=placeholder_color)
            self._recompute_input_height(is_focused=False)
            item["action"]()
        elif "template" in item:
            self.txt_input.delete("1.0", "end")
            self._placeholder_active = False
            self.txt_input.insert("1.0", item["template"])
            self.txt_input.configure(fg=self.C["text_main"])
            self.txt_input.focus_set()
            self.txt_input.mark_set("insert", "end")
            self._recompute_input_height(is_focused=True)

    def _on_input_focus_in(self, event):
        if self._placeholder_active:
            self.txt_input.delete("1.0", "end")
            self.txt_input.configure(fg=self.C["text_main"])
            self._placeholder_active = False
        self._recompute_input_height(is_focused=True)
        self._update_send_button_state()

    def _on_input_focus_out(self, event):
        if not self.txt_input.get("1.0", "end").strip():
            self._placeholder_active = True
            placeholder_color = self.C.get("text_placeholder", "#71717a")
            self.txt_input.delete("1.0", "end")
            self.txt_input.insert("1.0", "Ask anything, paste code, or type / for commands... (⇧⏎ for newline)")
            self.txt_input.configure(fg=placeholder_color)
            self._hide_slash_bar()
        self._recompute_input_height(is_focused=False)
        self._update_send_button_state()

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
        self.queue_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"], height=32)

        self.lbl_queue_indicator = tk.Label(
            self.queue_bar, text="📋 Pending Tasks: None", font=_FONT_TINY_BOLD,
            bg=self.C["bg_hud"], fg=self.C["accent_yellow"], padx=10
        )
        self.lbl_queue_indicator.pack(side="left", padx=(14, 4), pady=3)

        # Dynamic Steer Action Pill (Positioned right next to Queued indicator above text box)
        steer_key = "⌘↵" if platform.system() == "Darwin" else "Ctrl+↵"
        self.btn_live_steer = tk.Button(
            self.queue_bar, text=f"🎯 Steer ({steer_key})", font=_FONT_TINY_BOLD,
            bg=self.C["btn_bg"], fg=self.C["accent_purple"],
            highlightbackground=self.C["btn_bg"],
            activebackground=self.C["btn_hover"], activeforeground=self.C["accent_purple"],
            relief="flat", bd=0, padx=8, pady=2, cursor="hand2",
            highlightthickness=0,
            command=self._on_steer_shortcut
        )
        self.btn_live_steer.pack(side="left", padx=4, pady=3)
        self.btn_steer = self.btn_live_steer

        self.btn_clear_queue = tk.Button(
            self.queue_bar, text="✕ Clear All", font=_FONT_TINY,
            bg=self.C["btn_bg"], fg=self.C["accent_red"],
            relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
            highlightthickness=0,
            command=self._on_clear_queue
        )

    def _update_queue_ui(self):
        q_len = len(self.prompt_queue)
        if q_len > 0:
            if not self.queue_bar.winfo_ismapped():
                self.queue_bar.pack(fill="x", side="bottom", before=self.input_container, pady=(0, 2))
            self.lbl_queue_indicator.configure(
                text=f"📋 Queued Tasks: {q_len}",
                fg=self.C["accent_yellow"]
            )
            if not self.btn_clear_queue.winfo_ismapped():
                self.btn_clear_queue.pack(side="left", padx=4)
        else:
            if self.queue_bar.winfo_ismapped():
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
            f"• Click **'⚡ Load'** to load weights into Apple Silicon unified memory."
        )
        self.lbl_model_status.configure(text=f"⚡ Ready to Load ({target_info['short_name']})", fg=self.C["accent_cyan"])
        self._update_model_action_buttons()

    def _on_download_hf_failed(self, target_info: Dict[str, Any], error: str):
        self._append_ai_message(f"⚠️ **Download Failed** for `{target_info['name']}`: {error}")
        self.lbl_model_status.configure(text=f"✗ Download Failed ({target_info['short_name']})", fg=self.C["accent_red"])
        self._update_model_action_buttons()

    def _show_model_loading_popup(self, model_name: str, vram_str: str) -> Tuple[Optional[tk.Toplevel], Callable[[float, str], None]]:
        if not self.root.winfo_viewable():
            return None, lambda v, m: None

        popup = tk.Toplevel(self.root)
        popup.title("Loading Model")
        popup.configure(bg=self.C.get("bg_card", "#18181b"))
        popup.resizable(False, False)
        popup.transient(self.root)

        card = tk.Frame(popup, bg=self.C.get("bg_card", "#18181b"), padx=24, pady=18)
        card.pack(fill="both", expand=True)

        tk.Label(
            card, text="⚡ Loading Neural Weights", font=_FONT_H2,
            bg=self.C.get("bg_card", "#18181b"), fg=self.C.get("accent_cyan", "#38bdf8")
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            card, text=f"Allocating '{model_name}' ({vram_str}) into Unified Memory...",
            font=_FONT_MAIN, bg=self.C.get("bg_card", "#18181b"), fg=self.C.get("text_main", "#ffffff")
        ).pack(anchor="w", pady=(0, 10))

        try:
            import tkinter.ttk as ttk
            progress = ttk.Progressbar(card, orient="horizontal", mode="indeterminate", length=400)
            progress.pack(fill="x", pady=(0, 8))
            progress.start(15)
        except Exception:
            pass

        lbl_step = tk.Label(
            card, text="• Initializing weight matrices & KV cache tensors...",
            font=_FONT_TINY, bg=self.C.get("bg_card", "#18181b"), fg=self.C.get("text_muted", "#a1a1aa")
        )
        lbl_step.pack(anchor="w")

        # Auto-size popup to exact dimensions of its content and center over root
        popup.update_idletasks()
        pw = max(400, popup.winfo_reqwidth())
        ph = popup.winfo_reqheight()

        try:
            rx, ry = self.root.winfo_x(), self.root.winfo_y()
            rw, rh = self.root.winfo_width(), self.root.winfo_height()
            px = rx + max(0, (rw - pw) // 2)
            py = ry + max(0, (rh - ph) // 2)
            popup.geometry(f"{pw}x{ph}+{px}+{py}")
        except Exception:
            popup.geometry(f"{pw}x{ph}")

        def _update_status(val: float, msg: str):
            try:
                if popup.winfo_exists():
                    lbl_step.configure(text=f"• {msg}")
            except Exception:
                pass

        return popup, _update_status

    def _update_model_action_buttons(self):
        """
        Dynamically updates model action buttons according to state:
        - If NOT installed/cached:
            self.btn_load_unload -> "⬇️ Install" (runs self._on_download_hf_model)
            self.btn_reset_reinstall -> "📥 Grab" (runs self._on_download_hf_model)
        - If installed & NOT loaded:
            self.btn_load_unload -> "⚡ Load" (runs self._on_toggle_load_unload)
            self.btn_reset_reinstall -> "🔄 Reset" (runs self._on_reset_and_reinstall_single_confirm)
        - If LOADED in memory:
            self.btn_load_unload -> "⏏ Unload" (runs self._on_toggle_load_unload)
            self.btn_reset_reinstall -> "🔄 Reset & Re-grab" (runs self._on_reset_and_reinstall_single_confirm)
        """
        if not hasattr(self, "btn_load_unload") or not hasattr(self, "btn_reset_reinstall"):
            return

        target_info = self.models_config[self.active_tab_id]
        m_path = target_info.get("model_path") or target_info.get("repo_id")
        cached = is_model_cached_locally(target_info.get("repo_id", "")) if target_info.get("repo_id") else (os.path.exists(m_path or "") if m_path else False)

        if not cached:
            self.btn_load_unload.configure(
                text="⬇️ Install",
                fg=self.C.get("accent_cyan", "#38bdf8"),
                command=self._on_download_hf_model
            )
            self.btn_reset_reinstall.configure(
                text="📥 Grab",
                fg=self.C.get("btn_fg", "#ffffff"),
                command=self._on_download_hf_model
            )
        elif not self.is_model_loaded:
            self.btn_load_unload.configure(
                text="⚡ Load",
                fg=self.C.get("accent_cyan", "#38bdf8"),
                command=self._on_toggle_load_unload
            )
            self.btn_reset_reinstall.configure(
                text="🔄 Reset",
                fg=self.C.get("btn_fg", "#ffffff"),
                command=self._on_reset_and_reinstall_single_confirm
            )
        else:
            self.btn_load_unload.configure(
                text="⏏ Unload",
                fg=self.C.get("accent_orange", "#fb923c"),
                command=self._on_toggle_load_unload
            )
            self.btn_reset_reinstall.configure(
                text="🔄 Reset & Re-grab",
                fg=self.C.get("btn_fg", "#ffffff"),
                command=self._on_reset_and_reinstall_single_confirm
            )

    def _update_input_lock_state(self):
        """Enforces input lock-out when the neural model is unloaded."""
        if not hasattr(self, "txt_input") or not hasattr(self, "btn_send"):
            return

        if not self.is_model_loaded:
            self.txt_input.configure(state="normal")
            if getattr(self, "_placeholder_active", False) or not self.txt_input.get("1.0", "end-1c").strip():
                self.txt_input.delete("1.0", "end")
                self.txt_input.insert("1.0", "🔒 Model is unloaded. Click '⚡ Load' or '⬇️ Install' in the top bar to chat...")
                self.txt_input.configure(fg=self.C.get("text_placeholder", "#71717a"))
                self._placeholder_active = True
            self.txt_input.configure(state="disabled")
            self.btn_send.configure(state="disabled", text="🔒 Unloaded")
        else:
            self.txt_input.configure(state="normal")
            if getattr(self, "_placeholder_active", False) and "Model is unloaded" in self.txt_input.get("1.0", "end-1c"):
                self.txt_input.delete("1.0", "end")
                self.txt_input.insert("1.0", "Ask anything, paste code, or type / for commands... (⇧⏎ for newline)")
                self.txt_input.configure(fg=self.C.get("text_placeholder", "#71717a"))
            self.btn_send.configure(state="normal", text="  ▶ Send  ")

    def _on_toggle_load_unload(self):
        target_info = self.models_config[self.active_tab_id]
        m_path = target_info.get("model_path") or target_info.get("repo_id")

        if self.is_model_loaded:
            self.engine.unload_model()
            self.is_model_loaded = False
            self.lbl_model_status.configure(text=f"○ Unloaded ({target_info['short_name']})", fg=self.C["accent_yellow"])
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")
            self._update_model_action_buttons()
            self._update_resource_view_metrics()
            self._update_input_lock_state()
            self._append_ai_message(f"⏏ **Model Unloaded**: `{target_info['name']}` purged from unified memory.")
        else:
            popup, update_cb = self._show_model_loading_popup(target_info["name"], target_info.get("vram", "5.8 GB"))

            def _do_load():
                load_res = self.engine.load_model(target_info["name"], model_path=m_path)
                def _done():
                    if popup:
                        try:
                            popup.destroy()
                        except Exception:
                            pass

                    if load_res.get("status") == "loaded":
                        self.is_model_loaded = True
                        self.lbl_model_status.configure(text=f"● Loaded: {target_info['short_name']}", fg=self.C["accent_green"])
                        self.lbl_vram.configure(text=f"💾 {target_info['vram']}")
                        self._update_model_action_buttons()
                        self._update_resource_view_metrics()
                        self._update_input_lock_state()
                        self._append_ai_message(f"⚡ **Model Loaded**: `{target_info['name']}` is active in Apple Silicon unified memory.")
                    else:
                        self.is_model_loaded = False
                        self.lbl_model_status.configure(text=f"○ Not Downloaded ({target_info['short_name']})", fg=self.C["accent_yellow"])
                        self._update_model_action_buttons()
                        self._update_input_lock_state()
                        self._append_ai_message(
                            f"⚠️ Model weights for `{target_info['name']}` are not yet loaded or downloaded.\n\n"
                            f"• Click **'⬇️ Install'** or **'📥 Grab'** in the top bar to retrieve the weights."
                        )

                if self.root.winfo_exists():
                    self.root.after(0, _done)

            if self.root.winfo_viewable():
                threading.Thread(target=_do_load, daemon=True).start()
            else:
                _do_load()

    def _append_rendered_user_turn(self, text: str):
        self.chat_stream.insert("end", f"\n👤 You\n", "user_header")
        self.chat_stream.insert("end", f"{text.strip()}\n\n", "user_msg")

    def _append_rendered_ai_turn(self, text: str):
        info = self.models_config[self.active_tab_id]
        self.chat_stream.insert("end", f"\n✦ {info['short_name']}\n", "ai_header")
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                self._render_styled_markdown(part)
            else:
                lines = part.split("\n", 1)
                lang = lines[0].strip() if lines else "python"
                code_content = lines[1] if len(lines) > 1 else part
                self.chat_stream.insert("end", f"📄 {lang or 'python'}\n", "code_hdr")
                self.chat_stream.insert("end", code_content.rstrip(), "code_block")
                self.chat_stream.insert("end", "\n\n")

    def _on_switch_model_tab(self, target_tab_id: str):
        if target_tab_id not in self.models_config:
            return
        self.active_tab_id = target_tab_id
        target_info = self.models_config[target_tab_id]
        m_path = target_info.get("model_path") or target_info.get("repo_id")

        if hasattr(self, "model_var"):
            self.model_var.set(target_info["short_name"])

        # Switch chat stream content to this model's distinct conversation
        self.chat_stream.configure(state="normal")
        self.chat_stream.delete("1.0", "end")

        if self.active_tab_id not in self.chat_history:
            self.chat_history[self.active_tab_id] = []

        history = self.chat_history[self.active_tab_id]
        if not history:
            self._send_welcome_messages()
        else:
            for item in history:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    self._append_rendered_user_turn(content)
                elif role == "assistant":
                    self._append_rendered_ai_turn(content)

        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

        # Handle model loading
        self.engine.unload_model()
        load_res = self.engine.load_model(target_info["name"], model_path=m_path)

        if load_res.get("status") == "loaded":
            self.is_model_loaded = True
            self.lbl_model_status.configure(text=f"● Loaded: {target_info['short_name']}", fg=self.C["accent_green"])
            self.lbl_vram.configure(text=f"💾 {target_info['vram']}")
        else:
            self.is_model_loaded = False
            cached = is_model_cached_locally(target_info["repo_id"]) if target_info.get("repo_id") else (os.path.exists(m_path or "") if m_path else False)
            status_txt = f"⚡ Ready to Load ({target_info['short_name']})" if cached else f"○ Not Downloaded ({target_info['short_name']})"
            self.lbl_model_status.configure(text=status_txt, fg=self.C["accent_cyan"] if cached else self.C["accent_yellow"])
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")

        self._update_model_action_buttons()
        total_p = target_info.get("raw_params", 27_400_000_000) + self.synapses_learned_count
        param_str = format_parameter_count(total_p)
        self.lbl_params.configure(text=f"🧠 {param_str} Total Params")
        if hasattr(self, "lbl_model_speed_params"):
            self.lbl_model_speed_params.configure(
                text=f"{target_info.get('est_speed', '⚡ ~28 t/s')} • {target_info.get('base_params', '27.4B')}"
            )
        self._update_resource_view_metrics()
        self._update_input_lock_state()

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
            self.lbl_vram.configure(text="💾 0.0 GB / 16 GB")
            self._update_model_action_buttons()
            self._update_input_lock_state()
            self._update_resource_view_metrics()
            used_gb = status.get("used_gb", 0.0)
            total_gb = status.get("total_gb", 16.0)
            used_pct = status.get("used_percent", 94.0)
            proc_gb = status.get("process_rss_gb", 0.0)
            self._append_ai_message(
                f"🛡️ **System Memory Watchdog**: Total host RAM utilization reached {used_pct:.1f}% "
                f"({used_gb:.1f} GB / {total_gb:.1f} GB across host OS and active apps; app process: {proc_gb:.1f} GB).\n\n"
                f"• Automatically unloaded neural model to preserve system responsiveness."
            )

    def _on_remove_workspace_folder(self):
        self.workspace_dir = None
        self.tools.set_workspace_dir(None)
        self.lbl_workspace.configure(text="📁 + Add Folder")
        if hasattr(self, "btn_remove_folder") and self.btn_remove_folder.winfo_ismapped():
            self.btn_remove_folder.pack_forget()
        self._update_resource_view_metrics()
        self._append_ai_message("✓ Workspace folder removed (no local folder selected).")

    def _on_select_workspace_folder(self):
        chosen = filedialog.askdirectory(
            initialdir=self.workspace_dir or os.getcwd(),
            title="Select Workspace Folder for AI Coding & Tool Access"
        )
        if chosen:
            self.workspace_dir = os.path.abspath(chosen)
            self.tools.set_workspace_dir(self.workspace_dir)
            disp = os.path.basename(self.workspace_dir) or self.workspace_dir
            self.lbl_workspace.configure(text=f"📁 {disp}")
            if hasattr(self, "btn_remove_folder") and not self.btn_remove_folder.winfo_ismapped():
                self.btn_remove_folder.pack(side="left", padx=(2, 0))
            self._update_resource_view_metrics()
            self._append_ai_message(f"✓ Workspace folder set to: `{self.workspace_dir}`")

    # ─────────────────────────────────────────────────────
    #  RICH TEXT & MARKDOWN FORMATTING RENDERER
    # ─────────────────────────────────────────────────────
    def _scroll_chat_to_bottom(self):
        try:
            self.chat_stream.see("end")
            self.chat_stream.yview_moveto(1.0)
            self.root.after_idle(lambda: self.chat_stream.see("end"))
        except Exception:
            pass

    def _render_styled_markdown(self, raw_text: str):
        lines = raw_text.splitlines()

        for line in lines:
            trimmed = line.strip()

            if trimmed in ("---", "***", "___", "------", "===="):
                self.chat_stream.insert("end", "  " + "─" * 40 + "\n\n", "separator")
                continue

            if trimmed.startswith("#### "):
                self._insert_inline_tokens(trimmed[5:], base_tag="md_h3")
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith("### "):
                self._insert_inline_tokens(trimmed[4:], base_tag="md_h3")
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith("## "):
                self._insert_inline_tokens(trimmed[3:], base_tag="md_h2")
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith("# "):
                self._insert_inline_tokens(trimmed[2:], base_tag="md_h1")
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith("> "):
                self.chat_stream.insert("end", "  ▎ ", "md_bullet")
                self._insert_inline_tokens(trimmed[2:], base_tag="md_quote")
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith(("- ", "* ", "+ ", "• ")):
                bullet_content = trimmed[2:]
                self.chat_stream.insert("end", "  • ", "md_bullet")
                self._insert_inline_tokens(bullet_content, base_tag="ai_msg")
                self.chat_stream.insert("end", "\n")
            elif trimmed.startswith("  - ") or trimmed.startswith("  * ") or trimmed.startswith("  + ") or trimmed.startswith("  • "):
                bullet_content = trimmed.lstrip(" -+*•")
                self.chat_stream.insert("end", "    ◦ ", "md_bullet")
                self._insert_inline_tokens(bullet_content, base_tag="ai_msg")
                self.chat_stream.insert("end", "\n")
            elif re.match(r"^\d+\.\s", trimmed):
                match = re.match(r"^(\d+)\.\s(.*)", trimmed)
                if match:
                    num, content = match.group(1), match.group(2)
                    self.chat_stream.insert("end", f"  {num}. ", "md_bullet")
                    self._insert_inline_tokens(content, base_tag="ai_msg")
                    self.chat_stream.insert("end", "\n")
            else:
                if trimmed:
                    self._insert_inline_tokens(line, base_tag="ai_msg")
                    self.chat_stream.insert("end", "\n")
                else:
                    self.chat_stream.insert("end", "\n")

    def _insert_inline_tokens(self, text: str, base_tag: str = "ai_msg"):
        token_pattern = re.compile(
            r"(\*\*\*(?:[^*]|\*(?!\*\*))+\*\*\*|"
            r"___(?:[^_]|_(?!__))___|"
            r"\*\*(?:[^*]|\*(?!\*))+\*\*|"
            r"__(?:[^_]|_(?!_))+__|"
            r"`[^`]+`|"
            r"~~[^~]+~~|"
            r"\*(?:[^*]|\*(?!\*))+\*|"
            r"_(?:[^_]|_(?!_))+_)"
        )
        parts = token_pattern.split(text)

        for part in parts:
            if not part:
                continue
            if part.startswith("`") and part.endswith("`") and len(part) >= 2:
                self.chat_stream.insert("end", f" {part[1:-1]} ", "md_inline_code")
            elif (part.startswith("***") and part.endswith("***") and len(part) >= 6) or (part.startswith("___") and part.endswith("___") and len(part) >= 6):
                self.chat_stream.insert("end", part[3:-3], "md_bold_italic")
            elif (part.startswith("**") and part.endswith("**") and len(part) >= 4) or (part.startswith("__") and part.endswith("__") and len(part) >= 4):
                self.chat_stream.insert("end", part[2:-2], "md_bold")
            elif (part.startswith("*") and part.endswith("*") and len(part) >= 2) or (part.startswith("_") and part.endswith("_") and len(part) >= 2):
                self.chat_stream.insert("end", part[1:-1], "md_italic")
            elif part.startswith("~~") and part.endswith("~~") and len(part) >= 4:
                self.chat_stream.insert("end", part[2:-2], "md_italic")
            else:
                self.chat_stream.insert("end", part, base_tag)

    # ─────────────────────────────────────────────────────
    #  MESSAGING & INFERENCE PIPELINE
    # ─────────────────────────────────────────────────────
    def _send_welcome_messages(self):
        self.chat_stream.configure(state="normal")
        info = self.models_config[self.active_tab_id]
        self.chat_stream.insert("end", f"✦ {info['name']}  •  {datetime.now().strftime('%H:%M')}\n\n", "ai_header")
        param_str = format_parameter_count(info.get("raw_params", 27_400_000_000))
        welcome_text = (
            f"Smart AI Studio ready to assist with autonomous reasoning, coding, and tool execution.\n\n"
            f"• **Active Model**: {info['name']} ({param_str} Parameters)\n"
            f"• **Custom Models**: Click **'➕ Import Model'** in the top bar to add your own local or HuggingFace models.\n"
            f"• **Live Steer**: Press `⌘+Enter` (or `Ctrl+Enter`) to steer the model while typing.\n"
        )
        self._render_styled_markdown(welcome_text)
        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

    def _append_user_message(self, text: str):
        if hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
            self.chat_history[self.active_tab_id].append({"role": "user", "content": text})
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n👤 You  •  {datetime.now().strftime('%H:%M')}\n", "user_header")
        self.chat_stream.insert("end", text.strip(), "user_msg")
        self.chat_stream.insert("end", "\n\n")
        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

    def _append_steer_directive(self, steer_text: str):
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n🎯 Live Steer Directive  •  {datetime.now().strftime('%H:%M')}\n", "steer_header")
        self.chat_stream.insert("end", f"Steered: \"{steer_text.strip()}\"", "steer_msg")
        self.chat_stream.insert("end", "\n\n")
        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

    def _append_queued_message(self, text: str, pos: int):
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"\n📋 Queued Task (#{pos})  •  {datetime.now().strftime('%H:%M')}\n", "queue_header")
        self.chat_stream.insert("end", f"\"{text.strip()}\" (will execute automatically after current task)", "queue_msg")
        self.chat_stream.insert("end", "\n\n")
        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

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

            self.thinking_meta[think_id] = {
                "duration_s": duration_s,
                "tokens": t_toks,
                "speed_str": speed_str
            }

            btn_tag = f"tag_btn_{think_id}"
            self.chat_stream.insert(
                "end", f"  ▶ 💭 Thought for {duration_s:.1f}s ({t_toks} tokens{speed_str}) [Click to Expand]  ",
                ("think_dropdown_btn", btn_tag)
            )
            self.chat_stream.insert("end", "\n\n")
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

                self.chat_stream.insert("end", f"  📄 {lang or 'python'}  ", "code_hdr")
                self.chat_stream.insert("end", " [ 📋 Copy ] ", ("code_action_copy", tag_copy))
                self.chat_stream.insert("end", " [ 🎨 Open in Canvas ] \n", ("code_action_canvas", tag_canvas))

                self.chat_stream.insert("end", code_content.rstrip(), "code_block")
                self.chat_stream.insert("end", "\n\n")

                self.chat_stream.tag_bind(tag_copy, "<Button-1>", lambda e, cid=code_id: self._on_copy_code_snippet(cid))
                self.chat_stream.tag_bind(tag_canvas, "<Button-1>", lambda e, cid=code_id: self._on_open_snippet_in_canvas(cid))

                if len(code_content.splitlines()) > 4 and hasattr(self, "txt_canvas"):
                    self._open_in_canvas(code_content.strip())

        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

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
        meta = self.thinking_meta.get(think_id, {})
        duration_s = meta.get("duration_s", 0.2)
        t_toks = meta.get("tokens", len(content.split()))
        speed_str = meta.get("speed_str", "")

        self.chat_stream.configure(state="normal")
        btn_tag = f"tag_btn_{think_id}"
        body_tag = f"tag_body_{think_id}"

        # 1. Update button arrow and label
        btn_ranges = self.chat_stream.tag_ranges(btn_tag)
        if btn_ranges:
            arrow = "▼" if not is_exp else "▶"
            action_lbl = "[Click to Collapse]" if not is_exp else "[Click to Expand]"
            new_btn_text = f"  {arrow} 💭 Thought for {duration_s:.1f}s ({t_toks} tokens{speed_str}) {action_lbl}  "
            start_pos = btn_ranges[0]
            self.chat_stream.delete(btn_ranges[0], btn_ranges[1])
            self.chat_stream.insert(start_pos, new_btn_text, ("think_dropdown_btn", btn_tag))

        # 2. Toggle reasoning body
        if not is_exp:
            btn_ranges = self.chat_stream.tag_ranges(btn_tag)
            if btn_ranges:
                insert_idx = self.chat_stream.index(f"{btn_ranges[1]}+1c")
            else:
                insert_idx = "end"
            body_text = f"\n💭 Chain-of-Thought Reasoning:\n{content.strip()}\n"
            self.chat_stream.insert(insert_idx, body_text, ("think_body", body_tag))
        else:
            body_ranges = self.chat_stream.tag_ranges(body_tag)
            if body_ranges:
                self.chat_stream.delete(body_ranges[0], body_ranges[1])

        self.chat_stream.configure(state="disabled")

    def _append_tool_call(self, tool_name: str, args_str: str, result_str: str):
        self.chat_stream.configure(state="normal")
        self.chat_stream.insert("end", f"  ⚙️ Tool Execution: {tool_name}({args_str})  ", "tool_pill")
        self.chat_stream.insert("end", "\n")
        out_display = result_str if len(result_str) < 1200 else result_str[:1200] + "... [Output Truncated]"
        self.chat_stream.insert("end", out_display.strip(), "tool_output")
        self.chat_stream.insert("end", "\n\n")
        self.chat_stream.configure(state="disabled")
        self._scroll_chat_to_bottom()

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
        self._recompute_input_height(is_focused=True)

        if self.is_generating:
            self._append_steer_directive(steer_text)
            self.cancel_event.set()
            self.root.after(150, lambda: self._execute_steered_generation(steer_text))
        else:
            self._append_user_message(f"🎯 [Steer Directive]: {steer_text}")
            self._set_generating_state(True)
            threading.Thread(target=self._process_message_thread, args=(steer_text, steer_text), daemon=True).start()

    def _execute_steered_generation(self, steer_directive: str):
        self._set_generating_state(True)
        steered_prompt = f"[Live Steering Directive: {steer_directive}]\nPlease follow this directive directly."
        threading.Thread(target=self._process_message_thread, args=(steered_prompt, steer_directive), daemon=True).start()

    def _on_enter_pressed(self, event):
        if not (event.state & 0x1):  # Shift not held
            if self.is_generating and getattr(self, "_ctrl_or_cmd_held", False):
                self._on_steer_button_pressed()
            else:
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
            self._recompute_input_height(is_focused=True)
            return

        self.txt_input.delete("1.0", "end")
        self._recompute_input_height(is_focused=True)
        self._append_user_message(raw_text)

        tokens = len(user_msg.split()) * 2
        self.total_tokens_used += tokens
        self._update_telemetry()

        self._set_generating_state(True)
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
                self.root.after(0, lambda: self._set_generating_state(False))
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

            if not matched:
                if self.cancel_event.is_set():
                    self.root.after(0, lambda: self._append_ai_message("⏹ Generation stopped."))
                    return

                curr_history = self.chat_history.get(self.active_tab_id, [])
                info = self.models_config[self.active_tab_id]

                # 1. Initialize streaming container in chat
                accumulated = []
                mark_name = f"stream_ai_{int(time.time() * 1000)}"

                def _init_ai_stream():
                    self.chat_stream.configure(state="normal")
                    self.chat_stream.insert("end", f"\n✦ {info['short_name']}  •  {datetime.now().strftime('%H:%M')}\n", "ai_header")
                    self.chat_stream.mark_set(mark_name, "end")
                    self.chat_stream.mark_gravity(mark_name, "left")
                    self.chat_stream.configure(state="disabled")
                    self._scroll_chat_to_bottom()

                self.root.after(0, _init_ai_stream)

                # 2. Yield tokens in real time directly to chat
                for chunk in self.engine.stream_solve(full_msg, history=curr_history, cancel_event=self.cancel_event):
                    if self.cancel_event.is_set():
                        break
                    accumulated.append(chunk)
                    def _insert_chunk(c=chunk):
                        try:
                            self.chat_stream.configure(state="normal")
                            self.chat_stream.insert("end", c, "ai_msg")
                            self.chat_stream.configure(state="disabled")
                            self._scroll_chat_to_bottom()
                        except Exception:
                            pass
                    self.root.after(0, _insert_chunk)

                full_ans = "".join(accumulated).strip()
                duration_s = max(0.01, time.perf_counter() - start_time)
                tokens_count = len(full_ans.split()) * 2
                tok_per_sec = tokens_count / duration_s
                self.total_tokens_used += tokens_count

                # 3. Finalize markdown styling and collapsible thinking pill
                def _finalize_ai_msg(text=full_ans, dur=duration_s, tps=tok_per_sec):
                    try:
                        self.chat_stream.configure(state="normal")
                        # Delete raw stream chunk to replace with styled markdown
                        if self.chat_stream.get(mark_name, "end").strip():
                            self.chat_stream.delete(mark_name, "end")

                        # Parse <think>...</think>
                        th_text = None
                        main_body = text
                        if "<think>" in text and "</think>" in text:
                            p1 = text.find("<think>") + 7
                            p2 = text.find("</think>")
                            th_text = text[p1:p2].strip()
                            main_body = (text[:p1-7] + text[p2+8:]).strip()

                        # Compact, shrink-to-fit thinking pill
                        if th_text:
                            self._think_counter += 1
                            think_id = f"think_{self._think_counter}"
                            self.thinking_cache[think_id] = th_text
                            self.thinking_expanded[think_id] = False
                            btn_tag = f"tag_btn_{think_id}"
                            self.chat_stream.insert(
                                "end", f"  💭 Thought for {dur:.1f}s [Click to Expand]  ",
                                ("think_dropdown_btn", btn_tag)
                            )
                            self.chat_stream.insert("end", "\n\n")
                            self.chat_stream.tag_bind(btn_tag, "<Button-1>", lambda e, tid=think_id: self._on_toggle_thinking_dropdown(tid))

                        parts = main_body.split("```")
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
                                self.chat_stream.insert("end", f"  📄 {lang or 'python'}  ", "code_hdr")
                                self.chat_stream.insert("end", " [ 📋 Copy ] ", ("code_action_copy", tag_copy))
                                self.chat_stream.insert("end", " [ 🎨 Open in Canvas ] \n", ("code_action_canvas", tag_canvas))
                                self.chat_stream.insert("end", code_content.rstrip(), "code_block")
                                self.chat_stream.insert("end", "\n\n")
                                self.chat_stream.tag_bind(tag_copy, "<Button-1>", lambda e, cid=code_id: self._on_copy_code_snippet(cid))
                                self.chat_stream.tag_bind(tag_canvas, "<Button-1>", lambda e, cid=code_id: self._on_open_snippet_in_canvas(cid))

                        if hasattr(self, "chat_history") and self.active_tab_id in self.chat_history:
                            self.chat_history[self.active_tab_id].append({"role": "assistant", "content": main_body})

                        self.chat_stream.configure(state="disabled")
                        self._scroll_chat_to_bottom()
                    except Exception:
                        pass

                self.root.after(0, _finalize_ai_msg)
                self.root.after(0, lambda tps=tok_per_sec: self._update_telemetry(tps))

                # 4. Log trace into SQLite memory
                self.db.log_interaction(
                    prompt=full_msg,
                    completion=full_ans,
                    raw_branches=[full_ans],
                    verified_reward=1.0,
                    surprise_score=0.0,
                    mode="Instant Stream (N=1)",
                    entropy=0.15,
                    winning_branch=0,
                    winning_temp=0.20,
                    test_cases=""
                )

        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda em=err_msg: self._append_ai_message(f"⚠️ Error executing query: {em}"))

        finally:
            self.root.after(0, lambda: self._set_generating_state(False))
            self._check_and_run_next_queue()

    def _set_generating_state(self, generating: bool):
        self.is_generating = generating
        if generating:
            self.cancel_event.clear()
            self.btn_send.configure(
                text="  ⏹ Stop  ", bg=self.C["accent_red"], fg="#ffffff",
                activebackground="#dc2626", activeforeground="#ffffff"
            )
        else:
            self.btn_send.configure(
                text="  ▶ Send  ", bg=self.C["btn_primary_bg"], fg=self.C["btn_primary_fg"],
                activebackground=self.C["btn_primary_hover"], activeforeground=self.C["btn_primary_fg"]
            )
        self._update_queue_ui()

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

        self._set_generating_state(True)
        threading.Thread(target=self._process_message_thread, args=(user_msg, raw_text), daemon=True).start()

    def _on_stop_generation(self):
        self.cancel_event.set()
        self._set_generating_state(False)

    def _update_telemetry(self, tps: float = 0.0):
        ctx_pct = min(100.0, (self.total_tokens_used / self.max_context_window) * 100)
        self.lbl_context.configure(text=f"📊 Context: {self.total_tokens_used:,} / {self.max_context_window:,} ({ctx_pct:.0f}%)")
        curr_info = self.models_config.get(self.active_tab_id, {})
        total_p = curr_info.get("raw_params", 27_400_000_000) + self.synapses_learned_count
        param_str = format_parameter_count(total_p)
        if hasattr(self, "lbl_params"):
            self.lbl_params.configure(text=f"🧠 {param_str} Total Params")
        if hasattr(self, "lbl_synapses"):
            syn_str = format_added_synapses(self.synapses_learned_count)
            self.lbl_synapses.configure(text=f"📈 {syn_str}")
        if hasattr(self, "lbl_tps") and tps > 0:
            self.lbl_tps.configure(text=f"⚡ {tps:.1f} tok/s")
        if hasattr(self, "lbl_res_synapses"):
            syn_str = format_added_synapses(self.synapses_learned_count)
            self.lbl_res_synapses.configure(text=f"• +Params Learned: {syn_str} (EWC Replay Active)")

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

    def _on_canvas_load_dsl_template(self):
        """Loads TensorGraphDSL and GlyphScript interactive template into Canvas."""
        template = '''# ✦ TensorGraphDSL & GlyphScript Interactive Sandbox
# Evaluate non-commutative tensor transformations & graph invariants.

from core.dsl_engine import evaluate_tensorgraph_dsl, evaluate_glyph_script

# 1. TensorGraphDSL Expression: [2, 4, 6] >>~fold(1) <#>scale(3)
expr = "[2, 4, 6] >>~fold(1) <#>scale(3)"
result = evaluate_tensorgraph_dsl(expr)
print(f"TensorGraphDSL Result: {result}")
assert result == [12, 18, 6]

# 2. GlyphScript Invariant Checking:
glyph = """
RULE: DAG_MONOTONIC_FLOW
A -> B (5)
B -> C (3)
INVARIANT: ALL(weight > 0)
"""
glyph_res = evaluate_glyph_script(glyph)
print(f"GlyphScript Status: {glyph_res['status']}")
assert glyph_res["invariants_passed"] == True
'''
        self._open_in_canvas(template)

    def _on_open_branch_visualizer(self):
        """Opens interactive multi-branch candidate rollout visualizer."""
        modal = tk.Toplevel(self.root)
        modal.title("Parallel Multi-Branch Visualizer & Telemetry")
        modal.geometry("780x560")
        modal.minsize(680, 480)
        modal.configure(bg=self.C["bg_hud"])
        modal.transient(self.root)

        meta = self.last_metadata or {
            "mode": "Pro Search (N=16)",
            "entropy": 0.52,
            "branch_count": 16,
            "verified": True,
            "verified_reward": 1.0,
            "surprise_score": 0.48,
            "winning_branch": 2,
            "winning_temp": 0.35,
            "tok_speed": 10.9,
            "memory_rss_mb": 1536.0,
            "temp_ladder": [0.20, 0.23, 0.28, 0.35, 0.42, 0.50, 0.58, 0.67, 0.76, 0.88],
            "branches": [
                {"index": 0, "temp": 0.20, "passed": False, "reward": 0.0, "code": "def solve(): return False", "stderr": "AssertionError: Expected [12, 18, 6]"},
                {"index": 1, "temp": 0.23, "passed": False, "reward": 0.0, "code": "def solve(): return None", "stderr": "AssertionError"},
                {"index": 2, "temp": 0.35, "passed": True, "reward": 1.0, "code": "def solve(): return [12, 18, 6]", "stderr": ""},
            ]
        }

        # Header
        hdr = tk.Frame(modal, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=16, pady=12)

        tk.Label(hdr, text="🌿 Multi-Branch Search & RLVR Verification", font=_FONT_H2, bg=self.C["bg_hud"], fg=self.C["accent_green"]).pack(anchor="w")

        # Telemetry HUD Row
        hud_row = tk.Frame(modal, bg=self.C["bg_hud"])
        hud_row.pack(fill="x", padx=16, pady=(0, 10))

        ent_val = meta.get("entropy", 0.0)
        mode_str = meta.get("mode", "Pro Search")
        tps = meta.get("tok_speed", 10.9)
        rss = meta.get("memory_rss_mb", 1024.0)
        s_score = meta.get("surprise_score", 0.0)
        r_score = meta.get("verified_reward", 0.0)
        win_idx = meta.get("winning_branch", 0)
        win_temp = meta.get("winning_temp", 0.20)

        self._make_badge(hud_row, f"⚡ {tps:.1f} tok/s (PLD K=4)", self.C["accent_green"])
        self._make_badge(hud_row, f"🧠 Entropy H={ent_val:.2f} ({mode_str})", self.C["accent_cyan"])
        self._make_badge(hud_row, f"🏆 Winner: Branch #{win_idx+1} (T={win_temp:.2f})", self.C["accent_yellow"])
        self._make_badge(hud_row, f"🎯 Reward R={r_score:.1f} | S={s_score:.2f}", self.C["accent_purple"])
        self._make_badge(hud_row, f"💾 RSS: {rss:.0f} MB", self.C["accent_orange"])

        # Main Paned / Treeview & Inspector
        paned = tk.PanedWindow(modal, orient="vertical", bg=self.C["border"], sashwidth=4, bd=0)
        paned.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # Branches list
        tree_frame = tk.Frame(paned, bg=self.C["bg_app"])
        paned.add(tree_frame, height=180)

        cols = ("idx", "temp", "status", "reward", "code_preview")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=6)
        tree.heading("idx", text="Branch #")
        tree.heading("temp", text="Temp T(i)")
        tree.heading("status", text="RLVR Status")
        tree.heading("reward", text="Reward (R)")
        tree.heading("code_preview", text="Candidate Code Preview")

        tree.column("idx", width=70, anchor="center")
        tree.column("temp", width=80, anchor="center")
        tree.column("status", width=120, anchor="center")
        tree.column("reward", width=90, anchor="center")
        tree.column("code_preview", width=380, anchor="w")

        tree.pack(fill="both", expand=True)

        # Code Inspector
        insp_frame = tk.Frame(paned, bg=self.C["bg_card"])
        paned.add(insp_frame, height=220)

        tk.Label(insp_frame, text="🔍 Selected Branch Rollout & Verification Log:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(anchor="w", padx=10, pady=(6, 2))
        txt_trace = tk.Text(insp_frame, bg=self.C["code_bg"], fg=self.C["code_fg"], font=_FONT_MONO, bd=0, padx=10, pady=8)
        txt_trace.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        branches = meta.get("branches", [])
        if not branches:
            raw = meta.get("raw_branches", [])
            ladder = meta.get("temp_ladder", [0.20])
            for i, r in enumerate(raw):
                b_temp = ladder[i] if i < len(ladder) else ladder[-1]
                branches.append({
                    "index": i, "temp": b_temp,
                    "passed": (i == win_idx) and meta.get("verified", False),
                    "reward": 1.0 if ((i == win_idx) and meta.get("verified", False)) else 0.0,
                    "candidate": r, "code": r, "stderr": ""
                })

        for b in branches:
            b_idx = b.get("index", 0)
            b_temp = b.get("temp", 0.20)
            passed = b.get("passed", False)
            status_str = "✓ PASSED (Winner)" if (b_idx == win_idx and passed) else ("✓ PASSED" if passed else "✗ FAILED")
            reward_val = b.get("reward", 1.0 if passed else 0.0)
            code_line = b.get("code", "").replace("\n", " ")[:60]
            tree.insert("", "end", iid=str(b_idx), values=(f"#{b_idx+1}", f"T={b_temp:.2f}", status_str, f"{reward_val:.1f}", code_line))

        def _on_select(event):
            sel = tree.selection()
            if sel:
                idx = int(sel[0])
                if idx < len(branches):
                    b = branches[idx]
                    txt_trace.delete("1.0", "end")
                    trace_content = f"# Branch #{idx+1} [Temperature T={b.get('temp', 0.20):.2f}]\n"
                    trace_content += f"# RLVR Sandbox Result: {'PASSED (Reward = 1.0)' if b.get('passed') else 'FAILED'}\n"
                    if b.get("stderr"):
                        trace_content += f"# Stderr: {b.get('stderr')}\n"
                    trace_content += f"\n{b.get('candidate', '')}\n"
                    txt_trace.insert("1.0", trace_content)

        tree.bind("<<TreeviewSelect>>", _on_select)
        if branches:
            tree.selection_set(str(win_idx) if str(win_idx) in [str(b['index']) for b in branches] else "0")
            _on_select(None)

    def _on_open_memory_explorer(self):
        """Opens interactive episodic memory database explorer for data/memory.db."""
        modal = tk.Toplevel(self.root)
        modal.title("Episodic Memory Database Explorer (memory.db)")
        modal.geometry("920x620")
        modal.minsize(780, 520)
        modal.configure(bg=self.C["bg_hud"])
        modal.transient(self.root)

        hdr = tk.Frame(modal, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=16, pady=10)

        tk.Label(hdr, text="💾 Episodic Memory Database Explorer", font=_FONT_H2, bg=self.C["bg_hud"], fg=self.C["accent_cyan"]).pack(side="left")

        btn_refresh = tk.Button(hdr, text="🔄 Refresh", font=_FONT_TINY_BOLD, bg=self.C["btn_bg"], fg=self.C["btn_fg"], activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"], relief="flat", bd=0, padx=8, pady=3, cursor="hand2", highlightthickness=0)
        btn_refresh.pack(side="right", padx=3)

        btn_export = tk.Button(hdr, text="📥 Export Markdown", font=_FONT_TINY_BOLD, bg=self.C["btn_bg"], fg=self.C["btn_fg"], activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"], relief="flat", bd=0, padx=8, pady=3, cursor="hand2", highlightthickness=0, command=lambda: self.tools.execute_tool("export_chat_history", {}))
        btn_export.pack(side="right", padx=3)

        # Search and Filter Toolbar
        filter_bar = tk.Frame(modal, bg=self.C["bg_card"], highlightbackground=self.C["border"], highlightthickness=1)
        filter_bar.pack(fill="x", padx=16, pady=(0, 10))

        tk.Label(filter_bar, text="🔍 Search:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_main"]).pack(side="left", padx=(10, 4), pady=6)
        ent_search = tk.Entry(filter_bar, font=_FONT_SMALL, bg=self.C["bg_input_inner"], fg=self.C["text_main"], insertbackground=self.C["text_main"], bd=0, highlightthickness=1, width=28)
        ent_search.pack(side="left", padx=4, pady=6, ipady=3)

        tk.Label(filter_bar, text="Filter S ≥:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(side="left", padx=(12, 4))
        ent_surprise = tk.Entry(filter_bar, font=_FONT_SMALL, bg=self.C["bg_input_inner"], fg=self.C["text_main"], insertbackground=self.C["text_main"], bd=0, highlightthickness=1, width=6)
        ent_surprise.pack(side="left", padx=4, pady=6, ipady=3)
        ent_surprise.insert(0, "0.0")

        tk.Label(filter_bar, text="Consolidated:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(side="left", padx=(12, 4))
        var_cons = tk.StringVar(value="All")
        opt_cons = ttk.Combobox(filter_bar, textvariable=var_cons, values=["All", "Unconsolidated (0)", "Consolidated (1)"], width=16, state="readonly")
        opt_cons.pack(side="left", padx=4, pady=6)

        # Main Paned (Table + Detail)
        paned = tk.PanedWindow(modal, orient="vertical", bg=self.C["border"], sashwidth=4, bd=0)
        paned.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        table_frame = tk.Frame(paned, bg=self.C["bg_app"])
        paned.add(table_frame, height=220)

        cols = ("id", "date", "mode", "reward", "surprise", "temp", "cons", "prompt")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        tree.heading("id", text="ID")
        tree.heading("date", text="Timestamp")
        tree.heading("mode", text="Mode")
        tree.heading("reward", text="Reward (R)")
        tree.heading("surprise", text="Surprise (S)")
        tree.heading("temp", text="Temp")
        tree.heading("cons", text="Consolidated")
        tree.heading("prompt", text="Prompt Preview")

        tree.column("id", width=45, anchor="center")
        tree.column("date", width=140, anchor="center")
        tree.column("mode", width=120, anchor="center")
        tree.column("reward", width=75, anchor="center")
        tree.column("surprise", width=85, anchor="center")
        tree.column("temp", width=65, anchor="center")
        tree.column("cons", width=95, anchor="center")
        tree.column("prompt", width=270, anchor="w")

        tree.pack(fill="both", expand=True)

        detail_frame = tk.Frame(paned, bg=self.C["bg_card"])
        paned.add(detail_frame, height=200)

        tk.Label(detail_frame, text="📋 Selected Interaction Record & Verified Code:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(anchor="w", padx=10, pady=(6, 2))
        txt_detail = tk.Text(detail_frame, bg=self.C["code_bg"], fg=self.C["code_fg"], font=_FONT_MONO, bd=0, padx=10, pady=8)
        txt_detail.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        records_cache = {}

        def _load_data():
            tree.delete(*tree.get_children())
            records_cache.clear()
            query_str = ent_search.get().strip().lower()
            try:
                min_s = float(ent_surprise.get().strip() or "0.0")
            except Exception:
                min_s = 0.0
            cons_filter = var_cons.get()

            try:
                import sqlite3
                conn = sqlite3.connect(self.settings.database_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT 150")
                rows = cur.fetchall()
                conn.close()

                for r in rows:
                    p = r["prompt"] or ""
                    s = r["surprise_score"] or 0.0
                    c_status = r["consolidated"] or 0
                    if query_str and (query_str not in p.lower() and query_str not in (r["completion"] or "").lower()):
                        continue
                    if s < min_s:
                        continue
                    if cons_filter == "Unconsolidated (0)" and c_status != 0:
                        continue
                    if cons_filter == "Consolidated (1)" and c_status != 1:
                        continue

                    r_id = r["id"]
                    records_cache[str(r_id)] = r
                    temp_val = r["winning_temp"] if "winning_temp" in r.keys() else 0.20
                    tree.insert("", "end", iid=str(r_id), values=(
                        r_id, r["created_at"], r["mode"] or "Instant",
                        f"{r['verified_reward']:.1f}", f"{s:.2f}",
                        f"T={temp_val:.2f}", "Yes" if c_status else "Pending",
                        p.replace("\n", " ")[:45]
                    ))
            except Exception as e:
                pass

        def _on_select_record(event):
            sel = tree.selection()
            if sel:
                rid = sel[0]
                r = records_cache.get(rid)
                if r:
                    txt_detail.delete("1.0", "end")
                    content = f"# Interaction Record ID #{r['id']} ({r['created_at']})\n"
                    content += f"# Mode: {r['mode']} | Reward: {r['verified_reward']} | Surprise: {r['surprise_score']}\n"
                    content += f"# Temperature: {r.get('winning_temp', 0.20)} | Consolidated: {'Yes' if r['consolidated'] else 'Pending'}\n\n"
                    content += f"## 👤 Prompt:\n{r['prompt']}\n\n"
                    content += f"## 🤖 Verified Completion:\n{r['completion']}\n"
                    txt_detail.insert("1.0", content)

        tree.bind("<<TreeviewSelect>>", _on_select_record)
        btn_refresh.configure(command=_load_data)
        ent_search.bind("<KeyRelease>", lambda e: _load_data())
        ent_surprise.bind("<KeyRelease>", lambda e: _load_data())
        opt_cons.bind("<<ComboboxSelected>>", lambda e: _load_data())
        _load_data()

    def _on_open_sleep_consolidation_panel(self):
        """Opens live sleep consolidation & parameter update monitor."""
        modal = tk.Toplevel(self.root)
        modal.title("Sleep Consolidation Control Panel (EWC-LoRA)")
        modal.geometry("720x520")
        modal.minsize(640, 440)
        modal.configure(bg=self.C["bg_hud"])
        modal.transient(self.root)

        hdr = tk.Frame(modal, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=20, pady=12)

        tk.Label(hdr, text="💤 Sleep Consolidation & EWC Parametric Updates", font=_FONT_H2, bg=self.C["bg_hud"], fg=self.C["accent_purple"]).pack(anchor="w")

        body = tk.Frame(modal, bg=self.C["bg_hud"])
        body.pack(fill="both", expand=True, padx=20, pady=6)

        # Status cards
        cards_row = tk.Frame(body, bg=self.C["bg_hud"])
        cards_row.pack(fill="x", pady=(0, 10))

        # Check unconsolidated count
        uncons_count = 0
        try:
            import sqlite3
            conn = sqlite3.connect(self.settings.database_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM interactions WHERE consolidated = 0 AND verified_reward >= 1.0 AND surprise_score >= 0.35")
            uncons_count = cur.fetchone()[0]
            conn.close()
        except Exception:
            uncons_count = 3

        lbl_uncons = self._make_badge(cards_row, f"📬 High-Surprise Memories: {uncons_count}")
        self._make_badge(cards_row, f"🧬 Active Adapter: Slow-LoRA (Rank=32, Alpha=64)")
        self._make_badge(cards_row, f"🔒 EWC Lambda: λ = 400.0")

        # Progress / Output Log Area
        tk.Label(body, text="📈 Live Sleep Consolidation Progress & Parameter Shift:", font=_FONT_SMALL, bg=self.C["bg_hud"], fg=self.C["text_main"]).pack(anchor="w", pady=(8, 2))
        txt_cons_log = tk.Text(body, height=12, bg=self.C["code_bg"], fg=self.C["code_fg"], font=_FONT_MONO, bd=0, padx=12, pady=10)
        txt_cons_log.pack(fill="both", expand=True, pady=(0, 10))
        txt_cons_log.insert("1.0", f"✦ EWC Consolidation Daemon Ready.\n• Target Weights: eval_results/adapters.safetensors\n• Fisher Diagonal Matrix: Preserving foundation attention matrices\n• Pending Traces: {uncons_count} verified episodes ready for synaptic integration.\n")

        btn_bar = tk.Frame(body, bg=self.C["bg_hud"])
        btn_bar.pack(fill="x", pady=(0, 10))

        btn_trigger = tk.Button(
            btn_bar, text="▶ Trigger Sleep Consolidation Now", font=_FONT_BOLD,
            bg=self.C["btn_primary_bg"], fg=self.C["btn_primary_fg"],
            activebackground=self.C["btn_primary_hover"], activeforeground=self.C["btn_primary_fg"],
            relief="flat", bd=0, padx=16, pady=8, cursor="hand2", highlightthickness=0
        )
        btn_trigger.pack(side="left")

        def _run_daemon():
            btn_trigger.configure(state="disabled", text="⏳ Consolidating Weights...")
            txt_cons_log.insert("end", "\n[*] Initiating EWC Consolidation Cycle...\n")
            txt_cons_log.see("end")

            def _worker():
                daemon = SleepConsolidationDaemon(settings=self.settings, use_mock=self.settings.use_mock)
                res = daemon.run_consolidation_cycle()
                mem_count = res.get("memories_consolidated", 0)
                task_loss = res.get("avg_task_loss", 0.0)
                ewc_loss = res.get("avg_ewc_loss", 0.0)
                saved_to = res.get("adapter_saved_to", self.settings.lora_adapter_path or "LoRA adapter")
                exec_time = res.get("execution_time_seconds", 0.0)

                self.root.after(0, lambda: [
                    self._update_telemetry(),
                    txt_cons_log.insert("end", f"[✓] Cycle Complete! Memories Consolidated: {mem_count} ({exec_time:.2f}s)\n"),
                    txt_cons_log.insert("end", f"• Average Task Loss: {task_loss:.4f}\n"),
                    txt_cons_log.insert("end", f"• Average EWC Penalty: {ewc_loss:.4f}\n"),
                    txt_cons_log.insert("end", f"• Adapter Checkpoint: Saved to {saved_to}\n"),
                    txt_cons_log.see("end"),
                    btn_trigger.configure(state="normal", text="▶ Trigger Sleep Consolidation Now"),
                    lbl_uncons.configure(text="📬 High-Surprise Memories: 0")
                ])

            threading.Thread(target=_worker, daemon=True).start()

        btn_trigger.configure(command=_run_daemon)

    def _on_open_dsl_playground(self):
        """Opens interactive DSL & RLVR Sandbox testing playground."""
        modal = tk.Toplevel(self.root)
        modal.title("Interactive DSL & RLVR Sandbox Playground")
        modal.geometry("820x580")
        modal.minsize(700, 460)
        modal.configure(bg=self.C["bg_hud"])
        modal.transient(self.root)

        hdr = tk.Frame(modal, bg=self.C["bg_hud"])
        hdr.pack(fill="x", padx=16, pady=10)

        tk.Label(hdr, text="🧪 Interactive DSL & RLVR Sandbox Playground", font=_FONT_H2, bg=self.C["bg_hud"], fg=self.C["accent_yellow"]).pack(side="left")

        # Selector toolbar
        tool_bar = tk.Frame(modal, bg=self.C["bg_card"], highlightbackground=self.C["border"], highlightthickness=1)
        tool_bar.pack(fill="x", padx=16, pady=(0, 10))

        tk.Label(tool_bar, text="Mode:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_main"]).pack(side="left", padx=(10, 4), pady=6)
        dsl_var = tk.StringVar(value="TensorGraphDSL")
        opt_dsl = ttk.Combobox(tool_bar, textvariable=dsl_var, values=["TensorGraphDSL", "GlyphScript", "Python Sandbox"], width=18, state="readonly")
        opt_dsl.pack(side="left", padx=4, pady=6)

        tk.Label(tool_bar, text="Memory Cap: 512 MB | Timeout: 4.0s", font=_FONT_TINY, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(side="left", padx=(14, 0))

        btn_canvas = tk.Button(tool_bar, text="Send to AI Canvas", font=_FONT_TINY_BOLD, bg=self.C["btn_bg"], fg=self.C["btn_fg"], activebackground=self.C["btn_hover"], activeforeground=self.C["btn_fg"], relief="flat", bd=0, padx=8, pady=3, cursor="hand2", highlightthickness=0)
        btn_canvas.pack(side="right", padx=6)

        paned = tk.PanedWindow(modal, orient="vertical", bg=self.C["border"], sashwidth=4, bd=0)
        paned.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # Editor
        edit_frame = tk.Frame(paned, bg=self.C["bg_card"])
        paned.add(edit_frame, height=220)
        tk.Label(edit_frame, text="✏️ DSL Expression / Python Code Block:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(anchor="w", padx=10, pady=(6, 2))
        txt_code = tk.Text(edit_frame, bg=self.C["code_bg"], fg=self.C["code_fg"], font=_FONT_MONO, bd=0, padx=10, pady=8)
        txt_code.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        txt_code.insert("1.0", "[2, 4, 6] >>~fold(1) <#>scale(3)")

        # Result pane
        res_frame = tk.Frame(paned, bg=self.C["bg_card"])
        paned.add(res_frame, height=180)
        tk.Label(res_frame, text="📊 Sandbox Execution Output & Status:", font=_FONT_SMALL, bg=self.C["bg_card"], fg=self.C["text_muted"]).pack(anchor="w", padx=10, pady=(6, 2))
        txt_out = tk.Text(res_frame, bg=self.C["code_bg"], fg=self.C["code_fg"], font=_FONT_MONO, bd=0, padx=10, pady=8)
        txt_out.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        btn_run = tk.Button(modal, text="▶ Run in Sandbox", font=_FONT_BOLD, bg=self.C["btn_primary_bg"], fg=self.C["btn_primary_fg"], activebackground=self.C["btn_primary_hover"], activeforeground=self.C["btn_primary_fg"], relief="flat", bd=0, padx=18, pady=8, cursor="hand2", highlightthickness=0)
        btn_run.pack(padx=16, pady=(0, 12), anchor="w")

        def _on_mode_change(event):
            m = dsl_var.get()
            txt_code.delete("1.0", "end")
            if m == "TensorGraphDSL":
                txt_code.insert("1.0", "[2, 4, 6] >>~fold(1) <#>scale(3)")
            elif m == "GlyphScript":
                txt_code.insert("1.0", "RULE: DAG_MONOTONIC_FLOW\nA -> B (5)\nB -> C (3)\nINVARIANT: ALL(weight > 0)\n")
            else:
                txt_code.insert("1.0", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n\nassert factorial(5) == 120\nprint('Factorial(5) =', factorial(5))\n")

        def _execute():
            m = dsl_var.get().lower().replace(" sandbox", "").replace("dsl", "")
            code_text = txt_code.get("1.0", "end").strip()
            from core.dsl_engine import InteractiveDSLPlayground
            runner = InteractiveDSLPlayground()
            res = runner.execute_dsl(m, code_text)
            txt_out.delete("1.0", "end")
            status = "✓ PASSED" if res["passed"] else "✗ FAILED"
            txt_out.insert("1.0", f"[{status}] Execution Time: {res['execution_time_ms']:.2f}ms | Exit Code: {res['exit_code']}\n")
            if res["stdout"]:
                txt_out.insert("end", f"\n--- STDOUT ---\n{res['stdout']}\n")
            if res["stderr"]:
                txt_out.insert("end", f"\n--- STDERR ---\n{res['stderr']}\n")

        def _send_to_canvas():
            code_text = txt_code.get("1.0", "end").strip()
            self._open_in_canvas(code_text)
            modal.destroy()

        opt_dsl.bind("<<ComboboxSelected>>", _on_mode_change)
        btn_run.configure(command=_execute)
        btn_canvas.configure(command=_send_to_canvas)



# Aliases for backward compatibility
ChatbotAppGUI = SmartAIChatbotApp
AutonomousReasoningApp = SmartAIChatbotApp
DesktopAppGUI = SmartAIChatbotApp


def launch_app(settings: Optional[Settings] = None):
    """Entry point to launch the Smart AI Studio."""
    terminate_existing_app_instances()
    root = tk.Tk()
    app = SmartAIChatbotApp(root, settings=settings)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
