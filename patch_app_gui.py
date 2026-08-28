import re

with open("app_gui.py", "r") as f:
    code = f.read()

# 1. Add conversation_list to __init__
code = code.replace('self.prompt_queue: List[Tuple[str, str]] = []', 'self.prompt_queue: List[Tuple[str, str]] = []\n        self.conversation_list = []')

# 2. Modify _build_ui for Sidebar
sidebar_code = """
        # Master PanedWindow for Sidebar + Main Content
        self.master_paned = tk.PanedWindow(self.main_container, orient="horizontal", bd=0, sashwidth=4, bg=self.C["border"])
        self.master_paned.pack(fill="both", expand=True)

        # 0. SIDEBAR
        self.sidebar_frame = tk.Frame(self.master_paned, bg=self.C["bg_hud"], width=220)
        self.master_paned.add(self.sidebar_frame, stretch="never", minsize=220)
        
        self.btn_new_chat = tk.Button(self.sidebar_frame, text="+ New Chat", bg=self.C["bg_card"], fg=self.C["text_main"], command=self._on_new_chat)
        self.btn_new_chat.pack(fill="x", padx=8, pady=8)
        
        self.sidebar_list_frame = tk.Frame(self.sidebar_frame, bg=self.C["bg_hud"])
        self.sidebar_list_frame.pack(fill="both", expand=True, padx=8)
        
        self.btn_settings = tk.Button(self.sidebar_frame, text="⚙ Settings", bg=self.C["bg_hud"], fg=self.C["text_muted"])
        self.btn_settings.pack(fill="x", side="bottom", padx=8, pady=8)

        self.main_content_frame = tk.Frame(self.master_paned, bg=self.C["bg_app"])
        self.master_paned.add(self.main_content_frame, stretch="always")

        # 1. TOP HUD BAR (Telemetry Badges, Folder, Canvas Toggle, Export)
"""
code = re.sub(r'# 1\. TOP HUD BAR \(Telemetry Badges, Folder, Canvas Toggle, Export\)', sidebar_code, code)
code = code.replace('self.hud_bar = tk.Frame(self.main_container, bg=self.C["bg_hud"], height=52)', 'self.hud_bar = tk.Frame(self.main_content_frame, bg=self.C["bg_hud"], height=52)')
code = code.replace('self.res_drawer = tk.Frame(\n            self.main_container', 'self.res_drawer = tk.Frame(\n            self.main_content_frame')
code = code.replace('self.tab_bar = tk.Frame(self.main_container', 'self.tab_bar = tk.Frame(self.main_content_frame')
code = code.replace('self.input_container = tk.Frame(\n            self.main_container', 'self.input_container = tk.Frame(\n            self.main_content_frame')
code = code.replace('self.attachment_bar = tk.Frame(self.main_container', 'self.attachment_bar = tk.Frame(self.main_content_frame')
code = code.replace('self.queue_bar = tk.Frame(self.main_container', 'self.queue_bar = tk.Frame(self.main_content_frame')
code = code.replace('self.content_paned = tk.PanedWindow(\n            self.main_container', 'self.content_paned = tk.PanedWindow(\n            self.main_content_frame')

# Update Conversation list ui method
code += """
    def _update_conversation_sidebar(self):
        for widget in self.sidebar_list_frame.winfo_children():
            widget.destroy()
        for conv in reversed(self.conversation_list):
            btn = tk.Button(self.sidebar_list_frame, text=conv["title"][:20], bg=self.C["bg_card"], fg=self.C["text_muted"], command=self._on_clear_chat, anchor="w", relief="flat")
            btn.pack(fill="x", pady=2)
"""

# Modify HUD
hud_replace = """
        self.btn_sidebar_toggle = tk.Button(brand_frame, text="☰", font=_FONT_TITLE, bg=self.C["bg_hud"], fg=self.C["text_main"], relief="flat", bd=0, command=lambda: self._toggle_sidebar())
        self.btn_sidebar_toggle.pack(side="left", padx=(0, 8))
        
        tk.Label(brand_frame, text="✦ Smart AI", font=_FONT_TITLE, bg=self.C["bg_hud"], fg=self.C["accent_cyan"]).pack(side="left", padx=(0, 8))
"""
code = re.sub(r'tk\.Label\(\n\s+brand_frame, text="✦ Smart AI".*?\.pack\(side="left", padx=\(0, 8\)\)', hud_replace, code, flags=re.DOTALL)

# Modify Input Area
input_replace = """
        self.input_container = tk.Frame(self.main_content_frame, bg=self.C["bg_input"], highlightbackground=self.C["border"], highlightthickness=1)
        self.input_container.pack(fill="x", side="bottom", padx=24, pady=(0, 14))

        # Modern Layout
        btn_attach = tk.Button(self.input_container, text="📎", font=(_FONT_FAMILY, 14), bg=self.C["bg_input"], fg=self.C["text_muted"], relief="flat", bd=0, command=self._on_upload_file)
        btn_attach.pack(side="left", padx=8, pady=8)

        self.txt_input = tk.Text(self.input_container, height=3, bg=self.C["bg_input"], fg=self.C["text_main"], insertbackground="#ffffff", font=_FONT_MAIN, bd=0, padx=14, pady=8, highlightthickness=0, wrap="word")
        self.txt_input.insert("1.0", "Message Smart AI...")
        self.txt_input.pack(fill="both", side="left", expand=True, padx=4, pady=8)
        self.txt_input.bind("<FocusIn>", lambda e: self.txt_input.delete("1.0", "end") if self.txt_input.get("1.0", "end-1c") == "Message Smart AI..." else None)

        # Action Buttons
        btn_box = tk.Frame(self.input_container, bg=self.C["bg_input"])
        btn_box.pack(side="right", padx=8, pady=8)

        self.btn_send = tk.Button(btn_box, text="↑", font=_FONT_BOLD, bg=self.C["bg_card"], fg="#ffffff", relief="flat", bd=0, command=self._on_send_message)
        self.btn_send.pack(side="top")

        self.btn_stop = tk.Button(btn_box, text="■", font=_FONT_BOLD, bg=self.C["bg_card"], fg=self.C["text_muted"], relief="flat", bd=0, command=self._on_stop_generation)
        
        self.btn_steer = tk.Button(btn_box, text="🎯", font=_FONT_BOLD, bg=self.C["bg_card"], fg=self.C["accent_purple"], relief="flat", bd=0, command=self._on_steer_button_pressed)
        
        lbl_hints = tk.Label(self.main_content_frame, text="↵ Send • ⌘↵ Steer • ⇧↵ New line", font=_FONT_TINY, bg=self.C["bg_app"], fg=self.C["text_muted"])
        lbl_hints.pack(side="bottom", pady=4)
"""
code = re.sub(r'self\.input_container = tk\.Frame\(.*?self\.btn_stop\.pack\(side="top"\)', input_replace, code, flags=re.DOTALL)

with open("app_gui.py", "w") as f:
    f.write(code)

