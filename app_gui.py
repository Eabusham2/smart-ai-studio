import os
import psutil
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from run_studio_complete import UnifiedMasterEngine, EngineSettings

class AppGUIDashboard:
    def __init__(self, engine: UnifiedMasterEngine):
        self.engine = engine
        self.root = None

    def launch(self):
        self.root = tk.Tk()
        self.root.title("Smart AI Studio: Continuous Learning & OGP Monitor")
        self.root.geometry("860x620")
        self.root.configure(bg="#1E1E2E")

        title = tk.Label(self.root, text="⚡ SMART AI STUDIO: CONTINUOUS LEARNING OGP DAEMON", font=("Helvetica", 14, "bold"), fg="#A6E3A1", bg="#1E1E2E")
        title.pack(pady=10)

        card = tk.Frame(self.root, bg="#313244", padx=10, pady=10)
        card.pack(fill="x", padx=15, pady=5)

        self.lbl_ogp = tk.Label(card, text="OGP Daemon: RUNNING (Zero Drift)", font=("Courier", 11, "bold"), fg="#89B4FA", bg="#313244")
        self.lbl_ogp.grid(row=0, column=0, sticky="w", padx=10, pady=4)

        self.lbl_ram = tk.Label(card, text="RAM RSS: 0.00 GB / 9.0 GB", font=("Courier", 11), fg="#F38BA8", bg="#313244")
        self.lbl_ram.grid(row=0, column=1, sticky="w", padx=10, pady=4)

        self.lbl_expert = tk.Label(card, text="Active Expert: [system]", font=("Courier", 11), fg="#CBA6F7", bg="#313244")
        self.lbl_expert.grid(row=1, column=0, sticky="w", padx=10, pady=4)

        self.lbl_ortho = tk.Label(card, text="Orthogonal Safety: 0.00e+00", font=("Courier", 11), fg="#F9E2AF", bg="#313244")
        self.lbl_ortho.grid(row=1, column=1, sticky="w", padx=10, pady=4)

        self.lbl_queue = tk.Label(card, text="Consolidation Queue: 0 items | Loss: 0.0000", font=("Courier", 10), fg="#A6ADC8", bg="#313244")
        self.lbl_queue.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=4)

        log_frame = tk.LabelFrame(self.root, text="Continuous Learning Telemetry Stream", font=("Helvetica", 10, "bold"), fg="#CDD6F4", bg="#1E1E2E", padx=8, pady=8)
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.txt_logs = scrolledtext.ScrolledText(log_frame, bg="#11111B", fg="#A6ADC8", font=("Courier", 10))
        self.txt_logs.pack(fill="both", expand=True)

        bottom = tk.Frame(self.root, bg="#1E1E2E")
        bottom.pack(fill="x", padx=15, pady=10)

        self.entry_query = tk.Entry(bottom, font=("Helvetica", 11), bg="#313244", fg="#CDD6F4", insertbackground="white")
        self.entry_query.insert(0, "Evaluate TensorGraphDSL: [1, 2, 3] >>~fold(1) <#>scale(2)")
        self.entry_query.pack(side="left", fill="x", expand=True, padx=(0, 10))

        btn = tk.Button(bottom, text="Run Query", font=("Helvetica", 10, "bold"), bg="#89B4FA", fg="#11111B", command=self._on_submit)
        btn.pack(side="right")

        self._tick()
        self.root.mainloop()

    def _tick(self):
        rss = psutil.Process().memory_info().rss / (1024 ** 3)
        self.lbl_ram.config(text=f"RAM RSS: {rss:.2f} GB / 9.0 GB")
        if self.engine.moe_router:
            self.lbl_expert.config(text=f"Active Expert: [{self.engine.moe_router.active_expert}]")
        if self.engine.ogp_daemon:
            q_len = getattr(self.engine.ogp_daemon, "queue_length", 0)
            loss = getattr(self.engine.ogp_daemon, "last_loss", 0.0)
            overlap = getattr(self.engine.ogp_daemon, "last_ortho_overlap", 0.0)
            self.lbl_ortho.config(text=f"Orthogonal Safety: {overlap:.2e} (<g, m_j> = 0)")
            self.lbl_queue.config(text=f"Consolidation Queue: {q_len} items | Loss: {loss:.4f} | Total: {self.engine.ogp_daemon.total_consolidations}")
        if self.root:
            self.root.after(1000, self._tick)

    def _on_submit(self):
        q = self.entry_query.get()
        if self.engine.moe_router:
            exp = self.engine.moe_router.route_prompt(q)
            self.txt_logs.insert(tk.END, f"{time.strftime('%H:%M:%S')} [Routed MoE: {exp}] {q}\n")
        out = self.engine.generate(q)
        self.txt_logs.insert(tk.END, f"{time.strftime('%H:%M:%S')} [Output]: {out}\n")
        self.txt_logs.see(tk.END)

if __name__ == "__main__":
    settings = EngineSettings(enable_awake_ogp_daemon=True)
    engine = UnifiedMasterEngine(settings)
    app = AppGUIDashboard(engine)
    app.launch()
