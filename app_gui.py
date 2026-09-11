"""Smart AI Studio UI plus continual-learning status controls."""
from __future__ import annotations
import platform
import tkinter as tk
import app_gui_base as _ui_base
from app_gui_base import *
from app_gui_base import SmartAIChatbotApp as _BaseSmartAIChatbotApp

_FONT_TINY_BOLD = _ui_base._FONT_TINY_BOLD
_FONT_TINY = _ui_base._FONT_TINY
_FONT_SMALL = _ui_base._FONT_SMALL
_FONT_H2 = _ui_base._FONT_H2
_FONT_MONO = _ui_base._FONT_MONO


class SmartAIChatbotApp(_BaseSmartAIChatbotApp):
    def __init__(self, root: tk.Tk, settings=None):
        super().__init__(root, settings=settings)
        self._install_continual_learning_status()
        try:self.btn_sleep_panel.configure(text="🧬 OGP Learn")
        except Exception:pass

    def _install_continual_learning_status(self):
        try:
            self._ogp_enabled_var = tk.BooleanVar(value=bool(getattr(self.settings, "enable_awake_ogp_daemon", True)))
            self.continual_status_frame = tk.Frame(self.hud_bar, bg=self.C["bg_hud"])
            self.continual_status_frame.pack(side="right", padx=(2, 6), pady=2)
            self.lbl_continual_status = tk.Label(self.continual_status_frame,text="🧬 OGP: standby",font=_FONT_TINY_BOLD,bg=self.C["badge_bg"],fg=self.C["badge_fg"],padx=7,pady=3)
            self.lbl_continual_status.pack(side="left", padx=(0, 2))
            self.chk_continual_learning = tk.Checkbutton(self.continual_status_frame,text="Continual",variable=self._ogp_enabled_var,command=self._toggle_continual_learning,bg=self.C["bg_hud"],fg=self.C["text_main"],activebackground=self.C["bg_hud"],activeforeground=self.C["text_main"],selectcolor=self.C["bg_card"],font=_FONT_TINY,bd=0,highlightthickness=0)
            self.chk_continual_learning.pack(side="left")
            self._refresh_continual_status()
        except Exception:pass

    def _on_open_sleep_consolidation_panel(self):
        modal=tk.Toplevel(self.root);modal.title("OGP Continual Learning");modal.geometry("620x420");modal.configure(bg=self.C["bg_hud"]);modal.transient(self.root)
        tk.Label(modal,text="🧬 Orthogonal Gradient Continual Learning",font=_FONT_H2,bg=self.C["bg_hud"],fg=self.C["accent_purple"]).pack(anchor="w",padx=18,pady=(16,6))
        tk.Label(modal,text="Verified high-surprise memories → shadow LoRA → baseline OGP projection → atomic promotion",font=_FONT_SMALL,bg=self.C["bg_hud"],fg=self.C["text_muted"],wraplength=570,justify="left").pack(anchor="w",padx=18,pady=(0,10))
        status=tk.Label(modal,text="",font=_FONT_MONO,bg=self.C["code_bg"],fg=self.C["code_fg"],justify="left",anchor="nw",padx=12,pady=12);status.pack(fill="both",expand=True,padx=18,pady=8)
        var=tk.BooleanVar(value=bool(getattr(self.settings,"enable_awake_ogp_daemon",True)))
        controls=tk.Frame(modal,bg=self.C["bg_hud"]);controls.pack(fill="x",padx=18,pady=(4,14))
        def toggle():
            enabled=bool(var.get());self.engine.set_awake_ogp_enabled(enabled);self._ogp_enabled_var.set(enabled);self._refresh_continual_status(reschedule=False)
        tk.Checkbutton(controls,text="Enable non-stop awake OGP learning",variable=var,command=toggle,bg=self.C["bg_hud"],fg=self.C["text_main"],selectcolor=self.C["bg_card"],activebackground=self.C["bg_hud"],activeforeground=self.C["text_main"]).pack(side="left")
        def refresh():
            daemon=getattr(self.engine,"ogp_daemon",None);st=daemon.status() if daemon is not None else {"running":False}
            overlap=float(st.get("orthogonal_overlap",0.0) or 0.0);safe=bool(st.get("anchor_count",0)) and overlap<=max(float(getattr(self.settings,"ogp_ortho_tolerance",1e-5))*10,1e-4)
            text=(f"Running: {bool(st.get('running'))}\n" f"Queue: {int(st.get('queue_length',0) or 0)} verified memories\n" f"Baseline anchor vectors: {int(st.get('anchor_count',0) or 0)}\n" f"Last projected loss: {float(st.get('last_loss',0.0) or 0.0):.6f}\n" f"Max anchor overlap: {overlap:.3e}\n" f"Orthogonal safety: {'PASS' if safe else 'WAITING/BLOCKED'}\n" f"Updates applied: {int(st.get('total_consolidations',0) or 0)}\n" f"Last error: {st.get('last_error') or st.get('anchor_error') or 'None'}")
            try:status.configure(text=text)
            except Exception:return
            try:
                if modal.winfo_exists():modal.after(1000,refresh)
            except Exception:pass
        refresh()

    def _toggle_continual_learning(self):
        enabled = bool(getattr(self, "_ogp_enabled_var", None).get()) if hasattr(self, "_ogp_enabled_var") else False
        try:self.engine.set_awake_ogp_enabled(enabled)
        except Exception:
            try:self.settings.enable_awake_ogp_daemon = enabled
            except Exception:pass
        self._refresh_continual_status(reschedule=False)

    def _refresh_continual_status(self, reschedule=True):
        try:
            status = {"running": False};daemon = getattr(self.engine, "ogp_daemon", None)
            if daemon is not None:
                try:status = daemon.status()
                except Exception:status = {"running": bool(daemon.is_alive())}
            expert = getattr(getattr(self.engine, "moe_lora_manager", None), "active_domain", None);enabled = bool(getattr(self.settings, "enable_awake_ogp_daemon", True));backend = getattr(self.engine, "active_backend", None)
            if status.get("running") and status.get("anchor_error"):text=f"🧬 OGP: blocked • {status.get('anchor_error')}";fg=self.C["accent_red"]
            elif status.get("running"):overlap=float(status.get("orthogonal_overlap",0.0) or 0.0);text=f"🧬 OGP: active • {expert or 'system'} • {overlap:.1e}";fg=self.C["accent_green"]
            elif not enabled:text="🧬 OGP: off";fg=self.C["text_muted"]
            elif backend and backend != "mlx":text=f"🧬 OGP: standby • {backend}";fg=self.C["accent_yellow"]
            else:text="🧬 OGP: standby";fg=self.C["accent_yellow"]
            self.lbl_continual_status.configure(text=text,bg=self.C["badge_bg"],fg=fg)
            self.chk_continual_learning.configure(bg=self.C["bg_hud"],fg=self.C["text_main"],activebackground=self.C["bg_hud"],activeforeground=self.C["text_main"],selectcolor=self.C["bg_card"])
        except Exception:pass
        if reschedule:
            try:
                if self.root.winfo_exists():self.root.after(1000, self._refresh_continual_status)
            except Exception:pass

    def _update_telemetry(self, tps: float = 0.0):
        try:
            backend=getattr(self.engine,"mlx_backend",None);measured=float(getattr(backend,"last_tok_per_sec",0.0) or 0.0)
            if measured>0:tps=measured
        except Exception:pass
        try:
            latest=getattr(self.engine,"last_result_metadata",None)
            if isinstance(latest,dict):self.last_metadata=latest
        except Exception:pass
        super()._update_telemetry(tps);self._refresh_continual_status(reschedule=False)

    def _update_resource_view_metrics(self):
        super()._update_resource_view_metrics()
        try:
            backend=getattr(self.engine,"active_backend",None) or getattr(self.settings,"backend","auto")
            self.lbl_res_arch.configure(text=f"• Platform: {platform.system()} {platform.machine()} • Backend: {backend}")
        except Exception:pass

ChatbotAppGUI = SmartAIChatbotApp
AutonomousReasoningApp = SmartAIChatbotApp
DesktopAppGUI = SmartAIChatbotApp

def launch_app(settings=None):
    terminate_existing_app_instances();root=tk.Tk();SmartAIChatbotApp(root, settings=settings);root.mainloop()

if __name__ == "__main__":launch_app()
