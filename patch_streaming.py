import re

with open("app_gui.py", "r") as f:
    code = f.read()

# Implement streaming in _process_message_thread
stream_logic = """
            if not matched:
                if self.cancel_event.is_set():
                    self.root.after(0, lambda: self._append_ai_message("⏹ Generation stopped."))
                    return

                curr_history = self.chat_history.get(self.active_tab_id, [])
                
                # Streaming Logic
                try:
                    import mlx.core as mx
                    has_mlx = True
                except ImportError:
                    has_mlx = False
                    
                if self.is_model_loaded and has_mlx and hasattr(self.engine, 'stream_solve'):
                    # Streaming path
                    
                    # 1. Typing indicator
                    self.root.after(0, lambda: self.chat_stream.configure(state="normal"))
                    self.root.after(0, lambda: self.chat_stream.insert("end", "\\n● Ternary Bonsai is thinking...\\n", "ai_msg"))
                    self.root.after(0, lambda: self.chat_stream.configure(state="disabled"))
                    self.root.after(0, lambda: self.chat_stream.see("end"))
                    
                    token_count = 0
                    streamed_text = ""
                    for token in self.engine.stream_solve(full_msg, history=curr_history, cancel_event=self.cancel_event):
                        if token_count == 0:
                            # Remove thinking indicator when first token arrives
                            def _remove_thinking():
                                self.chat_stream.configure(state="normal")
                                idx = self.chat_stream.search("● Ternary Bonsai is thinking...", "1.0", "end")
                                if idx:
                                    self.chat_stream.delete(idx, f"{idx} lineend + 1c")
                                self.chat_stream.configure(state="disabled")
                            self.root.after(0, _remove_thinking)
                            
                        streamed_text += token
                        token_count += 1
                        def _insert_tok(tok=token):
                            self.chat_stream.configure(state="normal")
                            self.chat_stream.insert("end", tok, "ai_msg")
                            self.chat_stream.configure(state="disabled")
                            self.chat_stream.see("end")
                        self.root.after(0, _insert_tok)
                        
                        duration = max(0.01, time.perf_counter() - start_time)
                        tok_per_sec = token_count / duration
                        self.root.after(0, lambda tps=tok_per_sec: self._update_telemetry(tps))
                        
                    duration_s = max(0.01, time.perf_counter() - start_time)
                    tok_per_sec = token_count / duration_s
                    response_text = streamed_text
                    thinking_text = None
                    thinking_tokens = 0
                    
                    # Once complete, clear the raw text and render proper markdown
                    def _re_render():
                        self.chat_stream.configure(state="normal")
                        # We just delete the last streamed part and use the final re-render below
                        pass
                    self.root.after(0, _re_render)
                    
                else:
                    # Batch fallback path
                    ans, meta = self.engine.solve(full_msg, history=curr_history, cancel_event=self.cancel_event)
                    response_text = ans
                    thinking_text = meta.get("thinking_text")
                    duration_s = max(0.01, time.perf_counter() - start_time)
                    thinking_tokens = len(thinking_text.split()) * 2 if thinking_text else max(24, len(ans.split()) // 2 + 18)
                    tok_per_sec = (len(ans.split()) * 2 + thinking_tokens) / duration_s
"""

code = code.replace("""
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
                tok_per_sec = (len(ans.split()) * 2 + thinking_tokens) / duration_s""", stream_logic)


# Bubbles styling
bubble_styling = """
        stream.tag_configure("user_msg", foreground="#ffffff", font=_FONT_MAIN, background=self.C["bg_user_bubble"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8, justify="right", relief="flat")
        stream.tag_configure("ai_msg", foreground=self.C["text_main"], font=_FONT_MAIN, background=self.C["bg_ai_bubble"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8, justify="left", relief="flat")
        stream.tag_configure("user_header", foreground=self.C["accent_cyan"], font=_FONT_H3, spacing1=18, spacing3=4, justify="right")
        stream.tag_configure("ai_header", foreground=self.C["accent_green"], font=_FONT_H3, spacing1=20, spacing3=4, justify="left")
"""
code = code.replace("""
        stream.tag_configure("user_header", foreground=self.C["accent_cyan"], font=_FONT_H3, spacing1=18, spacing3=4)
        stream.tag_configure("user_msg", foreground="#ffffff", font=_FONT_MAIN, background=self.C["bg_user_bubble"], lmargin1=14, lmargin2=14, rmargin=60, spacing1=8, spacing3=8)
        stream.tag_configure("ai_header", foreground=self.C["accent_green"], font=_FONT_H3, spacing1=20, spacing3=4)
        stream.tag_configure("ai_msg", foreground=self.C["text_main"], font=_FONT_MAIN, lmargin1=14, lmargin2=14, spacing1=3, spacing3=4)""", bubble_styling)

with open("app_gui.py", "w") as f:
    f.write(code)

