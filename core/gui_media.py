"""Large, interactive previews of real generated media in the Tk transcript."""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox


MEDIA_GENERATION_TOOLS = frozenset({
    "media_generate", "media_pro", "media_rsi", "generate_audio",
    "generate_image", "image_gen", "draw_image", "create_image",
    "generate_image_diffusion", "diffusion_image", "realvis", "z_image", "sdxl_image",
    "generate_video_diffusion", "video_gen", "ltx_video", "wan_video", "minimax_video",
    "edit_image_rapid",
})


def media_artifacts(result):
    """Extract actual artifact records from the generation result contract."""
    records, seen = [], set()
    def visit(item):
        if not isinstance(item, dict):
            return
        kind, path = item.get("kind"), item.get("path")
        if kind in {"image", "video", "audio"} and isinstance(path, str) and path and item.get("status", "success") == "success":
            key = (kind, path)
            if key not in seen:
                seen.add(key)
                records.append(item)
        for candidate in item.get("candidates", []) if isinstance(item.get("candidates"), list) else []:
            if isinstance(candidate, dict):
                visit(candidate.get("generation", candidate))
    visit(result)
    return records


def fit_size(size, bounds):
    width, height = size
    scale = min(bounds[0] / max(1, width), bounds[1] / max(1, height), 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def open_media_file(path):
    """Use the system player for full-resolution media, including video audio."""
    path = str(Path(path).resolve(strict=True))
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform == "win32":
        import os
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])


def save_media_copy(source, target):
    """Save the original bytes, never the scaled preview."""
    source, target = Path(source).resolve(strict=True), Path(target).expanduser().absolute()
    if source != target:
        shutil.copy2(source, target)
    return target


def waveform_points(samples, width, height):
    """Show the measured amplitude envelope; never synthesize a waveform."""
    import numpy as np
    values = np.asarray(samples, dtype=float)
    if values.ndim > 1:
        values = np.abs(values).max(axis=-1)
    values = np.nan_to_num(values.reshape(-1))
    if not values.size:
        return []
    bins = min(max(1, int(width) // 3), values.size)
    envelope = [float(np.abs(chunk).max()) for chunk in np.array_split(values, bins)]
    peak = max(max(envelope), 1e-8)
    center, extent = height / 2, max(1, height / 2 - 10)
    return [(i * width / bins, center - value / peak * extent,
             i * width / bins, center + value / peak * extent)
            for i, value in enumerate(envelope)]


def read_audio_samples(path):
    try:
        import soundfile as sf
        return sf.read(str(path), always_2d=True)
    except ImportError:
        # Generated PCM WAVs remain previewable in a minimal portable install.
        import wave
        import numpy as np
        with wave.open(str(path), "rb") as audio:
            width, channels, rate = audio.getsampwidth(), audio.getnchannels(), audio.getframerate()
            if width not in {1, 2, 4}:
                raise ValueError("This audio format needs the soundfile dependency")
            values = np.frombuffer(audio.readframes(audio.getnframes()),
                                   dtype={1: "u1", 2: "<i2", 4: "<i4"}[width]).astype(float)
        if width == 1:
            values -= 128
        return (values / (2 ** (8 * width - 1))).reshape(-1, channels), rate


class MediaPreview(tk.Frame):
    """Original image or decoded video, with playback, expansion and Save As."""

    def __init__(self, parent, path, kind, colors, *, bounds=(880, 620), expandable=True):
        super().__init__(parent, bg=colors.get("bg_card", "#172033"))
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file() or not self.path.stat().st_size:
            raise ValueError("The generator did not produce a nonempty media file")
        self.kind, self.colors, self.bounds = kind, colors, bounds
        self._reader = None
        self._job = None
        self._playing = False
        self._frame_index = 0
        self._duration = 0.0
        self._fps = 24.0
        self._updating_slider = False
        self._programmatic_position = None
        self._photo = None
        self._source_image = None
        self.surface = tk.Label(self, bg=colors.get("bg_chat", "#080c14"), cursor="hand2")
        self.surface.pack(fill="both", expand=True)
        self.controls = tk.Frame(self, bg=self["bg"])
        self.controls.pack(fill="x", pady=(8, 0))
        if kind == "video":
            import imageio.v2 as imageio
            self._reader = imageio.get_reader(str(self.path), format="ffmpeg")
            try:
                metadata = self._reader.get_meta_data()
                self._fps = max(1.0, float(metadata.get("fps") or 24.0))
                duration = float(metadata.get("duration") or 0)
                self._duration = duration if math.isfinite(duration) else 0.0
                self._show_video_frame(0)
            except Exception:
                self._reader.close()
                self._reader = None
                raise
            self.play_button = self._button("▶ Play", self.toggle_playback)
            self.surface.bind("<Button-1>", lambda _event: self.toggle_playback())
            self.position = tk.Scale(self.controls, from_=0, to=max(self._duration, 0.1),
                                     resolution=0.01, orient="horizontal", showvalue=False,
                                     bg=self["bg"], fg=colors.get("text_main", "white"),
                                     highlightthickness=0, command=self._seek)
            self.position.pack(side="left", fill="x", expand=True, padx=8)
        elif kind == "image":
            from PIL import Image, ImageOps
            with Image.open(self.path) as image:
                self._show_image(ImageOps.exif_transpose(image).convert("RGB"))
            self.surface.bind("<Button-1>", lambda _event: self.expand())
        elif kind == "audio":
            samples, sample_rate = read_audio_samples(self.path)
            self._duration = len(samples) / max(1, sample_rate)
            self.surface.destroy()
            self.surface = tk.Canvas(self, width=min(bounds[0], 880), height=150,
                                     bg=colors.get("bg_chat", "#080c14"), highlightthickness=0)
            self.surface.pack(fill="x", before=self.controls)
            for points in waveform_points(samples, min(bounds[0], 880), 150):
                self.surface.create_line(*points, fill=colors.get("accent_cyan", "#38bdf8"), width=2)
            self.surface.bind("<Button-1>", lambda _event: open_media_file(self.path))
            self._button("▶ Play audio", lambda: open_media_file(self.path))
        else:
            raise ValueError("Unsupported media preview type")
        if expandable and kind in {"image", "video"}:
            self._button("Expand", self.expand)
        self._button("Open original", lambda: open_media_file(self.path))
        self._button("Save As…", self.save_as)
        self.caption = tk.Label(self, text=self._caption(), anchor="w", bg=self["bg"],
                                fg=colors.get("text_muted", "#94a3b8"))
        self.caption.pack(fill="x", pady=(6, 0))
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Save As…", command=self.save_as)
        menu.add_command(label="Open original", command=lambda: open_media_file(self.path))
        self.surface.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))
        self.surface.bind("<Control-Button-1>", lambda event: menu.tk_popup(event.x_root, event.y_root))
        self.bind("<Destroy>", self._destroyed, add="+")

    def _button(self, text, command):
        button = tk.Button(self.controls, text=text, command=command, cursor="hand2")
        button.pack(side="left", padx=(0, 6))
        return button

    def _caption(self):
        detail = ""
        if self._source_image is not None:
            detail = f"{self._source_image.width} × {self._source_image.height}"
        if self._duration:
            detail += f"  ·  {self._duration:.1f}s"
        return f"{self.path.name}  ·  {detail}".rstrip(" ·")

    def _show_image(self, image):
        from PIL import Image, ImageTk
        self._source_image = image.copy()
        preview = image.resize(fit_size(image.size, self.bounds), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(preview, master=self)
        self.surface.configure(image=self._photo)

    def _show_video_frame(self, index):
        from PIL import Image
        frame = self._reader.get_data(index)
        self._frame_index = index
        self._show_image(Image.fromarray(frame))
        if hasattr(self, "position"):
            self._updating_slider = True
            self.position.set(index / self._fps)
            # Tk invokes Scale's command at idle, after this method returns.
            # Remember its rounded value so that callback cannot seek backward
            # (e.g. frame 1 at 16 fps -> 0.06 seconds -> int(0.06 * 16) == 0).
            self._programmatic_position = float(self.position.get())
            self._updating_slider = False

    def toggle_playback(self):
        self._playing = not self._playing
        self.play_button.configure(text="❚❚ Pause" if self._playing else "▶ Play")
        if self._playing:
            if self._duration and self._frame_index + 1 >= self._duration * self._fps:
                self._show_video_frame(0)
            self._tick()
        elif self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _tick(self):
        self._job = None
        if not self._playing or self._reader is None:
            return
        start = time.monotonic()
        try:
            self._show_video_frame(self._frame_index + 1)
        except (IndexError, StopIteration):
            self._playing = False
            self.play_button.configure(text="▶ Replay")
            return
        except Exception as error:
            self._playing = False
            self.play_button.configure(text="▶ Play")
            self.caption.configure(text=f"Playback stopped: {error}. Use Open original.")
            return
        delay = max(1, round(1000 / self._fps - (time.monotonic() - start) * 1000))
        self._job = self.after(delay, self._tick)

    def _seek(self, seconds):
        if not self._updating_slider and self._reader is not None:
            seconds = float(seconds)
            if self._programmatic_position is not None and math.isclose(
                    seconds, self._programmatic_position, abs_tol=1e-9):
                return
            try:
                index = max(0, min(int(seconds * self._fps), max(0, int(self._duration * self._fps) - 1)))
                self._show_video_frame(index)
            except (IndexError, StopIteration):
                pass

    def save_as(self):
        suffix = self.path.suffix.lower()
        target = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title="Save " + self.kind,
                    initialfile=self.path.name, defaultextension=suffix,
                    filetypes=[(self.kind.title(), "*" + suffix), ("All files", "*.*")])
        if target:
            try:
                save_media_copy(self.path, target)
            except OSError as error:
                messagebox.showerror("Save media", str(error), parent=self.winfo_toplevel())

    def expand(self):
        window = tk.Toplevel(self)
        window.title(self.path.name)
        bounds = (max(320, int(window.winfo_screenwidth() * 0.88)),
                  max(240, int(window.winfo_screenheight() * 0.80)))
        try:
            viewer = MediaPreview(window, self.path, self.kind, self.colors, bounds=bounds, expandable=False)
            viewer.pack(fill="both", expand=True, padx=12, pady=12)
        except Exception:
            window.destroy()
            raise
        return window

    def _destroyed(self, event):
        if event.widget is not self:
            return
        self._playing = False
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None
