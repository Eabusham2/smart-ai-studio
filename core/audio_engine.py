"""Local audio-generation runtime for Smart AI Studio.

Audio is deliberately separate from Pro/text reasoning. It lazily loads music/SFX
models, supports Apple-Silicon MLX backends when available, and never reports a
generation as successful unless an actual audio file was written.
"""
from __future__ import annotations

import glob
import os
import platform
import time
from typing import Any, Dict, Optional


class AudioGenerationEngine:
    def __init__(self) -> None:
        self.model = None
        self.backend = ""
        self.repo_id = ""
        self.model_info: Dict[str, Any] = {}

    @staticmethod
    def _is_apple_silicon() -> bool:
        return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")

    def unload_model(self) -> Dict[str, Any]:
        self.model = None
        self.backend = ""
        self.repo_id = ""
        self.model_info = {}
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        return {"status": "unloaded"}

    def load_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        self.unload_model()
        info = dict(model_info or {})
        repo_id = str(info.get("repo_id") or "").strip()
        if not repo_id:
            return {"status": "error", "error": "Audio model has no repository ID."}

        backend_hint = str(info.get("audio_backend") or "").strip().lower()
        self.repo_id = repo_id
        self.model_info = info

        if self._is_apple_silicon() and backend_hint in ("mlx_audio", "mlx-audio", "moss_mlx"):
            try:
                from mlx_audio.tts.utils import load_model
                self.model = load_model(repo_id)
                self.backend = "mlx_audio"
                return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
            except Exception as exc:
                return {
                    "status": "error",
                    "error": (
                        f"MLX audio runtime could not load {repo_id}: {exc}. "
                        "Install/upgrade mlx-audio for this Apple-Silicon audio preset."
                    ),
                }

        if backend_hint in ("stable_audio_3", "stable-audio-3"):
            try:
                from stable_audio_3 import StableAudioModel
                variant = str(info.get("audio_variant") or "small-music")
                self.model = StableAudioModel.from_pretrained(variant)
                self.backend = "stable_audio_3"
                return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
            except Exception as exc:
                return {
                    "status": "error",
                    "error": (
                        f"Stable Audio 3 runtime could not load {repo_id}: {exc}. "
                        "Install the Stability AI stable-audio-3 runtime."
                    ),
                }

        if backend_hint in ("moss", "moss_soundeffect", "moss_soundeffect_v2"):
            try:
                import torch
                from moss_soundeffect_v2 import MossSoundEffectPipeline
                device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
                self.model = MossSoundEffectPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=dtype,
                    device=device,
                )
                self.backend = "moss_soundeffect_v2"
                return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
            except Exception as exc:
                return {
                    "status": "error",
                    "error": (
                        f"MOSS SoundEffect runtime could not load {repo_id}: {exc}. "
                        "Install the MOSS SoundEffect v2 inference package for this preset."
                    ),
                }

        # Generic final fallback for compatible HF text-to-audio models.
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.model = pipeline("text-to-audio", model=repo_id, device=device)
            self.backend = "transformers_text_to_audio"
            return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
        except Exception as exc:
            return {
                "status": "error",
                "error": f"No compatible local audio backend could load {repo_id}: {exc}",
            }

    @staticmethod
    def _save_wave_array(audio: Any, sample_rate: int, output_path: str) -> None:
        import numpy as np
        arr = audio
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().float().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(arr)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        try:
            import soundfile as sf
            sf.write(output_path, arr, int(sample_rate))
            return
        except Exception:
            pass
        try:
            import scipy.io.wavfile
            clipped = np.clip(arr, -1.0, 1.0)
            scipy.io.wavfile.write(output_path, int(sample_rate), (clipped * 32767).astype(np.int16))
            return
        except Exception as exc:
            raise RuntimeError(f"Could not save generated waveform; install soundfile: {exc}") from exc

    def generate(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        duration_seconds: float = 10.0,
        steps: int = 50,
        cfg_scale: float = 5.0,
    ) -> Dict[str, Any]:
        if self.model is None or not self.backend:
            return {"status": "error", "error": "No audio model is loaded."}
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Audio generation prompt is empty."}

        output_path = os.path.abspath(
            output_path or f"generated_audio_{int(time.time())}.wav"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            if self.backend == "mlx_audio":
                from mlx_audio.tts.generate import generate_audio
                prefix = os.path.splitext(output_path)[0]
                before = set(glob.glob(prefix + "*"))
                try:
                    result = generate_audio(model=self.model, text=prompt, file_prefix=prefix)
                except TypeError:
                    result = generate_audio(model=self.model, text=prompt, ref_audio=None, file_prefix=prefix)

                candidates = [p for p in glob.glob(prefix + "*") if p not in before and os.path.isfile(p)]
                if os.path.isfile(output_path):
                    final_path = output_path
                elif candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    final_path = candidates[0]
                elif isinstance(result, str) and os.path.isfile(result):
                    final_path = os.path.abspath(result)
                else:
                    return {
                        "status": "error",
                        "error": "MLX audio backend returned without writing an audio file.",
                    }
                return {"status": "success", "path": final_path, "backend": self.backend}

            if self.backend == "stable_audio_3":
                audio = self.model.generate(
                    prompt=prompt,
                    duration=float(duration_seconds),
                    steps=int(steps),
                    cfg_scale=float(cfg_scale),
                )
                sample_rate = int(getattr(self.model, "sample_rate", 44100) or 44100)
                self._save_wave_array(audio, sample_rate, output_path)
                return {"status": "success", "path": output_path, "backend": self.backend}

            if self.backend == "moss_soundeffect_v2":
                audio = self.model(
                    prompt=prompt,
                    seconds=float(duration_seconds),
                    num_inference_steps=max(1, int(steps)),
                    cfg_scale=float(cfg_scale),
                )
                self.model.save_audio(audio, output_path)
                if not os.path.isfile(output_path):
                    return {"status": "error", "error": "MOSS backend returned without writing audio."}
                return {"status": "success", "path": output_path, "backend": self.backend}

            if self.backend == "transformers_text_to_audio":
                result = self.model(prompt)
                audio = result.get("audio") if isinstance(result, dict) else None
                rate = int(result.get("sampling_rate", 44100)) if isinstance(result, dict) else 44100
                if audio is None:
                    return {"status": "error", "error": "Transformers audio backend returned no waveform."}
                self._save_wave_array(audio, rate, output_path)
                return {"status": "success", "path": output_path, "backend": self.backend}

            return {"status": "error", "error": f"Unsupported audio backend: {self.backend}"}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "backend": self.backend}
