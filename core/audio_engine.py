"""Local audio-generation runtime for Smart AI Studio.

Audio is deliberately separate from Pro/text reasoning. It lazily loads music/SFX
models, supports Apple-Silicon MLX backends when available, and never reports a
generation as successful unless an actual audio file was written.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable, Dict, Optional

from core.hf_transfer import format_download_error, run_hf_operation, snapshot_download


class AudioGenerationEngine:
    @staticmethod
    def _saved_lora_files(repo_id: str):
        try:
            from core.media_learning import get_saved_media_adapter
            directory = get_saved_media_adapter(repo_id)
        except Exception:
            directory = None
        if directory is None:
            return []
        files = sorted(
            str(path)
            for path in directory.rglob("*.safetensors")
            if path.is_file() and path.stat().st_size > 0
        )
        return files

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
            from core.training_memory import clear_available_mps_cache
            clear_available_mps_cache(torch)
        except Exception:
            pass
        return {"status": "unloaded"}

    def owned_process_ids(self):
        from core.stable_audio_runtime import owned_process_ids
        return owned_process_ids()

    def load_model(self, model_info: Dict[str, Any], progress_callback=None, cancel_event=None) -> Dict[str, Any]:
        info = dict(model_info or {})
        if info.get("download_unavailable_reason"):
            return {"status": "error", "error": str(info["download_unavailable_reason"])}
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "error": "Audio loading cancelled."}
        self.unload_model()
        repo_id = str(info.get("repo_id") or "").strip()
        if not repo_id:
            return {"status": "error", "error": "Audio model has no repository ID."}

        backend_hint = str(info.get("audio_backend") or "").strip().lower()
        self.repo_id = repo_id
        self.model_info = info

        def selected_source(selection=None):
            selected = selection or info
            identifier = selected.get("repo_id") or repo_id
            if not selected.get("repo_revision") or not selected.get("download_required_files"):
                return identifier
            source = snapshot_download(identifier, revision=selected["repo_revision"],
                                       allow_patterns=selected.get("download_allow_patterns"),
                                       **({"cancel_event": cancel_event} if cancel_event is not None else {}),
                                       **({"progress_callback": lambda message, percent: progress_callback(percent / 100, message)}
                                          if progress_callback is not None else {}))
            missing = [name for name in selected["download_required_files"] if not (Path(source) / name).is_file()]
            if missing:
                raise RuntimeError("Pinned audio files are missing: " + ", ".join(missing))
            return str(source)

        saved_loras = self._saved_lora_files(repo_id)

        if (
            self._is_apple_silicon()
            and backend_hint in ("mlx_audio", "mlx-audio", "moss_mlx")
            and not saved_loras
        ):
            try:
                if info.get("audio_variant") in ("small-music", "small-sfx"):
                    try:
                        from mlx_audio.music import load as load_model
                    except ImportError:
                        from mlx_audio.tts.utils import load_model
                else:
                    from mlx_audio.tts.utils import load_model
                source = selected_source()
                self.model = run_hf_operation(lambda: load_model(source))
                self.backend = "mlx_audio"
                return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
            except Exception as exc:
                return {
                    "status": "error",
                    "error": (
                        f"MLX audio runtime could not load {repo_id}: {format_download_error(exc)}. "
                        "Install/upgrade mlx-audio for this Apple-Silicon audio preset."
                    ),
                }

        if backend_hint in ("stable_audio_3", "stable-audio-3") or (
            saved_loras and str(info.get("audio_variant") or "") in ("small-music", "small-sfx")
        ):
            native_selection = info
            if saved_loras and backend_hint in ("mlx_audio", "mlx-audio", "moss_mlx"):
                # The existing learned-adapter path uses the publisher's native
                # checkpoint, not tensors converted into MLX's weight layout.
                from config.media_models import media_download_metadata
                native_repo = "stabilityai/stable-audio-3-" + str(info["audio_variant"])
                native_selection = {**media_download_metadata(native_repo), "repo_id": native_repo}
            from core.stable_audio_runtime import MODEL_REPOS, prepare_stable_audio_model, StableAudioCancelled
            if native_selection.get("repo_id") in MODEL_REPOS:
                try:
                    self.model = prepare_stable_audio_model(native_selection, lora_files=saved_loras,
                                                           cancel_event=cancel_event, progress_callback=progress_callback)
                    self.backend = "stable_audio_3_isolated"
                    return {"status": "loaded", "backend": self.backend, "repo_id": repo_id,
                            "learned_adapter_loaded": bool(saved_loras)}
                except Exception as exc:
                    cancelled = isinstance(exc, StableAudioCancelled) or (cancel_event is not None and cancel_event.is_set())
                    return {"status": "cancelled" if cancelled else "error", "error": format_download_error(exc)}
            stable_audio_error = None
            try:
                from stable_audio_3 import StableAudioModel
                variant = str(info.get("audio_variant") or "small-music")
                source = selected_source(native_selection)
                if os.path.isdir(source):
                    import torch
                    from stable_audio_3.loading_utils import load_diffusion_cond
                    model_config = self._local_audio_config(source)
                    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                    half = torch.cuda.is_available()
                    model = run_hf_operation(lambda: load_diffusion_cond(
                        model_config, str(Path(source) / "model.safetensors"), device=device, model_half=half,
                    ))
                    model.use_lora, model.lora_names = False, []
                    self.model = StableAudioModel(model, model_config, device, half)
                else:
                    self.model = run_hf_operation(lambda: StableAudioModel.from_pretrained(variant))
                if saved_loras:
                    self.model.load_lora(saved_loras)
                self.backend = "stable_audio_3"
                return {
                    "status": "loaded",
                    "backend": self.backend,
                    "repo_id": repo_id,
                    "learned_adapter_loaded": bool(saved_loras),
                }
            except Exception as exc:
                stable_audio_error = exc

            if saved_loras:
                return {
                    "status": "error",
                    "error": (
                        f"A learned Stable Audio adapter exists for {repo_id}, but the "
                        f"LoRA-capable stable_audio_3 runtime failed to load it: {stable_audio_error}"
                    ),
                }

            try:
                import torch
                from stable_audio_tools import get_pretrained_model
                source = selected_source(native_selection)
                if os.path.isdir(source):
                    from stable_audio_tools.models.factory import create_model_from_config
                    from stable_audio_tools.models.utils import load_ckpt_state_dict
                    model_config = self._local_audio_config(source)
                    model = run_hf_operation(lambda: create_model_from_config(model_config))
                    model.load_state_dict(load_ckpt_state_dict(str(Path(source) / "model.safetensors")))
                else:
                    model, model_config = run_hf_operation(lambda: get_pretrained_model(repo_id))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                self.model = (model, model_config, device)
                self.backend = "stable_audio_tools"
                return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
            except Exception as exc:
                return {
                    "status": "error",
                    "error": (
                        f"Stable Audio 3 runtime could not load {repo_id}. "
                        f"stable-audio-3 error: {format_download_error(stable_audio_error)}; "
                        f"stable-audio-tools error: {format_download_error(exc)}."
                    ),
                }

        if backend_hint in ("moss", "moss_soundeffect", "moss_soundeffect_v2"):
            try:
                import torch
                from moss_soundeffect_v2 import MossSoundEffectPipeline
                device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
                source = selected_source()
                self.model = run_hf_operation(lambda: MossSoundEffectPipeline.from_pretrained(
                    source,
                    torch_dtype=dtype,
                    device=device,
                ))
                self.backend = "moss_soundeffect_v2"
                return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
            except Exception as exc:
                return {
                    "status": "error",
                    "error": (
                        f"MOSS SoundEffect runtime could not load {repo_id}: {format_download_error(exc)}. "
                        "Install the MOSS SoundEffect v2 inference package for this preset."
                    ),
                }

        # Generic final fallback for compatible HF text-to-audio models.
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            source = selected_source()
            self.model = run_hf_operation(lambda: pipeline("text-to-audio", model=source, device=device))
            self.backend = "transformers_text_to_audio"
            return {"status": "loaded", "backend": self.backend, "repo_id": repo_id}
        except Exception as exc:
            return {
                "status": "error",
                "error": f"No compatible local audio backend could load {repo_id}: {format_download_error(exc)}",
            }

    @staticmethod
    def _local_audio_config(source):
        """Bind the publisher's text conditioner to the same pinned snapshot."""
        root = Path(source)
        config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
        conditioners = config.get("model", {}).get("conditioning", {}).get("configs", [])
        for conditioner in conditioners:
            options = conditioner.get("config") or {}
            subfolder = str(options.get("subfolder") or "")
            if conditioner.get("type") == "t5gemma" and subfolder == "t5gemma-b-b-ul2":
                if not (root / subfolder / "model.safetensors").is_file():
                    raise RuntimeError("The pinned audio text conditioner is missing")
                options["repo_id"] = str(root.resolve())
        return config

    @staticmethod
    def _wave_array(audio: Any):
        import numpy as np
        arr = audio
        if hasattr(arr, "detach"):
            arr = arr.detach().float().cpu().numpy()
        arr = np.asarray(arr)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("Audio runtime returned an empty or non-finite waveform")
        return arr

    @classmethod
    def _preview_waveform(cls, audio, sample_rate, callback, label="Completed audio waveform"):
        if callback is None:
            return
        try:
            import numpy as np
            arr = cls._wave_array(audio)
            mono = arr.mean(axis=1) if arr.ndim == 2 else arr.reshape(-1)
            indices = np.linspace(0, len(mono) - 1, min(2048, len(mono)), dtype=int)
            callback({"kind": "audio", "samples": np.clip(mono[indices], -1, 1).astype(float).tolist(),
                      "sample_rate": int(sample_rate), "duration_seconds": len(mono) / int(sample_rate),
                      "label": label})
        except Exception:
            # A preview must not discard a successfully generated audio file.
            pass

    @staticmethod
    def _preview_audio_file(path, callback):
        if callback is None:
            return
        try:
            import numpy as np
            import soundfile as sf
            with sf.SoundFile(path) as handle:
                if not len(handle):
                    return
                indices = np.linspace(0, len(handle) - 1, min(2048, len(handle)), dtype=int)
                samples = []
                for index in indices:
                    handle.seek(int(index))
                    samples.append(float(handle.read(1, dtype="float32", always_2d=True).mean()))
                callback({"kind": "audio", "samples": samples, "sample_rate": int(handle.samplerate),
                          "duration_seconds": len(handle) / handle.samplerate, "label": "Completed audio waveform"})
        except Exception:
            pass

    @classmethod
    def _save_wave_array(cls, audio: Any, sample_rate: int, output_path: str) -> None:
        import numpy as np
        arr = cls._wave_array(audio)
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
        except Exception:
            pass
        try:
            import wave
            with wave.open(output_path, "wb") as handle:
                handle.setnchannels(arr.shape[1] if arr.ndim == 2 else 1)
                handle.setsampwidth(2)
                handle.setframerate(int(sample_rate))
                handle.writeframes((np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes())
        except Exception as exc:
            raise RuntimeError(f"Could not save generated waveform: {exc}") from exc

    def generate(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        duration_seconds: float = 10.0,
        steps: int = 50,
        cfg_scale: float = 5.0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event=None,
    ) -> Dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "error": "Audio generation cancelled."}
        if self.model is None or not self.backend:
            return {"status": "error", "error": "No audio model is loaded."}
        if self.model_info.get("download_unavailable_reason"):
            return {"status": "error", "error": str(self.model_info["download_unavailable_reason"])}
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Audio generation prompt is empty."}

        output_path = os.path.abspath(
            output_path or f"generated_audio_{int(time.time())}.wav"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            if progress_callback:
                progress_callback(0.05, "Preparing audio generation")
            if self.backend == "stable_audio_3_isolated":
                from core.stable_audio_runtime import generate_stable_audio
                return generate_stable_audio(self.model, prompt, output_path, duration=duration_seconds,
                                             steps=steps, cfg_scale=cfg_scale, progress_callback=progress_callback,
                                             preview_callback=preview_callback, cancel_event=cancel_event)
            if self.backend == "mlx_audio":
                if self.model_info.get("audio_variant") in ("small-music", "small-sfx"):
                    import numpy as np
                    chunks, sample_rate = [], int(getattr(self.model, "sample_rate", 44100))
                    for result in self.model.generate(text=prompt, duration=float(duration_seconds),
                                                      steps=max(1, int(steps)), cfg_scale=float(cfg_scale)):
                        if cancel_event is not None and cancel_event.is_set():
                            return {"status": "cancelled", "error": "Audio generation cancelled."}
                        audio = getattr(result, "audio", None)
                        if audio is None:
                            raise RuntimeError("MLX audio returned a result without a waveform")
                        sample_rate = int(getattr(result, "sample_rate", sample_rate))
                        chunks.append(self._wave_array(audio))
                        self._preview_waveform(audio, sample_rate, preview_callback, "Generated audio waveform")
                    if not chunks:
                        raise RuntimeError("MLX audio returned no waveform")
                    audio = np.concatenate(chunks, axis=0)
                    self._save_wave_array(audio, sample_rate, output_path)
                    if progress_callback:
                        progress_callback(1.0, "Audio complete")
                    return {"status": "success", "path": output_path, "backend": self.backend}
                from mlx_audio.tts.generate import generate_audio
                prefix = os.path.splitext(output_path)[0]
                before = set(glob.glob(prefix + "*"))
                if progress_callback:
                    progress_callback(0.18, "Synthesizing waveform")
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
                self._preview_audio_file(final_path, preview_callback)
                if progress_callback:
                    progress_callback(1.0, "Audio complete")
                return {"status": "success", "path": final_path, "backend": self.backend}

            if self.backend == "stable_audio_3":
                if progress_callback:
                    progress_callback(0.18, "Generating music / sound")
                audio = self.model.generate(
                    prompt=prompt,
                    duration=float(duration_seconds),
                    steps=int(steps),
                    cfg_scale=float(cfg_scale),
                )
                sample_rate = int(getattr(self.model, "sample_rate", None) or getattr(self.model, "model_config", {}).get("sample_rate", 44100))
                self._preview_waveform(audio, sample_rate, preview_callback)
                self._save_wave_array(audio, sample_rate, output_path)
                if progress_callback:
                    progress_callback(1.0, "Audio complete")
                return {"status": "success", "path": output_path, "backend": self.backend}

            if self.backend == "stable_audio_tools":
                import torch
                from stable_audio_tools.inference.generation import generate_diffusion_cond
                model, model_config, device = self.model
                sample_rate = int(model_config.get("sample_rate", 44100) or 44100)
                sample_size = max(1, int(float(duration_seconds) * sample_rate))
                conditioning = [{
                    "prompt": prompt,
                    "seconds_start": 0,
                    "seconds_total": float(duration_seconds),
                }]
                if progress_callback:
                    progress_callback(0.18, "Generating music / sound")
                audio = generate_diffusion_cond(
                    model,
                    steps=max(1, int(steps)),
                    cfg_scale=float(cfg_scale),
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sample_rate=sample_rate,
                    device=device,
                )
                peak = torch.max(torch.abs(audio)).clamp_min(1e-8)
                audio = audio.to(torch.float32).div(peak).clamp(-1, 1)
                self._preview_waveform(audio, sample_rate, preview_callback)
                self._save_wave_array(audio, sample_rate, output_path)
                if progress_callback:
                    progress_callback(1.0, "Audio complete")
                return {"status": "success", "path": output_path, "backend": self.backend}

            if self.backend == "moss_soundeffect_v2":
                if progress_callback:
                    progress_callback(0.18, "Generating sound effect")
                audio = self.model(
                    prompt=prompt,
                    seconds=float(duration_seconds),
                    num_inference_steps=max(1, int(steps)),
                    cfg_scale=float(cfg_scale),
                )
                self.model.save_audio(audio, output_path)
                if not os.path.isfile(output_path):
                    return {"status": "error", "error": "MOSS backend returned without writing audio."}
                self._preview_audio_file(output_path, preview_callback)
                if progress_callback:
                    progress_callback(1.0, "Audio complete")
                return {"status": "success", "path": output_path, "backend": self.backend}

            if self.backend == "transformers_text_to_audio":
                if progress_callback:
                    progress_callback(0.18, "Generating audio")
                result = self.model(prompt)
                audio = result.get("audio") if isinstance(result, dict) else None
                rate = int(result.get("sampling_rate", 44100)) if isinstance(result, dict) else 44100
                if audio is None:
                    return {"status": "error", "error": "Transformers audio backend returned no waveform."}
                self._preview_waveform(audio, rate, preview_callback)
                self._save_wave_array(audio, rate, output_path)
                if progress_callback:
                    progress_callback(1.0, "Audio complete")
                return {"status": "success", "path": output_path, "backend": self.backend}

            return {"status": "error", "error": f"Unsupported audio backend: {self.backend}"}
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                return {"status": "cancelled", "error": "Audio generation cancelled.", "backend": self.backend}
            return {"status": "error", "error": str(exc), "backend": self.backend}
