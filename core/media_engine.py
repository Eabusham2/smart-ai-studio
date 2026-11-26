"""Real lazy image/video generation backend for Smart AI Studio.

Selected media presets never enter the text Pro engine. This backend loads a
Diffusers-compatible pipeline only for the active generation request, writes a real
artifact, then releases it so 16 GB systems do not keep multiple media models resident.
"""
from __future__ import annotations

import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from typing import Any, Callable, Dict, Optional


def validate_media_file(path: str, kind: str) -> Dict[str, Any]:
    """Decode the written artifact before a generation route reports success."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"The {kind} pipeline did not write a nonempty output file.")
    if kind == "image":
        from PIL import Image
        formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
        expected = formats.get(source.suffix.lower())
        if not expected:
            raise ValueError("Generated images must be saved as PNG, JPEG, or WebP.")
        with Image.open(source) as image:
            if image.format != expected:
                raise ValueError("The generated image contents do not match its filename format.")
            image.verify()
        with Image.open(source) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                raise ValueError("The generated image has no pixels.")
            return {"format": expected, "width": image.width, "height": image.height}
    if kind != "video" or source.suffix.lower() != ".mp4":
        raise ValueError("Generated videos must be saved as MP4.")
    import imageio.v2 as imageio
    with imageio.get_reader(str(source), format="ffmpeg") as reader:
        # An MP4 header, a still image renamed to .mp4, or a nonexistent filename
        # is not evidence of a completed video generation.
        first, second = reader.get_data(0), reader.get_data(1)
        if first.ndim != 3 or second.shape != first.shape or min(first.shape[:2]) < 1:
            raise ValueError("The video does not contain decodable image frames.")
        fps = float(reader.get_meta_data().get("fps", 0))
        if not 0 < fps < 1000:
            raise ValueError("The generated video has an invalid frame rate.")
        return {"format": "MP4", "width": first.shape[1], "height": first.shape[0], "fps": fps}


class MediaGenerationEngine:
    def __init__(self) -> None:
        self.active_pipeline = None
        self.active_repo_id = ""
        self._process_lock = threading.Lock()
        self._owned_processes = {}

    def owned_process_ids(self):
        """Only live media workers started by this engine count toward its RAM."""
        with self._process_lock:
            processes = list(self._owned_processes.values())
        return [process.pid for process in processes
                if process.poll() is None and isinstance(process.pid, int)
                and not isinstance(process.pid, bool) and process.pid > 0]

    @staticmethod
    def _torch_device():
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def unload_model(self) -> None:
        self.active_pipeline = None
        self.active_repo_id = ""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from core.training_memory import clear_available_mps_cache
            clear_available_mps_cache(torch)
        except Exception:
            pass

    @staticmethod
    def _model_info(model_info):
        from config.media_models import media_download_metadata
        info = dict(model_info) if isinstance(model_info, dict) else {"repo_id": str(model_info)}
        catalog = media_download_metadata(str(info.get("repo_id") or ""))
        info = {**catalog, **info}
        reason = catalog.get("download_unavailable_reason") or info.get("download_unavailable_reason")
        if reason:
            raise ValueError(str(reason))
        return info

    @staticmethod
    def _check_cancelled(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Media generation cancelled.")

    def _source(self, info, progress_callback=None, cancel_event=None):
        """Resolve the selected immutable component set, never a sibling model."""
        from core.hf_transfer import hf_hub_download, snapshot_download
        self._check_cancelled(cancel_event)
        repo_id = info["repo_id"]
        local = str(info.get("model_path") or "")
        if local and Path(local).is_dir():
            source = Path(local)
            index = source / "model_index.json"
        elif info.get("download_required_files") and info.get("repo_revision"):
            source = Path(snapshot_download(
                repo_id=repo_id, revision=info["repo_revision"],
                allow_patterns=info.get("download_allow_patterns") or info["download_required_files"],
                cancel_event=cancel_event,
                progress_callback=(lambda message, _pct: progress_callback(0.05, message)) if progress_callback else None,
            ))
            index = source / "model_index.json"
        else:
            source = repo_id
            index = Path(hf_hub_download(
                repo_id=repo_id, filename="model_index.json", revision=info.get("repo_revision"),
                cancel_event=cancel_event,
            ))
        if isinstance(source, Path):
            missing = [name for name in info.get("download_required_files", [])
                       if not (source / name).is_file() or (source / name).stat().st_size == 0]
            if missing:
                raise ValueError("The selected model snapshot is incomplete: " + ", ".join(missing[:5]))
        self._check_cancelled(cancel_event)
        return str(source), index

    @staticmethod
    def _pipeline_index(index):
        data = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("The model's Diffusers pipeline manifest is invalid.")
        components = {key for key, value in data.items() if not key.startswith("_")
                      and isinstance(value, list) and len(value) == 2 and all(value)}
        if not components:
            raise ValueError(
                "This repository does not declare loadable Diffusers components. "
                "A packed MLX model needs its publisher's matching runtime; it cannot be loaded as a Diffusers pipeline."
            )
        required = {
            "Flux2KleinPipeline": {"scheduler", "text_encoder", "tokenizer", "transformer", "vae"},
            "WanPipeline": {"scheduler", "text_encoder", "tokenizer", "transformer", "vae"},
            "CogVideoXPipeline": {"scheduler", "text_encoder", "tokenizer", "transformer", "vae"},
            "StableDiffusionXLPipeline": {"scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"},
        }.get(data.get("_class_name"), set())
        missing = required - components
        if missing:
            raise ValueError("The Diffusers pipeline manifest is missing components: " + ", ".join(sorted(missing)))
        return data

    def _load_pipeline(self, model_info, progress_callback=None, cancel_event=None):
        import torch
        from diffusers import DiffusionPipeline
        from core.hf_transfer import run_hf_operation

        info = self._model_info(model_info)
        repo_id = info["repo_id"]
        source, index_path = self._source(info, progress_callback, cancel_event)
        index = self._pipeline_index(index_path)
        device = self._torch_device()
        dtype_name = info.get("diffusers_dtype", "float16") if device in ("cuda", "mps") else "float32"
        dtype = getattr(torch, dtype_name)
        kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True, "use_safetensors": True}
        if Path(source).is_dir():
            kwargs["local_files_only"] = True
        elif info.get("repo_revision"):
            kwargs["revision"] = info["repo_revision"]
        if info.get("diffusers_variant"):
            kwargs["variant"] = info["diffusers_variant"]

        def load():
            # Wan's published recipe keeps its VAE in FP32. Casting that component
            # to the transformer's BF16/FP16 dtype can corrupt decoded frames.
            if info.get("diffusers_vae_dtype"):
                import diffusers
                vae_class = getattr(diffusers, index["vae"][1])
                vae_kwargs = {key: value for key, value in kwargs.items() if key != "torch_dtype"}
                kwargs["vae"] = vae_class.from_pretrained(
                    source, subfolder="vae", torch_dtype=getattr(torch, info["diffusers_vae_dtype"]), **vae_kwargs,
                )
            return DiffusionPipeline.from_pretrained(source, **kwargs)

        pipe = run_hf_operation(load, cancel_event=cancel_event)
        from core.media_learning import apply_saved_media_adapter
        apply_saved_media_adapter(pipe, repo_id)

        # CUDA can genuinely offload model components. MPS shares unified RAM, so
        # moving the pipeline to MPS is simpler and avoids duplicate CPU/GPU copies.
        if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        if hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

        self.active_pipeline = pipe
        self.active_repo_id = repo_id
        return pipe

    @staticmethod
    def _call_pipeline(pipe, kwargs, callback):
        # Retrying every TypeError used to run inference twice, masking errors in
        # the actual model. Only pass the callback when this pipeline accepts it.
        parameters = inspect.signature(pipe.__call__).parameters
        if "callback_on_step_end" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            kwargs["callback_on_step_end"] = callback
        return pipe(**kwargs)

    @staticmethod
    def _preview(path, kind, callback):
        if callback is None:
            return
        from PIL import Image
        if kind == "image":
            with Image.open(path) as image:
                preview = image.convert("RGB")
        else:
            import imageio.v2 as imageio
            with imageio.get_reader(path, format="ffmpeg") as reader:
                preview = Image.fromarray(reader.get_data(0)).convert("RGB")
        preview.thumbnail((768, 768))
        try:
            callback({"kind": kind, "image": preview, "label": "Generated image" if kind == "image" else "Generated video frame"})
        except Exception:
            # Closing a preview widget cannot invalidate a completed media file.
            pass

    def _finish_output(self, temporary, output_path, kind, repo_id, backend,
                       progress_callback, preview_callback, cancel_event):
        details = validate_media_file(temporary, kind)
        self._check_cancelled(cancel_event)
        os.replace(temporary, output_path)
        self._preview(output_path, kind, preview_callback)
        if progress_callback:
            progress_callback(1.0, f"{kind.title()} complete")
        return {"status": "success", "path": output_path, "repo_id": repo_id, "backend": backend, **details}

    def _generator(self, model_info):
        import torch
        device = "cuda" if self._torch_device() == "cuda" else "cpu"
        return torch.Generator(device=device).manual_seed(int(model_info.get("seed", 42)))

    def generate_image(
        self,
        model_info: Dict[str, Any],
        prompt: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event=None,
    ) -> Dict[str, Any]:
        repo_id = str(model_info.get("repo_id") or "").strip()
        if not repo_id:
            return {"status": "error", "error": "Image model has no repository ID."}
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Image prompt is empty."}

        output_path = os.path.abspath(output_path or f"image_{uuid.uuid4().hex}.png")
        temporary = str(Path(output_path).with_name(f".smartai-{uuid.uuid4().hex}{Path(output_path).suffix}"))

        try:
            model_info = self._model_info(model_info)
            if Path(output_path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError("Generated images must be saved as PNG, JPEG, or WebP.")
            steps = max(1, int(model_info.get("inference_steps", 30) or 30))
            self._check_cancelled(cancel_event)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if progress_callback:
                progress_callback(0.03, "Preparing image pipeline")
            backend_hint = str(model_info.get("image_backend") or "diffusers").lower()
            if backend_hint == "bonsai_mlx":
                from core.bonsai_image_runtime import generate_bonsai_image
                result = generate_bonsai_image(model_info, prompt, temporary, steps=steps,
                                              progress_callback=progress_callback, cancel_event=cancel_event)
                if result.get("status") != "success":
                    return result
                return self._finish_output(temporary, output_path, "image", repo_id, "bonsai_mlx",
                                           progress_callback, preview_callback, cancel_event)
            if backend_hint == "mflux":
                if importlib.util.find_spec("mflux") is None:
                    return {
                        "status": "error",
                        "error": "mflux is not installed in Studio's Python runtime. Install the macOS media dependency and retry.",
                        "repo_id": repo_id,
                    }
                source, _index = self._source(model_info, progress_callback, cancel_event)
                cmd = [
                    # Console-script shebangs may refer to the CI build path
                    # after relocation; the app's owned interpreter is stable.
                    sys.executable, "-m", "mflux.models.flux2.cli.flux2_generate",
                    "--model", source,
                    "--base-model", "flux2-klein-4b",
                    "--prompt", prompt,
                    "--steps", str(max(1, steps)),
                    "--seed", str(int(model_info.get("seed", 42))),
                    "--output", temporary,
                ]
                for key, flag in (("width", "--width"), ("height", "--height"), ("guidance_scale", "--guidance")):
                    if model_info.get(key) is not None:
                        cmd.extend([flag, str(model_info[key])])
                try:
                    from core.media_learning import get_saved_media_adapter
                    adapter_dir = get_saved_media_adapter(repo_id)
                except Exception:
                    adapter_dir = None
                if adapter_dir is not None:
                    lora_files = sorted(
                        str(path)
                        for path in adapter_dir.rglob("*.safetensors")
                        if path.is_file() and path.stat().st_size > 0
                    )
                    if not lora_files:
                        return {
                            "status": "error",
                            "error": "A learned mflux adapter exists but no reloadable .safetensors LoRA was found.",
                            "repo_id": repo_id,
                        }
                    cmd.extend(["--lora-paths", lora_files[-1], "--lora-scales", "1.0"])
                if progress_callback:
                    progress_callback(0.12, "Rendering image")
                from core.hf_transfer import hf_subprocess_command
                process = subprocess.Popen(hf_subprocess_command(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                with self._process_lock:
                    self._owned_processes[process.pid] = process
                try:
                    while True:
                        self._check_cancelled(cancel_event)
                        try:
                            stdout, stderr = process.communicate(timeout=0.5)
                            break
                        except subprocess.TimeoutExpired:
                            if progress_callback:
                                progress_callback(0.12, "Rendering image with mflux")
                finally:
                    try:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.communicate(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.communicate()
                    finally:
                        with self._process_lock:
                            self._owned_processes.pop(process.pid, None)
                if process.returncode != 0:
                    return {
                        "status": "error",
                        "error": (stderr or stdout or "mflux generation failed").strip(),
                        "repo_id": repo_id,
                    }
                return self._finish_output(temporary, output_path, "image", repo_id, "mflux",
                                           progress_callback, preview_callback, cancel_event)

            if backend_hint != "diffusers":
                raise ValueError(f"Unsupported image runtime: {backend_hint}.")
            pipe = self._load_pipeline(model_info, progress_callback, cancel_event)
            if progress_callback:
                progress_callback(0.12, "Model ready")

            def _step_callback(_pipe, step, _timestep, callback_kwargs):
                self._check_cancelled(cancel_event)
                if progress_callback:
                    frac = min(0.95, 0.12 + 0.80 * ((int(step) + 1) / max(1, steps)))
                    progress_callback(frac, f"Denoising {int(step) + 1}/{max(1, steps)}")
                return callback_kwargs

            call_kwargs = {
                "prompt": prompt,
                "num_inference_steps": max(1, steps),
                "generator": self._generator(model_info),
            }
            for key in ("guidance_scale", "width", "height"):
                if model_info.get(key) is not None:
                    call_kwargs[key] = model_info[key]
            result = self._call_pipeline(pipe, call_kwargs, _step_callback)
            images = getattr(result, "images", None)
            if images is None or len(images) == 0:
                return {"status": "error", "error": "Image pipeline returned no image."}
            images[0].save(temporary)
            return self._finish_output(temporary, output_path, "image", repo_id, "diffusers",
                                       progress_callback, preview_callback, cancel_event)
        except Exception as exc:
            return {"status": "error", "error": str(exc), "repo_id": repo_id}
        finally:
            Path(temporary).unlink(missing_ok=True)
            self.unload_model()

    def generate_video(
        self,
        model_info: Dict[str, Any],
        prompt: str,
        output_path: Optional[str] = None,
        frames: int = 49,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event=None,
    ) -> Dict[str, Any]:
        repo_id = str(model_info.get("repo_id") or "").strip()
        if not repo_id:
            return {"status": "error", "error": "Video model has no repository ID."}
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Video prompt is empty."}

        output_path = os.path.abspath(output_path or f"video_{uuid.uuid4().hex}.mp4")
        temporary = str(Path(output_path).with_name(f".smartai-{uuid.uuid4().hex}.mp4"))

        try:
            model_info = self._model_info(model_info)
            if Path(output_path).suffix.lower() != ".mp4":
                raise ValueError("Generated videos must be saved as MP4.")
            steps = max(1, int(model_info.get("inference_steps", 30) or 30))
            self._check_cancelled(cancel_event)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if progress_callback:
                progress_callback(0.03, "Preparing video pipeline")
            from diffusers.utils import export_to_video
            pipe = self._load_pipeline(model_info, progress_callback, cancel_event)
            if progress_callback:
                progress_callback(0.12, "Model ready")

            def _step_callback(_pipe, step, _timestep, callback_kwargs):
                self._check_cancelled(cancel_event)
                if progress_callback:
                    frac = min(0.92, 0.12 + 0.75 * ((int(step) + 1) / max(1, steps)))
                    progress_callback(frac, f"Generating video {int(step) + 1}/{max(1, steps)}")
                return callback_kwargs

            call_kwargs = {
                "prompt": prompt,
                "num_frames": max(2, int(frames)),
                "num_inference_steps": max(1, steps),
                "generator": self._generator(model_info),
            }
            for key in ("guidance_scale", "width", "height"):
                if model_info.get(key) is not None:
                    call_kwargs[key] = model_info[key]
            result = self._call_pipeline(pipe, call_kwargs, _step_callback)
            frame_sets = getattr(result, "frames", None)
            if frame_sets is None or len(frame_sets) == 0:
                return {"status": "error", "error": "Video pipeline returned no frames."}
            # Diffusers can return a batch as a 5D numpy array, a nested list of
            # PIL frames, or an unbatched sequence. Numpy truth testing is invalid.
            first = frame_sets[0]
            batched = getattr(frame_sets, "ndim", 0) == 5 or isinstance(first, (list, tuple)) or getattr(first, "ndim", 0) == 4
            frames_out = first if batched else frame_sets
            if len(frames_out) < 2:
                raise ValueError("Video pipeline returned fewer than two frames.")
            if progress_callback:
                progress_callback(0.95, "Encoding video")
            export_to_video(frames_out, temporary, fps=int(model_info.get("fps", 16)))
            return self._finish_output(temporary, output_path, "video", repo_id, "diffusers",
                                       progress_callback, preview_callback, cancel_event)
        except Exception as exc:
            return {"status": "error", "error": str(exc), "repo_id": repo_id}
        finally:
            Path(temporary).unlink(missing_ok=True)
            self.unload_model()
