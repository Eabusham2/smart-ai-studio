"""Real lazy image/video generation backend for Smart AI Studio.

Selected media presets never enter the text Pro engine. This backend loads a
Diffusers-compatible pipeline only for the active generation request, writes a real
artifact, then releases it so 16 GB systems do not keep multiple media models resident.
"""
from __future__ import annotations

import gc
import os
import platform
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional


class MediaGenerationEngine:
    def __init__(self) -> None:
        self.active_pipeline = None
        self.active_repo_id = ""

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
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def _load_pipeline(self, repo_id: str):
        import torch
        from diffusers import DiffusionPipeline

        device = self._torch_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        pipe = DiffusionPipeline.from_pretrained(repo_id, **kwargs)
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

    def generate_image(
        self,
        model_info: Dict[str, Any],
        prompt: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        repo_id = str(model_info.get("repo_id") or "").strip()
        if not repo_id:
            return {"status": "error", "error": "Image model has no repository ID."}
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Image prompt is empty."}

        output_path = os.path.abspath(
            output_path or f"image_{int(time.time())}.png"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        steps = int(model_info.get("inference_steps", 4) or 4)

        try:
            if progress_callback:
                progress_callback(0.03, "Preparing image pipeline")
            backend_hint = str(model_info.get("image_backend") or "diffusers").lower()
            if backend_hint == "mflux":
                exe = shutil.which("mflux-generate-flux2")
                if not exe:
                    return {
                        "status": "error",
                        "error": "mflux CLI is not installed. Install the macOS media dependency and retry.",
                        "repo_id": repo_id,
                    }
                cmd = [
                    exe,
                    "--model", repo_id,
                    "--prompt", prompt,
                    "--steps", str(max(1, steps)),
                    "--seed", str(int(model_info.get("seed", 42) or 42)),
                    "--output", output_path,
                ]
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
                run = subprocess.run(cmd, capture_output=True, text=True)
                if run.returncode != 0:
                    return {
                        "status": "error",
                        "error": (run.stderr or run.stdout or "mflux generation failed").strip(),
                        "repo_id": repo_id,
                    }
                if not os.path.isfile(output_path):
                    return {
                        "status": "error",
                        "error": "mflux returned successfully but did not write the output image.",
                        "repo_id": repo_id,
                    }
                if progress_callback:
                    progress_callback(1.0, "Image complete")
                return {"status": "success", "path": output_path, "repo_id": repo_id}

            pipe = self._load_pipeline(repo_id)
            if progress_callback:
                progress_callback(0.12, "Model ready")

            def _step_callback(_pipe, step, _timestep, callback_kwargs):
                if progress_callback:
                    frac = min(0.95, 0.12 + 0.80 * ((int(step) + 1) / max(1, steps)))
                    progress_callback(frac, f"Denoising {int(step) + 1}/{max(1, steps)}")
                return callback_kwargs

            call_kwargs = {
                "prompt": prompt,
                "num_inference_steps": max(1, steps),
            }
            try:
                result = pipe(
                    **call_kwargs,
                    callback_on_step_end=_step_callback,
                )
            except TypeError:
                result = pipe(**call_kwargs)
            images = getattr(result, "images", None)
            if not images:
                return {"status": "error", "error": "Image pipeline returned no image."}
            images[0].save(output_path)
            if not os.path.isfile(output_path):
                return {"status": "error", "error": "Image pipeline did not write the output file."}
            if progress_callback:
                progress_callback(1.0, "Image complete")
            return {"status": "success", "path": output_path, "repo_id": repo_id}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "repo_id": repo_id}
        finally:
            self.unload_model()

    def generate_video(
        self,
        model_info: Dict[str, Any],
        prompt: str,
        output_path: Optional[str] = None,
        frames: int = 49,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        repo_id = str(model_info.get("repo_id") or "").strip()
        if not repo_id:
            return {"status": "error", "error": "Video model has no repository ID."}
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Video prompt is empty."}

        output_path = os.path.abspath(
            output_path or f"video_{int(time.time())}.mp4"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        steps = int(model_info.get("inference_steps", 30) or 30)

        try:
            if progress_callback:
                progress_callback(0.03, "Preparing video pipeline")
            from diffusers.utils import export_to_video
            pipe = self._load_pipeline(repo_id)
            if progress_callback:
                progress_callback(0.12, "Model ready")

            def _step_callback(_pipe, step, _timestep, callback_kwargs):
                if progress_callback:
                    frac = min(0.92, 0.12 + 0.75 * ((int(step) + 1) / max(1, steps)))
                    progress_callback(frac, f"Generating video {int(step) + 1}/{max(1, steps)}")
                return callback_kwargs

            call_kwargs = {
                "prompt": prompt,
                "num_frames": max(1, int(frames)),
                "num_inference_steps": max(1, steps),
            }
            try:
                result = pipe(
                    **call_kwargs,
                    callback_on_step_end=_step_callback,
                )
            except TypeError:
                result = pipe(**call_kwargs)
            frame_sets = getattr(result, "frames", None)
            if not frame_sets:
                return {"status": "error", "error": "Video pipeline returned no frames."}
            frames_out = frame_sets[0] if isinstance(frame_sets, (list, tuple)) and frame_sets else frame_sets
            if progress_callback:
                progress_callback(0.95, "Encoding video")
            export_to_video(frames_out, output_path, fps=16)
            if not os.path.isfile(output_path):
                return {"status": "error", "error": "Video pipeline did not write the output file."}
            if progress_callback:
                progress_callback(1.0, "Video complete")
            return {"status": "success", "path": output_path, "repo_id": repo_id}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "repo_id": repo_id}
        finally:
            self.unload_model()
