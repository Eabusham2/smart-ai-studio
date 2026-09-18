"""Real lazy image/video generation backend for Smart AI Studio.

Selected media presets never enter the text Pro engine. This backend loads a
Diffusers-compatible pipeline only for the active generation request, writes a real
artifact, then releases it so 16 GB systems do not keep multiple media models resident.
"""
from __future__ import annotations

import gc
import os
import platform
import time
from typing import Any, Dict, Optional


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
            pipe = self._load_pipeline(repo_id)
            result = pipe(
                prompt=prompt,
                num_inference_steps=max(1, steps),
            )
            images = getattr(result, "images", None)
            if not images:
                return {"status": "error", "error": "Image pipeline returned no image."}
            images[0].save(output_path)
            if not os.path.isfile(output_path):
                return {"status": "error", "error": "Image pipeline did not write the output file."}
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
            from diffusers.utils import export_to_video
            pipe = self._load_pipeline(repo_id)
            result = pipe(
                prompt=prompt,
                num_frames=max(1, int(frames)),
                num_inference_steps=max(1, steps),
            )
            frame_sets = getattr(result, "frames", None)
            if not frame_sets:
                return {"status": "error", "error": "Video pipeline returned no frames."}
            frames_out = frame_sets[0] if isinstance(frame_sets, (list, tuple)) and frame_sets else frame_sets
            export_to_video(frames_out, output_path, fps=16)
            if not os.path.isfile(output_path):
                return {"status": "error", "error": "Video pipeline did not write the output file."}
            return {"status": "success", "path": output_path, "repo_id": repo_id}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "repo_id": repo_id}
        finally:
            self.unload_model()
