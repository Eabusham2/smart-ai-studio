"""Run the selected Wan 2.2 Remix experts with the native GGUF media runtime.

The catalog/downloader resolves the complete pinned component set first. This
adapter never fetches model weights or substitutes a smaller Wan checkpoint.
stable-diffusion.cpp writes real decoded video frames to AVI; ffmpeg encodes
those frames to MP4, and the result is checked before replacing the destination.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile
import time


MODEL_REPO = "freeguyfroverrrr/Wan-2.2-Remix-GGUF"
MODEL_REVISION = "caa6f468c3a8e01a82cc73aa029fbaf0f239d9b4"
HIGH_NOISE_FILE = "T2V/v2.0/High/wan22RemixT2VI2V_t2vHighV20-Q4_K_M.gguf"
LOW_NOISE_FILE = "T2V/v2.0/Low/wan22RemixT2VI2V_t2vLowV20-Q4_K_M.gguf"
COMPONENTS = {
    "t5xxl": {
        "repo_id": "city96/umt5-xxl-encoder-gguf",
        "revision": "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7",
        "required_files": ["umt5-xxl-encoder-Q4_K_M.gguf"],
        "size_bytes": 3_655_145_312,
    },
    "vae": {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "revision": "d6258323e18f14c60d37899da97ca555f3118259",
        "required_files": ["split_files/vae/wan_2.1_vae.safetensors"],
        "size_bytes": 253_815_318,
    },
}
_FILE_SIZES = {
    HIGH_NOISE_FILE: 9_650_100_736,
    LOW_NOISE_FILE: 9_650_100_736,
    **{spec["required_files"][0]: spec["size_bytes"] for spec in COMPONENTS.values()},
}


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Wan video generation cancelled.")


def owned_process_ids():
    """The shared runner owns the native generator and MP4 encoder processes."""
    from core.gguf_image_runtime import owned_process_ids as native_process_ids
    return native_process_ids()


def _checked_file(directory, filename):
    if not directory:
        raise ValueError("The Wan component is not installed: " + filename)
    path = Path(directory).expanduser().absolute() / filename
    if not path.is_file() or path.stat().st_size != _FILE_SIZES[filename]:
        raise ValueError("The selected Wan component is missing or incomplete: " + filename)
    with path.open("rb") as stream:
        header = stream.read(9)
    if path.suffix == ".gguf":
        valid = header[:4] == b"GGUF"
    else:
        # A downloaded Git LFS pointer or an empty/padded file is not a VAE.
        length = int.from_bytes(header[:8], "little")
        valid = 2 < length < path.stat().st_size - 8 and header[8:9] == b"{"
    if not valid:
        raise ValueError("The selected Wan component has an invalid file header: " + filename)
    return path


def _component_files(info):
    if (info.get("repo_id") != MODEL_REPO
            or info.get("repo_revision") != MODEL_REVISION
            or set(info.get("download_required_files") or ()) != {HIGH_NOISE_FILE, LOW_NOISE_FILE}):
        raise ValueError("Wan generation requires the exact selected Remix T2V Q4 expert pair.")
    main = info.get("model_path") or info.get("path")
    files = {
        "high_noise": _checked_file(main, HIGH_NOISE_FILE),
        "low_noise": _checked_file(main, LOW_NOISE_FILE),
    }
    manifests = info.get("download_components") or []
    paths = info.get("component_paths") or {}
    selected = info.get("component_files") or {}
    for role, expected in COMPONENTS.items():
        matches = [entry for entry in manifests if isinstance(entry, dict) and entry.get("role") == role]
        if len(matches) != 1 or any(matches[0].get(key) != expected[key]
                                    for key in ("repo_id", "revision", "required_files")):
            raise ValueError("The selected Wan " + role + " component manifest does not match the supported runtime.")
        filename = expected["required_files"][0]
        files[role] = _checked_file(paths.get(role), filename)
        if role in selected and [str(Path(p).expanduser().absolute()) for p in selected[role]] != [str(files[role])]:
            raise ValueError("The selected Wan " + role + " file does not match its component manifest.")
    return files


def _integer(value, label, minimum):
    try:
        parsed = int(value)
        valid = not isinstance(value, bool) and float(value) == parsed and parsed >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Wan " + label + " must be an integer of at least " + str(minimum) + ".")
    return parsed


def _generation_options(info, frames):
    width = _integer(info.get("width", 832), "width", 16)
    height = _integer(info.get("height", 480), "height", 16)
    if width % 16 or height % 16:
        raise ValueError("Wan video dimensions must be multiples of 16.")
    frames = _integer(frames, "frame count", 5)
    if (frames - 1) % 4:
        raise ValueError("Wan video frame count must be 4n+1, such as 33, 49, or 81.")
    fps = _integer(info.get("fps", 16), "frame rate", 1)
    if fps > 120:
        raise ValueError("Wan video frame rate must not exceed 120.")
    steps = _integer(info.get("inference_steps", 18), "sampling steps", 2)
    high_steps = _integer(info.get("high_noise_steps", -1), "high-noise steps", -1)
    if high_steps == 0:
        raise ValueError("The selected Wan expert pair requires high-noise sampling steps.")
    cfg = float(info.get("guidance_scale", 3.5))
    shift = float(info.get("flow_shift", 3.0))
    if not math.isfinite(cfg) or cfg < 0 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Wan guidance and flow shift must be finite valid numbers.")
    seed = _integer(info.get("seed", 42), "seed", -1)
    return dict(width=width, height=height, frames=frames, fps=fps,
                steps=steps, high_steps=high_steps, cfg=cfg, shift=shift, seed=seed)


def _generation_command(executable, files, prompt, info, options, output, preview=None):
    command = [
        str(executable), "-M", "vid_gen",
        "--diffusion-model", str(files["low_noise"]),
        "--high-noise-diffusion-model", str(files["high_noise"]),
        "--t5xxl", str(files["t5xxl"]), "--vae", str(files["vae"]),
        "-p", prompt, "-n", str(info.get("negative_prompt") or ""),
        "-W", str(options["width"]), "-H", str(options["height"]),
        "--video-frames", str(options["frames"]), "--fps", str(options["fps"]),
        "--steps", str(options["steps"]), "--high-noise-steps", str(options["high_steps"]),
        "--sampling-method", "euler", "--cfg-scale", str(options["cfg"]),
        "--flow-shift", str(options["shift"]), "--seed", str(options["seed"]),
        "--offload-to-cpu", "-o", str(output),
    ]
    if info.get("vae_tiling", True):
        command.append("--vae-tiling")
    if preview is not None and options["fps"] >= 4:
        # This is an actual latent projection, clearly labelled as a sampling
        # preview. Final output always comes from the full selected Wan VAE.
        command += ["--preview", "proj", "--preview-path", str(preview), "--preview-interval", "2"]
    return command


def _preview_frame(path, callback, label):
    if callback is None or not Path(path).is_file():
        return
    try:
        import imageio.v2 as imageio
        from PIL import Image
        with imageio.get_reader(str(path), format="ffmpeg") as reader:
            frame = Image.fromarray(reader.get_data(0)).convert("RGB")
        frame.thumbnail((768, 768))
        callback({"kind": "video", "image": frame, "label": label})
    except Exception:
        # A preview may be read while the native process is replacing it, or its
        # widget may have closed. Neither invalidates the final generated video.
        pass


def _encode_mp4(source, destination, options, run_command, cancel_event):
    if not source.is_file() or not source.stat().st_size:
        raise RuntimeError("The native Wan runtime returned without a generated video.")
    import imageio_ffmpeg
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-hide_banner",
        "-loglevel", "error", "-xerror", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", str(destination),
    ]
    output = run_command(command, cancel_event=cancel_event,
                         label="Encoding Wan video", timeout=300)
    counts = re.findall(r"(?m)^frame=\s*(\d+)\s*$", str(output))
    if not counts or int(counts[-1]) != options["frames"]:
        raise RuntimeError("The Wan runtime did not produce the complete requested video frame sequence.")
    from core.media_engine import validate_media_file
    details = validate_media_file(str(destination), "video")
    if (details["width"] != options["width"] or details["height"] != options["height"]
            or not math.isclose(details["fps"], options["fps"], abs_tol=0.01)):
        raise RuntimeError("The generated Wan video does not match the requested dimensions or frame rate.")
    return {**details, "frames": int(counts[-1])}


def generate_wan_gguf_video(model_info, prompt, output_path, *, frames=33,
                            progress_callback=None, preview_callback=None, cancel_event=None):
    """Generate a checked MP4 from the installed exact Remix component set."""
    try:
        _check_cancelled(cancel_event)
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Video prompt is empty.")
        destination = Path(output_path).expanduser().absolute()
        if destination.suffix.lower() != ".mp4":
            raise ValueError("Generated videos must be saved as MP4.")
        options = _generation_options(model_info, frames)
        files = _component_files(model_info)
        from core.gguf_image_runtime import ensure_gguf_media_runtime, run_gguf_media_command
        executable = ensure_gguf_media_runtime(
            cancel_event=cancel_event,
            progress_callback=(lambda _fraction, message: progress_callback(0.05, message)) if progress_callback else None,
        )
        _check_cancelled(cancel_event)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".smartai-wan-", dir=destination.parent) as work:
            work = Path(work)
            native_video, encoded, preview = work / "generated.avi", work / "generated.mp4", work / "preview.avi"
            last_preview = [None, 0.0]

            def on_progress(fraction, message):
                _check_cancelled(cancel_event)
                if progress_callback:
                    progress_callback(0.10 + 0.79 * max(0.0, min(1.0, fraction)), message)
                now = time.monotonic()
                if preview_callback and now - last_preview[1] >= 1 and preview.is_file():
                    stamp = preview.stat().st_mtime_ns
                    if stamp != last_preview[0]:
                        last_preview[:] = [stamp, now]
                        _preview_frame(preview, preview_callback, "Wan sampling preview (latent projection)")

            if progress_callback:
                progress_callback(0.10, "Loading Wan Remix Q4 experts and generating video")
            command = _generation_command(executable, files, prompt, model_info, options,
                                          native_video, preview if preview_callback else None)
            run_gguf_media_command(command, cancel_event=cancel_event, progress_callback=on_progress,
                                   label="Generating Wan Remix video", timeout=None)
            _check_cancelled(cancel_event)
            if progress_callback:
                progress_callback(0.92, "Encoding generated Wan video")
            details = _encode_mp4(native_video, encoded, options, run_gguf_media_command, cancel_event)
            _check_cancelled(cancel_event)
            os.replace(encoded, destination)
        _preview_frame(destination, preview_callback, "Generated Wan video frame")
        if progress_callback:
            progress_callback(1.0, "Video complete")
        return {"status": "success", "path": str(destination), "backend": "wan_gguf",
                "repo_id": MODEL_REPO, **details}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "repo_id": model_info.get("repo_id", MODEL_REPO)}
