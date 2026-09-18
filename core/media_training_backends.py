"""Optional real media training backends.

These adapters wrap upstream trainers instead of reimplementing their optimization
loops. They are only advertised when the corresponding trusted trainer script exists.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from core.media_learning import register_media_training_backend


def _find_script(env_var: str, package: str, relatives: list[str]) -> Path | None:
    env = os.environ.get(env_var)
    candidates: list[Path] = []
    if env:
        root = Path(env).expanduser().resolve()
        candidates.extend(root / rel for rel in relatives)
    try:
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            pkg = Path(spec.origin).resolve()
            roots = [pkg.parent, pkg.parent.parent, pkg.parent.parent.parent]
            for root in roots:
                candidates.extend(root / rel for rel in relatives)
    except Exception:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _run_cancelable(command: list[str], cancel_event=None, cwd: Path | None = None):
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                raise RuntimeError("Media training cancelled")
            if proc.stdout is not None:
                line = proc.stdout.readline()
                if line:
                    lines.append(line.rstrip())
            time.sleep(0.05)
        if proc.stdout is not None:
            lines.extend(line.rstrip() for line in proc.stdout.readlines())
        if proc.returncode != 0:
            tail = "\n".join(lines[-40:])
            raise RuntimeError(f"Upstream trainer exited with code {proc.returncode}:\n{tail}")
        return lines
    finally:
        if proc.poll() is None:
            proc.kill()


def _verify_safetensors(directory: Path) -> bool:
    files = list(directory.rglob("*.safetensors"))
    if not files:
        return False
    try:
        from safetensors import safe_open
        for file in files:
            if file.stat().st_size <= 0:
                continue
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                if list(handle.keys()):
                    return True
    except Exception:
        return any(file.stat().st_size > 0 for file in files)
    return False


def _copy_pairs(samples, target: Path, kind: str):
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, sample in enumerate(samples):
        src = Path(sample["path"]).resolve()
        suffix = src.suffix.lower()
        dst = target / f"{index:04d}{suffix}"
        shutil.copy2(src, dst)
        (target / f"{index:04d}.txt").write_text(str(sample["caption"]), encoding="utf-8")
        copied.append((dst, str(sample["caption"])))
    return copied


# ---- Stable Audio 3 -------------------------------------------------------

def _stable_audio_script():
    return _find_script(
        "STABLE_AUDIO_3_ROOT",
        "stable_audio_3",
        ["scripts/train_lora.py", "../scripts/train_lora.py"],
    )


def _stable_audio_available(_info):
    return _stable_audio_script() is not None


def stable_audio3_factory(info, _media_engine, _audio_engine):
    script = _stable_audio_script()
    if script is None:
        raise RuntimeError("Stable Audio 3 LoRA trainer is not installed")
    variant = str(info.get("audio_variant") or "small-music")
    if variant not in ("small-music", "small-sfx"):
        raise RuntimeError(f"Unsupported Stable Audio 3 training variant: {variant}")

    def train(samples, output_dir: Path, cancel_event=None):
        data = output_dir / "data"
        _copy_pairs(samples, data, "audio")
        command = [
            sys.executable,
            str(script),
            "--model", variant,
            "--data_dir", str(data),
            "--rank", "8",
            "--adapter_type", "lora-xs",
            "--base_precision", "bf16",
            "--steps", str(max(8, len(samples) * 4)),
            "--save_dir", str(output_dir),
            "--checkpoint_every", str(max(8, len(samples) * 4)),
            "--demo_every", "1000000",
            "--logger", "csv",
            "--name", "smart_ai_media",
            "--num_workers", "0",
            "--batch_size", "1",
            "--lr", "0.0001",
        ]
        lines = _run_cancelable(command, cancel_event, cwd=script.parent.parent)
        return {"trainer": "stable-audio-3", "log_tail": lines[-12:]}

    return {
        "backend": "stable_audio3_lora",
        "external_train": train,
        "verify_saved": _verify_safetensors,
    }

def _stable_audio_matches(info):
    repo = str(info.get("repo_id", "") or "").lower()
    name = str(info.get("name", "") or "").lower()
    kind = str(info.get("model_type", "") or "").lower()
    variant = str(info.get("audio_variant", "") or "")
    blob = " ".join((repo, name))
    return kind == "audio" and (
        "stable-audio-3" in blob or variant in ("small-music", "small-sfx")
    )


stable_audio3_factory.available = _stable_audio_available
register_media_training_backend(
    "stable_audio3_lora",
    stable_audio3_factory,
    matcher=_stable_audio_matches,
)


# ---- CogVideoX ------------------------------------------------------------

def _cogvideo_script():
    return _find_script(
        "DIFFUSERS_SOURCE_ROOT",
        "diffusers",
        [
            "examples/cogvideo/train_cogvideox_lora.py",
            "../examples/cogvideo/train_cogvideox_lora.py",
            "../../examples/cogvideo/train_cogvideox_lora.py",
        ],
    )


def _cogvideo_available(_info):
    return _cogvideo_script() is not None and shutil.which("accelerate") is not None


def cogvideox_factory(info, _media_engine, _audio_engine):
    script = _cogvideo_script()
    accelerate = shutil.which("accelerate")
    if script is None or accelerate is None:
        raise RuntimeError("CogVideoX LoRA trainer requires a Diffusers source checkout and accelerate")
    repo = str(info.get("repo_id") or "zai-org/CogVideoX-2b")

    def train(samples, output_dir: Path, cancel_event=None):
        data = output_dir / "data"
        videos = data / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        video_lines, prompt_lines = [], []
        for index, sample in enumerate(samples):
            src = Path(sample["path"]).resolve()
            dst = videos / f"{index:04d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            video_lines.append(f"videos/{dst.name}")
            prompt_lines.append(str(sample["caption"]))
        (data / "videos.txt").write_text("\n".join(video_lines) + "\n", encoding="utf-8")
        (data / "prompts.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

        steps = max(8, len(samples) * 4)
        command = [
            accelerate, "launch", str(script),
            "--pretrained_model_name_or_path", repo,
            "--instance_data_root", str(data),
            "--caption_column", "prompts.txt",
            "--video_column", "videos.txt",
            "--height", "256",
            "--width", "384",
            "--fps", "8",
            "--max_num_frames", "17",
            "--train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--max_train_steps", str(steps),
            "--learning_rate", "0.0001",
            "--rank", "8",
            "--lora_alpha", "8",
            "--output_dir", str(output_dir),
            "--mixed_precision", "fp16",
            "--seed", "42",
        ]
        lines = _run_cancelable(command, cancel_event, cwd=script.parent)
        return {"trainer": "diffusers-cogvideox", "log_tail": lines[-12:]}

    return {
        "backend": "cogvideox_lora",
        "external_train": train,
        "verify_saved": _verify_safetensors,
    }

def _cogvideo_matches(info):
    repo = str(info.get("repo_id", "") or "").lower()
    name = str(info.get("name", "") or "").lower()
    kind = str(info.get("model_type", "") or "").lower()
    return kind == "video" and "cogvideox" in (repo + " " + name)


cogvideox_factory.available = _cogvideo_available
register_media_training_backend(
    "cogvideox_lora",
    cogvideox_factory,
    matcher=_cogvideo_matches,
)
