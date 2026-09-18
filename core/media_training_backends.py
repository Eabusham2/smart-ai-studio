"""Optional real media training backends.

These adapters wrap upstream trainers instead of reimplementing their optimization
loops. They are only advertised when the corresponding trusted trainer script exists.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from core.media_learning import register_media_training_backend


_TRAINING_REPO_CACHE: dict[str, str] = {}
_PIPELINE_CLASS_CACHE: dict[str, str] = {}


def _training_repo(info) -> str:
    """Resolve a differentiable base checkpoint for quantized/MLX/GGUF derivatives."""
    explicit = str(info.get("training_repo_id") or "").strip()
    if explicit:
        return explicit
    repo = str(info.get("repo_id") or "").strip()
    if not repo or os.path.exists(repo):
        return repo
    if repo in _TRAINING_REPO_CACHE:
        return _TRAINING_REPO_CACHE[repo]
    resolved = repo
    try:
        from huggingface_hub import HfApi
        meta = HfApi().model_info(repo)
        card = getattr(meta, "card_data", None)
        if hasattr(card, "to_dict"):
            card = card.to_dict()
        if not isinstance(card, dict):
            card = {}
        base = card.get("base_model")
        if isinstance(base, (list, tuple)):
            base = next((x for x in base if isinstance(x, str) and x.strip()), None)
        if not base:
            for tag in list(getattr(meta, "tags", None) or []):
                if str(tag).startswith("base_model:"):
                    base = str(tag).split(":", 1)[1].strip()
                    break
        if isinstance(base, str) and "/" in base:
            resolved = base
    except Exception:
        pass
    _TRAINING_REPO_CACHE[repo] = resolved
    return resolved


def _pipeline_class(info) -> str:
    explicit = str(info.get("pipeline_class") or "").strip()
    if explicit:
        return explicit
    repo = _training_repo(info)
    if not repo:
        return ""
    if repo in _PIPELINE_CLASS_CACHE:
        return _PIPELINE_CLASS_CACHE[repo]
    name = ""
    try:
        path = Path(repo).expanduser()
        if path.is_dir() and (path / "model_index.json").is_file():
            config = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download
            cfg = hf_hub_download(repo_id=repo, filename="model_index.json")
            config = json.loads(Path(cfg).read_text(encoding="utf-8"))
        name = str(config.get("_class_name") or "")
    except Exception:
        pass
    _PIPELINE_CLASS_CACHE[repo] = name
    return name


def _arch_blob(info) -> str:
    return " ".join((
        str(info.get("name") or ""),
        str(info.get("precision") or ""),
        str(info.get("training_family") or ""),
        _pipeline_class(info),
    )).lower()


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



# ---- Generic mflux image LoRA (FLUX.2 / Z-Image) --------------------------

def _mflux_image_matches(info):
    kind = str(info.get("model_type", "") or "").lower()
    if kind != "image":
        return False
    blob = " ".join(
        str(info.get(key, "") or "").lower()
        for key in ("repo_id", "name", "precision", "image_backend")
    )
    return any(marker in blob for marker in ("flux.2", "flux2", "flux-2", "z-image", "z_image", "mflux"))


def _mflux_image_available(info):
    if shutil.which("mflux-train") is None:
        return False
    blob = " ".join(
        str(info.get(key, "") or "").lower()
        for key in ("repo_id", "name", "precision")
    )
    # mflux currently trains Z-Image and FLUX.2 *base* variants. Distilled
    # FLUX.2 checkpoints can still generate but are not falsely advertised trainable.
    return (
        "z-image" in blob
        or "z_image" in blob
        or "flux2-klein-base" in blob
        or "flux.2 klein base" in blob
        or bool(info.get("mflux_training_model"))
    )


def mflux_image_factory(info, _media_engine, _audio_engine):
    exe = shutil.which("mflux-train")
    if exe is None:
        raise RuntimeError("mflux-train is not installed")

    blob = " ".join(
        str(info.get(key, "") or "").lower()
        for key in ("repo_id", "name", "precision")
    )
    configured = str(info.get("mflux_training_model", "") or "").strip()
    if configured:
        model_key = configured
    elif "z-image" in blob or "z_image" in blob:
        model_key = "z-image-turbo" if "turbo" in blob else "z-image"
    elif "flux2-klein-base" in blob or "flux.2 klein base" in blob:
        model_key = "flux2-klein-base-9b" if "9b" in blob else "flux2-klein-base-4b"
    else:
        raise RuntimeError(
            "This mflux checkpoint can generate, but its exact training base is not "
            "known to be adapter-compatible. Set mflux_training_model only when verified."
        )

    def train(samples, output_dir: Path, cancel_event=None):
        data = output_dir / "data"
        _copy_pairs(samples, data, "image")
        train_out = output_dir / "training"
        config = {
            "model": model_key,
            "data": str(data),
            "seed": 42,
            "steps": 9 if "z-image" in model_key else 40,
            "guidance": 0.0 if "z-image" in model_key else 1.0,
            "quantize": None,
            "low_ram": True,
            "gradient_checkpointing": True,
            "max_resolution": 768,
            "training_loop": {
                "num_epochs": max(1, min(8, len(samples) * 2)),
                "batch_size": 1,
            },
            "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
            "checkpoint": {
                "output_path": str(train_out),
                "save_frequency": 1,
            },
        }
        config_path = output_dir / "mflux_train.json"
        import json
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        lines = _run_cancelable([exe, "--config", str(config_path)], cancel_event)
        return {"trainer": "mflux-train", "model": model_key, "log_tail": lines[-12:]}

    return {
        "backend": "mflux_image_lora",
        "external_train": train,
        "verify_saved": _verify_safetensors,
    }


mflux_image_factory.available = _mflux_image_available
register_media_training_backend(
    "mflux_image_lora",
    mflux_image_factory,
    matcher=_mflux_image_matches,
)


# ---- Generic Wan2.1 LoRA through AI Toolkit -------------------------------

def _ai_toolkit_root():
    env = os.environ.get("AI_TOOLKIT_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "run.py").is_file():
            return root
    return None


def _wan21_matches(info):
    kind = str(info.get("model_type", "") or "").lower()
    blob = " ".join(
        str(info.get(key, "") or "").lower()
        for key in ("repo_id", "name", "precision")
    )
    return kind == "video" and any(marker in blob for marker in ("wan2.1", "wan-2.1", "wan2_1"))


def _wan21_available(_info):
    root = _ai_toolkit_root()
    if root is None:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def wan21_factory(info, _media_engine, _audio_engine):
    root = _ai_toolkit_root()
    if root is None:
        raise RuntimeError("Wan2.1 LoRA training requires AI_TOOLKIT_ROOT pointing to an ai-toolkit checkout")
    repo = str(info.get("repo_id") or "").strip()
    if not repo:
        raise RuntimeError("Wan trainer requires a model repository/path")

    def train(samples, output_dir: Path, cancel_event=None):
        data = output_dir / "data"
        _copy_pairs(samples, data, "video")
        steps = max(8, len(samples) * 4)
        name = "smart_ai_wan21"
        config = {
            "job": "extension",
            "config": {
                "name": name,
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": str(output_dir),
                    "device": "cuda:0",
                    "network": {"type": "lora", "linear": 8, "linear_alpha": 8},
                    "save": {
                        "dtype": "float16",
                        "save_every": steps,
                        "max_step_saves_to_keep": 1,
                    },
                    "datasets": [{
                        "folder_path": str(data),
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.0,
                        "cache_latents_to_disk": True,
                    }],
                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "timestep_type": "sigmoid",
                        "optimizer": "adamw8bit",
                        "lr": 1e-4,
                        "disable_sampling": True,
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": repo,
                        "arch": "wan21",
                        "quantize_te": True,
                    },
                }],
            },
        }
        import json
        config_path = output_dir / "wan21_train.yaml"
        # JSON is valid YAML and avoids adding a YAML dependency.
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        lines = _run_cancelable(
            [sys.executable, str(root / "run.py"), str(config_path)],
            cancel_event,
            cwd=root,
        )
        return {"trainer": "ai-toolkit-wan21", "log_tail": lines[-12:]}

    return {
        "backend": "wan21_lora",
        "external_train": train,
        "verify_saved": _verify_safetensors,
    }


wan21_factory.available = _wan21_available
register_media_training_backend(
    "wan21_lora",
    wan21_factory,
    matcher=_wan21_matches,
)
