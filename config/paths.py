"""
Portable Path & Storage Manager.
Resolves portable data directory:
- If inside macOS .app bundle: stores data in <app_folder>/SmartAI_Data or ~/.smartai
- If running from local directory: stores data in ./data or ./SmartAI_Data
- Supports custom MLX model folder metadata inspection (config.json, weights.safetensors, etc.)
"""

import json
import os
import platform
import sys
from typing import Any, Dict, Optional, Tuple


def get_base_dir() -> str:
    """Returns the base directory of the project/application."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_portable_data_dir() -> str:
    """
    Returns the root directory for persistent writable data (memory.db, custom_models.json, custom LoRA weights).
    - If running inside a macOS .app bundle:
      Determines the directory containing the .app bundle (e.g. ~/Desktop or /Applications)
      and uses <app_parent>/SmartAI_Data if writable, or ~/.smartai.
    - If running portable/standalone:
      Uses ./data within the project root.
    """
    env_custom = os.getenv("SMARTAI_DATA_DIR")
    if env_custom:
        os.makedirs(env_custom, exist_ok=True)
        return env_custom

    base = get_base_dir()

    # Check if we are inside a .app bundle (e.g. SmartAI.app/Contents/Resources/app)
    if ".app/Contents/Resources" in base or ".app/Contents/MacOS" in base:
        app_parent = os.path.abspath(os.path.join(base, "../../../.."))
        if os.access(app_parent, os.W_OK):
            target = os.path.join(app_parent, "SmartAI_Data")
            os.makedirs(target, exist_ok=True)
            return target
        else:
            target = os.path.expanduser("~/.smartai")
            os.makedirs(target, exist_ok=True)
            return target

    # Normal local or portable folder execution
    target = os.path.join(base, "data")
    try:
        os.makedirs(target, exist_ok=True)
        return target
    except Exception:
        target = os.path.expanduser("~/.smartai")
        os.makedirs(target, exist_ok=True)
        return target


def get_custom_models_file() -> str:
    """Returns the persistent path for custom_models.json."""
    data_dir = get_portable_data_dir()
    return os.path.join(data_dir, "custom_models.json")


def inspect_mlx_model_folder(folder_path: str) -> Dict[str, Any]:
    """
    Inspects a local MLX or SafeTensors model directory.
    Extracts metadata from config.json (model architecture, parameter scale, quantization, context window).
    """
    if not os.path.isdir(folder_path):
        return {"valid": False, "error": f"Path `{folder_path}` is not a directory."}

    config_path = os.path.join(folder_path, "config.json")
    model_name = os.path.basename(os.path.abspath(folder_path))
    model_type = "MLX"
    raw_params = 7_000_000_000
    param_str = "7B"
    context_window = 32_768
    precision = "MLX Native"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            raw_mtype = cfg.get("model_type", "mlx")
            model_type = str(raw_mtype).capitalize()
            max_pos = cfg.get("max_position_embeddings") or cfg.get("seq_length") or cfg.get("max_sequence_length") or 32768
            context_window = int(max_pos)

            # Estimate parameters from hidden layers & hidden size
            hidden = cfg.get("hidden_size") or 4096
            layers = cfg.get("num_hidden_layers") or 32
            vocab = cfg.get("vocab_size") or 32000

            est_params = int((layers * 12 * (hidden ** 2)) + (2 * vocab * hidden))
            if est_params > 0:
                raw_params = est_params
                if est_params < 1_000_000_000:
                    param_str = f"{est_params / 1_000_000:.0f}M"
                elif est_params < 1_000_000_000_000:
                    param_str = f"{est_params / 1_000_000_000:.1f}B"
                else:
                    param_str = f"{est_params / 1_000_000_000_000:.2f}T"

            if "quantization" in cfg:
                q = cfg["quantization"]
                if isinstance(q, dict):
                    bits = q.get("bits", 4)
                    group_size = q.get("group_size", 64)
                    precision = f"{bits}-bit MLX (G{group_size})"
                else:
                    precision = f"{q} MLX"
            elif "1.58" in folder_path.lower() or "bitnet" in folder_path.lower():
                precision = "1.58-Bit BitLinear"
            elif "4bit" in folder_path.lower() or "4-bit" in folder_path.lower():
                precision = "4-bit MLX"
            elif "8bit" in folder_path.lower() or "8-bit" in folder_path.lower():
                precision = "8-bit MLX"
            elif "2bit" in folder_path.lower() or "2-bit" in folder_path.lower():
                precision = "2-bit MLX"
            else:
                precision = "MLX / SafeTensors"
        except Exception:
            pass

    weight_files = [
        f for f in os.listdir(folder_path)
        if f.endswith((".safetensors", ".npz", ".gguf", ".bin", ".pt"))
    ]
    has_weights = len(weight_files) > 0 or os.path.exists(config_path)

    return {
        "valid": True,
        "name": model_name,
        "path": os.path.abspath(folder_path),
        "model_type": model_type,
        "param_str": param_str,
        "raw_params": raw_params,
        "context_window": context_window,
        "precision": precision,
        "has_weights": has_weights,
        "weight_files_count": len(weight_files)
    }
