"""Memory placement for the selected Transformers checkpoint, without recasting it.

Accelerate owns layer dispatch. These budgets leave live headroom for the OS,
activations and KV cache; they are planning limits, not hard allocation ceilings.
Nothing in this module imports a neural runtime merely to inspect a model.
"""
from __future__ import annotations

import math
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


GIB = 1024 ** 3


def estimate_transformers_kv_bytes(source: str, context_tokens: int) -> int | None:
    """Conservative full-attention KV estimate from a local checkpoint's config.

Unknown architectures remain unknown. This does not invent a layer count from a
model name or change the requested context/cache precision to make a model fit.
"""
    try:
        path = Path(source) / "config.json"
        if not path.is_file() or path.stat().st_size > 2 * 1024 ** 2:
            return None
        config = json.loads(path.read_text(encoding="utf-8"))
        config = config.get("text_config", config)
        layers = int(config.get("num_hidden_layers", config.get("n_layer", 0)))
        heads = int(config.get("num_attention_heads", config.get("n_head", 0)))
        kv_heads = int(config.get("num_key_value_heads", heads))
        hidden = int(config.get("hidden_size", config.get("n_embd", 0)))
        head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
        if min(layers, heads, kv_heads, head_dim, int(context_tokens)) <= 0:
            return None
        dtype = str(config.get("dtype", config.get("torch_dtype", "float32"))).lower()
        width = 2 if dtype in {"float16", "bfloat16", "torch.float16", "torch.bfloat16"} else 8 if "64" in dtype else 4
        return 2 * layers * kv_heads * head_dim * int(context_tokens) * width
    except (OSError, TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None


def transformers_memory_limits(
    torch_module: Any, max_process_gb: float | None = None, *, kv_bytes: int | None = None,
) -> dict:
    """Give Accelerate live GPU/CPU budgets, counting unified RAM only once."""
    import psutil

    memory = psutil.virtual_memory()
    available = int(memory.available)
    kv_reserve = max(0, int(kv_bytes or 0))
    reserve = max(GIB, int(memory.total * 0.05)) + kv_reserve
    host_budget = max(0, available - reserve)
    if max_process_gb is not None:
        if isinstance(max_process_gb, bool):
            raise ValueError("The application memory limit must be a positive finite number.")
        cap = float(max_process_gb)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("The application memory limit must be a positive finite number.")
        from core.training_memory import process_memory_bytes
        host_budget = min(host_budget, max(0, int(cap * GIB) - process_memory_bytes() - GIB // 4 - kv_reserve))
    if host_budget < GIB // 4:
        raise RuntimeError("Too little available RAM remains for model execution after memory headroom and the application cap.")

    budgets: dict[Any, int] = {"cpu": host_budget}
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        for index in range(int(cuda.device_count())):
            try:
                free, total = cuda.mem_get_info(index)
                gpu_budget = max(0, int(free) - max(GIB, int(total * 0.10)) - kv_reserve)
            except Exception:
                # Total capacity is not free memory. Do not place weights on an
                # unmeasured accelerator and risk overcommitting another app.
                continue
            if gpu_budget >= GIB // 4:
                budgets[index] = gpu_budget
        return budgets

    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        # Metal and CPU allocations share the same physical pool. Separate
        # full-size CPU/GPU budgets would incorrectly promise twice the RAM.
        metal_budget = int(host_budget * 0.75)
        budgets = {"mps": metal_budget, "cpu": host_budget - metal_budget}
    return budgets


def _device_name(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    return str(value).lower()


def describe_transformers_placement(model: Any, budgets: Mapping | None = None) -> dict:
    """Report actual dispatch rather than claiming a disk tier from requested flags."""
    raw = getattr(model, "hf_device_map", None)
    mapping = {str(key): _device_name(value) for key, value in raw.items()} if isinstance(raw, Mapping) else {}
    devices = set(mapping.values())
    if not devices:
        device = getattr(model, "device", None)
        if device is not None:
            devices.add(_device_name(device))
    offloaded = "disk" in devices or "meta" in devices or len(devices) > 1
    hook = getattr(model, "_hf_hook", None)
    offloaded = offloaded or bool(getattr(hook, "offload", False))
    message = "Automatic checkpoint placement: " + (", ".join(sorted(devices)) or "runtime did not report its devices") + "."
    if "disk" in devices:
        message += " Some layers stream from disk; throughput depends on storage and the selected model."
    if offloaded:
        message += " This dispatched model is available for inference only."
    return {
        "runtime": "transformers",
        "mode": "auto",
        "device_map": mapping,
        "devices": sorted(devices),
        "disk_offload": "disk" in devices,
        "inference_only": bool(offloaded),
        "max_memory_bytes": {str(key): int(value) for key, value in (budgets or {}).items()},
        "measured_speedup": False,
        "message": message,
    }


def transformers_input_device(model: Any) -> Any:
    """Never move prompts to a meta/disk placeholder in an offloaded model."""
    hook = getattr(model, "_hf_hook", None)
    hooks = [hook, *(getattr(hook, "hooks", ()) or ())]
    for item in hooks:
        device = getattr(item, "execution_device", None)
        if device is not None and _device_name(device) not in {"disk", "meta"}:
            return device
    device = getattr(model, "device", None)
    if device is not None and _device_name(device) not in {"disk", "meta"}:
        return device
    mapping = getattr(model, "hf_device_map", None)
    if isinstance(mapping, Mapping):
        for candidate in mapping.values():
            if _device_name(candidate) not in {"disk", "meta", "cpu"}:
                return candidate
    return "cpu"
