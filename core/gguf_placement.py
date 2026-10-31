"""Load-time placement of the exact GGUF weights across GPU and mapped CPU memory.

This is a conservative allocation estimate, not an SSD expert-cache implementation
or a performance benchmark. The native loader remains the compatibility authority.
"""
from __future__ import annotations

import gc
import math
import os
import platform
import shutil
import subprocess
import sys

from core.gguf_metadata import inspect_gguf


GIB = 1024 ** 3


def placement_hardware(*, owner_pid=None) -> dict:
    """Read live memory; unknown accelerator capacity safely selects CPU placement."""
    import psutil

    vm = psutil.virtual_memory()
    pids = {os.getpid()}
    if type(owner_pid) is int and owner_pid > 0:
        pids.add(owner_pid)
    rss = 0
    for pid in pids:
        rss += psutil.Process(pid).memory_info().rss
    unified = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    result = {"ram_available_bytes": int(vm.available), "ram_total_bytes": int(vm.total),
              "process_rss_bytes": int(rss), "unified_memory": unified,
              "gpu_free_bytes": 0, "gpu_total_bytes": 0, "gpu_backend": "cpu"}
    if unified:
        # CPU and GPU share this capacity; the planner never adds the pools.
        result.update(gpu_free_bytes=int(vm.available), gpu_total_bytes=int(vm.total * 0.65), gpu_backend="metal")
        return result
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            result.update(gpu_free_bytes=int(free), gpu_total_bytes=int(total), gpu_backend="cuda")
            return result
    except Exception:
        pass
    # Native GGUF workers need not import PyTorch solely to query NVIDIA VRAM.
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
            if not device or device == "-1":
                return result
            query = subprocess.run(
                [executable, "--id=" + device, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            free, total = [int(value.strip()) * 1024 ** 2 for value in query.stdout.strip().split(",")]
            if 0 <= free <= total:
                result.update(gpu_free_bytes=free, gpu_total_bytes=total, gpu_backend="cuda")
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return result


def plan_gguf_placement(metadata: dict, *, context_tokens: int, memory_limit_gb=None,
                        hardware=None, gpu_offload_supported=True, mmap_allowed=True,
                        logits_all=False, projector_bytes=0, max_gpu_layers=None) -> dict:
    """Choose the most GPU layers within the estimated live working-set budgets.

    KV stays FP16 and the caller's context is unchanged. For mapped CPU weights,
    allow a larger file than RAM only when a conservative layer working set fits.
    Memory pressure can still change after this snapshot; no hard RSS guarantee is
    made for memory-mapped files or backend workspaces.
    """
    if type(context_tokens) is not int or context_tokens <= 0:
        raise ValueError("GGUF context must be a positive integer.")
    if memory_limit_gb is not None and (
        isinstance(memory_limit_gb, bool) or not isinstance(memory_limit_gb, (int, float))
        or not math.isfinite(memory_limit_gb) or memory_limit_gb <= 0
    ):
        raise ValueError("The application memory limit must be positive and finite.")
    hw = dict(hardware if hardware is not None else placement_hardware())
    layers = list(metadata.get("layer_bytes") or [])
    count = metadata.get("layer_count", 0)
    embedding = metadata.get("embedding_length", 0)
    if not count or len(layers) != count or any(size <= 0 for size in layers) or not embedding:
        raise ValueError("This GGUF lacks the layer/embedding metadata needed for safe automatic placement.")
    available, total = int(hw.get("ram_available_bytes", 0)), int(hw.get("ram_total_bytes", 0))
    if available <= 0 or total <= 0:
        raise ValueError("Live host-memory capacity is unavailable; automatic GGUF placement cannot be verified.")
    host_budget = max(0, available - max(512 * 1024 ** 2, min(2 * GIB, total // 20)))
    if memory_limit_gb is not None:
        host_budget = min(host_budget, max(0, int(memory_limit_gb * GIB) - int(hw.get("process_rss_bytes", 0))))
    heads = int(metadata.get("head_count") or 1)
    kv_heads = int(metadata.get("head_count_kv") or heads)
    # Heterogeneous attention can list a head count per layer. Use its minimum
    # for the maximum possible head dimension, never a smaller KV estimate.
    key = int(metadata.get("key_length") or math.ceil(embedding / int(metadata.get("head_count_min") or heads)))
    value = int(metadata.get("value_length") or key)
    kv_per_layer = context_tokens * kv_heads * (key + value) * 2
    kv_total = kv_per_layer * count
    vocab = int(metadata.get("vocab_size") or 0)
    logits = (context_tokens if logits_all else min(context_tokens, 512)) * vocab * 4
    workspace = max(512 * 1024 ** 2, min(2 * GIB, embedding * 512 * 32))
    projector = max(0, int(projector_bytes))
    fixed_host = logits + workspace + projector
    gpu_free = max(0, int(hw.get("gpu_free_bytes", 0))) if gpu_offload_supported else 0
    gpu_total = max(0, int(hw.get("gpu_total_bytes", 0))) if gpu_offload_supported else 0
    gpu_budget = max(0, min(gpu_free, gpu_total) - max(512 * 1024 ** 2, gpu_total // 10) - projector)
    unified = bool(hw.get("unified_memory"))
    weights = int(metadata["tensor_bytes"])
    non_layer, output = int(metadata["non_layer_bytes"]), int(metadata["output_bytes"])
    upper = count + 1 if gpu_budget else 0
    if max_gpu_layers is not None:
        upper = min(upper, max(0, int(max_gpu_layers)))
    selected = None
    for number in range(upper, -1, -1):
        offloaded = min(number, count)
        gpu_weights = sum(layers[count - offloaded:]) + (output if number > count else 0)
        gpu_kv = offloaded * kv_per_layer
        gpu_allocation = math.ceil(gpu_weights * 1.08) + gpu_kv
        if number and gpu_allocation > gpu_budget:
            continue
        cpu_weights = max(0, weights - gpu_weights)
        # One complete CPU layer and the non-layer tensors are a conservative
        # hot-set estimate; the remaining mapped weights can be paged by the OS.
        hot_weights = (max(layers[:count - offloaded], default=0)
                       + max(0, non_layer - (output if number > count else 0)))
        cpu_kv = kv_total - gpu_kv
        minimum_host = fixed_host + cpu_kv + math.ceil((hot_weights if mmap_allowed else cpu_weights) * 1.08)
        full_host = fixed_host + cpu_kv + math.ceil(cpu_weights * 1.08)
        if unified:
            minimum_host += gpu_allocation
            full_host += gpu_allocation
        if minimum_host <= host_budget:
            selected = {"n_gpu_layers": number, "gpu_weight_bytes": gpu_weights,
                        "cpu_weight_bytes": cpu_weights, "gpu_kv_bytes": gpu_kv, "cpu_kv_bytes": cpu_kv,
                        "minimum_host_bytes_estimate": minimum_host, "resident_host_bytes_estimate": full_host,
                        "file_backed_paging": bool(mmap_allowed and full_host > host_budget)}
            break
    if selected is None:
        raise MemoryError(
            "The selected GGUF and unchanged context have no verified working-set placement within current memory. "
            "No model, quantization, or context was substituted. A compatible managed Strata runtime may support "
            "selective expert loading; ordinary GGUF mapping does not provide that guarantee."
        )
    mode = "GPU + mapped CPU weights" if selected["n_gpu_layers"] else "mapped CPU weights"
    if not mmap_allowed:
        mode = "GPU + CPU weights" if selected["n_gpu_layers"] else "CPU weights"
    message = f"Automatic placement: {selected['n_gpu_layers']}/{count + 1} GPU layers; {mode}. Context stays {context_tokens:,} tokens."
    if selected["file_backed_paging"]:
        message += " Weights exceed the resident budget and may page from disk; speed is unmeasured and depends on storage."
    return {"status": "planned", **selected, "context_tokens": context_tokens,
            "layer_count": count, "use_mmap": bool(mmap_allowed), "use_mlock": False,
            "kv_precision": "f16", "kv_bytes_estimate": kv_total, "logits_bytes_estimate": logits,
            "host_budget_bytes": host_budget, "gpu_budget_bytes": gpu_budget,
            "unified_memory": unified, "measured": False, "hard_rss_cap": False, "message": message}


def _allocation_failure(exc):
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or any(text in message for text in (
        "cuda out of memory", "cuda error: out of memory", "failed to allocate", "std::bad_alloc",
    ))


def load_gguf_with_placement(factory, *, model_path, n_ctx, n_gpu_layers=-1,
                             auto_memory_placement=True, memory_limit_gb=None,
                             gpu_offload_supported=True, owner_pid=None, metadata=None, **options):
    """Inspect first, then load with at most one allocation-only placement retry."""
    metadata = inspect_gguf(model_path) if metadata is None else metadata
    mmap_allowed = not bool(options.get("lora_path"))
    projector = int(options.pop("placement_projector_bytes", 0))
    plan = {"status": "manual", "n_gpu_layers": n_gpu_layers, "context_tokens": n_ctx,
            "use_mmap": mmap_allowed, "use_mlock": False, "measured": False,
            "message": "Using the explicitly selected native memory placement."}
    attempts = []
    for attempt in range(2):
        if auto_memory_placement:
            plan = plan_gguf_placement(
                metadata, context_tokens=n_ctx, memory_limit_gb=memory_limit_gb,
                hardware=placement_hardware(owner_pid=owner_pid), gpu_offload_supported=gpu_offload_supported,
                mmap_allowed=mmap_allowed, logits_all=bool(options.get("logits_all", False)),
                projector_bytes=projector, max_gpu_layers=(attempts[0]["n_gpu_layers"] // 2 if attempts else None),
            )
        try:
            model = factory(model_path=metadata["model_path"], n_ctx=n_ctx, n_gpu_layers=plan["n_gpu_layers"],
                            use_mmap=plan["use_mmap"], use_mlock=False, **options)
        except Exception as exc:
            retry = bool(auto_memory_placement and not attempt and plan["n_gpu_layers"] > 0 and _allocation_failure(exc))
            attempts.append({"n_gpu_layers": plan["n_gpu_layers"], "error": str(exc)[:300]})
            if not retry:
                raise RuntimeError(f"The selected GGUF could not load with its unchanged context: {exc}") from exc
        else:
            plan.update(status="applied", model_path=metadata["model_path"], attempts=attempts)
            return model, plan
        # The failed constructor and its exception have left scope before another
        # native allocation is attempted. Never retry a format/architecture error.
        gc.collect()
    raise RuntimeError("GGUF allocation retries were exhausted.")


def last_token_logits(model):
    """Read real last-token logits without retaining a context-by-vocabulary matrix."""
    ctx = getattr(model, "_ctx", None)
    getter = getattr(ctx, "get_logits_ith", None)
    if callable(getter):
        pointer = getter(-1)
        return [float(pointer[index]) for index in range(int(model.n_vocab()))]
    if bool(getattr(model, "_logits_all", False)):
        return model._scores[-1]
    raise RuntimeError("This llama.cpp runtime does not expose last-token logits.")
